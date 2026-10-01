"""
quiz_generator.py - creates quizzes from the uploaded notes and grades them.

Flow:  topic -> retrieve chunks from ChromaDB -> Gemini writes questions (JSON)
       -> student answers -> grade -> (performance_tracker saves the result)
"""

import re

from modules import rag_engine
from utils.helpers import (
    MSG_NOT_FOUND, AppError, call_llm_json, format_context, format_sources, normalize_topic,
)

QUESTION_TYPES = ["Multiple Choice", "True/False", "Short Answer"]
DIFFICULTIES = ["Easy", "Medium", "Hard"]


# ----------------------------------------------------------------------
# Generating
# ----------------------------------------------------------------------
def _format_instructions(question_type):
    if question_type == "Multiple Choice":
        return ('Each question object: {"question": "...", "options": {"A": "...", "B": "...", '
                '"C": "...", "D": "..."}, "answer": "A", "explanation": "..."}. '
                'Exactly one option must be correct. "answer" is the letter A, B, C or D.')
    if question_type == "True/False":
        return ('Each question object: {"question": "a statement", "answer": "True", "explanation": "..."}. '
                '"answer" must be exactly "True" or "False". Mix true and false statements.')
    return ('Each question object: {"question": "...", "answer": "a short model answer (1-3 sentences)", '
            '"explanation": "..."}.')


def generate_quiz(topic, num_questions=5, difficulty="Medium", question_type="Multiple Choice"):
    """
    Build a quiz from the course material.
    Returns {"topic", "difficulty", "question_type", "questions": [...], "sources": [...]}.
    """
    docs = rag_engine.retrieve_for_topic(topic, k=8)
    if not docs:
        raise AppError(MSG_NOT_FOUND)

    topic_line = (f'The quiz topic is: "{topic.strip()}".' if topic and topic.strip()
                  else "No topic was given: pick the main topic covered by the context and name it in the \"topic\" field.")

    prompt = f"""You are creating a practice quiz for a college student.
Use ONLY the course material context below. Do not use outside knowledge.

COURSE MATERIAL CONTEXT:
{format_context(docs)}

{topic_line}
Create exactly {num_questions} {question_type} questions. Difficulty: {difficulty}.
{_format_instructions(question_type)}
Every "explanation" must briefly say why the answer is correct, based on the context.

If the context does not contain enough information about the topic, return exactly:
{{"error": "NOT_FOUND"}}

Otherwise return ONLY valid JSON (no markdown fences) in this shape:
{{"topic": "short topic name", "questions": [ ... ]}}"""

    data = call_llm_json(prompt, temperature=0.4)
    if not isinstance(data, dict) or data.get("error") == "NOT_FOUND":
        raise AppError(MSG_NOT_FOUND)

    questions = _validate_questions(data.get("questions", []), question_type)
    if not questions:
        raise AppError("The AI could not build a proper quiz this time. Please try again.")

    final_topic = normalize_topic(topic if topic and topic.strip() else data.get("topic", "General"))
    return {
        "topic": final_topic,
        "difficulty": difficulty,
        "question_type": question_type,
        "questions": questions[:num_questions],
        "sources": format_sources(docs),
    }


def _validate_questions(raw_questions, question_type):
    """Keep only well-formed questions (protects the app from odd AI output)."""
    clean = []
    for item in raw_questions if isinstance(raw_questions, list) else []:
        if not isinstance(item, dict) or not str(item.get("question", "")).strip():
            continue
        question = {
            "question": str(item["question"]).strip(),
            "type": question_type,
            "explanation": str(item.get("explanation", "")).strip(),
        }
        answer = str(item.get("answer", "")).strip()

        if question_type == "Multiple Choice":
            options = item.get("options", {})
            if not isinstance(options, dict):
                continue
            options = {str(k).strip().upper()[:1]: str(v).strip() for k, v in options.items()}
            if not all(letter in options for letter in "ABCD"):
                continue
            letter = answer.upper()[:1]
            if letter not in "ABCD" or not letter:
                continue
            question["options"] = {letter_: options[letter_] for letter_ in "ABCD"}
            question["answer"] = letter
        elif question_type == "True/False":
            if answer.lower() not in ("true", "false"):
                continue
            question["answer"] = answer.capitalize()
        else:
            if not answer:
                continue
            question["answer"] = answer
        clean.append(question)
    return clean


# ----------------------------------------------------------------------
# Grading
# ----------------------------------------------------------------------
def _keyword_overlap(reference, student):
    """Backup grading for short answers if the AI is unavailable."""
    words = lambda text: {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 3}
    ref, stu = words(reference), words(student)
    return bool(ref) and len(ref & stu) / len(ref) >= 0.5


def _grade_short_answers(quiz, answers, indexes):
    """Ask Gemini to mark short answers by comparing them with the reference answer."""
    items = [{
        "index": i,
        "question": quiz["questions"][i]["question"],
        "reference_answer": quiz["questions"][i]["answer"],
        "student_answer": answers.get(i, ""),
    } for i in indexes]

    prompt = f"""Mark these short answers. Compare each student answer with the reference answer
(which comes from the student's course notes). Mark "correct" true if the student's answer
captures the main idea, even if worded differently; otherwise false.

{items}

Return ONLY valid JSON: {{"results": [{{"index": 0, "correct": true, "feedback": "one short sentence"}}]}}"""
    try:
        data = call_llm_json(prompt, temperature=0.0)
        return {int(r["index"]): (bool(r.get("correct")), str(r.get("feedback", ""))) for r in data.get("results", [])}
    except (AppError, KeyError, TypeError, ValueError):
        return {}


def grade_quiz(quiz, answers):
    """
    answers: {question_index: student_answer_text}
      - Multiple choice: the letter "A".."D"
      - True/False: "True" or "False"
      - Short answer: free text
    Returns a result dict with score, total, percentage and per-question details.
    """
    questions = quiz["questions"]
    qtype = quiz["question_type"]

    short_feedback = {}
    if qtype == "Short Answer":
        answered = [i for i in range(len(questions)) if str(answers.get(i, "")).strip()]
        short_feedback = _grade_short_answers(quiz, answers, answered) if answered else {}

    details, score = [], 0
    for i, q in enumerate(questions):
        given = str(answers.get(i, "")).strip()
        feedback = ""
        if not given:
            correct = False
        elif qtype == "Short Answer":
            if i in short_feedback:
                correct, feedback = short_feedback[i]
            else:
                correct = _keyword_overlap(q["answer"], given)
        else:
            correct = given.lower() == q["answer"].lower()

        score += 1 if correct else 0
        details.append({
            "number": i + 1,
            "question": q["question"],
            "options": q.get("options"),
            "your_answer": given or "(not answered)",
            "correct_answer": q["answer"],
            "is_correct": correct,
            "explanation": q.get("explanation", ""),
            "feedback": feedback,
        })

    total = len(questions)
    return {
        "score": score,
        "total": total,
        "percentage": round(100.0 * score / total, 1) if total else 0.0,
        "details": details,
        "incorrect": [d for d in details if not d["is_correct"]],
    }


def incorrect_for_storage(result):
    """Small version of the wrong answers that we save in performance.json."""
    return [{
        "question": d["question"],
        "your_answer": d["your_answer"],
        "correct_answer": d["correct_answer"],
        "explanation": d["explanation"],
    } for d in result["incorrect"]]
