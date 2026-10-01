"""
answer_evaluator.py - marks a student's written (descriptive) answer.

The marking is based on the student's own course notes:
    question -> retrieve relevant notes -> Gemini compares the answer with the notes
"""

from modules import performance_tracker, rag_engine
from utils.helpers import (
    MSG_NOT_FOUND, AppError, call_llm_json, format_context, format_sources, normalize_topic,
)


def _as_text_list(value):
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


def evaluate_answer(question, student_answer, max_marks=10, save=True):
    """
    Evaluate a written answer.
    Returns {"topic", "score", "total", "correct_points", "missing_points",
             "improvement", "sources", "percentage"}.
    """
    question = (question or "").strip()
    student_answer = (student_answer or "").strip()
    if not question:
        raise AppError("Please enter the question first.")
    if not student_answer:
        raise AppError("Please write your answer first.")

    docs = rag_engine.retrieve(question, k=6)
    if not docs:
        raise AppError(MSG_NOT_FOUND)

    prompt = f"""You are a fair college examiner. Mark the student's answer using ONLY the course material below
as the source of truth. Do not use outside knowledge.

COURSE MATERIAL CONTEXT:
{format_context(docs)}

QUESTION:
{question}

STUDENT'S ANSWER:
{student_answer}

Give a score from 0 to {max_marks} (whole number). Be fair: reward correct ideas even if worded differently.
List what the student got right and what important points from the course material are missing or wrong.
Give one practical improvement tip.

If the course material does not contain enough information to judge this question, return exactly:
{{"error": "NOT_FOUND"}}

Otherwise return ONLY valid JSON (no markdown fences):
{{"topic": "short topic name (1-3 words)",
  "score": 0,
  "correct_points": ["..."],
  "missing_points": ["..."],
  "improvement": "..."}}"""

    data = call_llm_json(prompt, temperature=0.1)
    if not isinstance(data, dict) or data.get("error") == "NOT_FOUND":
        raise AppError(MSG_NOT_FOUND)

    try:
        score = int(round(float(data.get("score", 0))))
    except (TypeError, ValueError):
        raise AppError("The AI returned an unexpected score. Please try again.")
    score = max(0, min(max_marks, score))

    result = {
        "topic": normalize_topic(data.get("topic", "General")),
        "score": score,
        "total": max_marks,
        "percentage": round(100.0 * score / max_marks, 1),
        "correct_points": _as_text_list(data.get("correct_points")),
        "missing_points": _as_text_list(data.get("missing_points")),
        "improvement": str(data.get("improvement", "")).strip(),
        "sources": format_sources(docs),
    }

    if save:
        performance_tracker.record_written(
            result["topic"], question, score, max_marks, result["missing_points"]
        )
    return result
