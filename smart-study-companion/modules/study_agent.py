"""
study_agent.py - the AI Study Agent.

A chatbot only answers. An AGENT understands what the student wants,
CHOOSES tools, RUNS them on the course data, and then decides the NEXT ACTION.

    USER REQUEST
      -> 1. understand intent (Gemini writes a small plan)
      -> 2. select tools from the tool list below
      -> 3. run each tool using the student's course material
      -> 4. (after the quiz) update performance
      -> 5. recommend what to study next

We use a reliable "planner + tool router" design instead of a free-running
autonomous agent, so the app stays stable and easy to explain in a viva.
"""

from modules import answer_evaluator, performance_tracker, quiz_generator, rag_engine
from utils.helpers import (
    MSG_NOT_FOUND, AppError, call_llm_json, format_context, format_sources, normalize_topic,
    require_api_key,
)

# The six tools the agent can use (name -> what it does)
TOOLS = {
    "retrieve_course_material": "Search the uploaded notes for passages about a topic.",
    "answer_question": "Explain a topic or answer a question using only the retrieved notes.",
    "generate_quiz": "Create a practice quiz from the notes.",
    "evaluate_answer": "Mark a written answer the student gives, using the notes.",
    "analyze_performance": "Look at saved quiz / written results and find strong and weak topics.",
    "recommend_study_topic": "Recommend what the student should revise next.",
}
CONTENT_TOOLS = {"answer_question", "generate_quiz", "evaluate_answer"}


# ----------------------------------------------------------------------
# Step 1 + 2: understand intent and choose tools
# ----------------------------------------------------------------------
def plan_request(message):
    """Ask Gemini to turn the student's message into a plan (a list of tools)."""
    tool_lines = "\n".join(f"- {name}: {desc}" for name, desc in TOOLS.items())
    prompt = f"""You are the planner of an AI study agent. Read the student's message and decide
which tools to run, in order.

TOOLS:
{tool_lines}

RULES:
- Explain a topic -> ["retrieve_course_material", "answer_question"]
- Explain AND test / quiz me afterwards -> ["retrieve_course_material", "answer_question", "generate_quiz"]
- Only wants a quiz -> ["retrieve_course_material", "generate_quiz"]
- Gives a question AND their own answer to be marked -> ["retrieve_course_material", "evaluate_answer"]
- Asks how they are doing / weak topics -> ["analyze_performance"]
- Asks what to study next -> ["analyze_performance", "recommend_study_topic"]
- Use only tool names from the list.

STUDENT MESSAGE:
\"\"\"{message}\"\"\"

Return ONLY valid JSON:
{{"reasoning": "one short sentence about what the student wants",
  "topic": "the main topic in 1-4 words, or empty string",
  "steps": ["tool_name", "..."],
  "question": "a clear standalone question to answer (for explain requests), or empty",
  "student_answer": "the student's own answer text if they gave one, else empty",
  "num_questions": 5,
  "difficulty": "Easy, Medium or Hard"}}"""
    try:
        plan = call_llm_json(prompt, temperature=0.0)
        if not isinstance(plan, dict):
            raise AppError("bad plan")
    except AppError as error:
        if "API key" in str(error) or "rate limit" in str(error):
            raise
        plan = heuristic_plan(message)   # backup so the agent never gets stuck
    return _clean_plan(plan, message)


def heuristic_plan(message):
    """Simple keyword routing used only if the AI planner fails."""
    text = message.lower()
    steps = []
    wants_quiz = any(w in text for w in ("quiz", "test me", "test my", "practice questions", "mcq"))
    wants_explain = any(w in text for w in ("explain", "what is", "what are", "don't understand",
                                            "dont understand", "difference", "teach", "how does", "define"))
    if any(w in text for w in ("weak", "progress", "performance", "how am i doing", "my score")):
        steps = ["analyze_performance"]
    elif any(w in text for w in ("what should i study", "what to study", "revise next", "recommend")):
        steps = ["analyze_performance", "recommend_study_topic"]
    elif wants_explain and wants_quiz:
        steps = ["retrieve_course_material", "answer_question", "generate_quiz"]
    elif wants_quiz:
        steps = ["retrieve_course_material", "generate_quiz"]
    else:
        steps = ["retrieve_course_material", "answer_question"]
    return {"reasoning": "Keyword-based routing.", "topic": "", "steps": steps, "question": message}


def _clean_plan(plan, message):
    steps = [s for s in plan.get("steps", []) if s in TOOLS]
    if not steps:
        steps = heuristic_plan(message)["steps"]
    # Content tools need the retrieval step first.
    if any(s in CONTENT_TOOLS for s in steps) and "retrieve_course_material" not in steps:
        steps.insert(0, "retrieve_course_material")
    ordered = []
    for step in steps:
        if step not in ordered:
            ordered.append(step)
    try:
        num = int(plan.get("num_questions", 5))
    except (TypeError, ValueError):
        num = 5
    difficulty = str(plan.get("difficulty", "Medium")).strip().capitalize()
    return {
        "reasoning": str(plan.get("reasoning", "")).strip(),
        "topic": str(plan.get("topic", "")).strip(),
        "steps": ordered,
        "question": str(plan.get("question", "")).strip() or message,
        "student_answer": str(plan.get("student_answer", "")).strip(),
        "num_questions": num if num in (5, 10, 15) else 5,
        "difficulty": difficulty if difficulty in quiz_generator.DIFFICULTIES else "Medium",
    }


# ----------------------------------------------------------------------
# Step 3: run the tools
# ----------------------------------------------------------------------
def run_agent(message):
    """
    Run the full agent for one student message.
    Returns {"plan", "trace", "reply_md", "sources", "quiz", "topic"}.
    """
    message = (message or "").strip()
    if not message:
        raise AppError("Please type a request for the agent.")
    require_api_key()

    plan = plan_request(message)
    topic = plan["topic"]
    trace = [{"tool": "understand_intent",
              "detail": f"{plan['reasoning'] or 'Analysed your request.'} "
                        f"→ plan: {' → '.join(plan['steps'])}"}]
    parts, sources, quiz, found = [], [], None, True

    for step in plan["steps"]:
        if not found and step in CONTENT_TOOLS:
            continue

        if step == "retrieve_course_material":
            docs = rag_engine.retrieve_for_topic(topic or plan["question"], k=6)
            if docs:
                files = sorted({d.metadata.get("filename", "?") for d in docs})
                trace.append({"tool": step, "detail": f"Found {len(docs)} relevant passages in {', '.join(files)}."})
                sources = format_sources(docs)
            else:
                trace.append({"tool": step, "detail": "No relevant passages found."})
                parts.append(f"⚠️ {MSG_NOT_FOUND}")
                found = False

        elif step == "answer_question":
            question = plan["question"] if not topic else f"Explain {topic} in simple words. {plan['question']}"
            result = rag_engine.answer_question(question)
            performance_tracker.record_question(message)
            if result["found"]:
                trace.append({"tool": step, "detail": "Wrote an explanation grounded in your notes."})
                parts.append("### 📘 Explanation\n" + result["answer"])
                sources = result["sources"]
            else:
                trace.append({"tool": step, "detail": "Notes did not contain enough information."})
                parts.append(f"⚠️ {MSG_NOT_FOUND}")
                found = False

        elif step == "generate_quiz":
            quiz = quiz_generator.generate_quiz(
                topic, plan["num_questions"], plan["difficulty"], "Multiple Choice")
            trace.append({"tool": step, "detail": f"Created a {len(quiz['questions'])}-question "
                                                  f"{quiz['difficulty']} quiz on {quiz['topic']}."})
            parts.append(f"### 📝 Quick check\nI made a **{len(quiz['questions'])}-question quiz on "
                         f"{quiz['topic']}**. Answer it below and press **Submit**. I will grade it, "
                         f"update your progress and tell you what to do next.")
            topic = topic or quiz["topic"]

        elif step == "evaluate_answer":
            if not plan["student_answer"]:
                trace.append({"tool": step, "detail": "No answer found in your message."})
                parts.append("To mark an answer, send both the question and your answer, e.g. "
                             "*\"Evaluate my answer. Question: … My answer: …\"* "
                             "(or use the **Answer Evaluator** page).")
            else:
                result = answer_evaluator.evaluate_answer(plan["question"], plan["student_answer"])
                trace.append({"tool": step, "detail": f"Marked your answer: {result['score']}/{result['total']}."})
                parts.append(_format_evaluation(result))
                sources = result["sources"]

        elif step == "analyze_performance":
            text = analyze_performance_text()
            trace.append({"tool": step, "detail": "Read your saved quiz and written-answer results."})
            parts.append("### 📊 Your performance\n" + text)

        elif step == "recommend_study_topic":
            rec = recommend_study_topic(topic or None)
            trace.append({"tool": step, "detail": "Created a personalised revision plan."})
            parts.append("### 🎯 Recommendation\n" + rec["message_md"])

    return {"plan": plan, "trace": trace, "reply_md": "\n\n".join(parts),
            "sources": sources if found else [], "quiz": quiz, "topic": topic}


def _format_evaluation(result):
    lines = [f"### ✍️ Answer evaluation — Score: **{result['score']}/{result['total']}**"]
    if result["correct_points"]:
        lines.append("**Correct points:**\n" + "\n".join(f"- ✓ {p}" for p in result["correct_points"]))
    if result["missing_points"]:
        lines.append("**Missing points:**\n" + "\n".join(f"- ✗ {p}" for p in result["missing_points"]))
    if result["improvement"]:
        lines.append(f"**Improvement:** {result['improvement']}")
    return "\n\n".join(lines)


# ----------------------------------------------------------------------
# Step 4 + 5: after the quiz -> update performance -> recommend next action
# ----------------------------------------------------------------------
def finish_quiz(quiz, answers, source="agent"):
    """Grade the quiz, save the result, and recommend revision if the score is low."""
    result = quiz_generator.grade_quiz(quiz, answers)
    performance_tracker.record_quiz(
        quiz["topic"], result["score"], result["total"],
        quiz_generator.incorrect_for_storage(result),
        quiz["difficulty"], quiz["question_type"], source=source,
    )
    recommendation = None
    if result["percentage"] < performance_tracker.LOW_SCORE_THRESHOLD:
        try:
            recommendation = recommend_study_topic(quiz["topic"])
        except AppError as error:
            recommendation = {"topic": quiz["topic"], "message_md": f"⚠️ {error}", "revise": []}
    return {"result": result, "recommendation": recommendation}


# ----------------------------------------------------------------------
# Tools that use saved performance
# ----------------------------------------------------------------------
def analyze_performance_text():
    """Plain-language summary calculated from real stored results."""
    stats = performance_tracker.topic_stats()
    overall = performance_tracker.overall_stats()
    if not stats:
        return "You have no results yet. Complete a quiz or a written answer to see your performance."
    lines = [f"Your average score is **{overall['average_score']:.0f}%** across "
             f"{overall['quizzes_completed']} quiz(zes) and {overall['written_completed']} written answer(s)."]
    strong, weak = performance_tracker.strong_topics(), performance_tracker.weak_topics()
    if strong:
        lines.append("**Strong topics:** " + ", ".join(f"{r['topic']} ({r['percentage']:.0f}%)" for r in strong))
    if weak:
        lines.append("**Needs improvement:** " + ", ".join(f"{r['topic']} ({r['percentage']:.0f}%)" for r in weak))
    return "\n\n".join(lines)


def recommend_study_topic(topic=None):
    """
    Recommend what to revise. Uses:
      quiz + written results, the weak topics, wrong answers, and the course material.
    Returns {"topic", "percentage", "revise": [...], "message_md": str}.
    """
    require_api_key()
    stats = performance_tracker.topic_stats()
    stat = None

    if topic and topic.strip():
        stat = performance_tracker.find_topic_stat(topic)
        topic_name = stat["topic"] if stat else normalize_topic(topic)
    else:
        weak = performance_tracker.weak_topics()
        if not stats:
            return {"topic": None, "percentage": None, "revise": [],
                    "message_md": "Complete a quiz or a written answer first, then I can recommend what to study next."}
        if not weak:
            return {"topic": None, "percentage": None, "revise": [],
                    "message_md": "🎉 All your topics are strong right now. Try a harder quiz or a new topic to keep improving."}
        stat = weak[0]
        topic_name = stat["topic"]

    docs = rag_engine.retrieve_for_topic(topic_name, k=6)
    if not docs:
        raise AppError(MSG_NOT_FOUND)

    wrong = performance_tracker.incorrect_for_topic(topic_name)
    missing = performance_tracker.missing_points_for_topic(topic_name)
    wrong_text = "\n".join(f"- {w['question']} (correct: {w['correct_answer']})" for w in wrong) or "none recorded"
    missing_text = "\n".join(f"- {m}" for m in missing) or "none recorded"
    prompt = f"""A student is revising "{topic_name}". Using ONLY the course material below,
list the 3 to 5 most important sub-topics or concepts they should revise.
Focus on the ones related to their mistakes when possible.

COURSE MATERIAL CONTEXT:
{format_context(docs)}

QUESTIONS THE STUDENT GOT WRONG:
{wrong_text}

POINTS MISSING IN THEIR WRITTEN ANSWERS:
{missing_text}

Return ONLY valid JSON: {{"revise": ["short concept name", "..."]}}"""
    data = call_llm_json(prompt, temperature=0.2)
    revise = [str(r).strip() for r in data.get("revise", []) if str(r).strip()][:5] if isinstance(data, dict) else []
    if not revise:
        raise AppError("Could not create a recommendation this time. Please try again.")

    numbered = "\n".join(f"{i}. {item}" for i, item in enumerate(revise, 1))
    if stat:
        opening = f"Your performance in **{topic_name}** is currently **{stat['percentage']:.0f}%**."
    else:
        opening = f"You have not been tested on **{topic_name}** yet."
    message = (f"{opening}\n\nYou should revise:\n\n{numbered}\n\n"
               f"After revision, attempt a **5-question quiz on {topic_name}**.")
    return {"topic": topic_name, "percentage": stat["percentage"] if stat else None,
            "revise": revise, "message_md": message, "sources": format_sources(docs)}
