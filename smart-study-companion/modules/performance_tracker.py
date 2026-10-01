"""
performance_tracker.py - remembers how the student is doing.

Everything is saved in ONE JSON file (data/performance/performance.json),
so the data is still there after you restart the app.

All numbers shown on the dashboards (averages, strong / weak topics, charts)
are calculated from this file. Nothing is hard-coded.
"""

import uuid

from utils.helpers import (
    PERF_DIR, ensure_dirs, normalize_topic, now_str, read_json, topic_key, write_json,
)

PERF_FILE = PERF_DIR / "performance.json"

STRONG_THRESHOLD = 75      # topic percentage at or above this = strong
LOW_SCORE_THRESHOLD = 70   # a single result below this triggers revision advice


def _empty():
    return {"quizzes": [], "written": [], "questions": [], "activity": []}


def load_data():
    ensure_dirs()
    data = read_json(PERF_FILE, _empty())
    if not isinstance(data, dict):
        data = _empty()
    for key in _empty():
        data.setdefault(key, [])
    return data


def save_data(data):
    write_json(PERF_FILE, data)


def _percentage(score, total):
    return round(100.0 * score / total, 1) if total else 0.0


# ----------------------------------------------------------------------
# Recording
# ----------------------------------------------------------------------
def log_activity(kind, description):
    data = load_data()
    data["activity"].append({"date": now_str(), "type": kind, "description": description})
    data["activity"] = data["activity"][-200:]
    save_data(data)


def record_question(question):
    data = load_data()
    data["questions"].append({"date": now_str(), "question": question})
    data["activity"].append({"date": now_str(), "type": "question", "description": f"Asked: {question[:80]}"})
    save_data(data)


def record_quiz(topic, score, total, incorrect_questions, difficulty="", question_type="", source="quiz"):
    """Save one finished quiz. `incorrect_questions` is a list of dicts."""
    topic = normalize_topic(topic)
    entry = {
        "id": uuid.uuid4().hex[:8],
        "date": now_str(),
        "topic": topic,
        "score": score,
        "total": total,
        "percentage": _percentage(score, total),
        "difficulty": difficulty,
        "question_type": question_type,
        "source": source,
        "incorrect_questions": incorrect_questions,
    }
    data = load_data()
    data["quizzes"].append(entry)
    data["activity"].append({
        "date": entry["date"], "type": "quiz",
        "description": f"Quiz on {topic}: {score}/{total} ({entry['percentage']:.0f}%)",
    })
    save_data(data)
    return entry


def record_written(topic, question, score, total, missing_points=None):
    topic = normalize_topic(topic)
    entry = {
        "id": uuid.uuid4().hex[:8],
        "date": now_str(),
        "topic": topic,
        "question": question,
        "score": score,
        "total": total,
        "percentage": _percentage(score, total),
        "missing_points": missing_points or [],
    }
    data = load_data()
    data["written"].append(entry)
    data["activity"].append({
        "date": entry["date"], "type": "written",
        "description": f"Written answer on {topic}: {score}/{total}",
    })
    save_data(data)
    return entry


def reset_all():
    save_data(_empty())


# ----------------------------------------------------------------------
# Reading / calculating
# ----------------------------------------------------------------------
def topic_stats():
    """
    Combine quiz + written results per topic.
    Returns a list of dicts sorted from weakest to strongest:
    {topic, percentage, attempts, earned, total, status}
    """
    data = load_data()
    grouped = {}
    for item in data["quizzes"] + data["written"]:
        key = topic_key(item["topic"])
        row = grouped.setdefault(key, {"topic": item["topic"], "earned": 0.0, "total": 0.0, "attempts": 0})
        row["earned"] += item["score"]
        row["total"] += item["total"]
        row["attempts"] += 1
        row["topic"] = item["topic"]  # keep the most recent spelling

    stats = []
    for row in grouped.values():
        row["percentage"] = _percentage(row["earned"], row["total"])
        row["status"] = "Strong" if row["percentage"] >= STRONG_THRESHOLD else "Needs Improvement"
        stats.append(row)
    stats.sort(key=lambda r: r["percentage"])
    return stats


def strong_topics():
    return sorted([r for r in topic_stats() if r["status"] == "Strong"],
                  key=lambda r: -r["percentage"])


def weak_topics():
    return [r for r in topic_stats() if r["status"] == "Needs Improvement"]


def overall_stats():
    """Numbers for the four dashboard cards."""
    data = load_data()
    results = data["quizzes"] + data["written"]
    average = round(sum(r["percentage"] for r in results) / len(results), 1) if results else None
    return {
        "questions_asked": len(data["questions"]),
        "quizzes_completed": len(data["quizzes"]),
        "written_completed": len(data["written"]),
        "average_score": average,
        "has_data": bool(results),
    }


def quiz_history():
    """All quizzes, oldest first (used for the history chart)."""
    return list(load_data()["quizzes"])


def recent_quizzes(n=5):
    return list(reversed(load_data()["quizzes"]))[:n]


def recent_activity(n=8):
    return list(reversed(load_data()["activity"]))[:n]


def find_topic_stat(topic):
    key = topic_key(topic)
    for row in topic_stats():
        if topic_key(row["topic"]) == key:
            return row
    return None


def incorrect_for_topic(topic, limit=10):
    """Questions the student got wrong on this topic (used for recommendations)."""
    key = topic_key(topic)
    wrong = []
    for quiz in reversed(load_data()["quizzes"]):
        if topic_key(quiz["topic"]) == key:
            wrong.extend(quiz.get("incorrect_questions", []))
    return wrong[:limit]


def missing_points_for_topic(topic, limit=10):
    key = topic_key(topic)
    points = []
    for item in reversed(load_data()["written"]):
        if topic_key(item["topic"]) == key:
            points.extend(item.get("missing_points", []))
    return points[:limit]
