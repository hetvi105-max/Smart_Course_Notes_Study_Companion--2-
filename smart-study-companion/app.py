"""
app.py - Smart Course Notes & Study Companion (AI-Powered RAG Study Agent)

Start the app with:   streamlit run app.py

This file only contains the USER INTERFACE (pages, buttons, charts).
The real work is done by the files inside the `modules/` folder.
"""

import html
import traceback
import uuid

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Must be the first Streamlit command
st.set_page_config(page_title="Smart Study Companion", page_icon="📚", layout="wide")

from modules import (  # noqa: E402
    answer_evaluator, document_processor, performance_tracker, quiz_generator,
    rag_engine, study_agent,
)
from utils.helpers import (  # noqa: E402
    MSG_NO_KEY, AppError, ensure_dirs, friendly_error, has_api_key,
)

ensure_dirs()

PRIMARY, PURPLE, GREEN, RED, AMBER = "#5b5bd6", "#7c5cff", "#16a34a", "#ef4444", "#f59e0b"

# ======================================================================
# Styling (CSS)
# ======================================================================
CSS = """
<style>
html, body, [class*="css"] { font-family: 'Segoe UI', system-ui, -apple-system, Roboto, sans-serif; }
.stApp { background: #f5f6fb; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }
#MainMenu, footer { visibility: hidden; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e6e8f2; }
section[data-testid="stSidebar"] .stRadio > div { gap: 4px; }
section[data-testid="stSidebar"] .stRadio label {
    background: #f5f6fb; border-radius: 10px; padding: 8px 12px; width: 100%;
    border: 1px solid transparent; transition: all .15s;
}
section[data-testid="stSidebar"] .stRadio label:hover { background: #ececfb; border-color: #d8d9f5; }
section[data-testid="stSidebar"] .stRadio label:has(input:checked) {
    background: linear-gradient(90deg, #5b5bd6, #7c5cff); border-color: transparent;
}
section[data-testid="stSidebar"] .stRadio label:has(input:checked) p { color: #ffffff !important; font-weight: 600; }

.brand { font-size: 1.25rem; font-weight: 700; color: #3f3fb5; margin-bottom: 2px; }
.brand-sub { color: #6b7194; font-size: .85rem; margin-bottom: 14px; }

/* Hero + headers */
.hero { background: linear-gradient(120deg, #4f56d9 0%, #7c5cff 100%); color: #fff;
        padding: 28px 32px; border-radius: 20px; margin-bottom: 22px;
        box-shadow: 0 8px 24px rgba(91,91,214,.25); }
.hero h1 { color: #fff; margin: 0; font-size: 2rem; }
.hero p { margin: 4px 0 0 0; opacity: .92; font-size: 1.05rem; }
.page-title { font-size: 1.7rem; font-weight: 700; color: #262b57; margin: 0; }
.page-sub { color: #6b7194; margin: 2px 0 18px 0; }

/* Cards */
.metric-card { background: #fff; border-radius: 16px; padding: 18px 20px; border: 1px solid #e8eaf5;
               box-shadow: 0 2px 10px rgba(40,44,100,.06); }
.metric-icon { font-size: 1.5rem; }
.metric-value { font-size: 2rem; font-weight: 700; color: #3f3fb5; line-height: 1.2; }
.metric-label { color: #6b7194; font-size: .9rem; }
.section-title { font-size: 1.15rem; font-weight: 650; color: #262b57; margin: 6px 0 10px 0; }
.panel { background: #fff; border-radius: 16px; padding: 18px 20px; border: 1px solid #e8eaf5;
         box-shadow: 0 2px 10px rgba(40,44,100,.05); margin-bottom: 14px; }
.topic-row { margin-bottom: 12px; }
.topic-head { display: flex; justify-content: space-between; font-size: .93rem; color: #262b57; margin-bottom: 4px; }
.bar { background: #eceef8; border-radius: 99px; height: 9px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 99px; }
.activity { padding: 8px 0; border-bottom: 1px solid #eef0f8; font-size: .92rem; color: #33385f; }
.activity:last-child { border-bottom: none; }
.activity small { color: #8a90b3; display: block; }
.pill { display: inline-block; padding: 3px 10px; border-radius: 99px; font-size: .78rem; font-weight: 600; }
.pill-ok { background: #e3f6ea; color: #167a3b; }
.pill-bad { background: #fde8e8; color: #b42323; }
.source-line { color: #4a4f7c; font-size: .9rem; }
.big-score { font-size: 2.6rem; font-weight: 800; color: #3f3fb5; }

/* Buttons */
.stButton > button, .stFormSubmitButton > button { border-radius: 10px; font-weight: 600; padding: .5rem 1.1rem; }
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
    background: linear-gradient(90deg, #5b5bd6, #7c5cff); border: none; color: #fff; }
div[data-testid="stMetric"] { background: #fff; border: 1px solid #e8eaf5; border-radius: 14px; padding: 12px 16px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

PAGES = [
    "🏠 Dashboard",
    "📄 Course Materials",
    "🤖 Study Assistant",
    "🧭 AI Study Agent",
    "📝 Quiz Generator",
    "✍️ Answer Evaluator",
    "📊 Knowledge Dashboard",
]
TOOL_ICONS = {
    "understand_intent": "🧠", "retrieve_course_material": "🔎", "answer_question": "💬",
    "generate_quiz": "📝", "evaluate_answer": "✍️", "analyze_performance": "📊",
    "recommend_study_topic": "🎯",
}


# ======================================================================
# Small UI helpers
# ======================================================================
def go_to(page, quiz_topic=None):
    """Button callback: jump to another page (optionally pre-filling the quiz topic)."""
    st.session_state["page"] = page
    if quiz_topic is not None:
        st.session_state["quiz_topic"] = quiz_topic


def handle_error(error):
    """Show a friendly message; the technical details only go to the terminal."""
    if not isinstance(error, AppError):
        traceback.print_exc()
    st.error(friendly_error(error))


def try_call(kwargs_options, function, *args):
    """Call a Streamlit function trying newer/older argument names (works across versions)."""
    for kwargs in kwargs_options:
        try:
            return function(*args, **kwargs)
        except Exception:  # noqa: BLE001 - this Streamlit version does not know that argument
            continue
    return function(*args)


def show_chart(fig):
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=50, b=10),
                      font=dict(family="Segoe UI, sans-serif", color="#262b57"))
    try_call([{"width": "stretch"}, {"use_container_width": True}], st.plotly_chart, fig)


def show_table(df):
    try_call([{"hide_index": True, "width": "stretch"},
              {"hide_index": True, "use_container_width": True},
              {"hide_index": True}], st.dataframe, df)


def page_header(title, subtitle):
    st.markdown(f'<div class="page-title">{title}</div><div class="page-sub">{subtitle}</div>',
                unsafe_allow_html=True)


def api_key_banner():
    if not has_api_key():
        st.error(
            f"🔑 **{MSG_NO_KEY}**\n\n"
            "1. Get a free key at https://aistudio.google.com/apikey\n"
            "2. In the project folder, copy `.env.example` to a new file named `.env`\n"
            "3. Write `GOOGLE_API_KEY=your_key_here` inside `.env` and save\n"
            "4. Stop the app (Ctrl+C) and run `streamlit run app.py` again"
        )


def ensure_ready():
    """True if we have an API key and processed documents; otherwise show a friendly message."""
    try:
        rag_engine.check_ready()
        return True
    except AppError as error:
        if str(error) == MSG_NO_KEY:
            api_key_banner()
        else:
            st.warning(f"📄 {error}")
            st.button("Go to Course Materials", on_click=go_to, args=("📄 Course Materials",),
                      key=f"goto_materials_{st.session_state.get('page')}")
        return False


def metric_card(icon, value, label):
    return (f'<div class="metric-card"><div class="metric-icon">{icon}</div>'
            f'<div class="metric-value">{html.escape(str(value))}</div>'
            f'<div class="metric-label">{html.escape(label)}</div></div>')


def topic_bar(name, percentage, color):
    return (f'<div class="topic-row"><div class="topic-head"><span>{html.escape(name)}</span>'
            f'<span><b>{percentage:.0f}%</b></span></div>'
            f'<div class="bar"><div class="bar-fill" style="width:{min(percentage, 100)}%;'
            f'background:{color}"></div></div></div>')


def show_sources(sources):
    if sources:
        st.markdown("**Sources:**")
        for line in sources:
            st.markdown(f'<div class="source-line">{html.escape(line)}</div>', unsafe_allow_html=True)


def show_trace(trace):
    if trace:
        with st.expander("🔍 Agent reasoning — tools used", expanded=False):
            for step in trace:
                icon = TOOL_ICONS.get(step["tool"], "🔧")
                st.markdown(f"{icon} **{step['tool']}** — {step['detail']}")


# ======================================================================
# Quiz widgets (used by the Quiz page and the Agent page)
# ======================================================================
def quiz_form(quiz, quiz_id):
    """Draw the quiz. Returns the student's answers when Submit is pressed, else None."""
    answers = {}
    with st.form(f"quiz_form_{quiz_id}"):
        for i, q in enumerate(quiz["questions"]):
            st.markdown(f"**Q{i + 1}.** {q['question']}")
            key = f"{quiz_id}_q{i}"
            if quiz["question_type"] == "Multiple Choice":
                labels = [f"{letter}. {text}" for letter, text in q["options"].items()]
                choice = st.radio("Choose one", labels, index=None, key=key, label_visibility="collapsed")
                answers[i] = choice[0] if choice else ""
            elif quiz["question_type"] == "True/False":
                choice = st.radio("Choose one", ["True", "False"], index=None, key=key,
                                  label_visibility="collapsed")
                answers[i] = choice or ""
            else:
                answers[i] = st.text_area("Your answer", key=key, label_visibility="collapsed", height=90)
            st.write("")
        submitted = st.form_submit_button("✅ Submit Quiz", type="primary")
    return answers if submitted else None


def _answer_label(detail, letter_or_text):
    options = detail.get("options")
    if options and letter_or_text in options:
        return f"{letter_or_text}. {options[letter_or_text]}"
    return letter_or_text


def quiz_result_view(result):
    """Score, percentage, correct / incorrect answers with explanations."""
    correct_count = result["score"]
    wrong_count = result["total"] - result["score"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Score", f"{result['score']}")
    c2.metric("Total marks", f"{result['total']}")
    c3.metric("Percentage", f"{result['percentage']:.0f}%")
    c4.metric("Correct / Incorrect", f"{correct_count} / {wrong_count}")

    correct = [d for d in result["details"] if d["is_correct"]]
    wrong = result["incorrect"]
    if correct:
        st.markdown("#### ✅ Correct answers")
        for d in correct:
            st.markdown(f"**Q{d['number']}.** {d['question']}  \n"
                        f"Your answer: {_answer_label(d, d['your_answer'])}")
    if wrong:
        st.markdown("#### ❌ Incorrect answers")
        for d in wrong:
            with st.container(border=True):
                st.markdown(f"**Q{d['number']}.** {d['question']}")
                st.markdown(f"Your answer: {_answer_label(d, d['your_answer'])}  \n"
                            f"Correct answer: **{_answer_label(d, d['correct_answer'])}**")
                if d["feedback"]:
                    st.caption(d["feedback"])
                if d["explanation"]:
                    st.info(f"💡 {d['explanation']}")
    else:
        st.success("🎉 Perfect score! Every answer was correct.")


# ======================================================================
# PAGE 1 - Dashboard
# ======================================================================
def page_dashboard():
    st.markdown('<div class="hero"><h1>📚 Smart Study Companion</h1>'
                '<p>AI-Powered RAG Study Agent</p></div>', unsafe_allow_html=True)
    api_key_banner()

    stats = performance_tracker.overall_stats()
    average = f"{stats['average_score']:.0f}%" if stats["average_score"] is not None else "—"
    cols = st.columns(4)
    cards = [("📄", document_processor.count_processed(), "Documents Processed"),
             ("🧠", stats["questions_asked"], "Questions Asked"),
             ("📝", stats["quizzes_completed"], "Quizzes Completed"),
             ("🎯", average, "Average Score")]
    for col, (icon, value, label) in zip(cols, cards):
        col.markdown(metric_card(icon, value, label), unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([3, 2])
    with left:
        st.markdown('<div class="section-title">🧠 Knowledge Overview</div>', unsafe_allow_html=True)
        strong, weak = performance_tracker.strong_topics(), performance_tracker.weak_topics()
        if not strong and not weak:
            st.info("Complete a quiz to see your performance analytics.")
        else:
            a, b = st.columns(2)
            with a:
                st.markdown("**💪 Strong Topics**")
                html_bars = "".join(topic_bar(r["topic"], r["percentage"], GREEN) for r in strong[:5])
                st.markdown(f'<div class="panel">{html_bars or "<i>None yet</i>"}</div>', unsafe_allow_html=True)
            with b:
                st.markdown("**⚠️ Needs Improvement**")
                html_bars = "".join(topic_bar(r["topic"], r["percentage"], RED) for r in weak[:5])
                st.markdown(f'<div class="panel">{html_bars or "<i>Nothing — great job!</i>"}</div>',
                            unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section-title">🕒 Recent Activity</div>', unsafe_allow_html=True)
        activity = performance_tracker.recent_activity(6)
        if activity:
            rows = "".join(f'<div class="activity">{html.escape(a["description"])}'
                           f'<small>{html.escape(a["date"])}</small></div>' for a in activity)
            st.markdown(f'<div class="panel">{rows}</div>', unsafe_allow_html=True)
        else:
            st.info("No activity yet. Upload your notes to get started.")

    st.markdown('<div class="section-title">📈 Recent Quiz Scores</div>', unsafe_allow_html=True)
    quizzes = performance_tracker.recent_quizzes(8)
    if quizzes:
        df = pd.DataFrame(quizzes[::-1])
        df["label"] = df["topic"] + " (" + df["date"].str[5:10] + ")"
        fig = px.bar(df, x="label", y="percentage", text="percentage", range_y=[0, 105],
                     color_discrete_sequence=[PRIMARY], labels={"label": "", "percentage": "Score (%)"})
        fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
        show_chart(fig)
    else:
        st.info("Complete a quiz to see your performance analytics.")


# ======================================================================
# PAGE 2 - Course Materials
# ======================================================================
STATUS_ICONS = {"Processed": "✅ Processed", "Pending": "🕒 Ready to process", "Failed": "❌ Failed"}


def page_materials():
    page_header("📄 Course Materials", "Upload your PDFs, then process them so the AI can search them.")
    api_key_banner()

    uploader_key = f"uploader_{st.session_state.get('uploader_version', 0)}"
    uploaded = st.file_uploader("Upload PDF lecture notes, textbooks or handouts",
                                type=["pdf"], accept_multiple_files=True, key=uploader_key)
    for file in uploaded or []:
        try:
            document_processor.register_upload(file.name, file.getvalue())
        except OSError:
            st.error(f"Could not save {file.name}. Please try again.")

    entries = document_processor.get_registry()
    to_process = [e for e in entries if e["status"] != "Processed"]

    if st.button("⚙️ Process Documents", type="primary"):
        if not entries:
            st.warning("📄 Please upload course material first.")
        elif not has_api_key():
            api_key_banner()
        elif not to_process:
            st.info("All your documents are already processed. Nothing to do.")
        else:
            for entry in to_process:
                st.write(f"**Processing {entry['filename']}...**")
                bar = st.progress(0.0)
                try:
                    done = rag_engine.process_document(entry, progress=bar.progress)
                    bar.progress(1.0)
                    st.success(f"✅ Document processed successfully: {done['filename']} "
                               f"({done['pages']} pages, {done['chunks']} chunks)")
                    performance_tracker.log_activity("document", f"Processed {done['filename']}")
                except Exception as error:  # noqa: BLE001
                    st.error(f"{entry['filename']}: {friendly_error(error)}")
                    if not isinstance(error, AppError):
                        traceback.print_exc()
            entries = document_processor.get_registry()

    st.markdown('<div class="section-title">Your documents</div>', unsafe_allow_html=True)
    if not entries:
        st.info("No documents yet. Upload a PDF above to begin.")
        return

    df = pd.DataFrame([{
        "Filename": e["filename"],
        "Status": STATUS_ICONS.get(e["status"], e["status"]),
        "Pages": e["pages"] if e["pages"] is not None else "—",
        "Chunks": e["chunks"] if e["chunks"] is not None else "—",
        "Size (KB)": e["size_kb"],
        "Processed at": e["processed_at"] or "—",
    } for e in entries])
    show_table(df)

    failed = [e for e in entries if e["status"] == "Failed" and e.get("error")]
    for e in failed:
        st.warning(f"**{e['filename']}** — {e['error']}")

    with st.expander("🗑️ Remove documents"):
        for e in entries:
            c1, c2 = st.columns([4, 1])
            c1.write(e["filename"])
            if c2.button("Remove", key=f"remove_{e['doc_id']}"):
                rag_engine.remove_document(e["doc_id"])
                # a new uploader key clears the upload box so the file is not re-added
                st.session_state["uploader_version"] = st.session_state.get("uploader_version", 0) + 1
                st.rerun()


# ======================================================================
# PAGE 3 - Study Assistant (RAG question answering)
# ======================================================================
def render_message(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            show_sources(message["sources"])


def page_assistant():
    page_header("🤖 Study Assistant", "Ask questions. Answers come only from your uploaded notes.")
    history = st.session_state.setdefault("qa_history", [])
    if not ensure_ready():
        return

    top_left, top_right = st.columns([5, 1])
    top_left.caption("Try: “What is normalization?” · “Explain 2NF in simple words.” · "
                     "“What is the difference between BFS and DFS?”")
    if top_right.button("🧹 Clear chat"):
        st.session_state["qa_history"] = []
        st.rerun()

    for message in history:
        render_message(message)

    question = st.chat_input("Ask a question about your course material...")
    if question:
        user_msg = {"role": "user", "content": question}
        render_message(user_msg)
        history.append(user_msg)
        with st.chat_message("assistant"):
            with st.spinner("Searching your notes..."):
                try:
                    result = rag_engine.answer_question(question)
                    performance_tracker.record_question(question)
                    reply = {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
                except Exception as error:  # noqa: BLE001
                    if not isinstance(error, AppError):
                        traceback.print_exc()
                    reply = {"role": "assistant", "content": f"⚠️ {friendly_error(error)}", "sources": []}
            st.markdown(reply["content"])
            show_sources(reply["sources"])
        history.append(reply)


# ======================================================================
# PAGE 4 - AI Study Agent
# ======================================================================
EXAMPLE_PROMPTS = [
    "I don't understand normalization. Explain it and then test me.",
    "Quiz me on the main topic of my notes.",
    "What are my weak topics and what should I study next?",
]


def set_agent_prompt(text):
    st.session_state["agent_prompt"] = text


def render_agent_message(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        show_trace(message.get("trace"))
        show_sources(message.get("sources"))


def page_agent():
    page_header("🧭 AI Study Agent",
                "Tell the agent what you need. It picks the right tools, uses your notes and tracks your progress.")
    history = st.session_state.setdefault("agent_history", [])
    if not ensure_ready():
        return

    if not history:
        st.markdown("**Try one of these:**")
        for i, text in enumerate(EXAMPLE_PROMPTS):
            st.button(text, key=f"example_{i}", on_click=set_agent_prompt, args=(text,))

    for message in history:
        render_agent_message(message)

    prompt = st.chat_input("Tell the study agent what you need...")
    if st.session_state.get("agent_prompt"):
        prompt = st.session_state.pop("agent_prompt")

    if prompt:
        user_msg = {"role": "user", "content": prompt}
        render_agent_message(user_msg)
        history.append(user_msg)
        with st.chat_message("assistant"):
            with st.spinner("Agent is understanding your request and choosing tools..."):
                try:
                    out = study_agent.run_agent(prompt)
                    reply = {"role": "assistant", "content": out["reply_md"] or "Done.",
                             "trace": out["trace"], "sources": out["sources"]}
                    if out["quiz"]:
                        st.session_state["agent_quiz"] = {"id": uuid.uuid4().hex[:8], "quiz": out["quiz"]}
                except Exception as error:  # noqa: BLE001
                    if not isinstance(error, AppError):
                        traceback.print_exc()
                    reply = {"role": "assistant", "content": f"⚠️ {friendly_error(error)}"}
            st.markdown(reply["content"])
            show_trace(reply.get("trace"))
            show_sources(reply.get("sources"))
        history.append(reply)

    # A quiz created by the agent is shown here so the student can attempt it.
    pending = st.session_state.get("agent_quiz")
    if pending:
        st.markdown(f"#### 📝 Quiz: {pending['quiz']['topic']}")
        answers = quiz_form(pending["quiz"], pending["id"])
        if answers is not None:
            with st.spinner("Grading your quiz and updating your progress..."):
                try:
                    fin = study_agent.finish_quiz(pending["quiz"], answers, source="agent")
                except Exception as error:  # noqa: BLE001
                    handle_error(error)
                    return
            r = fin["result"]
            text = (f"### 📝 Quiz result: **{r['score']}/{r['total']} ({r['percentage']:.0f}%)**\n\n"
                    f"✅ Correct: {r['score']}  ·  ❌ Incorrect: {r['total'] - r['score']}\n\n")
            for d in r["incorrect"]:
                text += (f"- **Q{d['number']}.** {d['question']}  \n"
                         f"  Correct answer: **{_answer_label(d, d['correct_answer'])}**. {d['explanation']}\n")
            trace = [{"tool": "evaluate_answer", "detail": "Graded your quiz answers."},
                     {"tool": "analyze_performance", "detail": "Saved the result and updated your topic scores."}]
            if fin["recommendation"]:
                text += "\n### 🎯 Recommendation\n" + fin["recommendation"]["message_md"]
                trace.append({"tool": "recommend_study_topic",
                              "detail": "Score was below the target, so I created a revision plan."})
            else:
                text += "\n🎉 Great score! Try a harder quiz or a new topic next."
            history.append({"role": "assistant", "content": text, "trace": trace})
            st.session_state["agent_quiz"] = None
            st.rerun()


# ======================================================================
# PAGE 5 - Quiz Generator
# ======================================================================
def new_quiz():
    st.session_state["quiz_state"] = None


def page_quiz():
    page_header("📝 Quiz Generator", "Create a practice quiz from your notes, attempt it and get instant marks.")
    if not ensure_ready():
        return

    with st.container(border=True):
        topic = st.text_input("Topic (leave empty to let the AI pick a topic from your notes)",
                              key="quiz_topic", placeholder="e.g. Normalization")
        c1, c2, c3 = st.columns(3)
        count = c1.selectbox("Number of questions", [5, 10, 15])
        difficulty = c2.selectbox("Difficulty", quiz_generator.DIFFICULTIES, index=1)
        qtype = c3.selectbox("Question type", quiz_generator.QUESTION_TYPES)
        if st.button("🎯 Generate Quiz", type="primary"):
            with st.spinner("Reading your notes and writing questions..."):
                try:
                    quiz = quiz_generator.generate_quiz(topic, count, difficulty, qtype)
                    st.session_state["quiz_state"] = {
                        "id": uuid.uuid4().hex[:8], "quiz": quiz, "result": None, "recommendation": None}
                except Exception as error:  # noqa: BLE001
                    handle_error(error)

    state = st.session_state.get("quiz_state")
    if not state:
        return
    quiz = state["quiz"]

    if state["result"] is None:
        st.markdown(f"### Quiz: {quiz['topic']}")
        st.caption(f"{len(quiz['questions'])} questions · {quiz['difficulty']} · {quiz['question_type']}")
        answers = quiz_form(quiz, state["id"])
        if answers is not None:
            with st.spinner("Grading your answers..."):
                try:
                    fin = study_agent.finish_quiz(quiz, answers, source="quiz")
                    state["result"], state["recommendation"] = fin["result"], fin["recommendation"]
                except Exception as error:  # noqa: BLE001
                    handle_error(error)
            if state["result"] is not None:
                st.rerun()   # redraw the page to show the results
    else:
        st.markdown(f"### Results: {quiz['topic']}")
        quiz_result_view(state["result"])
        rec = state["recommendation"]
        if rec:
            with st.container(border=True):
                st.markdown("### 🎯 Personalised recommendation")
                st.markdown(rec["message_md"])
        else:
            st.success("Great work! Your score is above the target. Try a harder quiz or a new topic.")
        b1, b2, _ = st.columns([1, 1.4, 3])
        b1.button("🔁 New quiz", on_click=new_quiz)
        b2.button("📊 Knowledge Dashboard", on_click=go_to, args=("📊 Knowledge Dashboard",))


# ======================================================================
# PAGE 6 - Answer Evaluator
# ======================================================================
def page_evaluator():
    page_header("✍️ Answer Evaluator",
                "Write a descriptive answer. The AI marks it using your course material.")
    if not ensure_ready():
        return

    question = st.text_area("Question", placeholder="e.g. Explain the difference between 2NF and 3NF.", height=80)
    answer = st.text_area("Your answer", placeholder="Write your answer here...", height=200)
    marks = st.selectbox("Maximum marks", [5, 10, 15, 20], index=1)

    if st.button("✅ Evaluate My Answer", type="primary"):
        with st.spinner("Comparing your answer with your course notes..."):
            try:
                st.session_state["evaluation"] = answer_evaluator.evaluate_answer(question, answer, marks)
            except Exception as error:  # noqa: BLE001
                st.session_state["evaluation"] = None
                handle_error(error)

    result = st.session_state.get("evaluation")
    if result:
        st.markdown("---")
        left, right = st.columns([1, 3])
        left.markdown(f'<div class="panel"><div class="metric-label">Score</div>'
                      f'<div class="big-score">{result["score"]}/{result["total"]}</div>'
                      f'<div class="metric-label">Topic: {html.escape(result["topic"])}</div></div>',
                      unsafe_allow_html=True)
        with right:
            st.markdown("**Correct points:**")
            for p in result["correct_points"] or ["(none identified)"]:
                st.markdown(f"✓ {p}")
            st.markdown("**Missing points:**")
            for p in result["missing_points"] or ["Nothing important was missing."]:
                st.markdown(f"✗ {p}")
            if result["improvement"]:
                st.markdown("**Improvement:**")
                st.info(result["improvement"])
            show_sources(result["sources"])
        st.caption("✔ This result was saved to your performance history.")


# ======================================================================
# PAGE 7 - Knowledge Dashboard
# ======================================================================
def page_knowledge():
    page_header("📊 Knowledge Dashboard", "Your real performance, calculated from your saved quiz and written results.")
    stats = performance_tracker.overall_stats()
    if not stats["has_data"]:
        st.info("Complete a quiz to see your performance analytics.")
        st.button("Go to Quiz Generator", on_click=go_to, args=("📝 Quiz Generator",))
        return

    topics = performance_tracker.topic_stats()
    strong, weak = performance_tracker.strong_topics(), performance_tracker.weak_topics()

    c1, c2 = st.columns(2)
    with c1:   # 1. Overall performance
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=stats["average_score"],
            number={"suffix": "%"}, title={"text": "Overall performance"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": PRIMARY},
                   "steps": [{"range": [0, 50], "color": "#fde8e8"},
                             {"range": [50, 75], "color": "#fff3d6"},
                             {"range": [75, 100], "color": "#e3f6ea"}]}))
        show_chart(fig)
    with c2:   # 2. Topic-wise performance
        df = pd.DataFrame(topics)
        fig = px.bar(df, x="percentage", y="topic", orientation="h", color="status", text="percentage",
                     color_discrete_map={"Strong": GREEN, "Needs Improvement": RED},
                     range_x=[0, 105], title="Topic-wise performance",
                     labels={"percentage": "Score (%)", "topic": ""})
        fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
        fig.update_layout(legend_title_text="")
        show_chart(fig)

    c3, c4 = st.columns(2)
    with c3:   # 3. Quiz history
        history = pd.DataFrame(performance_tracker.quiz_history())
        if history.empty:
            st.info("Complete a quiz to see your quiz history.")
        else:
            history["attempt"] = range(1, len(history) + 1)
            fig = px.line(history, x="attempt", y="percentage", markers=True, range_y=[0, 105],
                          hover_data=["topic", "date"], title="Quiz history",
                          labels={"attempt": "Quiz number", "percentage": "Score (%)"})
            fig.update_traces(line_color=PRIMARY)
            fig.add_hline(y=performance_tracker.STRONG_THRESHOLD, line_dash="dot", line_color=GREEN,
                          annotation_text="Strong level")
            show_chart(fig)
    with c4:   # 4. Strong vs weak
        counts = pd.DataFrame({"status": ["Strong", "Needs Improvement"], "topics": [len(strong), len(weak)]})
        fig = px.pie(counts, names="status", values="topics", hole=0.55, title="Strong vs weak topics",
                     color="status", color_discrete_map={"Strong": GREEN, "Needs Improvement": RED})
        show_chart(fig)

    a, b = st.columns(2)
    with a:
        st.markdown('<div class="section-title">💪 Strong Topics</div>', unsafe_allow_html=True)
        bars = "".join(topic_bar(r["topic"], r["percentage"], GREEN) for r in strong)
        st.markdown(f'<div class="panel">{bars or "<i>No strong topics yet</i>"}</div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="section-title">⚠️ Needs Improvement</div>', unsafe_allow_html=True)
        bars = "".join(topic_bar(r["topic"], r["percentage"], RED) for r in weak)
        st.markdown(f'<div class="panel">{bars or "<i>Nothing — great job!</i>"}</div>', unsafe_allow_html=True)

    # Personalised recommendations
    st.markdown('<div class="section-title">🎯 Personalised Study Recommendations</div>', unsafe_allow_html=True)
    if not weak:
        st.success("All your topics are strong right now. Try a harder quiz or a new topic!")
    else:
        for row in weak[:3]:
            with st.container(border=True):
                st.markdown(f"Your performance in **{row['topic']}** is currently **{row['percentage']:.0f}%** "
                            f"({row['attempts']} attempt(s)). Revise it, then attempt a 5-question quiz.")
                st.button(f"📝 Quiz me on {row['topic']}", key=f"quizme_{row['topic']}",
                          on_click=go_to, args=("📝 Quiz Generator", row["topic"]))

        st.markdown("**Get a detailed revision plan from your course material**")
        choice = st.selectbox("Topic", [r["topic"] for r in weak], key="rec_topic")
        if st.button("🧠 Generate AI Recommendation", type="primary"):
            if not ensure_ready():
                return
            with st.spinner("Analysing your mistakes and your notes..."):
                try:
                    st.session_state["recommendation"] = study_agent.recommend_study_topic(choice)
                except Exception as error:  # noqa: BLE001
                    handle_error(error)
        rec = st.session_state.get("recommendation")
        if rec:
            with st.container(border=True):
                st.markdown(rec["message_md"])
                show_sources(rec.get("sources"))

    with st.expander("📋 All topic scores (table)"):
        table = pd.DataFrame([{"Topic": r["topic"], "Score (%)": r["percentage"], "Attempts": r["attempts"],
                               "Status": r["status"]} for r in topics])
        show_table(table)

    with st.expander("⚠️ Reset performance data"):
        st.write("This deletes all saved quiz and written-answer results.")
        confirm = st.checkbox("Yes, I want to delete my performance history")
        if st.button("Delete performance data", disabled=not confirm):
            performance_tracker.reset_all()
            st.session_state["recommendation"] = None
            st.rerun()


# ======================================================================
# Sidebar + page routing
# ======================================================================
st.session_state.setdefault("page", PAGES[0])

with st.sidebar:
    st.markdown('<div class="brand">📚 Smart Study Companion</div>'
                '<div class="brand-sub">AI-Powered RAG Study Agent</div>', unsafe_allow_html=True)
    st.radio("Navigation", PAGES, key="page", label_visibility="collapsed")
    st.markdown("---")
    if has_api_key():
        st.markdown('<span class="pill pill-ok">● Gemini API key found</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill pill-bad">● API key missing</span>', unsafe_allow_html=True)
    st.caption(f"📄 {document_processor.count_processed()} document(s) ready")

ROUTES = {
    "🏠 Dashboard": page_dashboard,
    "📄 Course Materials": page_materials,
    "🤖 Study Assistant": page_assistant,
    "🧭 AI Study Agent": page_agent,
    "📝 Quiz Generator": page_quiz,
    "✍️ Answer Evaluator": page_evaluator,
    "📊 Knowledge Dashboard": page_knowledge,
}
ROUTES[st.session_state["page"]]()
