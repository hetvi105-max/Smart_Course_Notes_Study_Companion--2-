# 📚 Smart Course Notes & Study Companion
### AI-Powered RAG Study Agent

A college AI project. Upload your own PDF notes, ask questions, take quizzes generated from **your** material, get written answers marked, and let an **AI Study Agent** decide what you should do next.

---

## 1. Project Overview

Students study from many PDFs: lecture notes, textbooks, handouts. This app turns those PDFs into a personal AI tutor. It uses **Retrieval-Augmented Generation (RAG)**, so every answer, quiz question and mark comes from the student's own course material, not from the AI's general memory. It also tracks performance, finds weak topics and recommends what to revise.

## 2. Problem Statement

- Searching through hundreds of PDF pages for one explanation is slow.
- Normal chatbots may **hallucinate** (invent facts) and do not know *your* syllabus.
- Students rarely know which topics they are actually weak in.

## 3. Objectives

1. Let students upload and index their own PDFs.
2. Answer questions using **only** that material and show the source pages.
3. Generate and automatically grade quizzes (MCQ, True/False, Short answer).
4. Evaluate descriptive answers against the notes.
5. Track results and find weak topics from real data.
6. Provide an AI **agent** that picks the right action for each request.

## 4. Features

| Feature | What it does |
|---|---|
| 📄 Course Materials | Upload multiple PDFs, process them once, see pages/chunks/status, remove documents |
| 🤖 Study Assistant | RAG question answering with source pages; says *"I couldn't find sufficient information…"* when notes don't cover it |
| 🧭 AI Study Agent | Understands a request, selects tools, runs them, grades the quiz, updates performance and recommends next steps |
| 📝 Quiz Generator | 5/10/15 questions, Easy/Medium/Hard, MCQ / True-False / Short answer, automatic marking with explanations |
| ✍️ Answer Evaluator | Marks a written answer using the notes: score, correct points, missing points, improvement tip |
| 📊 Knowledge Dashboard | Plotly charts, strong/weak topics and personalised recommendations, all from stored results |
| 🏠 Dashboard | Documents processed, questions asked, quizzes completed, average score, recent activity |

## 5. Technology Stack

| Part | Technology |
|---|---|
| Language | Python 3.11 or 3.12 |
| Interface | Streamlit |
| LLM | Google Gemini API |
| RAG framework | LangChain (`langchain-google-genai`, `langchain-chroma`, `langchain-community`, `langchain-text-splitters`) |
| Vector database | ChromaDB (stored in the `vectorstore/` folder) |
| PDF reading | PyPDF (`PyPDFLoader`) |
| Embeddings | Google Gemini embeddings (`gemini-embedding-001`) |
| Data storage | JSON files (simple, no database setup) |
| Charts | Plotly |
| Secrets | python-dotenv (`.env` file) |

## 6. System Architecture

```
                         ┌───────────────────────────────┐
                         │        Streamlit UI (app.py)   │
                         └───────────────┬───────────────┘
        ┌───────────────┬───────────────┼───────────────┬────────────────┐
        ▼               ▼               ▼               ▼                ▼
 document_processor  rag_engine    quiz_generator  answer_evaluator  study_agent
   (PDF → chunks)   (Chroma + Gemini)  (quiz + grading) (written marks) (planner + tools)
        │               │               │               │                │
        └───────────────┴───────┬───────┴───────────────┴────────────────┘
                                ▼
                       performance_tracker  →  data/performance/performance.json
```

## 7. RAG Explanation (the pipeline in this project)

```
PDF Upload → Extract Text → Split into Chunks (1000 chars, 150 overlap)
   → Generate Embeddings (Gemini) → Store in ChromaDB (with filename + page metadata)

Question → Embed the question → Similarity search in ChromaDB → Top relevant chunks
   → Send chunks + question to Gemini ("use only this context")
   → Grounded answer + Sources (file and page)
```

If Gemini finds the answer is not in the retrieved chunks, the app shows: *"I couldn't find sufficient information about this topic in your uploaded course material."* and no invented answer.

## 8. AI Agent Explanation

The **AI Study Agent** page follows this loop:

```
USER REQUEST → AGENT UNDERSTANDS INTENT → AGENT SELECTS TOOLS → TOOLS USE COURSE DATA
   → AI GENERATES RESULT → PERFORMANCE IS UPDATED → AGENT RECOMMENDS NEXT ACTION
```

Six tools are available:

| Tool | Purpose |
|---|---|
| `retrieve_course_material` | Search ChromaDB for relevant passages |
| `answer_question` | Explain a topic from the notes |
| `generate_quiz` | Build a quiz from the notes |
| `evaluate_answer` | Mark a written answer |
| `analyze_performance` | Read saved results, find strong/weak topics |
| `recommend_study_topic` | Create a revision plan |

Example: *"I don't understand normalization. Explain it and then test me."* → the agent plans `retrieve_course_material → answer_question → generate_quiz`, shows the explanation, shows a quiz. When you submit, it grades the quiz, saves the score, and if the score is below 70% it runs `recommend_study_topic` to tell you what to revise.

The **"🔍 Agent reasoning — tools used"** panel under each agent reply shows exactly which tools were chosen. This is useful in your demo and viva.

We use a *planner + tool router* (Gemini writes a small JSON plan, Python runs the tools) instead of a free-running autonomous agent. It is much more reliable and easier to explain. If the planner's output is unusable, a keyword-based backup router takes over so the agent never gets stuck.

## 9. Folder Structure

```
smart-study-companion/
├── app.py                      ← Streamlit interface (run this)
├── requirements.txt
├── .env.example                ← template for your API key
├── .gitignore
├── README.md
├── .streamlit/config.toml      ← light theme colours
├── sample_notes/               ← a small sample PDF for testing
├── data/
│   ├── uploaded_notes/         ← your uploaded PDFs are saved here
│   └── performance/            ← performance.json (quiz + written results)
├── vectorstore/                ← ChromaDB files (created automatically)
├── modules/
│   ├── document_processor.py   ← PDF → chunks, document registry
│   ├── rag_engine.py           ← embeddings, ChromaDB, retrieval, answers
│   ├── quiz_generator.py       ← quiz creation + grading
│   ├── answer_evaluator.py     ← written answer marking
│   ├── performance_tracker.py  ← saves results, topic scores, weak topics
│   └── study_agent.py          ← planner, tools, recommendations
└── utils/
    └── helpers.py              ← paths, API key, Gemini calls, friendly errors
```

## 10. Installation on Windows (VS Code): step by step

**You need:** Python 3.11 or 3.12 from https://www.python.org/downloads/ (tick **"Add python.exe to PATH"** during installation) and VS Code.

1. **Extract** the ZIP. You get a folder called `smart-study-companion`.
2. Open VS Code → **File → Open Folder…** → choose `smart-study-companion`.
3. Open the terminal: **Terminal → New Terminal**.
4. Create a virtual environment (a private box for this project's packages):
   ```
   python -m venv venv
   ```
5. Activate it:
   ```
   venv\Scripts\activate
   ```
   You should now see `(venv)` at the start of the line.
   > If PowerShell says *"running scripts is disabled"*, run this once and try again:
   > `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
   > Or switch the terminal to **Command Prompt** (the ▾ next to the + in the terminal panel).
6. Install the packages (takes a few minutes):
   ```
   pip install -r requirements.txt
   ```

## 11. Gemini API Key Setup

1. Go to **https://aistudio.google.com/apikey** and sign in with a Google account.
2. Click **Create API key** and copy it. It is free to start.
3. In VS Code's file list, **copy `.env.example` and rename the copy to `.env`**
   (or in the terminal: `copy .env.example .env`).
4. Open `.env` and replace the text after the `=`:
   ```
   GOOGLE_API_KEY=AIza...your-real-key...
   ```
   No quotes, no spaces. Save the file.
5. **Never share `.env` or upload it to GitHub.** It is already listed in `.gitignore`.

Optional lines you can add to `.env` if Google renames models in the future:
```
GEMINI_MODEL=gemini-flash-latest
EMBEDDING_MODEL=gemini-embedding-001
```

## 12. How to Run

With `(venv)` active in the VS Code terminal:

```
streamlit run app.py
```

Your browser opens at `http://localhost:8501`. To stop the app press **Ctrl + C** in the terminal. Next time, you only need steps 3, 5 and this command (activate `venv`, then run).

## 13. How to Use

1. **📄 Course Materials** → upload one or more PDFs (try `sample_notes/DBMS_and_Graphs_Sample_Notes.pdf`) → click **Process Documents** and wait for ✅.
2. **🤖 Study Assistant** → ask *"What is normalization?"*. You get an answer and source pages.
3. **📝 Quiz Generator** → enter a topic (or leave it empty), choose count, difficulty, type → **Generate Quiz** → answer → **Submit Quiz**.
4. **✍️ Answer Evaluator** → type a question and your answer → **Evaluate My Answer**.
5. **🧭 AI Study Agent** → type a request such as *"I don't understand normalization. Explain it and then test me."*
6. **📊 Knowledge Dashboard** → see charts, weak topics and recommendations, all built from your saved results.

## 14. Testing Procedure

| # | Test | Steps | Expected result |
|---|---|---|---|
| 1 | Upload PDF | Upload the sample PDF → Process Documents | "✅ Document processed successfully", table shows 4 pages and a chunk count |
| 2 | Question in PDF | Ask "What is the difference between 2NF and 3NF?" | Correct answer + sources like `📄 …pdf — Page 2` |
| 3 | Question not in PDF | Ask "Who won the 2010 football World Cup?" | "I couldn't find sufficient information about this topic in your uploaded course material." |
| 4 | Generate quiz | Topic "Normalization", 5 questions, Medium, MCQ | 5 questions based on the notes, answers hidden |
| 5 | Submit quiz | Answer and submit | Score, total, percentage, correct/incorrect list with explanations |
| 6 | Written answer | Question "Explain the difference between 2NF and 3NF." + your answer | Score out of 10, correct points ✓, missing points ✗, improvement tip |
| 7 | Multiple quizzes | Do 2–3 quizzes on different topics | Knowledge Dashboard charts and strong/weak lists update |
| 8 | Study Agent | "I don't understand normalization. Explain it and then test me." | Reasoning panel shows tools; explanation + quiz; after submit, score is saved and a recommendation appears if the score is under 70% |
| 9 | Missing key | Rename `.env`, restart | Friendly "Google API key is missing…" message |
| 10 | No documents | Delete all documents, open Study Assistant | "Please upload course material first." |
| 11 | Restart | Stop and start the app | Documents and performance data are still there |

## 15. Troubleshooting

| Problem | Fix |
|---|---|
| `python` is not recognised | Reinstall Python and tick **Add python.exe to PATH**, then reopen VS Code. |
| PowerShell blocks `venv\Scripts\activate` | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use Command Prompt. |
| `pip install` fails on `chromadb` / `hnswlib` (needs C++ build tools) | Use Python **3.11 or 3.12** (not the newest version), run `python -m pip install --upgrade pip`, retry. If it still fails, install "Microsoft C++ Build Tools" from Microsoft. |
| "Google API key is missing" | Check the file is named exactly `.env` (not `.env.txt`), it is in the project's main folder, and the line is `GOOGLE_API_KEY=your_key`. Restart the app after editing. |
| "API key looks invalid" | Create a new key at aistudio.google.com/apikey and paste it again. |
| "Rate limit reached" | Free plans have limits. Wait a minute and retry. Large PDFs take longer because they are embedded in batches. |
| "Model name was not found" | Google renamed a model. Set `GEMINI_MODEL` / `EMBEDDING_MODEL` in `.env` (see section 11). If you change `EMBEDDING_MODEL`, remove your documents and process them again. |
| "PDF has no readable text" | The PDF is a scan (a picture). Use a PDF with selectable text. |
| Answers say "couldn't find sufficient information" for things in the PDF | Make sure the document shows ✅ Processed. Try wording closer to the notes. |
| Want to start completely fresh | Stop the app, delete the contents of `vectorstore/`, `data/uploaded_notes/`, `data/performance/` and the file `data/documents.json`. |

## 16. Future Improvements

- Support for DOCX and PPTX files and OCR for scanned PDFs
- Spaced-repetition reminders based on weak topics
- Flashcard generation and export of quizzes to PDF
- Multi-user login with separate notes and progress
- Hybrid search (keyword + vector) and re-ranking for better retrieval
- A fully autonomous LangChain/LangGraph agent with memory

## 17. Project Demonstration Flow (5–7 minutes)

| Time | Action | What to say |
|---|---|---|
| 0:00 | Open the **Dashboard** | "This is an AI-powered RAG study agent. Nothing is uploaded yet." |
| 0:30 | **Course Materials**: upload the PDF, click **Process Documents** | "The PDF is split into 1000-character chunks, converted to embeddings and stored in ChromaDB." |
| 1:30 | **Study Assistant**: ask "What is normalization?" | "It searches my notes, retrieves the best chunks and sends only those to Gemini." |
| 2:15 | Point at **Sources** | "Every answer shows the file and page. That proves it is grounded." |
| 2:30 | Ask something not in the notes | "It refuses to guess. That is how we avoid hallucination." |
| 3:00 | **Quiz Generator**: 5 MCQs on Normalization → answer → submit | "The quiz is created from my notes and graded automatically." |
| 4:00 | Show score + explanations | "Wrong answers come with explanations from the notes." |
| 4:30 | **Answer Evaluator**: 2NF vs 3NF | "It scores my written answer against the course material." |
| 5:15 | **Knowledge Dashboard** | "These charts come from saved results, not fixed numbers. Here is my weak topic and the recommendation." |
| 6:00 | **AI Study Agent**: "I don't understand normalization. Explain it and then test me." | "The agent understood the intent, chose tools, explained, generated a quiz, and after grading it updates my performance and recommends revision." |
| 6:45 | Open **Agent reasoning** panel | "A chatbot only replies. An agent decides which tools to use, acts on my data and chooses the next step." |

## 18. Simple AI Concept Explanations (for your viva)

**What is RAG?**
Retrieval-Augmented Generation. Instead of asking the AI to answer from memory, we first *retrieve* the relevant paragraphs from our own documents and then ask the AI to *generate* an answer using only those paragraphs. It is like an open-book exam where the AI is only allowed to use your notes.

**What are embeddings?**
An embedding turns text into a long list of numbers that captures its *meaning*. Sentences with similar meaning get similar numbers. "What is normalization?" and a paragraph explaining normalization end up close together even if they share few words.

**What is a vector database?**
A database built to store embeddings and quickly find the ones closest to a given embedding. It answers "which stored paragraphs mean something similar to this question?".

**What is ChromaDB?**
A free, open-source vector database that runs locally on your computer. We store it in the `vectorstore/` folder, so no cloud setup is needed and data survives restarts.

**What is LangChain?**
A Python framework that makes it easy to connect the parts of an AI application: document loaders, text splitters, embeddings, vector stores and language models. We use it to read PDFs, split text, talk to Gemini and use ChromaDB.

**What is Gemini?**
Google's family of large language models. In this project Gemini writes explanations, creates quizzes and marks answers (the "generation" part). Gemini embedding models create the embeddings.

**What is an AI Agent?**
A system that does not just answer: it understands a goal, chooses among tools, uses them, looks at the results and decides what to do next.

**How does the Study Agent work?**
Gemini reads the request and writes a small JSON plan (topic + list of tools). Python runs those tools in order (retrieve → explain → quiz). After the quiz it grades, saves the score, checks whether the score is low, and if so calls the recommendation tool.

**Why is RAG useful for course notes?**
Course notes are specific to your syllabus and teacher. RAG makes the AI use exactly those notes, shows the page it used, and can say "not found" instead of guessing.

**Why is this better than a normal chatbot?**
A normal chatbot answers from general training data, can hallucinate, has no idea of your notes, cannot quiz you from them and does not remember how you are performing. This project is grounded in your material, cites sources, tests you, tracks weak topics and acts as an agent.

## 19. Viva Questions and Answers

1. **Why RAG instead of fine-tuning a model?** Fine-tuning is expensive and must be repeated for every new PDF. RAG only needs indexing, works instantly for new notes and can show sources.
2. **Why do you split documents into chunks?** Models and embeddings work best on small focused pieces. Chunks of 1000 characters with 150 overlap keep each idea together and stop a sentence from being cut off between two chunks.
3. **What is the chunk overlap for?** It repeats the end of one chunk at the start of the next so context at the boundary is not lost.
4. **How do you prevent hallucination?** Gemini is told to use only the retrieved context and to reply `NOT_FOUND` otherwise; the app then shows the "couldn't find sufficient information" message.
5. **How does similarity search work?** The question is embedded into a vector; ChromaDB returns the stored chunks whose vectors are closest (cosine distance).
6. **What metadata do you store?** Filename, page number, chunk index, total chunks and document id. That is how we show sources.
7. **How is this an agent and not a chatbot?** It chooses tools based on intent, runs several steps, uses stored performance data and decides the next action (recommendation).
8. **How are weak topics found?** For each topic we add up marks earned and marks available across all quizzes and written answers. Below 75% is "Needs Improvement". Nothing is hard-coded.
9. **How are short answers graded?** Gemini compares the student's answer with the reference answer created from the notes; if Gemini is unavailable, a keyword-overlap backup is used.
10. **How is the API key protected?** It lives in `.env` (ignored by Git), is read with python-dotenv and is never shown in the UI.
11. **Why JSON instead of a database?** It is simple, needs no setup and is enough for one student. SQLite or PostgreSQL would be used for many users.
12. **Why not process a PDF twice?** Each file gets a content hash. If the same file is uploaded again it is recognised and skipped.
13. **What are the limitations?** Scanned PDFs need OCR, free API limits slow big documents, and answer quality depends on the retrieved chunks.
14. **How would you improve it?** Hybrid search, re-ranking, OCR, more file types, multi-user login and a LangGraph agent with memory.

---
*Built with Python, Streamlit, LangChain, ChromaDB and Google Gemini.*
