"""
rag_engine.py - the heart of the project (Retrieval-Augmented Generation).

    chunks -> embeddings -> ChromaDB                       (indexing)
    question -> similar chunks -> Gemini -> grounded answer (answering)
"""

import random
import time

from modules import document_processor as docs_module
from utils.helpers import (
    MSG_NO_DOCS, MSG_NOT_FOUND, MSG_NOT_PROCESSED, VECTOR_DIR, AppError,
    call_llm, ensure_dirs, format_context, format_sources, friendly_error,
    get_embed_model_name, is_rate_limit_error, now_str, require_api_key,
)

COLLECTION_NAME = "course_notes"
MAX_DISTANCE = 1.2    # only very unrelated chunks are dropped; Gemini decides the rest
BATCH_SIZE = 20       # chunks embedded per request (keeps free-tier limits happy)

_store_cache = {}


# ----------------------------------------------------------------------
# Vector store (ChromaDB) + embeddings (Google)
# ----------------------------------------------------------------------
def get_vectorstore():
    """Open (or create) the local ChromaDB folder using Gemini embeddings."""
    key = require_api_key()
    model = get_embed_model_name()
    cache_key = (key, model)
    if cache_key not in _store_cache:
        ensure_dirs()
        from langchain_chroma import Chroma
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        embeddings = GoogleGenerativeAIEmbeddings(model=model, google_api_key=key)
        _store_cache[cache_key] = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(VECTOR_DIR),
            collection_metadata={"hnsw:space": "cosine"},
        )
    return _store_cache[cache_key]


def delete_document_vectors(doc_id):
    """Remove all chunks of one document from ChromaDB."""
    store = get_vectorstore()
    try:
        store._collection.delete(where={"doc_id": doc_id})
    except Exception as error:  # noqa: BLE001
        print(f"[Chroma delete warning] {error}")


# ----------------------------------------------------------------------
# Indexing
# ----------------------------------------------------------------------
def index_chunks(chunks, doc_id, progress=None):
    """Embed chunks in small batches and store them in ChromaDB."""
    store = get_vectorstore()
    delete_document_vectors(doc_id)  # avoids duplicates when re-processing

    total = len(chunks)
    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        ids = [f"{doc_id}_{c.metadata['chunk_index']}" for c in batch]

        for attempt in range(4):
            try:
                store.add_documents(batch, ids=ids)
                break
            except Exception as error:  # noqa: BLE001
                print(f"[Embedding error, attempt {attempt + 1}] {error}")
                if attempt == 3:
                    raise AppError(friendly_error(error))
                time.sleep(20 if is_rate_limit_error(error) else 3)

        if progress:
            progress(min(start + BATCH_SIZE, total) / total)
        time.sleep(0.5)


def process_document(entry, progress=None):
    """
    Full indexing pipeline for one PDF:
    extract text -> split -> embed -> store -> update registry.
    Returns the updated registry entry.
    """
    require_api_key()
    try:
        chunks, page_count = docs_module.load_and_split(entry)
        index_chunks(chunks, entry["doc_id"], progress)
    except AppError as error:
        entry.update({"status": "Failed", "error": str(error)})
        docs_module.update_entry(entry)
        raise
    except Exception as error:  # noqa: BLE001
        print(f"[Processing error] {error}")
        entry.update({"status": "Failed", "error": friendly_error(error)})
        docs_module.update_entry(entry)
        raise AppError(friendly_error(error))

    entry.update({
        "status": "Processed", "pages": page_count, "chunks": len(chunks),
        "error": "", "processed_at": now_str(),
    })
    docs_module.update_entry(entry)
    return entry


def remove_document(doc_id):
    """Delete a document from ChromaDB, the registry and the disk."""
    try:
        delete_document_vectors(doc_id)
    except AppError:
        pass  # if there is no API key we can still remove the file + registry entry
    docs_module.remove_entry(doc_id)


# ----------------------------------------------------------------------
# Checks used by the pages
# ----------------------------------------------------------------------
def check_ready():
    """Raise a friendly error if the app is not ready to answer questions."""
    require_api_key()
    if not docs_module.get_registry():
        raise AppError(MSG_NO_DOCS)
    if docs_module.count_processed() == 0:
        raise AppError(MSG_NOT_PROCESSED)


# ----------------------------------------------------------------------
# Retrieval
# ----------------------------------------------------------------------
def retrieve(query, k=5):
    """Similarity search: return the k chunks most similar to the query."""
    check_ready()
    store = get_vectorstore()
    try:
        results = store.similarity_search_with_score(query, k=k)
    except Exception as error:  # noqa: BLE001
        print(f"[Retrieval error] {error}")
        raise AppError(friendly_error(error))
    return [doc for doc, distance in results if distance <= MAX_DISTANCE]


def sample_chunks(n=10):
    """Random chunks from all documents (used when the student gives no topic)."""
    check_ready()
    store = get_vectorstore()
    try:
        data = store.get(limit=500, include=["documents", "metadatas"])
    except Exception as error:  # noqa: BLE001
        raise AppError(friendly_error(error))
    from langchain_core.documents import Document

    items = [Document(page_content=text, metadata=meta or {})
             for text, meta in zip(data.get("documents", []), data.get("metadatas", []))]
    random.shuffle(items)
    return items[:n]


def retrieve_for_topic(topic, k=8):
    """Chunks about a topic; if the topic is empty, a random sample of the notes."""
    if topic and topic.strip():
        return retrieve(topic.strip(), k=k)
    return sample_chunks(k)


# ----------------------------------------------------------------------
# Answering
# ----------------------------------------------------------------------
ANSWER_SYSTEM = (
    "You are a friendly study assistant for a college student. "
    "You may ONLY use the COURSE MATERIAL CONTEXT given to you. "
    "Never use outside knowledge and never guess. "
    "If the context does not contain enough information to answer, reply with exactly: NOT_FOUND"
)


def answer_question(question, history_text=""):
    """
    Answer a student's question using only their uploaded notes.
    Returns {"answer": str, "sources": [str], "found": bool, "docs": [...]}.
    """
    question = (question or "").strip()
    if not question:
        raise AppError("Please type a question first.")

    docs = retrieve(question, k=5)
    if not docs:
        return {"answer": MSG_NOT_FOUND, "sources": [], "found": False, "docs": []}

    prompt = f"""COURSE MATERIAL CONTEXT:
{format_context(docs)}

{history_text}STUDENT QUESTION:
{question}

Instructions:
- Answer clearly in simple, student-friendly words, using only the context above.
- Use short paragraphs or bullet points. Add a tiny example only if the context has one.
- If the context does not contain the answer, reply with exactly: NOT_FOUND"""

    answer = call_llm(prompt, system=ANSWER_SYSTEM, temperature=0.2)
    if "NOT_FOUND" in answer:
        return {"answer": MSG_NOT_FOUND, "sources": [], "found": False, "docs": []}
    return {"answer": answer, "sources": format_sources(docs), "found": True, "docs": docs}
