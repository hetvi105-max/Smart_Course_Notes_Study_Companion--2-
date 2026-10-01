"""
document_processor.py - handles PDF files.

Steps done here (first half of the RAG pipeline):
    PDF upload  ->  extract text  ->  split into chunks (with metadata)

It also keeps a small "registry" (data/documents.json) that remembers
which PDFs were uploaded and processed, so we never process the same PDF twice.
"""

from pathlib import Path

from utils.helpers import (
    DATA_DIR, MSG_BAD_PDF, UPLOAD_DIR, AppError, ensure_dirs, file_hash,
    now_str, read_json, safe_filename, write_json,
)

CHUNK_SIZE = 1000      # characters per chunk
CHUNK_OVERLAP = 150    # characters shared between neighbouring chunks
REGISTRY_FILE = DATA_DIR / "documents.json"


# ----------------------------------------------------------------------
# Registry (list of documents)
# ----------------------------------------------------------------------
def get_registry():
    """Return the list of all known documents."""
    ensure_dirs()
    data = read_json(REGISTRY_FILE, [])
    return data if isinstance(data, list) else []


def save_registry(entries):
    write_json(REGISTRY_FILE, entries)


def find_entry(doc_id):
    for entry in get_registry():
        if entry["doc_id"] == doc_id:
            return entry
    return None


def update_entry(entry):
    """Add a new entry or replace the existing one with the same doc_id."""
    entries = [e for e in get_registry() if e["doc_id"] != entry["doc_id"]]
    entries.append(entry)
    entries.sort(key=lambda e: e.get("filename", "").lower())
    save_registry(entries)


def remove_entry(doc_id):
    """Delete the registry entry and the saved PDF file."""
    entry = find_entry(doc_id)
    if entry:
        try:
            Path(entry.get("path", "")).unlink(missing_ok=True)
        except OSError:
            pass
    save_registry([e for e in get_registry() if e["doc_id"] != doc_id])


def processed_documents():
    return [e for e in get_registry() if e.get("status") == "Processed"]


def count_processed():
    return len(processed_documents())


# ----------------------------------------------------------------------
# Saving uploads
# ----------------------------------------------------------------------
def register_upload(filename, data: bytes):
    """
    Save an uploaded PDF into data/uploaded_notes and add it to the registry.
    If the same file (same content) was uploaded before, nothing changes.
    Returns the registry entry.
    """
    ensure_dirs()
    doc_id = file_hash(data)
    existing = find_entry(doc_id)
    if existing and Path(existing["path"]).exists():
        return existing

    name = safe_filename(filename)
    target = UPLOAD_DIR / name
    if target.exists():  # same name but different content -> keep both
        target = UPLOAD_DIR / f"{target.stem}_{doc_id}{target.suffix}"
    target.write_bytes(data)

    entry = {
        "doc_id": doc_id,
        "filename": name,
        "path": str(target),
        "size_kb": round(len(data) / 1024, 1),
        "pages": None,
        "chunks": None,
        "status": "Pending",
        "error": "",
        "uploaded_at": now_str(),
        "processed_at": "",
    }
    update_entry(entry)
    return entry


# ----------------------------------------------------------------------
# Reading + splitting
# ----------------------------------------------------------------------
def load_and_split(entry):
    """
    Read the PDF page by page, then cut the text into overlapping chunks.
    Every chunk keeps: filename, page number, chunk index, doc_id.
    Returns (chunks, number_of_pages).
    """
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    try:
        pages = PyPDFLoader(entry["path"]).load()
    except Exception as error:  # noqa: BLE001 - corrupted / encrypted / not a PDF
        print(f"[PDF error] {entry['filename']}: {error}")
        raise AppError(MSG_BAD_PDF)

    if not pages:
        raise AppError(MSG_BAD_PDF)
    if not any(p.page_content and p.page_content.strip() for p in pages):
        raise AppError(
            "This PDF has no readable text (it may be a scanned image). "
            "Please upload a PDF with selectable text."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    chunks = [c for c in chunks if c.page_content.strip()]

    for index, chunk in enumerate(chunks):
        page_zero_based = chunk.metadata.get("page", 0)
        chunk.metadata = {
            "filename": entry["filename"],
            "page": int(page_zero_based) + 1,   # humans count pages from 1
            "chunk_index": index,
            "total_chunks": len(chunks),
            "doc_id": entry["doc_id"],
        }
    return chunks, len(pages)
