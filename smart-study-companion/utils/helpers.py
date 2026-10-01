"""
helpers.py - small shared tools used by every other file.

Contains:
  * folder paths for the project
  * API key loading (from the .env file)
  * a safe way to call the Gemini model (with retries)
  * JSON reading / writing
  * friendly error messages (so students never see raw Python errors)
"""

import hashlib
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# ----------------------------------------------------------------------
# Folder paths (all inside the project folder, nothing outside)
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploaded_notes"
PERF_DIR = DATA_DIR / "performance"
VECTOR_DIR = BASE_DIR / "vectorstore"
ENV_FILE = BASE_DIR / ".env"

# ----------------------------------------------------------------------
# Standard messages (wording taken from the project requirements)
# ----------------------------------------------------------------------
MSG_NO_KEY = "Google API key is missing. Please add GOOGLE_API_KEY to your .env file."
MSG_NO_DOCS = "Please upload course material first."
MSG_NOT_PROCESSED = "Please process your course material before asking questions."
MSG_BAD_PDF = "Unable to process this PDF. Please upload a valid PDF."
MSG_NOT_FOUND = "I couldn't find sufficient information about this topic in your uploaded course material."

# You can change the models in the .env file if Google renames them.
# "gemini-flash-latest" is Google's alias that always points to the newest Flash model,
# so the project keeps working when older models are retired.
DEFAULT_LLM_MODEL = "gemini-flash-latest"
DEFAULT_EMBED_MODEL = "gemini-embedding-001"
# Tried automatically (in this order) if the chosen model is reported as "not found".
FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-3-flash-preview", "gemini-3.1-flash-lite"]


class AppError(Exception):
    """An error whose message is safe and friendly to show to the student."""


def ensure_dirs():
    """Create the data folders if they do not exist yet."""
    for folder in (DATA_DIR, UPLOAD_DIR, PERF_DIR, VECTOR_DIR):
        folder.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# API key handling
# ----------------------------------------------------------------------
def get_api_key():
    """Return the Gemini API key from .env, or '' if it is missing."""
    load_dotenv(ENV_FILE, override=False)
    key = os.getenv("GOOGLE_API_KEY", "").strip().strip('"').strip("'")
    if not key or key.startswith("PASTE_YOUR") or key.startswith("YOUR_API_KEY"):
        return ""
    return key


def has_api_key():
    return bool(get_api_key())


def require_api_key():
    key = get_api_key()
    if not key:
        raise AppError(MSG_NO_KEY)
    return key


def get_llm_model_name():
    load_dotenv(ENV_FILE, override=True)
    return os.getenv("GEMINI_MODEL", DEFAULT_LLM_MODEL).strip() or DEFAULT_LLM_MODEL


def get_embed_model_name():
    load_dotenv(ENV_FILE, override=True)
    return os.getenv("EMBEDDING_MODEL", DEFAULT_EMBED_MODEL).strip() or DEFAULT_EMBED_MODEL


# ----------------------------------------------------------------------
# Friendly errors
# ----------------------------------------------------------------------
def friendly_error(error):
    """Turn any exception into a short message a student can understand."""
    if isinstance(error, AppError):
        return str(error)
    text = str(error).lower()
    if "api key not valid" in text or "api_key_invalid" in text or "invalid api key" in text:
        return "Your Google API key looks invalid. Please check GOOGLE_API_KEY in your .env file."
    if "429" in text or "quota" in text or "resource_exhausted" in text or "rate limit" in text:
        return ("The Gemini API rate limit was reached. Please wait about a minute and try again "
                "(the free plan has limits).")
    if "404" in text and "model" in text:
        return ("The Gemini model name was not found. Check GEMINI_MODEL / EMBEDDING_MODEL "
                "in your .env file.")
    if "permission" in text or "403" in text:
        return "Gemini refused the request (permission denied). Check that your API key is enabled."
    if "timeout" in text or "timed out" in text or "connect" in text or "network" in text or "dns" in text:
        return "Could not reach the Gemini service. Please check your internet connection and try again."
    if "safety" in text or "blocked" in text:
        return "Gemini blocked this request for safety reasons. Try rephrasing your request."
    return "Something went wrong while talking to the AI service. Please try again in a moment."


def is_rate_limit_error(error):
    text = str(error).lower()
    return "429" in text or "quota" in text or "resource_exhausted" in text or "rate limit" in text


# ----------------------------------------------------------------------
# Talking to Gemini
# ----------------------------------------------------------------------
_llm_cache = {}
_working_model = {}   # remembers which model worked, so we do not retry broken ones


def get_llm(temperature=0.3, model=None):
    """Create (or reuse) the Gemini chat model."""
    key = require_api_key()
    model = model or get_llm_model_name()
    cache_key = (key, model, temperature)
    if cache_key not in _llm_cache:
        from langchain_google_genai import ChatGoogleGenerativeAI
        _llm_cache[cache_key] = ChatGoogleGenerativeAI(
            model=model, temperature=temperature, google_api_key=key
        )
    return _llm_cache[cache_key]


def message_to_text(response):
    """Newer LangChain versions may return a list of parts - always give back plain text."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "".join(parts).strip()
    return str(content).strip()


def is_model_not_found_error(error):
    text = str(error).lower()
    return ("404" in text or "not found" in text or "no longer available" in text
            or "is not supported" in text) and "model" in text


def call_llm(prompt, system=None, temperature=0.3, retries=3):
    """
    Send a prompt to Gemini and return the answer as text.
      * retries automatically if the free-tier rate limit is hit
      * switches to a backup model if the chosen model has been retired
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    require_api_key()
    messages = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(content=prompt))

    configured = get_llm_model_name()
    candidates = [_working_model.get("name", configured)]
    candidates += [m for m in [configured] + FALLBACK_MODELS if m not in candidates]

    last_error = None
    for model in candidates:
        llm = get_llm(temperature, model)
        for attempt in range(retries):
            try:
                answer = message_to_text(llm.invoke(messages))
                _working_model["name"] = model
                return answer
            except Exception as error:  # noqa: BLE001 - converted to a friendly message below
                last_error = error
                print(f"[Gemini error, model={model}, attempt {attempt + 1}] {error}")  # terminal only
                if is_model_not_found_error(error):
                    break            # try the next model
                if attempt < retries - 1:
                    time.sleep(15 * (attempt + 1) if is_rate_limit_error(error) else 2)
        else:
            break                    # real error (not "model not found") -> stop trying models
    raise AppError(friendly_error(last_error))


def extract_json(text):
    """Pull a JSON object/array out of the model's reply (it may add ```json fences)."""
    if not text:
        raise AppError("The AI returned an empty reply. Please try again.")
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    starts = [i for i in (cleaned.find("{"), cleaned.find("[")) if i != -1]
    if not starts:
        raise AppError("The AI returned an unexpected format. Please try again.")
    start = min(starts)
    end = max(cleaned.rfind("}"), cleaned.rfind("]"))
    if end <= start:
        raise AppError("The AI returned an unexpected format. Please try again.")
    try:
        return json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        raise AppError("The AI returned an unexpected format. Please try again.")


def call_llm_json(prompt, system=None, temperature=0.3, attempts=2):
    """Call Gemini and parse the answer as JSON (tries again once if the format is bad)."""
    last = None
    for _ in range(attempts):
        try:
            return extract_json(call_llm(prompt, system=system, temperature=temperature))
        except AppError as error:
            last = error
            if "format" not in str(error) and "empty" not in str(error):
                raise
    raise last


# ----------------------------------------------------------------------
# JSON files
# ----------------------------------------------------------------------
def read_json(path, default):
    path = Path(path)
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return default


def write_json(path, data):
    """Write safely: save to a temp file first, then replace (avoids corrupted files)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    os.replace(temp, path)


# ----------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def file_hash(data: bytes):
    return hashlib.sha1(data).hexdigest()[:12]


def safe_filename(name):
    name = Path(name).name
    return re.sub(r"[^A-Za-z0-9._\- ]", "_", name) or "document.pdf"


def normalize_topic(topic):
    """'normalization' -> 'Normalization' (keeps words like SQL, 2NF, BFS as they are)."""
    topic = re.sub(r"\s+", " ", (topic or "").strip())
    if not topic:
        return "General"
    words = []
    for word in topic.split(" "):
        if word.isupper() or any(ch.isdigit() for ch in word):
            words.append(word)
        else:
            words.append(word[:1].upper() + word[1:])
    return " ".join(words)


def topic_key(topic):
    return normalize_topic(topic).lower()


def _page_number(line):
    try:
        return int(line.rsplit("Page ", 1)[1])
    except (ValueError, IndexError):
        return 0


def format_sources(docs):
    """Turn retrieved chunks into unique lines like '📄 file.pdf — Page 7'."""
    seen, lines = set(), []
    for doc in docs:
        meta = doc.metadata or {}
        line = f"📄 {meta.get('filename', 'Unknown file')} — Page {meta.get('page', '?')}"
        if line not in seen:
            seen.add(line)
            lines.append(line)
    lines.sort(key=lambda s: (s.split(" — Page ")[0], _page_number(s)))
    return lines


def format_context(docs, max_chars=9000):
    """Combine retrieved chunks into one text block that we send to Gemini."""
    blocks, total = [], 0
    for doc in docs:
        meta = doc.metadata or {}
        block = f"[Source: {meta.get('filename', '?')}, Page {meta.get('page', '?')}]\n{doc.page_content.strip()}"
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)
    return "\n\n---\n\n".join(blocks)
