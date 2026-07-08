"""
app.py
------
Streamlit frontend for the Enterprise AI Documentation Assistant.

Provides:
  - Sidebar: upload documentation, index documents, clear database.
  - Main page: ask a natural language question and view the generated
    answer alongside its supporting source chunks.

Run with:
    streamlit run frontend/app.py
"""

import os

import requests
import streamlit as st

# The frontend talks to the backend purely over HTTP, so it stays fully
# decoupled and can be deployed/scaled independently of the API.
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
REQUEST_TIMEOUT_SECONDS = 120

# ---------------------------------------------------------------------------
# Page configuration & theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Enterprise AI Documentation Assistant",
    page_icon="📘",
    layout="wide",
)

CUSTOM_CSS = """
<style>
    .main { background-color: #ffffff; }
    .stButton>button {
        background-color: #1a73e8;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1.25rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #1558b0;
        color: white;
    }
    .source-chunk {
        background-color: #f0f6ff;
        border-left: 4px solid #1a73e8;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.75rem;
        font-size: 0.9rem;
    }
    h1, h2, h3 { color: #0b3d91; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Cached across reruns within a session; only refreshed when the user
# re-indexes documentation (see the "Index Documents" button below).
if "sample_questions" not in st.session_state:
    st.session_state.sample_questions = None
if "sample_questions_error" not in st.session_state:
    st.session_state.sample_questions_error = None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def api_upload(files) -> dict:
    payload = [("files", (f.name, f.getvalue())) for f in files]
    response = requests.post(f"{BACKEND_URL}/upload", files=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def api_ingest() -> dict:
    response = requests.post(f"{BACKEND_URL}/ingest", timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def api_query(question: str) -> dict:
    response = requests.post(
        f"{BACKEND_URL}/query",
        json={"question": question},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def api_clear() -> dict:
    response = requests.delete(f"{BACKEND_URL}/clear", timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def api_suggest_questions(num_questions: int = 6) -> dict:
    response = requests.post(
        f"{BACKEND_URL}/suggest-questions",
        json={"num_questions": num_questions},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def friendly_error_message(exc: requests.exceptions.RequestException) -> str:
    """Translate backend/network errors into user-friendly messages.

    Important: a ConnectionError here means Streamlit couldn't reach the
    FastAPI backend itself (e.g. it isn't running) — it is NOT the same as
    the backend failing to reach Ollama. The backend already returns a 503
    with "Unable to connect to local LLM." as its detail message in that
    case, which is surfaced via the HTTPError branch below.
    """
    if isinstance(exc, requests.exceptions.ConnectionError):
        return (
            f"Unable to reach the backend server at {BACKEND_URL}. "
            "Make sure it's running (uvicorn backend.api:app --reload)."
        )

    if isinstance(exc, requests.exceptions.Timeout):
        return "The request timed out. The backend or local LLM may be taking too long to respond."

    if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
        try:
            detail = exc.response.json().get("detail", "")
        except ValueError:
            detail = ""
        if detail:
            return detail

    return "Something went wrong. Please try again."


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📂 Documentation Management")

    uploaded_files = st.file_uploader(
        "Upload Documentation",
        type=["pdf", "md", "markdown", "txt"],
        accept_multiple_files=True,
    )

    if st.button("⬆️ Upload Files", use_container_width=True):
        if not uploaded_files:
            st.warning("Please select at least one file to upload.")
        else:
            try:
                with st.spinner("Uploading..."):
                    result = api_upload(uploaded_files)
                if result["uploaded_files"]:
                    st.success(f"Uploaded: {', '.join(result['uploaded_files'])}")
                if result["skipped_files"]:
                    st.warning(f"Skipped (unsupported type): {', '.join(result['skipped_files'])}")
            except requests.exceptions.RequestException as exc:
                st.error(friendly_error_message(exc))

    st.divider()

    if st.button("📚 Index Documents", use_container_width=True):
        try:
            with st.spinner("Indexing documentation... this may take a moment."):
                result = api_ingest()
            st.success(f"{result['message']} ({result['chunks_indexed']} chunks indexed)")

            # Refresh sample questions so they reflect the newly indexed
            # content instead of whatever was indexed previously.
            st.session_state.sample_questions = None
            st.session_state.sample_questions_error = None
            try:
                with st.spinner("Generating sample questions from your documentation..."):
                    suggestions = api_suggest_questions()
                st.session_state.sample_questions = suggestions["questions"]
            except requests.exceptions.RequestException as exc:
                # Indexing itself succeeded, so don't fail the whole action —
                # just note that suggestions couldn't be generated this time.
                st.session_state.sample_questions_error = friendly_error_message(exc)

        except requests.exceptions.RequestException as exc:
            st.error(friendly_error_message(exc))

    st.divider()

    if st.button("🗑️ Clear Database", use_container_width=True):
        try:
            with st.spinner("Clearing database..."):
                result = api_clear()
            st.success(result["message"])
            st.session_state.sample_questions = None
            st.session_state.sample_questions_error = None
        except requests.exceptions.RequestException as exc:
            st.error(friendly_error_message(exc))

    st.divider()
    st.caption("Backend: " + BACKEND_URL)


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.title("📘 Enterprise AI Documentation Assistant")
st.caption("Ask questions about your uploaded documentation and get grounded, sourced answers.")

question = st.text_input("Ask your question...", placeholder="e.g. How do I authenticate a user?")

col1, _ = st.columns([1, 4])
with col1:
    generate_clicked = st.button("🔍 Generate Answer", use_container_width=True)

if generate_clicked:
    if not question or not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Searching documentation and generating an answer..."):
                result = api_query(question)

            st.subheader("Answer")
            st.write(result["answer"])

            st.subheader("Source Documents")
            if result["sources"]:
                for i, source in enumerate(result["sources"], start=1):
                    st.markdown(
                        f'<div class="source-chunk"><strong>Chunk {i}</strong><br>{source}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No source chunks were returned for this answer.")

        except requests.exceptions.RequestException as exc:
            st.error(friendly_error_message(exc))

st.divider()
with st.expander("💡 Sample questions", expanded=True):
    if st.session_state.sample_questions:
        st.caption("Generated from your indexed documentation:")
        st.markdown(
            "\n".join(f"- {q}" for q in st.session_state.sample_questions)
        )
    elif st.session_state.sample_questions_error:
        st.warning(
            "Couldn't generate sample questions this time "
            f"({st.session_state.sample_questions_error}). "
            "Try clicking 'Index Documents' again, or just ask your own question above."
        )
    else:
        st.info(
            "Upload and index your documentation, and relevant example "
            "questions will appear here automatically."
        )
