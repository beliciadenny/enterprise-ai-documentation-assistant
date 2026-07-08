"""
rag.py
------
The core Retrieval-Augmented Generation pipeline:
  1. Embed the user's question.
  2. Retrieve the top-K most relevant chunks from FAISS.
  3. Construct a grounded prompt.
  4. Send the prompt to the locally hosted Ollama LLM.
  5. Return the generated answer along with its source chunks.
"""

from dataclasses import dataclass
from typing import List

import requests
from langchain_community.vectorstores import FAISS

from backend.config import (
    LLM_TEMPERATURE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT_SECONDS,
    TOP_K_RESULTS,
)
from backend.ingest import load_index
from backend.prompts import FALLBACK_ANSWER, build_prompt, build_sample_questions_prompt
from backend.utils import (
    NoRelevantChunksError,
    OllamaConnectionError,
    get_logger,
)

logger = get_logger(__name__)

DEFAULT_NUM_SAMPLE_QUESTIONS = 6
MAX_CHUNKS_FOR_SAMPLE_QUESTIONS = 6


@dataclass
class QueryResult:
    """Structured result returned by the RAG pipeline for a single query."""

    answer: str
    sources: List[str]


def _call_ollama(prompt: str) -> str:
    """
    Send the constructed prompt to the local Ollama server and return the
    generated text.

    Raises OllamaConnectionError if the local LLM server cannot be reached.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": LLM_TEMPERATURE},
    }

    try:
        response = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()

    except requests.exceptions.ConnectionError as exc:
        logger.error("Could not connect to Ollama at %s: %s", url, exc)
        raise OllamaConnectionError("Unable to connect to local LLM.") from exc

    except requests.exceptions.Timeout as exc:
        logger.error("Ollama request timed out: %s", exc)
        raise OllamaConnectionError("Unable to connect to local LLM.") from exc

    except requests.exceptions.RequestException as exc:
        logger.error("Ollama request failed: %s", exc)
        raise OllamaConnectionError("Unable to connect to local LLM.") from exc


def answer_question(question: str, top_k: int = TOP_K_RESULTS) -> QueryResult:
    """
    Run the full RAG pipeline for a single question and return the answer
    plus the supporting source chunks.
    """
    if not question or not question.strip():
        raise ValueError("Question must not be empty.")

    vectorstore = load_index()

    results = vectorstore.similarity_search(question, k=top_k)

    if not results:
        raise NoRelevantChunksError("No relevant information found.")

    chunks = [doc.page_content for doc in results]

    # Each source string pairs the originating file (and page, for PDFs)
    # with its chunk text, so the UI can display "Chunk 1", "Chunk 2", etc.
    # alongside a clear reference to where the content came from.
    sources = []
    for doc in results:
        origin = doc.metadata.get("source", "unknown")
        if "page" in doc.metadata:
            origin = f"{origin} (page {doc.metadata['page']})"
        sources.append(f"[{origin}] {doc.page_content}")

    prompt = build_prompt(question, chunks)
    logger.info("Sending prompt to Ollama model '%s'", OLLAMA_MODEL)
    answer = _call_ollama(prompt)

    if not answer:
        answer = FALLBACK_ANSWER

    return QueryResult(answer=answer, sources=sources)


def _sample_chunks_from_index(vectorstore: FAISS, max_chunks: int) -> List[str]:
    """
    Pull a spread of real chunks out of the FAISS docstore, so sample
    question generation is grounded in what was actually uploaded rather
    than biased toward any particular search query.

    Chunks are sampled at evenly spaced positions across the full set,
    which tends to surface content from different parts of (and across)
    the uploaded document(s) rather than clustering on the first file.
    """
    docstore = vectorstore.docstore
    doc_ids = list(vectorstore.index_to_docstore_id.values())

    if not doc_ids:
        return []

    if len(doc_ids) <= max_chunks:
        selected_ids = doc_ids
    else:
        step = len(doc_ids) / max_chunks
        selected_ids = [doc_ids[int(i * step)] for i in range(max_chunks)]

    chunks = []
    for doc_id in selected_ids:
        doc = docstore.search(doc_id)
        # InMemoryDocstore.search returns a Document, or an error string
        # if the id isn't found — guard against the latter defensively.
        if hasattr(doc, "page_content"):
            chunks.append(doc.page_content)

    return chunks


def _parse_questions(raw_text: str, num_questions: int) -> List[str]:
    """
    Parse the LLM's line-per-question output into a clean list, stripping
    any numbering, bullets, or stray formatting the model adds despite
    instructions not to.
    """
    questions = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Strip common leading list markers like "1.", "1)", "-", "*", "Q:"
        for marker in (". ", ") "):
            if marker in line[:4]:
                _, _, remainder = line.partition(marker)
                if remainder:
                    line = remainder.strip()
                break
        line = line.lstrip("-*•").strip()
        if line.lower().startswith("q:"):
            line = line[2:].strip()

        if line:
            questions.append(line)

    return questions[:num_questions]


def generate_sample_questions(
    num_questions: int = DEFAULT_NUM_SAMPLE_QUESTIONS,
) -> List[str]:
    """
    Generate example questions that are genuinely answerable from whatever
    documentation is currently indexed, replacing any static/hardcoded
    question list with ones grounded in the real uploaded content.

    Raises NoDocumentsError (via load_index) if nothing has been indexed
    yet, and OllamaConnectionError if the local LLM can't be reached.
    """
    vectorstore = load_index()
    chunks = _sample_chunks_from_index(vectorstore, MAX_CHUNKS_FOR_SAMPLE_QUESTIONS)

    if not chunks:
        raise NoRelevantChunksError("No relevant information found.")

    prompt = build_sample_questions_prompt(chunks, num_questions)
    logger.info("Generating %d sample questions from indexed content.", num_questions)
    raw_output = _call_ollama(prompt)

    questions = _parse_questions(raw_output, num_questions)

    if not questions:
        logger.warning("LLM returned no parsable sample questions.")

    return questions
