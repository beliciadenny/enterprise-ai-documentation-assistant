"""
api.py
------
FastAPI backend exposing the Enterprise AI Documentation Assistant's
REST API: document upload, ingestion, querying, database clearing,
and a health check.

Run with:
    uvicorn backend.api:app --host 0.0.0.0 --port 8000
"""

from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.config import DOCS_DIR
from backend.ingest import clear_database, run_ingestion_pipeline
from backend.rag import answer_question, generate_sample_questions
from backend.utils import (
    NoDocumentsError,
    NoRelevantChunksError,
    OllamaConnectionError,
    get_logger,
    is_supported_file,
    list_uploaded_documents,
)

logger = get_logger(__name__)

app = FastAPI(
    title="Enterprise AI Documentation Assistant",
    description=(
        "RAG-powered API for searching and understanding technical "
        "documentation using natural language."
    ),
    version="1.0.0",
)

# Allow the Streamlit frontend (and other local clients) to call the API
# freely. In a production deployment this should be restricted to known
# origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's natural language question.")


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]


class SuggestQuestionsRequest(BaseModel):
    num_questions: int = Field(default=6, ge=1, le=15)


class SuggestQuestionsResponse(BaseModel):
    questions: List[str]


class UploadResponse(BaseModel):
    uploaded_files: List[str]
    skipped_files: List[str]


class IngestResponse(BaseModel):
    message: str
    chunks_indexed: int


class MessageResponse(BaseModel):
    message: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", response_model=MessageResponse, tags=["Health"])
def root() -> MessageResponse:
    """Basic root endpoint confirming the API is running."""
    return MessageResponse(message="Enterprise AI Documentation Assistant API is running.")


@app.get("/health", response_model=MessageResponse, tags=["Health"])
def health_check() -> MessageResponse:
    """Simple health check used by monitoring and the frontend."""
    return MessageResponse(message="ok")


@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_documents(files: List[UploadFile] = File(...)) -> UploadResponse:
    """
    Upload one or more documentation files (PDF, Markdown, or TXT).

    Files are stored under docs/ ready for ingestion. Unsupported file
    types are skipped and reported back to the caller.
    """
    uploaded, skipped = [], []

    for upload in files:
        if not is_supported_file(upload.filename):
            logger.warning("Skipping unsupported file: %s", upload.filename)
            skipped.append(upload.filename)
            continue

        destination = DOCS_DIR / upload.filename
        try:
            contents = await upload.read()
            destination.write_bytes(contents)
            uploaded.append(upload.filename)
            logger.info("Saved uploaded file to %s", destination)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to save %s: %s", upload.filename, exc)
            skipped.append(upload.filename)

    return UploadResponse(uploaded_files=uploaded, skipped_files=skipped)


@app.get("/documents", response_model=List[str], tags=["Documents"])
def get_documents() -> List[str]:
    """List all currently uploaded documentation files."""
    return [p.name for p in list_uploaded_documents(DOCS_DIR)]


@app.post("/ingest", response_model=IngestResponse, tags=["Documents"])
def ingest_documents() -> IngestResponse:
    """
    Run the ingestion pipeline: load uploaded documents, split them into
    chunks, generate embeddings, and (re)build the FAISS index.
    """
    try:
        chunk_count = run_ingestion_pipeline()
    except NoDocumentsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ingestion pipeline failed.")
        raise HTTPException(status_code=500, detail="Failed to index documents.") from exc

    return IngestResponse(
        message="Documents indexed successfully.",
        chunks_indexed=chunk_count,
    )


@app.post("/query", response_model=QueryResponse, tags=["Query"])
def query_documents(request: QueryRequest) -> QueryResponse:
    """
    Answer a natural language question using the indexed documentation,
    via the RAG pipeline (retrieve top-K chunks -> prompt -> local LLM).
    """
    try:
        result = answer_question(request.question)
    except NoDocumentsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NoRelevantChunksError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except OllamaConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Query failed unexpectedly.")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.") from exc

    return QueryResponse(answer=result.answer, sources=result.sources)


@app.post("/suggest-questions", response_model=SuggestQuestionsResponse, tags=["Query"])
def suggest_questions(request: SuggestQuestionsRequest) -> SuggestQuestionsResponse:
    """
    Generate example questions that are genuinely answerable from the
    currently indexed documentation. Intended to be called right after
    /ingest succeeds, so the UI can show suggestions grounded in whatever
    the user actually uploaded instead of a generic, unrelated list.
    """
    try:
        questions = generate_sample_questions(num_questions=request.num_questions)
    except NoDocumentsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NoRelevantChunksError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except OllamaConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Sample question generation failed unexpectedly.")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.") from exc

    return SuggestQuestionsResponse(questions=questions)


@app.delete("/clear", response_model=MessageResponse, tags=["Documents"])
def clear_documents() -> MessageResponse:
    """Clear all uploaded documents and the FAISS vector index."""
    try:
        clear_database()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to clear database.")
        raise HTTPException(status_code=500, detail="Failed to clear database.") from exc

    return MessageResponse(message="Database cleared successfully.")
