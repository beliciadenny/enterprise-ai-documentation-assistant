"""
ingest.py
---------
Handles the document ingestion pipeline:
  1. Load raw documents (PDF, Markdown, TXT) from disk.
  2. Split them into overlapping chunks.
  3. Embed the chunks and build/update a FAISS vector index.
  4. Persist the index to disk and reload it on startup.
"""

import shutil
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCS_DIR,
    FAISS_INDEX_NAME,
    VECTORSTORE_DIR,
)
from backend.embeddings import get_embedding_model
from backend.utils import (
    NoDocumentsError,
    get_logger,
    list_uploaded_documents,
)

logger = get_logger(__name__)

_INDEX_PATH = VECTORSTORE_DIR / FAISS_INDEX_NAME


def _load_single_document(path: Path) -> List[Document]:
    """Load one file into LangChain Document objects, based on its extension."""
    suffix = path.suffix.lower()

    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        elif suffix in (".md", ".markdown"):
            try:
                # Imported lazily: `unstructured` is an optional extra, so
                # importing it only when a .md file actually shows up means
                # PDF/TXT-only deployments never need it installed at all.
                from langchain_community.document_loaders import (
                    UnstructuredMarkdownLoader,
                )

                loader = UnstructuredMarkdownLoader(str(path))
            except Exception:
                # Fall back to plain text loading if the unstructured
                # markdown parser / its dependencies aren't available.
                loader = TextLoader(str(path), encoding="utf-8")
        else:  # .txt and anything else falls back to plain text
            loader = TextLoader(str(path), encoding="utf-8")

        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = path.name
        return docs

    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load document %s: %s", path.name, exc)
        return []


def load_all_documents(docs_dir: Path = DOCS_DIR) -> List[Document]:
    """Load every supported document currently stored in docs_dir."""
    files = list(list_uploaded_documents(docs_dir))

    if not files:
        raise NoDocumentsError("Please upload documentation first.")

    all_docs: List[Document] = []
    for file_path in files:
        logger.info("Loading document: %s", file_path.name)
        all_docs.extend(_load_single_document(file_path))

    return all_docs


def split_documents(documents: List[Document]) -> List[Document]:
    """Split loaded documents into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split %d documents into %d chunks.", len(documents), len(chunks))
    return chunks


def build_and_save_index(chunks: List[Document]) -> FAISS:
    """Embed chunks and build a fresh FAISS index, saving it to disk."""
    embedding_model = get_embedding_model()
    logger.info("Building FAISS index from %d chunks...", len(chunks))
    vectorstore = FAISS.from_documents(chunks, embedding_model)

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR), index_name=FAISS_INDEX_NAME)
    logger.info("FAISS index saved to %s", VECTORSTORE_DIR)
    return vectorstore


def load_index() -> FAISS:
    """
    Load an existing FAISS index from disk.

    Raises NoDocumentsError if no index has been built yet.
    """
    index_file = VECTORSTORE_DIR / f"{FAISS_INDEX_NAME}.faiss"
    if not index_file.exists():
        raise NoDocumentsError(
            "No index found. Please upload and index documentation first."
        )

    embedding_model = get_embedding_model()
    vectorstore = FAISS.load_local(
        str(VECTORSTORE_DIR),
        embedding_model,
        index_name=FAISS_INDEX_NAME,
        # Safe here: the index is created and controlled entirely by this
        # application, never loaded from an untrusted third-party source.
        allow_dangerous_deserialization=True,
    )
    logger.info("FAISS index loaded from %s", VECTORSTORE_DIR)
    return vectorstore


def run_ingestion_pipeline(docs_dir: Path = DOCS_DIR) -> int:
    """
    Full pipeline: load -> split -> embed -> build index -> save.

    Returns the number of chunks indexed.
    """
    documents = load_all_documents(docs_dir)
    chunks = split_documents(documents)
    build_and_save_index(chunks)
    return len(chunks)


def clear_database() -> None:
    """
    Remove the FAISS index and all uploaded documents, resetting the
    application to a clean state. Backs the "Clear Database" UI button.
    """
    if VECTORSTORE_DIR.exists():
        shutil.rmtree(VECTORSTORE_DIR)
        VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Cleared vector store at %s", VECTORSTORE_DIR)

    if DOCS_DIR.exists():
        for f in DOCS_DIR.iterdir():
            if f.is_file():
                f.unlink()
        logger.info("Cleared uploaded documents at %s", DOCS_DIR)

    # Loading the index reads from disk each time, so nothing further to
    # invalidate here; a fresh load_index() call will correctly raise
    # NoDocumentsError until documents are re-ingested.
