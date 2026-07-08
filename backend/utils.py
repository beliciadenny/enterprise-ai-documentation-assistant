"""
utils.py
--------
Shared utility functions: logging configuration, file helpers, and
custom exceptions used across the backend.
"""

import logging
import sys
from pathlib import Path
from typing import Iterable

from backend.config import LOG_LEVEL, SUPPORTED_EXTENSIONS


def get_logger(name: str) -> logging.Logger:
    """
    Create (or retrieve) a configured logger.

    Using a factory function here means every module gets a consistently
    formatted logger without duplicating handler setup code.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(LOG_LEVEL)
        logger.propagate = False

    return logger


class DocumentAssistantError(Exception):
    """Base exception for all application-specific errors."""


class NoDocumentsError(DocumentAssistantError):
    """Raised when a query or ingestion is attempted with no documents uploaded."""


class NoRelevantChunksError(DocumentAssistantError):
    """Raised when a query returns no relevant chunks from the vector store."""


class OllamaConnectionError(DocumentAssistantError):
    """Raised when the local Ollama LLM server cannot be reached."""


class UnsupportedFileTypeError(DocumentAssistantError):
    """Raised when an uploaded file type is not supported."""


def is_supported_file(filename: str) -> bool:
    """Return True if the file extension is one we know how to ingest."""
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS


def list_uploaded_documents(docs_dir: Path) -> Iterable[Path]:
    """Return all supported documentation files currently stored in docs_dir."""
    if not docs_dir.exists():
        return []
    return sorted(
        p for p in docs_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
