"""
config.py
---------
Centralized configuration for the Enterprise AI Documentation Assistant.

All tunable settings (paths, model names, chunking parameters, etc.) live
here and are loaded from environment variables (via a .env file) with
sensible defaults, so the rest of the codebase never hardcodes values.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a .env file in the project root, if present.
load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DOCS_DIR = Path(os.getenv("DOCS_DIR", BASE_DIR / "docs"))
VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_DIR", BASE_DIR / "vectorstore"))
FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", "faiss_index")

# Ensure required directories exist at import time.
DOCS_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Document processing / chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt"}

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", 3))

# ---------------------------------------------------------------------------
# LLM (Ollama)
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", 60))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.2))

# ---------------------------------------------------------------------------
# API / server
# ---------------------------------------------------------------------------
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
BACKEND_URL = os.getenv("BACKEND_URL", f"http://localhost:{API_PORT}")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
