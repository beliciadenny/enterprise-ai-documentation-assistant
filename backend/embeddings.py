"""
embeddings.py
-------------
Wraps the sentence-transformers embedding model (all-MiniLM-L6-v2) behind
a LangChain-compatible embeddings interface, so it can be plugged directly
into a FAISS vector store.
"""

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from backend.config import EMBEDDING_MODEL_NAME
from backend.utils import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Load and cache the sentence-transformers embedding model.

    lru_cache ensures the (relatively expensive) model load only happens
    once per process, regardless of how many times this function is called.
    """
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
    model = HuggingFaceEmbeddings(
        model_name=f"sentence-transformers/{EMBEDDING_MODEL_NAME}"
        if "/" not in EMBEDDING_MODEL_NAME
        else EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    logger.info("Embedding model loaded successfully.")
    return model
