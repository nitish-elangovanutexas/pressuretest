"""
Lazy singletons for the heavy NLP models used in Phase 2.

Both loaders return ``None`` if the underlying library is missing, so callers
can degrade gracefully (useful for unit tests that don't need the real model).
"""
from __future__ import annotations

import logging
from functools import lru_cache

from .question_scorer import _load_finbert  # re-export

log = logging.getLogger(__name__)

EMBEDDER_MODEL_ID = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def load_embedder():
    """Lazy-load the sentence-transformers embedder; returns None on failure."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log.warning(
            "sentence-transformers not installed; embeddings disabled"
        )
        return None

    try:
        return SentenceTransformer(EMBEDDER_MODEL_ID)
    except Exception as exc:                                  # pragma: no cover
        log.warning("Failed to initialise embedder (%s): %s", EMBEDDER_MODEL_ID, exc)
        return None


def load_finbert():
    """Public alias for the FinBERT loader defined in question_scorer."""
    return _load_finbert()


__all__ = ["load_embedder", "load_finbert", "EMBEDDER_MODEL_ID"]
