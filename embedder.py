"""Embedder utility — generates dense vectors for text using either
1. Local Sentence‑Transformer (default) or
2. Hugging Face Inference API (fallback).

Fix: previously calling `.tolist()` on a Python list raised `AttributeError`.
The code now detects ndarray vs list and coerces correctly.
"""
from __future__ import annotations

import logging
from typing import List, Sequence

import requests
from sentence_transformers import SentenceTransformer
import numpy as np

from models import db_settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ──────────────────────────────────────────────────────────────────────────────
# Local Sentence‑Transformer model
# ──────────────────────────────────────────────────────────────────────────────
try:
    _local_model: SentenceTransformer | None = SentenceTransformer(
        db_settings.EMBEDDING_MODEL_NAME, trust_remote_code=True
    )
    logger.info("Loaded local embedding model '%s'", db_settings.EMBEDDING_MODEL_NAME)
except Exception as exc:  # pragma: no cover
    logger.warning("Failed to load local embedding model: %s", exc)
    _local_model = None

# ──────────────────────────────────────────────────────────────────────────────
# Hugging Face Inference API fallback
# ──────────────────────────────────────────────────────────────────────────────
_USE_HF_API = "false"
_HF_TOKEN = db_settings.HF_TOKEN
_HF_ENDPOINT = (
    "https://api-inference.huggingface.co/pipeline/feature-extraction/"
    + db_settings.EMBEDDING_MODEL_NAME
)
if _USE_HF_API and not _HF_TOKEN:
    raise RuntimeError("USE_HF_INFERENCE_API true but HF_TOKEN missing")


def _embed_via_hf(texts: Sequence[str]) -> List[List[float]]:
    """Call HF Inference endpoint and return list of vectors."""
    headers = {"Authorization": f"Bearer {_HF_TOKEN}"}
    resp = requests.post(
        _HF_ENDPOINT,
        headers=headers,
        json={"inputs": list(texts), "options": {"wait_for_model": True}},
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"HF Inference API {resp.status_code}: {resp.text}")
    return resp.json()  # already list[list[float]]


# Normalise numpy → python list

def _to_list(vec) -> List[float]:
    if isinstance(vec, list):
        return vec
    if isinstance(vec, np.ndarray):
        return vec.astype(float).tolist()
    raise TypeError(f"Unexpected vector type {type(vec)}")


# ──────────────────────────────────────────────────────────────────────────────
# Public function
# ──────────────────────────────────────────────────────────────────────────────

def embed_text(text: str | Sequence[str]) -> List[float] | List[List[float]]:
    """Generate embedding(s) for a single string or a list of strings."""
    is_single = isinstance(text, str)
    sentences: List[str] = [text] if is_single else list(text)  # type: ignore[arg-type]

    try:
        if _USE_HF_API or _local_model is None:
            vectors = _embed_via_hf(sentences)
        else:
            # We request numpy for easy dtype management
            raw = _local_model.encode(sentences, convert_to_numpy=True)
            if raw.ndim == 1:
                vectors = [_to_list(raw)]
            else:
                vectors = [_to_list(row) for row in raw]
    except Exception as exc:
        logger.exception("Embedding failed")
        raise RuntimeError("Embedding generation error") from exc

    return vectors[0] if is_single else vectors
