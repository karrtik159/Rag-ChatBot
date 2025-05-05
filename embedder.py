"""Embedder utility — generates dense vectors for text using either
1. Local Sentence‑Transformer (default) or
2. Hugging Face Inference API (fallback).

Configure behaviour with environment variables:
- EMBEDDING_MODEL (model name, defaults in models.py)
- USE_HF_INFERENCE_API ("true"/"false")
- HF_TOKEN (required if using HF Inference API)
"""
from __future__ import annotations

import os
import logging
from typing import List

import requests
from sentence_transformers import SentenceTransformer

from models import db_settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ────────────────────────────────────────────────────────────────────────────────
# Local Sentence‑Transformer model
# ────────────────────────────────────────────────────────────────────────────────
try:
    _local_model: SentenceTransformer | None = SentenceTransformer(
        db_settings.EMBEDDING_MODEL_NAME,
        trust_remote_code=True,
    )
    logger.info(
        "Loaded local embedding model '%s'.", db_settings.EMBEDDING_MODEL_NAME
    )
except Exception as exc:  # pragma: no cover
    logger.warning("Failed to load local embedding model: %s", exc)
    _local_model = None

# ────────────────────────────────────────────────────────────────────────────────
# Hugging Face Inference‑API fallback configuration
# ────────────────────────────────────────────────────────────────────────────────
_USE_HF_API = os.getenv("USE_HF_INFERENCE_API", "false").lower() == "true"
_HF_TOKEN = db_settings.HF_TOKEN
_HF_ENDPOINT = (
    f"https://api-inference.huggingface.co/pipeline/feature-extraction/"
    f"{db_settings.EMBEDDING_MODEL_NAME}"
)

if _USE_HF_API and not _HF_TOKEN:
    raise RuntimeError(
        "USE_HF_INFERENCE_API is true but HF_TOKEN is not provided in environment"
    )

# ────────────────────────────────────────────────────────────────────────────────
# Internal helper for HF API calls
# ────────────────────────────────────────────────────────────────────────────────

def _embed_via_hf(texts: List[str]) -> List[List[float]]:
    headers = {"Authorization": f"Bearer {_HF_TOKEN}"}
    payload = {"inputs": texts, "options": {"wait_for_model": True}}
    resp = requests.post(_HF_ENDPOINT, headers=headers, json=payload, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(
            f"HF inference API error {resp.status_code}: {resp.text}"
        )
    return resp.json()

# ────────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────────

def embed_text(text: str | List[str]) -> List[float] | List[List[float]]:
    """Generate embedding(s) for text.

    Accepts either a single string or a list of strings. Returns a single
    vector (list) or a list of vectors accordingly.
    """
    single = isinstance(text, str)
    sentences = [text] if single else list(text)  # type: ignore[arg-type]

    try:
        if _USE_HF_API or _local_model is None:
            vectors = _embed_via_hf(sentences)
        else:
            vectors = (
                _local_model.encode(sentences, convert_to_numpy=False)  # type: ignore
            ).tolist()
    except Exception as exc:
        logger.error("Error generating embedding: %s", exc)
        raise

    return vectors[0] if single else vectors
