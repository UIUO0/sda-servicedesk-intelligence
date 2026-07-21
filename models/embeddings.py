"""Shared multilingual sentence-embedding helper (cached to disk).

Clustering and any future semantic model share one embedding backend so the
(slow) encode step runs once and is reused. The model is multilingual on
purpose: Arabic and English tickets about the same fault must land near each
other in vector space — the cross-lingual test in ``clustering.recurring`` is
the gate that proves it before any counts are trusted.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

import config

logger = logging.getLogger("embeddings")

# --- corporate-network shims (must run before any HuggingFace download) -----
# 1) Use the Windows certificate store so the corporate TLS-inspection proxy's
#    CA is trusted (same reason SDP_VERIFY_SSL=false for the ITSM server).
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:  # dev machine without the corporate proxy
    pass
# 2) hub 1.x's "xet" transfer uses an httpx client that closes mid-download on
#    this stack ("Cannot send a request, as the client has been closed").
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


# Small, fast, genuinely multilingual (50+ languages incl. Arabic). ~470 MB.
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _hf_cache_has(model_name: str) -> bool:
    """Filesystem check (no HF import) for a cached model snapshot."""
    home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    folder = "models--" + model_name.replace("/", "--")
    snap = Path(home) / "hub" / folder / "snapshots"
    return snap.is_dir() and any(snap.iterdir())


# 3) Once the model is cached, force offline BEFORE any HF import so each load
#    skips slow metadata round-trips through the corporate proxy (~100s -> instant).
if _hf_cache_has(DEFAULT_MODEL):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

CACHE_DIR = config.MODELS_DIR / "artifacts" / "emb_cache"

_model = None


def _load_model(name: str = DEFAULT_MODEL):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer  # lazy: heavy import
        logger.info("Loading embedding model %s%s…", name,
                    " (offline cache)" if os.environ.get("HF_HUB_OFFLINE") else " (first run downloads it)")
        _model = SentenceTransformer(name)
    return _model


def _cache_key(texts: List[str], model_name: str) -> str:
    h = hashlib.sha256(model_name.encode("utf-8"))
    for t in texts:
        h.update(b"\x00")
        h.update(t.encode("utf-8"))
    return h.hexdigest()[:16]


def embed(texts: List[str], *, model_name: str = DEFAULT_MODEL,
          use_cache: bool = True) -> np.ndarray:
    """Return L2-normalised embeddings for ``texts`` (cached by content hash)."""
    texts = [t if isinstance(t, str) else "" for t in texts]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file: Optional[Path] = None
    if use_cache:
        cache_file = CACHE_DIR / f"{_cache_key(texts, model_name)}.npy"
        if cache_file.exists():
            logger.info("Using cached embeddings (%s)", cache_file.name)
            return np.load(cache_file)

    model = _load_model(model_name)
    vecs = model.encode(texts, batch_size=64, show_progress_bar=True,
                        normalize_embeddings=True)
    vecs = np.asarray(vecs, dtype=np.float32)
    if cache_file is not None:
        np.save(cache_file, vecs)
        logger.info("Cached embeddings -> %s", cache_file.name)
    return vecs
