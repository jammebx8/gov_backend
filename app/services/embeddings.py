"""
Embedding service — BAAI/bge-small-en-v1.5 via ONNX Runtime.

Why ONNX instead of sentence-transformers / PyTorch?
  PyTorch alone is ~800 MB unpacked; Vercel Python functions cap at 500 MB.
  onnxruntime (~35 MB) + tokenizers (~3 MB) + the fp32 ONNX model (~127 MB)
  fits comfortably inside the limit with room for all other deps.

Model files are downloaded from HuggingFace Hub on the first request and
cached in /tmp (Vercel's writable scratch space).  Subsequent warm-instance
calls skip the download entirely.

The 384-d normalised vectors are identical to sentence-transformers output,
so all existing pgvector embeddings remain valid.
"""

import asyncio
import math
import os
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Files needed from the HuggingFace Hub revision
_HF_BASE = f"https://huggingface.co/{MODEL_ID}/resolve/main"
_CACHE_DIR = Path(os.environ.get("MODEL_CACHE_DIR", "/tmp/bge_small_en"))

_REQUIRED_FILES = {
    "onnx/model.onnx": f"{_HF_BASE}/onnx/model.onnx",
    "tokenizer.json": f"{_HF_BASE}/tokenizer.json",
    "tokenizer_config.json": f"{_HF_BASE}/tokenizer_config.json",
    "special_tokens_map.json": f"{_HF_BASE}/special_tokens_map.json",
}

MAX_LENGTH = 512  # model's max token length


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _ensure_files() -> Path:
    """Download model files to _CACHE_DIR if not already present."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    onnx_dir = _CACHE_DIR / "onnx"
    onnx_dir.mkdir(exist_ok=True)

    for rel_path, url in _REQUIRED_FILES.items():
        dest = _CACHE_DIR / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            urllib.request.urlretrieve(url, dest)

    return _CACHE_DIR


# ---------------------------------------------------------------------------
# Model + tokenizer loading (cached per process)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_session():
    """Load the ONNX inference session — runs once per worker process."""
    import onnxruntime as ort

    cache = _ensure_files()
    model_path = str(cache / "onnx" / "model.onnx")

    sess_opts = ort.SessionOptions()
    sess_opts.inter_op_num_threads = 1
    sess_opts.intra_op_num_threads = 1

    session = ort.InferenceSession(
        model_path,
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )
    return session


@lru_cache(maxsize=1)
def _load_tokenizer():
    """Load the fast HF tokenizer — runs once per worker process."""
    from tokenizers import Tokenizer

    cache = _ensure_files()
    tok = Tokenizer.from_file(str(cache / "tokenizer.json"))
    tok.enable_padding(pad_id=0, pad_token="[PAD]", length=MAX_LENGTH)
    tok.enable_truncation(max_length=MAX_LENGTH)
    return tok


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def _embed_sync(texts: List[str]) -> List[List[float]]:
    """Tokenise + run ONNX inference + L2-normalise. Fully synchronous."""
    tokenizer = _load_tokenizer()
    session = _load_session()

    encodings = tokenizer.encode_batch(texts)

    input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
    token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

    outputs = session.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
    )

    # outputs[0] is last_hidden_state [batch, seq, 384]
    # Mean-pool over non-padding tokens, then L2-normalise
    hidden = outputs[0]  # (batch, seq, 384)
    mask = attention_mask[:, :, np.newaxis].astype(np.float32)  # (batch, seq, 1)

    summed = (hidden * mask).sum(axis=1)          # (batch, 384)
    counts = mask.sum(axis=1).clip(min=1e-9)      # (batch, 1)
    pooled = summed / counts                       # (batch, 384)

    # L2 normalise
    norms = np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-9)
    normalised = pooled / norms

    return normalised.tolist()


# ---------------------------------------------------------------------------
# Public async API (same interface as before)
# ---------------------------------------------------------------------------

async def embed_text(text: str, is_query: bool = False) -> List[float]:
    """
    Embed a single string asynchronously.
    is_query=True adds the BGE retrieval prefix — use for search bar input.
    is_query=False is used when embedding scheme/user text.
    """
    if is_query:
        text = BGE_QUERY_PREFIX + text
    loop = asyncio.get_event_loop()
    vectors = await loop.run_in_executor(None, _embed_sync, [text])
    return vectors[0]


async def embed_batch(texts: List[str], is_query: bool = False) -> List[List[float]]:
    """Embed multiple strings at once."""
    if is_query:
        texts = [BGE_QUERY_PREFIX + t for t in texts]
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _embed_sync, texts)


# ---------------------------------------------------------------------------
# Text builders (unchanged)
# ---------------------------------------------------------------------------

def build_scheme_text(scheme: dict) -> str:
    parts = [
        scheme.get("scheme_name", ""),
        scheme.get("details", ""),
        scheme.get("benefits", ""),
        scheme.get("eligibility", ""),
        scheme.get("scheme_category", ""),
        scheme.get("level", ""),
    ]
    tags = scheme.get("tags") or []
    if tags:
        parts.append(" ".join(tags))
    return " ".join(p for p in parts if p).strip()


def build_user_query_text(user: dict, documents: list = None) -> str:
    from app.utils.prompt_builder import build_user_profile_text
    return build_user_profile_text(user, documents or [])
