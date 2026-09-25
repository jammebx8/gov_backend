"""
Embedding service using BAAI/bge-small-en-v1.5 via sentence-transformers.

The model is downloaded once on first use (~130 MB) and then cached in memory.
It produces 384-dimensional vectors that match the `vector` column in the
`government_schemes` table (pgvector with HNSW index).

Usage:
    from app.services.embeddings import embed_text, embed_batch
    vec = await embed_text("scholarships for girls in rajasthan")
"""

import asyncio
from functools import lru_cache
from typing import List

MODEL_NAME = "BAAI/bge-small-en-v1.5"

# BGE models perform best with this prefix for retrieval queries
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _load_model():
    """Load the sentence-transformer model once and keep it in memory."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    return model


def _embed_sync(texts: List[str], is_query: bool = False) -> List[List[float]]:
    """
    Synchronous embedding call — runs in the main thread.
    Adds the BGE retrieval prefix when encoding search queries so that
    query vectors are in the same semantic space as document vectors.
    """
    model = _load_model()

    if is_query:
        texts = [BGE_QUERY_PREFIX + t for t in texts]

    # normalize_embeddings=True ensures cosine similarity == dot product
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
    )
    return vectors.tolist()


async def embed_text(text: str, is_query: bool = False) -> List[float]:
    """
    Embed a single string asynchronously.
    `is_query=True` adds the BGE retrieval prefix — use this for search bar input.
    `is_query=False` is used when embedding scheme text for storage.
    """
    loop = asyncio.get_event_loop()
    vectors = await loop.run_in_executor(None, _embed_sync, [text], is_query)
    return vectors[0]


async def embed_batch(texts: List[str], is_query: bool = False) -> List[List[float]]:
    """Embed multiple strings at once — more efficient for backfill scripts."""
    loop = asyncio.get_event_loop()
    vectors = await loop.run_in_executor(None, _embed_sync, texts, is_query)
    return vectors


def build_scheme_text(scheme: dict) -> str:
    """
    Concatenate scheme fields into the string we embed for storage.
    Must match what `government_schemes_search_vector_update` function
    would use so that semantic search aligns with full-text search.
    """
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
    """
    Build a natural-language description of the user for recommendation.
    This is embedded and compared against scheme embeddings via cosine similarity.
    """
    from app.utils.prompt_builder import build_user_profile_text
    return build_user_profile_text(user, documents or [])
