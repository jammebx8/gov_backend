from fastapi import APIRouter, Depends, Query, BackgroundTasks, HTTPException
from typing import Optional
from app.models.scheme import SchemeWithEligibility
from app.utils.auth import get_current_user
from app.services.eligibility import refresh_user_eligibility, get_eligible_schemes
from app.services.embeddings import embed_text, build_user_query_text
from app.database import get_supabase

router = APIRouter(prefix="/schemes", tags=["schemes"])

CATEGORIES = [
    "Education", "Agriculture", "Housing", "Health",
    "Women & Child", "Social Welfare", "Employment",
    "Business & MSME", "Pension", "Scholarship",
    "Skill Development", "Minority Welfare", "Differently Abled",
    "Financial Inclusion",
]

# ── helpers ──────────────────────────────────────────────────────────────────

def _pgvector_literal(vec: list[float]) -> str:
    """Format a Python list as a pgvector literal string for RPC calls."""
    return "[" + ",".join(f"{v:.8f}" for v in vec) + "]"


# ── endpoints ────────────────────────────────────────────────────────────────

@router.get("/categories")
async def list_categories():
    return {"categories": CATEGORIES}


@router.get("/search")
async def vector_search(
    q: str = Query(..., min_length=1, description="Natural-language search query"),
    limit: int = Query(20, ge=1, le=50),
    category: Optional[str] = Query(None),
    level: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_user),
):
    """
    Semantic vector search over government schemes.

    The query string is embedded in real-time with BAAI/bge-small-en-v1.5
    and compared against pre-computed scheme embeddings stored in the
    `embedding` (vector) column using cosine similarity via pgvector HNSW.

    Falls back to full-text search (search_vector tsvector column) when
    a scheme has no embedding yet.
    """
    # Embed the query (is_query=True adds the BGE retrieval prefix)
    query_vec = await embed_text(q, is_query=True)
    vec_literal = _pgvector_literal(query_vec)

    db = get_supabase()

    # Build category / level filter fragment for the RPC
    # We call a Postgres function so we can use the HNSW index efficiently
    rpc_params: dict = {
        "query_embedding": vec_literal,
        "match_count": limit,
        "category_filter": category,
        "level_filter": level,
    }

    result = db.rpc("search_schemes_by_embedding", rpc_params).execute()
    schemes = result.data or []

    # Attach similarity score as a convenience field
    return {
        "schemes": schemes,
        "query": q,
        "total": len(schemes),
        "search_type": "vector",
    }


@router.get("/recommend")
async def recommend_for_user(
    limit: int = Query(15, ge=1, le=50),
    category: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_user),
):
    """
    Personalised scheme recommendations via vector similarity.

    Embeds the current user's profile (age, income, state, caste, occupation,
    extracted document fields) and finds the nearest scheme embeddings.
    This replaces the slow Groq LLM eligibility batch as the primary
    recommendation signal on the dashboard.
    """
    db = get_supabase()
    user_id = str(current_user["id"])

    # Load user + documents for profile text
    docs_result = db.table("user_documents").select("doc_type, extracted_data").eq("user_id", user_id).execute()
    documents = docs_result.data or []

    profile_text = build_user_query_text(current_user, documents)
    if not profile_text.strip():
        # Profile not filled yet — return popular/random schemes
        fallback = db.table("government_schemes").select(
            "id, scheme_name, slug, details, benefits, eligibility, level, scheme_category, tags, application, documents"
        ).limit(limit).execute()
        return {"schemes": fallback.data or [], "total": len(fallback.data or []), "recommendation_type": "popular"}

    # Embed the user profile (NOT a retrieval query — it's a document embedding)
    user_vec = await embed_text(profile_text, is_query=False)
    vec_literal = _pgvector_literal(user_vec)

    rpc_params: dict = {
        "query_embedding": vec_literal,
        "match_count": limit,
        "category_filter": category,
        "level_filter": None,
    }

    result = db.rpc("search_schemes_by_embedding", rpc_params).execute()
    schemes = result.data or []

    return {
        "schemes": schemes,
        "total": len(schemes),
        "recommendation_type": "vector_similarity",
    }


@router.get("/")
async def list_schemes(
    category: Optional[str] = Query(None),
    level: Optional[str] = Query(None),
    query: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=50),
):
    """
    List schemes with optional category/level filters and full-text search.
    For semantic search use GET /schemes/search?q=... instead.
    """
    db = get_supabase()
    offset = (page - 1) * page_size

    q = db.table("government_schemes").select(
        "id, scheme_name, slug, details, benefits, eligibility, level, scheme_category, tags, application, documents"
    )

    if category:
        q = q.eq("scheme_category", category)
    if level:
        q = q.eq("level", level)
    if query:
        # tsvector full-text search (fast, keyword-based)
        q = q.text_search("search_vector", query)

    q = q.range(offset, offset + page_size - 1)
    result = q.execute()

    return {
        "schemes": result.data or [],
        "page": page,
        "page_size": page_size,
    }


@router.get("/eligible")
async def get_my_eligible_schemes(
    current_user: dict = Depends(get_current_user),
):
    """
    Get schemes from the eligibility cache (Groq LLM computed).
    Falls back to vector recommendations if cache is empty.
    """
    db = get_supabase()
    user_id = str(current_user["id"])

    cache_result = db.table("user_eligibility").select("scheme_id").eq("user_id", user_id).limit(1).execute()

    if not cache_result.data:
        await refresh_user_eligibility(user_id, db)

    eligible = await get_eligible_schemes(user_id, db)

    schemes = []
    for row in eligible:
        scheme_data = row.get("government_schemes", {})
        if scheme_data:
            schemes.append({
                **scheme_data,
                "eligible": row.get("eligible", True),
                "eligibility_reason": row.get("eligibility_reason", ""),
            })

    return {"schemes": schemes, "total": len(schemes)}


@router.post("/eligible/refresh")
async def refresh_eligibility(
    bg: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
):
    db = get_supabase()
    user_id = str(current_user["id"])
    bg.add_task(refresh_user_eligibility, user_id, db)
    return {"message": "Eligibility refresh started", "user_id": user_id}


@router.get("/{slug}")
async def get_scheme_by_slug(
    slug: str,
    current_user: dict = Depends(get_current_user),
):
    db = get_supabase()
    result = (
        db.table("government_schemes")
        .select("*")
        .eq("slug", slug)
        .single()
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Scheme not found")

    scheme = result.data

    elig_result = (
        db.table("user_eligibility")
        .select("eligible, eligibility_reason")
        .eq("user_id", str(current_user["id"]))
        .eq("scheme_id", str(scheme["id"]))
        .execute()
    )
    if elig_result.data:
        scheme["eligible"] = elig_result.data[0]["eligible"]
        scheme["eligibility_reason"] = elig_result.data[0]["eligibility_reason"]

    return scheme
