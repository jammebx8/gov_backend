from fastapi import APIRouter, Depends, Query, BackgroundTasks, HTTPException
from typing import Optional
from app.models.scheme import SchemeWithEligibility
from app.utils.auth import get_current_user
from app.services.eligibility import refresh_user_eligibility, get_eligible_schemes
from app.services.summarizer import get_or_create_summary
from app.database import get_supabase

router = APIRouter(prefix="/schemes", tags=["schemes"])

CATEGORIES = [
    "Education", "Agriculture", "Housing", "Health",
    "Women & Child", "Social Welfare", "Employment",
    "Business & MSME", "Pension", "Scholarship",
    "Skill Development", "Minority Welfare", "Differently Abled",
    "Financial Inclusion",
]

# Occupation → likely relevant categories (used for profile-based matching)
_OCC_CATEGORIES: dict[str, list[str]] = {
    "farmer":    ["Agriculture", "Financial Inclusion", "Housing"],
    "student":   ["Education", "Scholarship", "Skill Development"],
    "business":  ["Business & MSME", "Financial Inclusion", "Skill Development"],
    "labourer":  ["Social Welfare", "Employment", "Health"],
    "self-employed": ["Business & MSME", "Financial Inclusion"],
    "unemployed":    ["Employment", "Skill Development", "Social Welfare"],
    "government":    ["Pension", "Health", "Housing"],
    "teacher":       ["Education", "Pension"],
}

# Caste → likely relevant categories
_CASTE_CATEGORIES: dict[str, list[str]] = {
    "SC":  ["Social Welfare", "Scholarship", "Housing", "Financial Inclusion", "Minority Welfare"],
    "ST":  ["Social Welfare", "Scholarship", "Housing", "Financial Inclusion", "Minority Welfare"],
    "OBC": ["Social Welfare", "Scholarship", "Employment", "Business & MSME"],
    "EWS": ["Housing", "Financial Inclusion", "Scholarship", "Health"],
}


def _pgvector_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in vec) + "]"


def _profile_category_hints(user: dict) -> list[str]:
    """
    Derive a prioritised list of scheme categories from the user's profile
    without any ML — purely rule-based.  Used as the fast recommendation
    fallback when the ONNX embedding model is unavailable on Vercel.
    """
    cats: list[str] = []

    # Occupation-based
    occ = (user.get("occupation") or "").lower()
    for key, cat_list in _OCC_CATEGORIES.items():
        if key in occ:
            cats.extend(cat_list)

    # Flag-based
    if user.get("is_student"):
        cats.extend(["Education", "Scholarship", "Skill Development"])
    if user.get("is_farmer"):
        cats.extend(["Agriculture", "Financial Inclusion", "Housing"])
    if user.get("disability_status"):
        cats.extend(["Differently Abled", "Social Welfare", "Health"])

    # Caste-based
    caste = (user.get("caste_category") or "").upper()
    cats.extend(_CASTE_CATEGORIES.get(caste, []))

    # Income-based
    income = user.get("annual_income") or 0
    if income < 150000:
        cats.extend(["Social Welfare", "Financial Inclusion", "Health", "Housing"])

    # Gender-based
    gender = (user.get("gender") or "").lower()
    if gender == "female":
        cats.extend(["Women & Child", "Scholarship", "Social Welfare"])

    # Always add some broad safety-net categories
    cats.extend(["Health", "Employment", "Social Welfare"])

    # Deduplicate while preserving order (most relevant first)
    seen: set[str] = set()
    ordered: list[str] = []
    for c in cats:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


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
    Semantic vector search.  Tries pgvector first; falls back to tsvector
    full-text search if ONNX embedding is unavailable on this instance.
    """
    db = get_supabase()

    # ── try vector path ───────────────────────────────────────────────────
    try:
        from app.services.embeddings import embed_text
        import asyncio

        query_vec  = await asyncio.wait_for(embed_text(q, is_query=True), timeout=8.0)
        vec_literal = _pgvector_literal(query_vec)

        rpc_params: dict = {
            "query_embedding": vec_literal,
            "match_count": limit,
            "category_filter": category,
            "level_filter": level,
        }
        result = db.rpc("search_schemes_by_embedding", rpc_params).execute()
        schemes = result.data or []

        if schemes:
            return {
                "schemes": schemes,
                "query": q,
                "total": len(schemes),
                "search_type": "vector",
            }
    except Exception:
        pass  # fall through to tsvector

    # ── tsvector fallback ─────────────────────────────────────────────────
    qb = db.table("government_schemes").select(
        "id, scheme_name, slug, details, benefits, eligibility, level, scheme_category, tags, application, documents"
    ).text_search("search_vector", q)

    if category:
        qb = qb.eq("scheme_category", category)
    if level:
        qb = qb.eq("level", level)

    result = qb.limit(limit).execute()
    schemes = result.data or []

    return {
        "schemes": schemes,
        "query": q,
        "total": len(schemes),
        "search_type": "fulltext",
    }


@router.get("/recommend")
async def recommend_for_user(
    limit: int = Query(15, ge=1, le=50),
    category: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_user),
):
    """
    Personalised scheme recommendations — three-tier strategy:

    1. Eligibility cache (LLM-verified) — fastest, most accurate.
    2. Profile-based category matching — rule-based, instant, no ML needed.
    3. Latest schemes fallback — when profile is completely empty.
    """
    db = get_supabase()
    user_id = str(current_user["id"])

    # ── 1. LLM eligibility cache ──────────────────────────────────────────
    elig_result = (
        db.table("user_eligibility")
        .select("scheme_id, eligibility_reason, eligible")
        .eq("user_id", user_id)
        .eq("eligible", True)
        .execute()
    )

    if elig_result.data:
        eligible_ids = [r["scheme_id"] for r in elig_result.data]
        reason_map   = {r["scheme_id"]: r.get("eligibility_reason", "")
                        for r in elig_result.data}

        qb = db.table("government_schemes").select(
            "id, scheme_name, slug, details, benefits, eligibility, "
            "level, scheme_category, tags, application, documents"
        ).in_("id", eligible_ids)

        if category:
            qb = qb.eq("scheme_category", category)

        schemes_result = qb.limit(limit).execute()
        schemes = schemes_result.data or []

        for s in schemes:
            s["eligible"]            = True
            s["eligibility_reason"]  = reason_map.get(str(s["id"]), "")

        return {
            "schemes": schemes,
            "total": len(schemes),
            "recommendation_type": "eligible",
        }

    # ── 2. Profile-based category matching (no ONNX) ──────────────────────
    # Derive categories from occupation / caste / income / gender / flags.
    # Then fetch schemes from those categories in priority order.
    hint_cats = _profile_category_hints(current_user)

    if hint_cats:
        # Build a prioritised result: fetch up to ceil(limit/len) per category
        # so the top categories dominate the list.
        per_cat   = max(3, (limit // max(len(hint_cats[:6]), 1)) + 1)
        collected: list[dict] = []
        seen_ids:  set[str]   = set()

        for cat in hint_cats[:8]:          # cap at 8 categories to stay fast
            if len(collected) >= limit:
                break

            qb = db.table("government_schemes").select(
                "id, scheme_name, slug, details, benefits, eligibility, "
                "level, scheme_category, tags, application, documents"
            ).eq("scheme_category", cat if not category else category)

            # Prefer Central schemes for broader coverage
            rows = qb.order("level").limit(per_cat).execute()
            for row in (rows.data or []):
                if row["id"] not in seen_ids:
                    seen_ids.add(row["id"])
                    collected.append(row)

            if category:
                break  # only one category requested

        if collected:
            return {
                "schemes": collected[:limit],
                "total": len(collected[:limit]),
                "recommendation_type": "profile_match",
            }

    # ── 3. Latest schemes fallback (empty profile) ────────────────────────
    qb = db.table("government_schemes").select(
        "id, scheme_name, slug, details, benefits, eligibility, "
        "level, scheme_category, tags, application, documents"
    )
    if category:
        qb = qb.eq("scheme_category", category)

    fallback = qb.order("created_at", desc=True).limit(limit).execute()
    return {
        "schemes": fallback.data or [],
        "total": len(fallback.data or []),
        "recommendation_type": "popular",
    }


@router.get("/")
async def list_schemes(
    category: Optional[str] = Query(None),
    level: Optional[str] = Query(None),
    query: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=50),
):
    db = get_supabase()
    offset = (page - 1) * page_size

    qb = db.table("government_schemes").select(
        "id, scheme_name, slug, details, benefits, eligibility, level, scheme_category, tags, application, documents"
    )

    if category:
        qb = qb.eq("scheme_category", category)
    if level:
        qb = qb.eq("level", level)
    if query:
        qb = qb.text_search("search_vector", query)

    qb = qb.range(offset, offset + page_size - 1)
    result = qb.execute()

    return {
        "schemes": result.data or [],
        "page": page,
        "page_size": page_size,
    }


@router.get("/eligible")
async def get_my_eligible_schemes(
    current_user: dict = Depends(get_current_user),
):
    db = get_supabase()
    user_id = str(current_user["id"])

    cache_result = (
        db.table("user_eligibility")
        .select("scheme_id")
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )

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
        scheme["eligible"]            = elig_result.data[0]["eligible"]
        scheme["eligibility_reason"]  = elig_result.data[0]["eligibility_reason"]

    # AI summary — cached after first call, fast on repeat visits
    try:
        ai = await get_or_create_summary(scheme, db)
        scheme["summary"]            = ai.get("summary", "")
        scheme["benefits_simple"]    = ai.get("benefits_simple", "")
        scheme["eligibility_simple"] = ai.get("eligibility_simple", "")
        scheme["documents_simple"]   = ai.get("documents_simple", "")
    except Exception:
        pass

    return scheme


@router.get("/{slug}/summary")
async def get_scheme_summary(
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

    ai = await get_or_create_summary(result.data, db)
    return {"scheme_id": str(result.data["id"]), **ai}
