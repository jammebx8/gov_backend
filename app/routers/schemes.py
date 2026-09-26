import re
from fastapi import APIRouter, Depends, Query, BackgroundTasks, HTTPException
from typing import Optional
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

_SCHEME_COLS = (
    "id, scheme_name, slug, details, benefits, eligibility, "
    "level, scheme_category, tags, application, documents"
)

_OCC_CATEGORIES: dict[str, list[str]] = {
    "farmer":        ["Agriculture", "Financial Inclusion", "Housing"],
    "student":       ["Education", "Scholarship", "Skill Development"],
    "business":      ["Business & MSME", "Financial Inclusion", "Skill Development"],
    "labourer":      ["Social Welfare", "Employment", "Health"],
    "self-employed": ["Business & MSME", "Financial Inclusion"],
    "unemployed":    ["Employment", "Skill Development", "Social Welfare"],
    "government":    ["Pension", "Health", "Housing"],
    "teacher":       ["Education", "Pension"],
}

_CASTE_CATEGORIES: dict[str, list[str]] = {
    "SC":  ["Social Welfare", "Scholarship", "Housing", "Financial Inclusion", "Minority Welfare"],
    "ST":  ["Social Welfare", "Scholarship", "Housing", "Financial Inclusion", "Minority Welfare"],
    "OBC": ["Social Welfare", "Scholarship", "Employment", "Business & MSME"],
    "EWS": ["Housing", "Financial Inclusion", "Scholarship", "Health"],
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_tsquery(q: str) -> str:
    """
    Convert a plain user query into a valid Postgres tsquery expression.
    Strips special chars, joins words with & (AND).

    "single parent"  →  "single & parent"
    "widow's scheme" →  "widow & s & scheme"  (apostrophe stripped)
    """
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", q)
    tokens  = [t for t in cleaned.split() if len(t) > 1]   # skip 1-char noise
    if not tokens:
        return "scheme"
    return " & ".join(tokens)


def _ilike_search(db, q: str, limit: int, category: Optional[str], level: Optional[str]) -> list[dict]:
    """
    Last-resort keyword search via ilike — never fails regardless of query format.
    Tries scheme_name first, then details.
    """
    keyword = q.strip().split()[0] if q.strip() else "scheme"

    qb = db.table("government_schemes").select(_SCHEME_COLS)
    if category:
        qb = qb.eq("scheme_category", category)
    if level:
        qb = qb.eq("level", level)
    rows = qb.ilike("scheme_name", f"%{keyword}%").limit(limit).execute()

    if rows.data:
        return rows.data

    # broaden to details column
    qb2 = db.table("government_schemes").select(_SCHEME_COLS)
    if category:
        qb2 = qb2.eq("scheme_category", category)
    rows2 = qb2.ilike("details", f"%{keyword}%").limit(limit).execute()
    return rows2.data or []


def _search(db, q: str, limit: int, category: Optional[str], level: Optional[str]) -> list[dict]:
    """
    Multi-layer search strategy (no ONNX — not viable on Vercel lambdas):

    1. Postgres full-text search (tsvector) — fast, stemmed, ranked.
    2. ilike fallback — catches cases where tsquery has no matches.

    IMPORTANT (supabase-py 2.x):
      text_search() returns SyncQueryRequestBuilder which only supports
      .execute().  All filters and modifiers MUST be chained before it.
    """
    tsq = _to_tsquery(q)

    try:
        qb = db.table("government_schemes").select(_SCHEME_COLS)
        if category:
            qb = qb.eq("scheme_category", category)
        if level:
            qb = qb.eq("level", level)
        # limit BEFORE text_search — cannot chain after
        qb     = qb.limit(limit)
        result = qb.text_search("search_vector", tsq).execute()
        data   = result.data or []
        if data:
            return data
    except Exception:
        pass

    # tsvector returned nothing or errored — fall back to ilike
    return _ilike_search(db, q, limit, category, level)


def _profile_category_hints(user: dict) -> list[str]:
    cats: list[str] = []

    occ = (user.get("occupation") or "").lower()
    for key, cat_list in _OCC_CATEGORIES.items():
        if key in occ:
            cats.extend(cat_list)

    if user.get("is_student"):
        cats.extend(["Education", "Scholarship", "Skill Development"])
    if user.get("is_farmer"):
        cats.extend(["Agriculture", "Financial Inclusion", "Housing"])
    if user.get("disability_status"):
        cats.extend(["Differently Abled", "Social Welfare", "Health"])

    caste = (user.get("caste_category") or "").upper()
    cats.extend(_CASTE_CATEGORIES.get(caste, []))

    income = user.get("annual_income") or 0
    if income < 150000:
        cats.extend(["Social Welfare", "Financial Inclusion", "Health", "Housing"])

    gender = (user.get("gender") or "").lower()
    if gender == "female":
        cats.extend(["Women & Child", "Scholarship", "Social Welfare"])

    cats.extend(["Health", "Employment", "Social Welfare"])

    seen:    set[str]  = set()
    ordered: list[str] = []
    for c in cats:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


# ── endpoints ─────────────────────────────────────────────────────────────────

@router.get("/categories")
async def list_categories():
    return {"categories": CATEGORIES}


@router.get("/search")
async def search_schemes(
    q: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=50),
    category: Optional[str] = Query(None),
    level: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_user),
):
    """
    Full-text search over government schemes.
    Uses Postgres tsvector with ilike fallback — no ONNX/ML dependency.
    """
    db      = get_supabase()
    schemes = _search(db, q, limit, category, level)
    return {"schemes": schemes, "query": q, "total": len(schemes), "search_type": "fulltext"}


@router.get("/recommend")
async def recommend_for_user(
    limit: int = Query(15, ge=1, le=50),
    category: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_user),
):
    """
    Personalised recommendations — three-tier strategy:
    1. LLM-verified eligibility cache (most accurate)
    2. Rule-based profile→category matching (instant, no ML)
    3. Latest schemes fallback (empty profile)
    """
    db      = get_supabase()
    user_id = str(current_user["id"])

    # ── 1. LLM eligibility cache ──────────────────────────────────────────
    elig = (
        db.table("user_eligibility")
        .select("scheme_id, eligibility_reason, eligible")
        .eq("user_id", user_id)
        .eq("eligible", True)
        .execute()
    )
    if elig.data:
        eligible_ids = [r["scheme_id"] for r in elig.data]
        reason_map   = {r["scheme_id"]: r.get("eligibility_reason", "") for r in elig.data}

        qb = db.table("government_schemes").select(_SCHEME_COLS).in_("id", eligible_ids)
        if category:
            qb = qb.eq("scheme_category", category)
        rows    = qb.limit(limit).execute()
        schemes = rows.data or []
        for s in schemes:
            s["eligible"]           = True
            s["eligibility_reason"] = reason_map.get(str(s["id"]), "")
        return {"schemes": schemes, "total": len(schemes), "recommendation_type": "eligible"}

    # ── 2. Profile-based category matching ───────────────────────────────
    hint_cats = _profile_category_hints(current_user)
    if hint_cats:
        per_cat   = max(3, (limit // max(len(hint_cats[:6]), 1)) + 1)
        collected: list[dict] = []
        seen_ids:  set[str]   = set()

        for cat in hint_cats[:8]:
            if len(collected) >= limit:
                break
            target_cat = category if category else cat
            rows = (
                db.table("government_schemes")
                .select(_SCHEME_COLS)
                .eq("scheme_category", target_cat)
                .order("level")
                .limit(per_cat)
                .execute()
            )
            for row in (rows.data or []):
                if row["id"] not in seen_ids:
                    seen_ids.add(row["id"])
                    collected.append(row)
            if category:
                break

        if collected:
            return {
                "schemes": collected[:limit],
                "total":   len(collected[:limit]),
                "recommendation_type": "profile_match",
            }

    # ── 3. Latest schemes fallback ────────────────────────────────────────
    qb = db.table("government_schemes").select(_SCHEME_COLS)
    if category:
        qb = qb.eq("scheme_category", category)
    fallback = qb.order("created_at", desc=True).limit(limit).execute()
    return {
        "schemes": fallback.data or [],
        "total":   len(fallback.data or []),
        "recommendation_type": "popular",
    }


@router.get("/")
async def list_schemes(
    category:  Optional[str] = Query(None),
    level:     Optional[str] = Query(None),
    query:     Optional[str] = Query(None),
    page:      int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=50),
):
    db     = get_supabase()
    offset = (page - 1) * page_size

    if query:
        # All modifiers BEFORE text_search (supabase-py 2.x constraint)
        qb = db.table("government_schemes").select(_SCHEME_COLS)
        if category:
            qb = qb.eq("scheme_category", category)
        if level:
            qb = qb.eq("level", level)
        qb     = qb.limit(page_size).offset(offset)
        tsq    = _to_tsquery(query)
        result = qb.text_search("search_vector", tsq).execute()
    else:
        qb = db.table("government_schemes").select(_SCHEME_COLS)
        if category:
            qb = qb.eq("scheme_category", category)
        if level:
            qb = qb.eq("level", level)
        result = qb.range(offset, offset + page_size - 1).execute()

    return {"schemes": result.data or [], "page": page, "page_size": page_size}


@router.get("/eligible")
async def get_my_eligible_schemes(
    current_user: dict = Depends(get_current_user),
):
    db      = get_supabase()
    user_id = str(current_user["id"])

    cache = (
        db.table("user_eligibility")
        .select("scheme_id")
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    if not cache.data:
        await refresh_user_eligibility(user_id, db)

    eligible = await get_eligible_schemes(user_id, db)
    schemes  = []
    for row in eligible:
        sd = row.get("government_schemes", {})
        if sd:
            schemes.append({
                **sd,
                "eligible":           row.get("eligible", True),
                "eligibility_reason": row.get("eligibility_reason", ""),
            })
    return {"schemes": schemes, "total": len(schemes)}


@router.post("/eligible/refresh")
async def refresh_eligibility(
    bg: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
):
    db      = get_supabase()
    user_id = str(current_user["id"])
    bg.add_task(refresh_user_eligibility, user_id, db)
    return {"message": "Eligibility refresh started", "user_id": user_id}


@router.get("/{slug}")
async def get_scheme_by_slug(
    slug: str,
    current_user: dict = Depends(get_current_user),
):
    db     = get_supabase()
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

    elig = (
        db.table("user_eligibility")
        .select("eligible, eligibility_reason")
        .eq("user_id", str(current_user["id"]))
        .eq("scheme_id", str(scheme["id"]))
        .execute()
    )
    if elig.data:
        scheme["eligible"]           = elig.data[0]["eligible"]
        scheme["eligibility_reason"] = elig.data[0]["eligibility_reason"]

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
    db     = get_supabase()
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
