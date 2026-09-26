import re
from fastapi import APIRouter, Depends, Query, BackgroundTasks, HTTPException
from typing import Optional
from app.utils.auth import get_current_user
from app.services.eligibility import refresh_user_eligibility, get_eligible_schemes
from app.services.summarizer import get_or_create_summary
from app.database import get_supabase

router = APIRouter(prefix="/schemes", tags=["schemes"])

# Real canonical category names as stored in government_schemes.scheme_category.
# Values in the DB can be comma-joined multi-category strings like:
#   "Education & Learning, Health & Wellness, Women and Child"
# The frontend filter uses containsCategory() to match against these.
CATEGORIES = [
    "Agriculture,Rural & Environment",
    "Banking,Financial Services and Insurance",
    "Business & Entrepreneurship",
    "Education & Learning",
    "Health & Wellness",
    "Housing & Shelter",
    "Public Safety,Law & Justice",
    "Science, IT & Communications",
    "Skills & Employment",
    "Social welfare & Empowerment",
    "Sports & Culture",
    "Transport & Infrastructure",
    "Travel & Tourism",
    "Utility & Sanitation",
    "Women and Child",
]

_SCHEME_COLS = (
    "id, scheme_name, slug, details, benefits, eligibility, "
    "level, scheme_category, tags, application, documents"
)

# ── Profile → category mapping (real DB names) ────────────────────────────────
_OCC_CATEGORIES: dict[str, list[str]] = {
    "farmer":        ["Agriculture,Rural & Environment", "Banking,Financial Services and Insurance", "Housing & Shelter"],
    "student":       ["Education & Learning", "Skills & Employment", "Banking,Financial Services and Insurance"],
    "business":      ["Business & Entrepreneurship", "Banking,Financial Services and Insurance", "Skills & Employment"],
    "labourer":      ["Social welfare & Empowerment", "Skills & Employment", "Health & Wellness"],
    "self-employed": ["Business & Entrepreneurship", "Banking,Financial Services and Insurance"],
    "unemployed":    ["Skills & Employment", "Social welfare & Empowerment", "Business & Entrepreneurship"],
    "government":    ["Social welfare & Empowerment", "Health & Wellness", "Housing & Shelter"],
    "teacher":       ["Education & Learning", "Social welfare & Empowerment"],
    "artisan":       ["Business & Entrepreneurship", "Skills & Employment", "Social welfare & Empowerment"],
    "fisher":        ["Agriculture,Rural & Environment", "Social welfare & Empowerment"],
}

_CASTE_CATEGORIES: dict[str, list[str]] = {
    "SC":  ["Social welfare & Empowerment", "Education & Learning", "Housing & Shelter",
            "Banking,Financial Services and Insurance", "Skills & Employment"],
    "ST":  ["Social welfare & Empowerment", "Education & Learning", "Housing & Shelter",
            "Agriculture,Rural & Environment", "Banking,Financial Services and Insurance"],
    "OBC": ["Social welfare & Empowerment", "Education & Learning", "Skills & Employment",
            "Business & Entrepreneurship"],
    "EWS": ["Housing & Shelter", "Banking,Financial Services and Insurance",
            "Education & Learning", "Health & Wellness"],
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_tsquery(q: str) -> str:
    """Convert plain query to valid Postgres tsquery (words joined with &)."""
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", q)
    tokens  = [t for t in cleaned.split() if len(t) > 1]
    return " & ".join(tokens) if tokens else "scheme"


def _ilike_search(db, q: str, limit: int, category: Optional[str], level: Optional[str]) -> list[dict]:
    """Keyword fallback using ilike — never fails on query format."""
    keyword = q.strip().split()[0] if q.strip() else "scheme"
    qb = db.table("government_schemes").select(_SCHEME_COLS)
    if category:
        qb = qb.ilike("scheme_category", f"%{category}%")
    if level:
        qb = qb.eq("level", level)
    rows = qb.ilike("scheme_name", f"%{keyword}%").limit(limit).execute()
    if rows.data:
        return rows.data
    qb2 = db.table("government_schemes").select(_SCHEME_COLS)
    if category:
        qb2 = qb2.ilike("scheme_category", f"%{category}%")
    rows2 = qb2.ilike("details", f"%{keyword}%").limit(limit).execute()
    return rows2.data or []


def _search(db, q: str, limit: int, category: Optional[str], level: Optional[str]) -> list[dict]:
    """FTS with ilike fallback. All modifiers before text_search (supabase-py 2.x)."""
    tsq = _to_tsquery(q)
    try:
        qb = db.table("government_schemes").select(_SCHEME_COLS)
        if category:
            # DB category field may be a multi-value CSV — use ilike for partial match
            qb = qb.ilike("scheme_category", f"%{category}%")
        if level:
            qb = qb.eq("level", level)
        qb     = qb.limit(limit)
        result = qb.text_search("search_vector", tsq).execute()
        data   = result.data or []
        if data:
            return data
    except Exception:
        pass
    return _ilike_search(db, q, limit, category, level)


def _compute_match_score(user: dict, scheme: dict) -> int:
    """
    Compute a 0-100 profile match score for a scheme based on the user's
    demographic data.  No ML — purely rule-based signal scoring.
    """
    score = 0
    cat   = (scheme.get("scheme_category") or "").lower()
    elig  = (scheme.get("eligibility") or "").lower()

    # ── Occupation / flags (up to 30 pts) ────────────────────────────────
    occ = (user.get("occupation") or "").lower()
    if user.get("is_farmer") or "farmer" in occ:
        if "agricultur" in cat or "rural" in cat:
            score += 30
        elif "agricultur" in elig or "farmer" in elig or "kisan" in elig:
            score += 20
    if user.get("is_student") or "student" in occ:
        if "education" in cat or "learning" in cat:
            score += 30
        elif "student" in elig or "education" in elig:
            score += 20
    if "business" in occ or "entrepreneur" in occ or "self-employ" in occ:
        if "business" in cat or "entrepreneur" in cat:
            score += 25
    if "labourer" in occ or "labour" in occ or "worker" in occ:
        if "skill" in cat or "employment" in cat or "social welfare" in cat:
            score += 25
    if user.get("disability_status"):
        if "differently abled" in elig or "disability" in elig or "divyangjan" in elig:
            score += 30

    # ── Gender (up to 25 pts) ─────────────────────────────────────────────
    gender = (user.get("gender") or "").lower()
    if gender == "female":
        if "women" in cat or "child" in cat:
            score += 25
        elif "women" in elig or "widow" in elig or "girl" in elig:
            score += 20

    # ── Caste (up to 20 pts) ─────────────────────────────────────────────
    caste = (user.get("caste_category") or "").upper()
    if caste in ("SC", "ST"):
        if "social welfare" in cat or "empowerment" in cat:
            score += 20
        elif any(k in elig for k in ("sc", "st", "scheduled", "tribal", "dalit")):
            score += 15
    elif caste == "OBC":
        if any(k in elig for k in ("obc", "backward")):
            score += 15
    elif caste == "EWS":
        if any(k in elig for k in ("ews", "economically weaker")):
            score += 15

    # ── Income (up to 20 pts) ─────────────────────────────────────────────
    income = user.get("annual_income") or 0
    if income > 0:
        if income < 100000:
            if any(k in elig for k in ("bpl", "below poverty", "annual income", "income limit")):
                score += 20
            elif "social welfare" in cat or "financial" in cat:
                score += 10
        elif income < 300000:
            if "financial" in cat or "banking" in cat:
                score += 10

    # ── State match (10 pts) ─────────────────────────────────────────────
    state = (user.get("state") or "").lower()
    if state and state in elig:
        score += 10

    # ── Base relevance: social welfare / health always broadly relevant ───
    if score == 0:
        if any(k in cat for k in ("health", "social welfare", "education")):
            score = 25
        else:
            score = 15

    return min(score, 100)


def _profile_category_hints(user: dict) -> list[str]:
    """Return prioritised list of DB category names from user profile."""
    cats: list[str] = []

    occ = (user.get("occupation") or "").lower()
    for key, cat_list in _OCC_CATEGORIES.items():
        if key in occ:
            cats.extend(cat_list)

    if user.get("is_student"):
        cats.extend(["Education & Learning", "Skills & Employment",
                     "Banking,Financial Services and Insurance"])
    if user.get("is_farmer"):
        cats.extend(["Agriculture,Rural & Environment",
                     "Banking,Financial Services and Insurance", "Housing & Shelter"])
    if user.get("disability_status"):
        cats.extend(["Social welfare & Empowerment", "Health & Wellness",
                     "Banking,Financial Services and Insurance"])

    caste = (user.get("caste_category") or "").upper()
    cats.extend(_CASTE_CATEGORIES.get(caste, []))

    income = user.get("annual_income") or 0
    if income < 150000:
        cats.extend(["Social welfare & Empowerment",
                     "Banking,Financial Services and Insurance",
                     "Health & Wellness", "Housing & Shelter"])

    gender = (user.get("gender") or "").lower()
    if gender == "female":
        cats.extend(["Women and Child", "Education & Learning",
                     "Social welfare & Empowerment"])

    cats.extend(["Health & Wellness", "Skills & Employment",
                 "Social welfare & Empowerment"])

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
    db      = get_supabase()
    schemes = _search(db, q, limit, category, level)
    return {"schemes": schemes, "query": q, "total": len(schemes), "search_type": "fulltext"}


@router.get("/recommend")
async def recommend_for_user(
    limit: int = Query(30, ge=1, le=100),
    category: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_user),
):
    """
    Personalised recommendations with profile match scores.

    Strategy:
    1. LLM eligibility cache — return only verified-eligible schemes + score.
    2. Profile-based category matching — rule-based, instant, no ML.
       Uses ilike on scheme_category to handle multi-value CSV DB values.
    3. Latest schemes fallback — empty profile.

    Every scheme gets a match_score (0-100) computed from user demographics.
    Results are sorted by score descending.
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
            qb = qb.ilike("scheme_category", f"%{category}%")
        rows    = qb.limit(limit).execute()
        schemes = rows.data or []

        for s in schemes:
            s["eligible"]           = True
            s["eligibility_reason"] = reason_map.get(str(s["id"]), "")
            s["match_score"]        = _compute_match_score(current_user, s)

        schemes.sort(key=lambda s: s["match_score"], reverse=True)
        return {"schemes": schemes, "total": len(schemes), "recommendation_type": "eligible"}

    # ── 2. Profile-based category matching ───────────────────────────────
    hint_cats = _profile_category_hints(current_user)
    if hint_cats:
        per_cat   = max(4, (limit // max(len(hint_cats[:8]), 1)) + 2)
        collected: list[dict] = []
        seen_ids:  set[str]   = set()

        for cat in hint_cats[:10]:
            if len(collected) >= limit:
                break
            target_cat = category if category else cat

            # Use ilike so "Education & Learning, Health & Wellness" matches "Education & Learning"
            rows = (
                db.table("government_schemes")
                .select(_SCHEME_COLS)
                .ilike("scheme_category", f"%{target_cat}%")
                .order("level")
                .limit(per_cat)
                .execute()
            )
            for row in (rows.data or []):
                if row["id"] not in seen_ids:
                    seen_ids.add(row["id"])
                    row["match_score"] = _compute_match_score(current_user, row)
                    collected.append(row)
            if category:
                break

        if collected:
            collected.sort(key=lambda s: s["match_score"], reverse=True)
            return {
                "schemes": collected[:limit],
                "total":   len(collected[:limit]),
                "recommendation_type": "profile_match",
            }

    # ── 3. Latest schemes fallback ────────────────────────────────────────
    qb = db.table("government_schemes").select(_SCHEME_COLS)
    if category:
        qb = qb.ilike("scheme_category", f"%{category}%")
    fallback = qb.order("created_at", desc=True).limit(limit).execute()
    schemes  = fallback.data or []
    for s in schemes:
        s["match_score"] = _compute_match_score(current_user, s)
    schemes.sort(key=lambda s: s["match_score"], reverse=True)
    return {
        "schemes": schemes,
        "total":   len(schemes),
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
        qb = db.table("government_schemes").select(_SCHEME_COLS)
        if category:
            qb = qb.ilike("scheme_category", f"%{category}%")
        if level:
            qb = qb.eq("level", level)
        qb     = qb.limit(page_size).offset(offset)
        tsq    = _to_tsquery(query)
        result = qb.text_search("search_vector", tsq).execute()
    else:
        qb = db.table("government_schemes").select(_SCHEME_COLS)
        if category:
            qb = qb.ilike("scheme_category", f"%{category}%")
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
