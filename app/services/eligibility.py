"""
Eligibility engine: uses Groq Cloud (openai/gpt-oss-120b) to match user
profiles against government schemes. Processes in batches and caches results.
"""
import json
from groq import AsyncGroq
from app.config import settings
from app.utils.prompt_builder import build_user_profile_text

# Shared async Groq client
_groq_client: AsyncGroq | None = None


def get_groq() -> AsyncGroq:
    global _groq_client
    if _groq_client is None:
        _groq_client = AsyncGroq(api_key=settings.groq_api_key)
    return _groq_client


def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


async def call_groq_text(prompt: str, system: str = "") -> str:
    """Call Groq text model and return raw response string."""
    client = get_groq()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = await client.chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        temperature=0.1,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


async def compute_eligibility(
    user: dict, schemes: list, documents: list = None
) -> list[dict]:
    """
    Use Groq to determine eligibility for a list of schemes.
    Returns list of {scheme_id, eligible, confidence, reason}.
    """
    user_profile_text = build_user_profile_text(user, documents or [])

    system_prompt = (
        "You are a government scheme eligibility expert for India. "
        "Given a user profile and scheme eligibility criteria, determine eligibility. "
        "Be practical — if most criteria are met, lean eligible. "
        "Return ONLY a JSON object with a \"results\" array."
    )

    all_results = []

    for batch in chunked(schemes, 15):
        scheme_list = "\n\n".join([
            f"SCHEME_ID: {s['id']}\n"
            f"Name: {s['scheme_name']}\n"
            f"Category: {s['scheme_category']}\n"
            f"Level: {s['level']}\n"
            f"Eligibility: {s['eligibility']}"
            for s in batch
        ])

        prompt = f"""User Profile:
{user_profile_text}

Evaluate eligibility for each scheme and return JSON:

{{
  "results": [
    {{
      "scheme_id": "uuid-here",
      "eligible": true,
      "confidence": "high",
      "reason": "brief 1-2 sentence explanation"
    }}
  ]
}}

Schemes to evaluate:
{scheme_list}"""

        try:
            raw = await call_groq_text(prompt, system_prompt)
            content = raw.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            data = json.loads(content.strip())
            all_results.extend(data.get("results", []))
        except Exception as e:
            for s in batch:
                all_results.append({
                    "scheme_id": str(s["id"]),
                    "eligible": False,
                    "confidence": "low",
                    "reason": f"Check failed: {str(e)[:100]}",
                })

    return all_results


async def refresh_user_eligibility(user_id: str, db) -> list[dict]:
    """Load user + docs + all schemes, run LLM check, cache results."""
    user_result = db.table("users").select("*").eq("id", user_id).single().execute()
    if not user_result.data:
        return []
    user = user_result.data

    docs_result = db.table("user_documents").select("*").eq("user_id", user_id).execute()
    documents = docs_result.data or []

    schemes_result = db.table("government_schemes").select(
        "id, scheme_name, scheme_category, level, eligibility"
    ).execute()
    schemes = schemes_result.data or []

    if not schemes:
        return []

    results = await compute_eligibility(user, schemes, documents)

    upsert_rows = [
        {
            "user_id": user_id,
            "scheme_id": r["scheme_id"],
            "eligible": r["eligible"],
            "eligibility_reason": r.get("reason", ""),
        }
        for r in results
    ]
    if upsert_rows:
        db.table("user_eligibility").upsert(upsert_rows).execute()

    return results


async def get_eligible_schemes(user_id: str, db) -> list[dict]:
    """Return eligible schemes with full details from cache."""
    result = (
        db.table("user_eligibility")
        .select("*, government_schemes(*)")
        .eq("user_id", user_id)
        .eq("eligible", True)
        .execute()
    )
    return result.data or []
