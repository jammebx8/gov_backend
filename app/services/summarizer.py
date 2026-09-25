"""
AI Summarizer — uses Groq to rewrite bureaucratic government scheme text
into plain, friendly English for regular citizens.

Results are cached in the `scheme_summaries` Supabase table so the LLM is
only ever called once per scheme.  Subsequent requests return the cached
version immediately.
"""
import json
from app.services.eligibility import get_groq
from app.config import settings


_SYSTEM = (
    "You are a friendly plain-English writer helping Indian citizens understand "
    "government schemes. Rewrite the given text so a Class-10 student can easily "
    "understand it. Use simple, short sentences. Avoid jargon and legal language. "
    "Write in active voice. Return ONLY a JSON object — no markdown, no extra text."
)


async def _call_groq(prompt: str) -> str:
    client = get_groq()
    response = await client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.3,
        max_tokens=1024,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or "{}"


async def summarize_scheme(scheme: dict) -> dict:
    """
    Given a raw scheme dict, return a dict with AI-simplified fields:
      summary, benefits_simple, eligibility_simple, documents_simple

    Calls Groq once, caches the result in `scheme_summaries`.
    """
    prompt = f"""Simplify the following government scheme information into plain English.

Scheme Name: {scheme.get('scheme_name', '')}

Original Description:
{scheme.get('details', '')}

Original Benefits:
{scheme.get('benefits', '')}

Original Eligibility Criteria:
{scheme.get('eligibility', '')}

Original Required Documents:
{scheme.get('documents', 'Not specified')}

Return a JSON object with exactly these four keys:
{{
  "summary": "2-3 sentence plain English summary of what this scheme is and who it helps",
  "benefits_simple": "Plain bullet-point list of what you get. Each point on a new line starting with •",
  "eligibility_simple": "Plain bullet-point list of who can apply. Each point on a new line starting with •",
  "documents_simple": "Plain bullet-point list of documents needed. Each point on a new line starting with •. If no documents specified write 'No specific documents required.'"
}}"""

    try:
        raw = await _call_groq(prompt)
        data = json.loads(raw.strip())
        # Validate expected keys are present
        required = {"summary", "benefits_simple", "eligibility_simple", "documents_simple"}
        if not required.issubset(data.keys()):
            raise ValueError("Missing keys in LLM response")
        return data
    except Exception:
        # Fallback: return original text so the page still works
        return {
            "summary": scheme.get("details", ""),
            "benefits_simple": scheme.get("benefits", ""),
            "eligibility_simple": scheme.get("eligibility", ""),
            "documents_simple": scheme.get("documents", ""),
        }


async def get_or_create_summary(scheme: dict, db) -> dict:
    """
    Return cached summary from `scheme_summaries` table if it exists,
    otherwise call Groq, store the result, and return it.
    """
    scheme_id = str(scheme.get("id", ""))
    if not scheme_id:
        return await summarize_scheme(scheme)

    # 1. Check cache
    try:
        cached = (
            db.table("scheme_summaries")
            .select("summary, benefits_simple, eligibility_simple, documents_simple")
            .eq("scheme_id", scheme_id)
            .single()
            .execute()
        )
        if cached.data:
            return cached.data
    except Exception:
        pass  # table may not exist yet — fall through to generate

    # 2. Generate with LLM
    result = await summarize_scheme(scheme)

    # 3. Persist so we never call LLM again for this scheme
    try:
        db.table("scheme_summaries").upsert({
            "scheme_id":          scheme_id,
            "summary":            result["summary"],
            "benefits_simple":    result["benefits_simple"],
            "eligibility_simple": result["eligibility_simple"],
            "documents_simple":   result["documents_simple"],
        }).execute()
    except Exception:
        pass  # best-effort cache write — don't fail the request

    return result
