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
    "You are a friendly plain-English assistant helping Indian citizens understand "
    "government schemes. Your job is to rewrite bureaucratic text so a Class-8 "
    "student can understand it easily.\n\n"
    "Rules:\n"
    "- Use very simple words. No jargon or legal language.\n"
    "- Write in active voice with short sentences.\n"
    "- For lists (benefits, eligibility, documents): ALWAYS output each item on its "
    "own line starting with '• ' (bullet + space). One item per line. Never combine "
    "multiple items into one line.\n"
    "- For documents: list EACH document separately. Never write them as a sentence.\n"
    "- Return ONLY a valid JSON object — no markdown, no explanation, no extra text."
)


async def _call_groq(prompt: str) -> str:
    client = get_groq()
    response = await client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.2,
        max_tokens=1500,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or "{}"


async def summarize_scheme(scheme: dict) -> dict:
    """
    Calls Groq to generate simplified scheme fields.
    Returns: summary, benefits_simple, eligibility_simple, documents_simple
    """
    prompt = f"""Simplify this government scheme into plain English.

Scheme Name: {scheme.get('scheme_name', '')}

Description:
{scheme.get('details', '')}

Benefits:
{scheme.get('benefits', '')}

Eligibility Criteria:
{scheme.get('eligibility', '')}

Required Documents:
{scheme.get('documents', 'Not specified')}

Return this exact JSON structure:
{{
  "summary": "2-3 simple sentences explaining what this scheme is, who it helps, and what they get.",
  "benefits_simple": "Each benefit on its own line starting with '• '. Example:\\n• You get ₹5,000 every month\\n• Free health insurance up to ₹2 lakh\\n• Subsidised housing loan",
  "eligibility_simple": "Each eligibility condition on its own line starting with '• '. Example:\\n• You must be between 18 and 45 years old\\n• Your family income must be below ₹1 lakh per year\\n• You must be a resident of the applying state",
  "documents_simple": "Each document on its own line starting with '• '. One document per line. Example:\\n• Aadhaar Card\\n• Income Certificate\\n• Bank passbook (first page)\\n• Passport size photograph\\nIf no specific documents are required, write: • No specific documents required"
}}

Important: In benefits_simple, eligibility_simple, and documents_simple — put EACH item on its OWN line. Never combine two items into one line."""

    try:
        raw  = await _call_groq(prompt)
        data = json.loads(raw.strip())
        required = {"summary", "benefits_simple", "eligibility_simple", "documents_simple"}
        if not required.issubset(data.keys()):
            raise ValueError("Missing keys")
        # Post-process: ensure bullets are clean
        for key in ("benefits_simple", "eligibility_simple", "documents_simple"):
            data[key] = _clean_bullets(data[key])
        return data
    except Exception:
        return {
            "summary":            scheme.get("details", ""),
            "benefits_simple":    scheme.get("benefits", ""),
            "eligibility_simple": scheme.get("eligibility", ""),
            "documents_simple":   scheme.get("documents", ""),
        }


def _clean_bullets(text: str) -> str:
    """
    Ensure each line that isn't already a bullet gets prefixed with '• '.
    Removes blank lines. Normalises inconsistent bullet formats (*, -, –).
    """
    if not text:
        return text
    lines = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        # Normalise existing bullet-like prefixes
        for prefix in ("• ", "* ", "- ", "– ", "— ", "> "):
            if line.startswith(prefix):
                line = "• " + line[len(prefix):]
                break
        else:
            # No bullet found — add one unless it looks like a heading
            if not line.startswith("•"):
                line = "• " + line
        lines.append(line)
    return "\n".join(lines)


async def get_or_create_summary(scheme: dict, db) -> dict:
    """
    Return cached summary if it exists, otherwise generate + cache it.
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
            # Apply bullet cleanup to cached data too (in case old entries are messy)
            d = cached.data
            for key in ("benefits_simple", "eligibility_simple", "documents_simple"):
                if d.get(key):
                    d[key] = _clean_bullets(d[key])
            return d
    except Exception:
        pass

    # 2. Generate with LLM
    result = await summarize_scheme(scheme)

    # 3. Cache
    try:
        db.table("scheme_summaries").upsert({
            "scheme_id":          scheme_id,
            "summary":            result["summary"],
            "benefits_simple":    result["benefits_simple"],
            "eligibility_simple": result["eligibility_simple"],
            "documents_simple":   result["documents_simple"],
        }).execute()
    except Exception:
        pass

    return result
