"""
Document field extraction using Groq Cloud vision model (qwen/qwen3.8-27b).
Parses uploaded government documents and returns structured JSON data.
"""
import json
import base64
from typing import Optional
from groq import AsyncGroq
from app.config import settings

_groq_client: AsyncGroq | None = None


def get_groq() -> AsyncGroq:
    global _groq_client
    if _groq_client is None:
        _groq_client = AsyncGroq(api_key=settings.groq_api_key)
    return _groq_client


EXTRACTION_PROMPTS = {
    "aadhaar": """\
Extract all visible fields from this Aadhaar card as JSON:
{
  "name": "full name as on card",
  "aadhaar_number": "12-digit number",
  "date_of_birth": "DD/MM/YYYY",
  "gender": "Male/Female/Other",
  "address": "full address",
  "state": "state name",
  "district": "district name",
  "pincode": "6-digit pincode"
}""",

    "pan": """\
Extract all visible fields from this PAN card as JSON:
{
  "name": "full name as on card",
  "father_name": "father's name",
  "pan_number": "10-character PAN",
  "date_of_birth": "DD/MM/YYYY"
}""",

    "income_certificate": """\
Extract all visible fields from this income certificate as JSON:
{
  "name": "certificate holder name",
  "annual_income": <numeric rupees>,
  "financial_year": "e.g. 2023-24",
  "issuing_authority": "issuing office/officer",
  "issue_date": "DD/MM/YYYY",
  "state": "state name",
  "district": "district name",
  "certificate_number": "cert number if visible"
}""",

    "caste_certificate": """\
Extract all visible fields from this caste certificate as JSON:
{
  "name": "certificate holder name",
  "caste_category": "SC/ST/OBC/EWS",
  "caste": "specific caste name",
  "sub_caste": "sub-caste if applicable",
  "state": "state name",
  "district": "district name",
  "issuing_authority": "issuing office",
  "issue_date": "DD/MM/YYYY",
  "certificate_number": "cert number if visible"
}""",

    "marksheet": """\
Extract all visible fields from this marksheet as JSON:
{
  "student_name": "student's full name",
  "roll_number": "roll/enrollment number",
  "exam_name": "name of examination",
  "board_university": "board or university name",
  "year_of_passing": "year",
  "percentage_marks": <numeric percentage>,
  "grade": "grade/division if applicable",
  "subjects": ["list of subjects with marks if visible"]
}""",

    "bank_passbook": """\
Extract all visible fields from this bank passbook as JSON:
{
  "account_holder_name": "account holder's name",
  "account_number": "account number",
  "bank_name": "name of bank",
  "branch": "branch name",
  "ifsc_code": "IFSC code if visible",
  "account_type": "Savings/Current/etc"
}""",

    "disability_certificate": """\
Extract all visible fields from this disability certificate as JSON:
{
  "name": "certificate holder name",
  "disability_type": "type of disability",
  "disability_percentage": <numeric percentage>,
  "issuing_authority": "issuing hospital/office",
  "issue_date": "DD/MM/YYYY",
  "certificate_number": "cert number if visible"
}""",

    "default": """\
Extract all visible text fields from this government document as a flat JSON object.
Include every field name and its value that you can read.""",
}


async def extract_document_fields(
    file_url: str,
    doc_type: str,
    file_bytes: Optional[bytes] = None,
) -> dict:
    """
    Use Groq vision model to extract structured data from a document image.
    Prefers raw bytes (base64) over URL to avoid auth issues with private storage.
    """
    extraction_schema = EXTRACTION_PROMPTS.get(doc_type, EXTRACTION_PROMPTS["default"])
    full_prompt = (
        "You are an expert at reading Indian government documents.\n"
        f"{extraction_schema}\n\n"
        "IMPORTANT RULES:\n"
        "- Return ONLY valid JSON, nothing else — no markdown fences, no explanation.\n"
        "- Omit fields that are not visible.\n"
        "- Use null for values that are present but unclear."
    )

    # Build image part — prefer bytes (avoids Supabase storage auth headers)
    if file_bytes:
        b64 = base64.b64encode(file_bytes).decode()
        # Detect format from magic bytes
        mime = "image/jpeg"
        if file_bytes[:4] == b"%PDF":
            # Groq vision doesn't accept PDFs directly — treat as jpeg placeholder
            # and return a note so caller can handle gracefully
            return {"extraction_note": "PDF uploaded; manual field entry may be needed", "parse_error": False}
        elif file_bytes[:8] == b"\x89PNG\r\n\x1a\n":
            mime = "image/png"
        image_url_val = f"data:{mime};base64,{b64}"
    else:
        image_url_val = file_url

    client = get_groq()
    response = await client.chat.completions.create(
        model=settings.vision_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url_val},
                    },
                    {"type": "text", "text": full_prompt},
                ],
            }
        ],
        temperature=0.1,
        max_tokens=1024,
    )

    content = response.choices[0].message.content.strip()

    # Strip markdown fences if model wraps output anyway
    if content.startswith("```"):
        lines = content.split("\n")
        # drop first (```json) and last (```) lines
        content = "\n".join(lines[1:-1]).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"raw_extraction": content, "parse_error": True}
