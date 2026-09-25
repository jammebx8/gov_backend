"""
Groq vision verifier: takes a screenshot of the filled form and cross-checks
all values against the user's actual data before the agent submits.
Uses qwen/qwen3.8-27b (vision model on Groq).
"""
import json
from groq import AsyncGroq
from app.config import settings

_groq_client: AsyncGroq | None = None


def get_groq() -> AsyncGroq:
    global _groq_client
    if _groq_client is None:
        _groq_client = AsyncGroq(api_key=settings.groq_api_key)
    return _groq_client


async def verify_form(
    screenshot_b64: str,
    user_data: dict,
    scheme_name: str,
) -> dict:
    """
    Visually verify a filled form against the user's actual data.

    Returns:
    {
        "all_correct": bool,
        "ready_to_submit": bool,
        "issues": ["..."],
        "corrections_needed": [{"field": "...", "current_value": "...", "expected_value": "..."}],
        "empty_required_fields": ["..."],
        "form_stage": "filling|review|confirmation|submitted"
    }
    """
    prompt = f"""You are verifying a filled government form for: "{scheme_name}"

The user's actual data:
{json.dumps(user_data, indent=2, default=str)}

Look at this screenshot carefully and:
1. Check each visible filled field against the user's data above.
2. List any empty required fields.
3. List any incorrect / mismatched values.
4. Check for visible error messages on the page.

Return ONLY this JSON (no markdown, no explanation):
{{
    "all_correct": true,
    "ready_to_submit": true,
    "issues": [],
    "corrections_needed": [],
    "empty_required_fields": [],
    "form_stage": "filling"
}}

If the form is correctly filled with no issues set all_correct=true and ready_to_submit=true.
"""

    client = get_groq()
    response = await client.chat.completions.create(
        model=settings.vision_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{screenshot_b64}",
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        temperature=0.1,
        max_tokens=1024,
    )

    content = response.choices[0].message.content.strip()

    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1]).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {
            "all_correct": False,
            "ready_to_submit": False,
            "issues": ["Verification response could not be parsed"],
            "corrections_needed": [],
            "empty_required_fields": [],
            "form_stage": "filling",
        }
