"""
Groq Cloud planner: given the current page state and a screenshot,
decides the single next action for the form-fill agent.
Uses openai/gpt-oss-120b for reasoning, with vision via qwen/qwen3.8-27b.
"""
import json
import re
import base64
from groq import AsyncGroq
from app.config import settings
from app.utils.prompt_builder import build_form_fill_system_prompt

_groq_client: AsyncGroq | None = None


def get_groq() -> AsyncGroq:
    global _groq_client
    if _groq_client is None:
        _groq_client = AsyncGroq(api_key=settings.groq_api_key)
    return _groq_client


async def plan_next_action(
    page_state: dict,
    screenshot_b64: str,
    user_data: dict,
    scheme_name: str,
    previous_steps: list[dict],
) -> dict:
    """
    Returns one of:
      {"type": "fill",        "selector": "...", "value": "...", "field_name": "..."}
      {"type": "click",       "selector": "...", "description": "..."}
      {"type": "select",      "selector": "...", "value": "..."}
      {"type": "upload",      "selector": "...", "doc_type": "aadhaar|pan|photo|..."}
      {"type": "scroll",      "direction": "down"}
      {"type": "verify",      "description": "cross-check all filled data"}
      {"type": "submit",      "selector": "...", "description": "submit the form"}
      {"type": "done",        "reference_number": "ref if visible"}
      {"type": "needs_human", "reason": "captcha/OTP/payment"}
      {"type": "wait",        "ms": 2000}
    """
    recent_steps = previous_steps[-8:] if len(previous_steps) > 8 else previous_steps

    prompt = f"""You are filling the government scheme application: "{scheme_name}"
Current URL: {page_state.get('url', 'unknown')}
Page title: {page_state.get('title', 'unknown')}

USER DATA (use ONLY this to fill fields):
{json.dumps(user_data, indent=2, default=str)}

VISIBLE FORM FIELDS (non-disabled):
{json.dumps([f for f in page_state.get('form_fields', []) if not f.get('disabled')], indent=2)}

VISIBLE PAGE TEXT (excerpt):
{page_state.get('visible_text', '')[:2000]}

ERROR INDICATORS:
{page_state.get('error_indicators', [])}

RECENT STEPS:
{json.dumps(recent_steps, indent=2) if recent_steps else "None yet"}

Look at the screenshot carefully. Decide the SINGLE best next action.

RULES:
- Unfilled required fields visible → fill one (required first)
- All visible fields filled + "Next"/"Continue" button → click it
- All fields filled + review page → return verify
- CAPTCHA / audio challenge → needs_human
- OTP / mobile verification input → needs_human
- Payment page → needs_human
- Success page / reference number visible → done with reference number
- After verify passed → submit

Selector priority: id > name > data-* attribute > CSS class.
Write CSS selectors only (e.g. "#applicantName", "input[name='dob']").

Return ONLY a single valid JSON object — no markdown, no explanation.
"""

    client = get_groq()

    # Use vision model so agent can actually see the screenshot
    response = await client.chat.completions.create(
        model=settings.vision_model,
        messages=[
            {"role": "system", "content": build_form_fill_system_prompt()},
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
            },
        ],
        temperature=0.05,
        max_tokens=512,
    )

    content = response.choices[0].message.content.strip()

    # Strip markdown fences
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1]).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Try to pull a JSON object out of free-text response
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        # Safe fallback — scroll down and try again next iteration
        return {"type": "scroll", "direction": "down", "description": "Could not parse action, scrolling"}
