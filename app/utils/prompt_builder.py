from typing import Optional


def build_user_profile_text(user: dict, documents: list[dict] = None) -> str:
    """
    Build a concise natural-language summary of a user's profile and documents
    for use in LLM prompts.
    """
    parts = []

    if user.get("full_name"):
        parts.append(f"Name: {user['full_name']}")

    if user.get("date_of_birth"):
        from datetime import date
        dob = user["date_of_birth"]
        if isinstance(dob, str):
            dob = date.fromisoformat(dob)
        age = (date.today() - dob).days // 365
        parts.append(f"Age: {age} years")

    if user.get("gender"):
        parts.append(f"Gender: {user['gender']}")

    if user.get("state"):
        parts.append(f"State: {user['state']}")

    if user.get("district"):
        parts.append(f"District: {user['district']}")

    if user.get("annual_income") is not None:
        income = user["annual_income"]
        if income < 100000:
            income_str = f"₹{income:,.0f} (Below 1 Lakh)"
        elif income < 500000:
            income_str = f"₹{income:,.0f} (1-5 Lakh)"
        else:
            income_str = f"₹{income:,.0f} (Above 5 Lakh)"
        parts.append(f"Annual Income: {income_str}")

    if user.get("caste_category"):
        parts.append(f"Category: {user['caste_category']}")

    if user.get("occupation"):
        parts.append(f"Occupation: {user['occupation']}")

    if user.get("is_student"):
        parts.append("Is a student: Yes")

    if user.get("is_farmer"):
        parts.append("Is a farmer: Yes")

    if user.get("disability_status"):
        parts.append("Has disability: Yes")

    # Add extracted document data
    if documents:
        for doc in documents:
            if doc.get("extracted_data"):
                parts.append(
                    f"[{doc['doc_type'].upper()} data]: {doc['extracted_data']}"
                )

    return "\n".join(parts)


def build_form_fill_system_prompt() -> str:
    return """You are an expert government form-filling agent. Your job is to autonomously fill 
government scheme application forms on behalf of users.

RULES:
1. Only fill fields with data explicitly available in the user's profile or documents.
2. For fields where data is unavailable, skip them (do not guess).
3. Always prefer official name spellings from Aadhaar/PAN documents.
4. Date formats: use DD/MM/YYYY unless the form clearly shows another format.
5. For dropdown fields, select the closest matching option.
6. After filling all fields, ALWAYS do a visual verification step before submitting.
7. If you encounter a CAPTCHA, OTP field, or payment page, stop and return needs_human.
8. Return ONLY valid JSON - no markdown, no explanation text.
"""
