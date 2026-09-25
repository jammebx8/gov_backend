from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from app.utils.auth import get_current_user
from app.utils.storage import upload_document
from app.services.document_parser import extract_document_fields
from app.models.document import SUPPORTED_DOC_TYPES
from app.database import get_supabase
import uuid

router = APIRouter(prefix="/documents", tags=["documents"])

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post("/upload")
async def upload_document_endpoint(
    doc_type: str = Form(...),
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """
    Upload a government document, store it in Supabase Storage,
    and extract all visible fields using the Groq vision model.
    """
    if doc_type not in SUPPORTED_DOC_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported doc_type. Allowed values: {SUPPORTED_DOC_TYPES}",
        )

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 10 MB)")

    user_id = str(current_user["id"])

    # 1. Upload to Supabase Storage
    file_url = await upload_document(
        file_bytes=file_bytes,
        user_id=user_id,
        doc_type=doc_type,
        original_filename=file.filename or f"{doc_type}.bin",
    )

    # 2. Extract fields with vision model (non-fatal — store error in DB if it fails)
    extracted_data: dict = {}
    try:
        extracted_data = await extract_document_fields(
            file_url=file_url,
            doc_type=doc_type,
            file_bytes=file_bytes,
        )
    except Exception as exc:
        extracted_data = {"extraction_error": str(exc)}

    # 3. If Aadhaar data was extracted, auto-populate user profile fields
    if doc_type == "aadhaar" and extracted_data and not extracted_data.get("parse_error"):
        _auto_populate_profile(user_id, extracted_data)

    # 4. Persist document record
    db = get_supabase()
    doc_id = str(uuid.uuid4())
    result = db.table("user_documents").insert({
        "id": doc_id,
        "user_id": user_id,
        "doc_type": doc_type,
        "file_url": file_url,
        "extracted_data": extracted_data,
        "verified": False,
    }).execute()

    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to save document record")

    return result.data[0]


@router.get("/")
async def list_my_documents(current_user: dict = Depends(get_current_user)):
    """List all uploaded documents for the current user."""
    db = get_supabase()
    result = (
        db.table("user_documents")
        .select("*")
        .eq("user_id", str(current_user["id"]))
        .order("uploaded_at", desc=True)
        .execute()
    )
    return result.data or []


@router.delete("/{doc_id}")
async def delete_document(
    doc_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Remove a document record (does not delete from Storage)."""
    db = get_supabase()
    db.table("user_documents").delete().eq("id", doc_id).eq(
        "user_id", str(current_user["id"])
    ).execute()
    return {"deleted": True}


# ── helpers ────────────────────────────────────────────────────────────────

def _auto_populate_profile(user_id: str, aadhaar_data: dict) -> None:
    """
    If the user's profile is missing basic fields that are present in their
    Aadhaar extraction, fill them in automatically.
    """
    db = get_supabase()
    user_result = db.table("users").select("*").eq("id", user_id).single().execute()
    if not user_result.data:
        return
    user = user_result.data

    update: dict = {}
    if not user.get("full_name") and aadhaar_data.get("name"):
        update["full_name"] = aadhaar_data["name"]
    if not user.get("date_of_birth") and aadhaar_data.get("date_of_birth"):
        update["date_of_birth"] = aadhaar_data["date_of_birth"]
    if not user.get("gender") and aadhaar_data.get("gender"):
        update["gender"] = aadhaar_data["gender"]
    if not user.get("state") and aadhaar_data.get("state"):
        update["state"] = aadhaar_data["state"]
    if not user.get("district") and aadhaar_data.get("district"):
        update["district"] = aadhaar_data["district"]

    if update:
        db.table("users").update(update).eq("id", user_id).execute()
