from pydantic import BaseModel
from typing import Optional, Any
from datetime import datetime
from uuid import UUID


class DocumentUploadResponse(BaseModel):
    id: UUID
    user_id: UUID
    doc_type: str
    file_url: Optional[str] = None
    digilocker_uri: Optional[str] = None
    extracted_data: Optional[dict] = None
    verified: bool = False
    uploaded_at: Optional[datetime] = None


class DocumentExtractRequest(BaseModel):
    doc_id: str


class DigiLockerCallback(BaseModel):
    code: str
    state: str  # user_id


SUPPORTED_DOC_TYPES = [
    "aadhaar",
    "pan",
    "income_certificate",
    "caste_certificate",
    "domicile_certificate",
    "birth_certificate",
    "marksheet",
    "bank_passbook",
    "photo",
    "signature",
    "disability_certificate",
    "farmer_id",
]
