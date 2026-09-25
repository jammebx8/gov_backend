from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from uuid import UUID


class SchemeBase(BaseModel):
    scheme_name: str
    slug: str
    details: str
    benefits: str
    eligibility: str
    application: Optional[str] = None
    documents: Optional[str] = None
    level: str
    scheme_category: str
    tags: Optional[List[str]] = None


class SchemeCreate(SchemeBase):
    pass


class Scheme(SchemeBase):
    id: UUID
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class SchemeWithEligibility(Scheme):
    eligible: Optional[bool] = None
    eligibility_reason: Optional[str] = None


class SchemeSearchParams(BaseModel):
    query: Optional[str] = None
    category: Optional[str] = None
    level: Optional[str] = None
    tags: Optional[List[str]] = None
    page: int = 1
    page_size: int = 20


class EligibilityResult(BaseModel):
    scheme_id: str
    eligible: bool
    reason: str
