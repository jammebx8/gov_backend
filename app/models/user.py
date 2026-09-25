from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import date, datetime
from uuid import UUID


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    phone: Optional[str] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    phone: Optional[str] = None
    date_of_birth: Optional[date] = None
    gender: Optional[str] = None
    state: Optional[str] = None
    district: Optional[str] = None
    annual_income: Optional[float] = None
    caste_category: Optional[str] = None  # GEN, OBC, SC, ST
    is_student: Optional[bool] = None
    is_farmer: Optional[bool] = None
    occupation: Optional[str] = None
    disability_status: Optional[bool] = None


class UserProfile(BaseModel):
    id: UUID
    email: str
    full_name: Optional[str] = None
    phone: Optional[str] = None
    date_of_birth: Optional[date] = None
    gender: Optional[str] = None
    state: Optional[str] = None
    district: Optional[str] = None
    annual_income: Optional[float] = None
    caste_category: Optional[str] = None
    is_student: Optional[bool] = None
    is_farmer: Optional[bool] = None
    occupation: Optional[str] = None
    disability_status: Optional[bool] = None
    profile_complete: bool = False
    created_at: Optional[datetime] = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserProfile
