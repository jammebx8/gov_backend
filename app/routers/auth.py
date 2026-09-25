from fastapi import APIRouter, HTTPException, status, Depends
from passlib.context import CryptContext
from app.models.user import UserCreate, UserLogin, TokenResponse, UserProfile
from app.database import get_supabase
from app.utils.auth import create_access_token, get_current_user
import uuid

router = APIRouter(prefix="/auth", tags=["auth"])
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


@router.post("/signup", response_model=TokenResponse, status_code=201)
async def signup(body: UserCreate):
    db = get_supabase()

    # Check if email already exists
    existing = db.table("users").select("id").eq("email", body.email).execute()
    if existing.data:
        raise HTTPException(status_code=400, detail="Email already registered")

    user_id = str(uuid.uuid4())
    hashed = hash_password(body.password)

    result = db.table("users").insert({
        "id": user_id,
        "email": body.email,
        "full_name": body.full_name,
        "phone": body.phone,
        "password_hash": hashed,
        "profile_complete": False,
    }).execute()

    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to create user")

    user_data = result.data[0]
    token = create_access_token(user_id, body.email)

    return TokenResponse(
        access_token=token,
        user=UserProfile(**user_data),
    )


@router.post("/login", response_model=TokenResponse)
async def login(body: UserLogin):
    db = get_supabase()

    result = db.table("users").select("*").eq("email", body.email).single().execute()
    if not result.data:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    user = result.data
    if not verify_password(body.password, user.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(str(user["id"]), user["email"])

    return TokenResponse(
        access_token=token,
        user=UserProfile(**user),
    )


@router.get("/me", response_model=UserProfile)
async def get_me(current_user: dict = Depends(get_current_user)):
    return UserProfile(**current_user)
