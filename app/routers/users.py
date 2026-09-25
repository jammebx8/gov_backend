from fastapi import APIRouter, Depends, HTTPException
from app.models.user import UserProfile, UserProfileUpdate
from app.utils.auth import get_current_user
from app.database import get_supabase

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me", response_model=UserProfile)
async def get_profile(current_user: dict = Depends(get_current_user)):
    return UserProfile(**current_user)


@router.patch("/me", response_model=UserProfile)
async def update_profile(
    body: UserProfileUpdate,
    current_user: dict = Depends(get_current_user),
):
    db = get_supabase()
    user_id = str(current_user["id"])

    update_data = body.model_dump(exclude_none=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")

    # Convert date to ISO string if present
    if "date_of_birth" in update_data and update_data["date_of_birth"]:
        update_data["date_of_birth"] = update_data["date_of_birth"].isoformat()

    # Check if profile is now complete
    result_before = db.table("users").select("*").eq("id", user_id).single().execute()
    merged = {**result_before.data, **update_data}
    required_fields = ["full_name", "date_of_birth", "gender", "state", "annual_income", "caste_category"]
    is_complete = all(merged.get(f) for f in required_fields)
    update_data["profile_complete"] = is_complete

    result = (
        db.table("users")
        .update(update_data)
        .eq("id", user_id)
        .execute()
    )

    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to update profile")

    return UserProfile(**result.data[0])


@router.get("/me/documents")
async def get_my_documents(current_user: dict = Depends(get_current_user)):
    db = get_supabase()
    result = (
        db.table("user_documents")
        .select("*")
        .eq("user_id", str(current_user["id"]))
        .execute()
    )
    return result.data or []
