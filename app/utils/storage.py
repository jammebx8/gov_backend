import uuid
import base64
from typing import Optional
from app.database import get_supabase
from app.config import settings


async def upload_file_bytes(
    file_bytes: bytes,
    filename: str,
    bucket: str,
    content_type: str = "application/octet-stream",
) -> str:
    """Upload bytes to Supabase Storage and return public URL."""
    db = get_supabase()
    path = f"{uuid.uuid4()}/{filename}"

    db.storage.from_(bucket).upload(
        path=path,
        file=file_bytes,
        file_options={"content-type": content_type},
    )

    result = db.storage.from_(bucket).get_public_url(path)
    return result


async def upload_screenshot(
    screenshot_bytes: bytes,
    job_id: str,
    step: int,
) -> str:
    """Upload a Playwright screenshot and return public URL."""
    filename = f"step_{step:03d}.png"
    path = f"jobs/{job_id}/{filename}"
    db = get_supabase()

    db.storage.from_(settings.screenshots_bucket).upload(
        path=path,
        file=screenshot_bytes,
        file_options={"content-type": "image/png"},
    )

    return db.storage.from_(settings.screenshots_bucket).get_public_url(path)


async def upload_document(
    file_bytes: bytes,
    user_id: str,
    doc_type: str,
    original_filename: str,
) -> str:
    """Upload a user document and return public URL."""
    ext = original_filename.rsplit(".", 1)[-1] if "." in original_filename else "bin"
    filename = f"{doc_type}_{uuid.uuid4()}.{ext}"
    path = f"users/{user_id}/{filename}"

    content_type_map = {
        "pdf": "application/pdf",
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
    }
    content_type = content_type_map.get(ext.lower(), "application/octet-stream")

    db = get_supabase()
    db.storage.from_(settings.storage_bucket).upload(
        path=path,
        file=file_bytes,
        file_options={"content-type": content_type},
    )

    return db.storage.from_(settings.storage_bucket).get_public_url(path)
