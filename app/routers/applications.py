import asyncio
import json
import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from app.utils.auth import get_current_user
from app.services.agent.orchestrator import run_agent
from app.database import get_supabase

router = APIRouter(prefix="/applications", tags=["applications"])


@router.post("/apply/{scheme_id}", status_code=202)
async def apply_to_scheme(
    scheme_id: str,
    bg: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
):
    """
    Trigger the agentic form-fill agent for a scheme.
    Creates a job and starts the background agent task.
    """
    db = get_supabase()
    user_id = str(current_user["id"])

    # Validate scheme exists
    scheme_result = db.table("government_schemes").select("id, scheme_name, application").eq("id", scheme_id).single().execute()
    if not scheme_result.data:
        raise HTTPException(status_code=404, detail="Scheme not found")

    scheme = scheme_result.data
    if not scheme.get("application"):
        raise HTTPException(
            status_code=400,
            detail="This scheme has no online application portal configured yet",
        )

    # Check if already running
    existing = (
        db.table("application_jobs")
        .select("id, status")
        .eq("user_id", user_id)
        .eq("scheme_id", scheme_id)
        .in_("status", ["queued", "running"])
        .execute()
    )
    if existing.data:
        return {
            "job_id": existing.data[0]["id"],
            "status": existing.data[0]["status"],
            "message": "Application already in progress",
        }

    # Create job record
    job_id = str(uuid.uuid4())
    db.table("application_jobs").insert({
        "id": job_id,
        "user_id": user_id,
        "scheme_id": scheme_id,
        "status": "queued",
        "agent_log": [],
        "screenshot_urls": [],
    }).execute()

    # Start background agent
    bg.add_task(run_agent, job_id, user_id, scheme_id)

    return {
        "job_id": job_id,
        "status": "queued",
        "scheme_name": scheme["scheme_name"],
        "message": "Agent started. Track progress via /applications/jobs/{job_id}/stream",
    }


@router.get("/jobs/{job_id}")
async def get_job(
    job_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Get the current state of an application job."""
    db = get_supabase()
    result = (
        db.table("application_jobs")
        .select("*")
        .eq("id", job_id)
        .eq("user_id", str(current_user["id"]))
        .single()
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Job not found")
    return result.data


@router.get("/jobs/{job_id}/stream")
async def stream_job_status(
    job_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Server-Sent Events stream for real-time agent progress.
    Frontend connects here and receives job updates every 1.5s.
    """
    db = get_supabase()
    user_id = str(current_user["id"])

    # Validate ownership
    ownership = (
        db.table("application_jobs")
        .select("id")
        .eq("id", job_id)
        .eq("user_id", user_id)
        .execute()
    )
    if not ownership.data:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        terminal_statuses = {"completed", "failed", "needs_review"}
        last_step_count = 0
        timeout_seconds = 600  # 10 minutes max stream
        elapsed = 0

        yield f"data: {json.dumps({'type': 'connected', 'job_id': job_id})}\n\n"

        while elapsed < timeout_seconds:
            try:
                result = (
                    db.table("application_jobs")
                    .select("*")
                    .eq("id", job_id)
                    .single()
                    .execute()
                )

                if result.data:
                    job = result.data
                    current_steps = len(job.get("agent_log") or [])

                    # Always send update (frontend uses latest)
                    payload = {
                        "type": "update",
                        "job_id": job_id,
                        "status": job["status"],
                        "agent_log": job.get("agent_log") or [],
                        "screenshot_urls": job.get("screenshot_urls") or [],
                        "application_ref_id": job.get("application_ref_id"),
                        "error_details": job.get("error_details"),
                        "completed_at": str(job.get("completed_at") or ""),
                    }
                    yield f"data: {json.dumps(payload)}\n\n"

                    if job["status"] in terminal_statuses:
                        yield f"data: {json.dumps({'type': 'terminal', 'status': job['status']})}\n\n"
                        break

                    last_step_count = current_steps

            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

            await asyncio.sleep(1.5)
            elapsed += 1.5

        if elapsed >= timeout_seconds:
            yield f"data: {json.dumps({'type': 'timeout'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/my")
async def list_my_applications(
    current_user: dict = Depends(get_current_user),
):
    """List all applications for the current user."""
    db = get_supabase()
    result = (
        db.table("application_jobs")
        .select("*, government_schemes(scheme_name, slug, scheme_category)")
        .eq("user_id", str(current_user["id"]))
        .order("created_at", desc=True)
        .execute()
    )
    return {"applications": result.data or []}


@router.delete("/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Cancel a queued job."""
    db = get_supabase()
    result = (
        db.table("application_jobs")
        .update({"status": "failed", "error_details": "Cancelled by user"})
        .eq("id", job_id)
        .eq("user_id", str(current_user["id"]))
        .eq("status", "queued")
        .execute()
    )
    return {"cancelled": bool(result.data)}
