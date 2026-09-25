from pydantic import BaseModel
from typing import Optional, List, Any
from datetime import datetime
from uuid import UUID


class AgentStep(BaseModel):
    step: int
    action_type: str
    description: str
    selector: Optional[str] = None
    value: Optional[str] = None
    screenshot_url: Optional[str] = None
    success: bool = True
    error: Optional[str] = None
    timestamp: Optional[str] = None


class ApplicationJob(BaseModel):
    id: UUID
    user_id: UUID
    scheme_id: UUID
    status: str  # queued, running, completed, failed, needs_review
    agent_log: Optional[List[AgentStep]] = None
    screenshot_urls: Optional[List[str]] = None
    application_ref_id: Optional[str] = None
    error_details: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: Optional[datetime] = None


class ApplicationJobCreate(BaseModel):
    scheme_id: str


class JobStatusUpdate(BaseModel):
    status: str
    agent_log: Optional[List[Any]] = None
    screenshot_urls: Optional[List[str]] = None
    application_ref_id: Optional[str] = None
    error_details: Optional[str] = None
