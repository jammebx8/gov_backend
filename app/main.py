from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.config import settings
from app.utils.auth import get_current_user
from app.routers import auth, users, documents, schemes, applications

app = FastAPI(
    title="GovAssist API",
    description="Agentic government scheme application system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url, "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers



@app.get("/")
async def root():
    return {
        "service": "GovAssist API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


# Fix the /auth/me endpoint properly
@app.get("/api/v1/auth/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    from app.models.user import UserProfile
    return UserProfile(**current_user)
