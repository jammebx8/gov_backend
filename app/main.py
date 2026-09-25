from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.config import settings
from app.utils.auth import get_current_user
from app.routers import auth, users, documents, schemes

app = FastAPI(
    title="GovAssist API",
    description="Agentic government scheme application system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
# allow_origins uses explicit origins so credentials work correctly.
# "*" is intentionally omitted because allow_credentials=True + "*" is
# forbidden by the CORS spec — browsers would still block it.
_cors_origins = list({
    settings.frontend_url,
    "https://scheme-sarthi-nine.vercel.app",
    "http://localhost:3000",
})

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Routers — all mounted under /api/v1
app.include_router(auth.router, prefix="/api/v1")
app.include_router(users.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")
app.include_router(schemes.router, prefix="/api/v1")


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
