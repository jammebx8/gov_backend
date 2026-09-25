from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.config import settings
from app.routers import auth, users, documents, schemes, applications

# ── Allowed origins ───────────────────────────────────────────────────────────
# Keep this list in sync with vercel.json headers.source origins.
# "https://scheme-sarthi-nine.vercel.app" is the production frontend.
# settings.frontend_url covers local dev and any custom domain set via env var.
_ALLOWED_ORIGINS = list({
    "https://scheme-sarthi-nine.vercel.app",
    "http://localhost:3000",
    "http://localhost:3001",
    settings.frontend_url,
})

app = FastAPI(
    title="GovAssist API",
    description="Agentic government scheme application system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS middleware ───────────────────────────────────────────────────────────
# NOTE: allow_origins must be explicit (not "*") when allow_credentials=True,
# otherwise browsers reject the response per the CORS spec.
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept", "X-Requested-With"],
    expose_headers=["*"],
    max_age=600,  # cache preflight for 10 minutes
)


# ── Explicit OPTIONS handler ──────────────────────────────────────────────────
# Vercel's edge sometimes swallows the CORSMiddleware response for preflight
# requests on cold-start lambdas.  This middleware intercepts every OPTIONS
# request before it reaches the router and returns the correct headers
# immediately, guaranteeing the browser never sees a missing CORS header.
@app.middleware("http")
async def handle_options_preflight(request: Request, call_next):
    if request.method == "OPTIONS":
        origin = request.headers.get("origin", "")
        if origin in _ALLOWED_ORIGINS:
            return JSONResponse(
                content=None,
                status_code=204,
                headers={
                    "Access-Control-Allow-Origin":      origin,
                    "Access-Control-Allow-Methods":     "GET, POST, PUT, PATCH, DELETE, OPTIONS",
                    "Access-Control-Allow-Headers":     "Authorization, Content-Type, Accept, X-Requested-With",
                    "Access-Control-Allow-Credentials": "true",
                    "Access-Control-Max-Age":           "600",
                    "Vary":                             "Origin",
                },
            )
    response = await call_next(request)
    return response


# ── Routers — all mounted under /api/v1 ──────────────────────────────────────
app.include_router(auth.router,         prefix="/api/v1")
app.include_router(users.router,        prefix="/api/v1")
app.include_router(documents.router,    prefix="/api/v1")
app.include_router(schemes.router,      prefix="/api/v1")
app.include_router(applications.router, prefix="/api/v1")


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
