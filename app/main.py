from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from app.config import settings
from app.routers import auth, users, documents, schemes, applications

# ── Allowed origins ───────────────────────────────────────────────────────────
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
# Must be added BEFORE the OPTIONS middleware so CORSMiddleware runs on the
# way OUT (adding headers to the response) while OPTIONS middleware runs on
# the way IN (short-circuiting preflight before it hits the router).
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept", "X-Requested-With"],
    expose_headers=["*"],
    max_age=600,
)


# ── Preflight short-circuit ───────────────────────────────────────────────────
# Vercel edge can sometimes forward an OPTIONS request to the lambda without
# the CORS headers already being applied.  This middleware intercepts OPTIONS
# *before* it reaches any router and returns 204 immediately.
# It does NOT use call_next (which caused the cascading exception in the logs)
# because there is nothing meaningful to forward for a preflight.
@app.middleware("http")
async def handle_options_preflight(request: Request, call_next):
    if request.method == "OPTIONS":
        origin = request.headers.get("origin", "")
        allowed = origin if origin in _ALLOWED_ORIGINS else _ALLOWED_ORIGINS[0]
        return Response(
            status_code=204,
            headers={
                "Access-Control-Allow-Origin":      allowed,
                "Access-Control-Allow-Methods":     "GET, POST, PUT, PATCH, DELETE, OPTIONS",
                "Access-Control-Allow-Headers":     "Authorization, Content-Type, Accept, X-Requested-With",
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Max-Age":           "600",
                "Vary":                             "Origin",
            },
        )
    return await call_next(request)


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth.router,         prefix="/api/v1")
app.include_router(users.router,        prefix="/api/v1")
app.include_router(documents.router,    prefix="/api/v1")
app.include_router(schemes.router,      prefix="/api/v1")
app.include_router(applications.router, prefix="/api/v1")


@app.get("/")
async def root():
    return {"service": "GovAssist API", "version": "1.0.0", "status": "running", "docs": "/docs"}


@app.get("/health")
async def health():
    return {"status": "healthy"}
