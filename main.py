"""
AI Job Hunter — FastAPI application entry point.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import APP_TITLE, APP_VERSION, ALLOWED_ORIGINS, BACKEND_PORT
from app.core.database import create_tables, close_connections
from app.routers import auth, profile, jobs


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create tables (idempotent)
    await create_tables()
    yield
    # Shutdown: clean up connections
    await close_connections()


app = FastAPI(title=APP_TITLE, version=APP_VERSION, lifespan=lifespan)

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(auth.router,    prefix="/api/v1")
app.include_router(profile.router, prefix="/api/v1")
app.include_router(jobs.router,    prefix="/api/v1")


@app.get("/health")
async def health():
    return {"status": "ok", "version": APP_VERSION}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=BACKEND_PORT, reload=True)