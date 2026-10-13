import os
from dotenv import load_dotenv

load_dotenv()

# ── App ──────────────────────────────────────────────────────────────────────
APP_TITLE = "AI Job Hunter API"
APP_VERSION = "1.0.0"
BACKEND_PORT = int(os.getenv("BACKEND_PORT", 1776))

# ── Security ─────────────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production-please")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = int(os.getenv("ACCESS_TOKEN_EXPIRE_HOURS", 24))

# ── PostgreSQL ────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://jobhunter:jobhunter_pass@localhost:5432/ai_job_hunter",
)

# ── Redis ─────────────────────────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CACHE_TTL_SECONDS = 6 * 60 * 60  # 6 hours

# ── Brave Search ──────────────────────────────────────────────────────────────
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")
BRAVE_SEARCH_URL = os.getenv(
    "BRAVE_SEARCH_URL", "https://api.search.brave.com/res/v1/web/search"
)
BRAVE_RESULTS_COUNT = int(os.getenv("BRAVE_RESULTS_COUNT", 20))

# ── OpenRouter ────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_TEXT_MODEL = os.getenv(
    "OPENROUTER_TEXT_MODEL", "mistralai/mistral-7b-instruct:free"
)

# ── CORS ──────────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
