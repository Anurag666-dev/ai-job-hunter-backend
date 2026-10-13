"""
SQLAlchemy async engine + session + Redis connection.
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
import redis.asyncio as aioredis
from app.core.config import DATABASE_URL, REDIS_URL

# ── SQLAlchemy ─────────────────────────────────────────────────────────────────
engine = create_async_engine(DATABASE_URL, echo=False, pool_pre_ping=True)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


async def get_db():
    """FastAPI dependency — yields an async session."""
    async with AsyncSessionLocal() as session:
        yield session


async def create_tables():
    """Create all tables on startup (idempotent)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# ── Redis (with in-memory fallback) ───────────────────────────────────────────
import time

class SimpleMemoryCache:
    def __init__(self):
        self._store: dict[str, tuple[str, float]] = {}

    async def get(self, key: str):
        if key in self._store:
            val, exp = self._store[key]
            if time.time() < exp:
                return val
            del self._store[key]
        return None

    async def setex(self, key: str, seconds: int, value: str):
        self._store[key] = (value, time.time() + seconds)

    async def aclose(self):
        self._store.clear()

_redis_client = None
_memory_cache = SimpleMemoryCache()

async def get_redis():
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    try:
        client = await aioredis.from_url(REDIS_URL, decode_responses=True)
        await client.ping()
        _redis_client = client
        return _redis_client
    except Exception:
        return _memory_cache

async def close_connections():
    global _redis_client
    await engine.dispose()
    if _redis_client:
        try:
            await _redis_client.aclose()
        except Exception:
            pass
