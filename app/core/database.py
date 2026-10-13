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


# ── Redis ──────────────────────────────────────────────────────────────────────
_redis_client: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = await aioredis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


async def close_connections():
    global _redis_client
    await engine.dispose()
    if _redis_client:
        await _redis_client.aclose()
