"""
ORM models — all tables in the ai_job_hunter database.
"""
from datetime import datetime, timezone
from sqlalchemy import (
    Integer, String, Text, Boolean, DateTime, ForeignKey, JSON, UniqueConstraint
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base


def _now():
    return datetime.now(timezone.utc)


# ─────────────────────────────────────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    first_name: Mapped[str] = mapped_column(String(100))
    last_name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    profile: Mapped["Profile | None"] = relationship("Profile", back_populates="user", uselist=False, cascade="all, delete-orphan")
    saved_jobs: Mapped[list["SavedJob"]] = relationship("SavedJob", back_populates="user", cascade="all, delete-orphan")
    visited_jobs: Mapped[list["VisitedJob"]] = relationship("VisitedJob", back_populates="user", cascade="all, delete-orphan")


# ─────────────────────────────────────────────────────────────────────────────
class Profile(Base):
    """Extracted resume data stored as JSON blobs per section."""
    __tablename__ = "profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), unique=True)
    full_name: Mapped[str | None] = mapped_column(String(255))
    email: Mapped[str | None] = mapped_column(String(255))
    phone: Mapped[str | None] = mapped_column(String(50))
    location: Mapped[str | None] = mapped_column(String(255))
    summary: Mapped[str | None] = mapped_column(Text)
    skills: Mapped[list] = mapped_column(JSON, default=list)
    education: Mapped[list] = mapped_column(JSON, default=list)
    experience: Mapped[list] = mapped_column(JSON, default=list)
    projects: Mapped[list] = mapped_column(JSON, default=list)
    achievements: Mapped[list] = mapped_column(JSON, default=list)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    user: Mapped["User"] = relationship("User", back_populates="profile")


# ─────────────────────────────────────────────────────────────────────────────
class SavedJob(Base):
    """Jobs the user explicitly saved — persisted forever."""
    __tablename__ = "saved_jobs"
    __table_args__ = (UniqueConstraint("user_id", "job_hash", name="uq_saved_user_job"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    job_hash: Mapped[str] = mapped_column(String(64), index=True)
    title: Mapped[str] = mapped_column(String(255))
    company: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text)
    salary: Mapped[str | None] = mapped_column(String(100))
    location: Mapped[str | None] = mapped_column(String(255))
    posted_date: Mapped[str | None] = mapped_column(String(100))
    url: Mapped[str] = mapped_column(Text)
    source: Mapped[str | None] = mapped_column(String(100))
    saved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    user: Mapped["User"] = relationship("User", back_populates="saved_jobs")


# ─────────────────────────────────────────────────────────────────────────────
class VisitedJob(Base):
    """Jobs the user clicked Apply on — persisted forever."""
    __tablename__ = "visited_jobs"
    __table_args__ = (UniqueConstraint("user_id", "job_hash", name="uq_visited_user_job"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    job_hash: Mapped[str] = mapped_column(String(64), index=True)
    title: Mapped[str] = mapped_column(String(255))
    company: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text)
    salary: Mapped[str | None] = mapped_column(String(100))
    location: Mapped[str | None] = mapped_column(String(255))
    posted_date: Mapped[str | None] = mapped_column(String(100))
    url: Mapped[str] = mapped_column(Text)
    source: Mapped[str | None] = mapped_column(String(100))
    visited_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    user: Mapped["User"] = relationship("User", back_populates="visited_jobs")
