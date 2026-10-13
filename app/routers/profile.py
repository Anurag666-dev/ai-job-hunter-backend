"""
Profile router — /api/v1/profile
Upload CV → extract via pdfplumber + OpenRouter → save to Postgres.
"""
import io
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import pdfplumber

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.models import User, Profile
from app.services.ai_service import extract_profile_from_text

router = APIRouter(prefix="/profile", tags=["Profile"])


def _extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract raw text from PDF bytes using pdfplumber."""
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
    return "\n".join(text_parts)


@router.post("/upload-cv")
async def upload_cv(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    file_bytes = await file.read()
    if len(file_bytes) > 10 * 1024 * 1024:  # 10 MB guard
        raise HTTPException(status_code=413, detail="File too large (max 10 MB)")

    # 1. Extract plain text from PDF
    resume_text = _extract_text_from_pdf(file_bytes)
    if not resume_text.strip():
        raise HTTPException(status_code=422, detail="Could not extract text from PDF")

    # 2. Use AI to parse structured profile
    parsed = await extract_profile_from_text(resume_text)
    if not parsed:
        raise HTTPException(status_code=502, detail="AI extraction failed, try again")

    user_id = int(current_user["id"])

    # 3. Upsert profile row
    result = await db.execute(select(Profile).where(Profile.user_id == user_id))
    profile = result.scalar_one_or_none()

    if profile:
        profile.full_name = parsed.get("full_name")
        profile.email = parsed.get("email")
        profile.phone = parsed.get("phone")
        profile.location = parsed.get("location")
        profile.summary = parsed.get("summary")
        profile.skills = parsed.get("skills", [])
        profile.education = parsed.get("education", [])
        profile.experience = parsed.get("experience", [])
        profile.projects = parsed.get("projects", [])
        profile.achievements = parsed.get("achievements", [])
    else:
        profile = Profile(
            user_id=user_id,
            full_name=parsed.get("full_name"),
            email=parsed.get("email"),
            phone=parsed.get("phone"),
            location=parsed.get("location"),
            summary=parsed.get("summary"),
            skills=parsed.get("skills", []),
            education=parsed.get("education", []),
            experience=parsed.get("experience", []),
            projects=parsed.get("projects", []),
            achievements=parsed.get("achievements", []),
        )
        db.add(profile)

    await db.commit()
    await db.refresh(profile)
    return {"message": "Profile extracted and saved", "profile": _profile_to_dict(profile)}


@router.get("/me")
async def get_my_profile(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user_id = int(current_user["id"])
    result = await db.execute(select(Profile).where(Profile.user_id == user_id))
    profile = result.scalar_one_or_none()
    if not profile:
        return {"profile": None}
    return {"profile": _profile_to_dict(profile)}


def _profile_to_dict(p: Profile) -> dict:
    return {
        "full_name": p.full_name,
        "email": p.email,
        "phone": p.phone,
        "location": p.location,
        "summary": p.summary,
        "skills": p.skills or [],
        "education": p.education or [],
        "experience": p.experience or [],
        "projects": p.projects or [],
        "achievements": p.achievements or [],
    }
