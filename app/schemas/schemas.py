from pydantic import BaseModel, EmailStr, Field
from typing import Optional


# ── Auth ──────────────────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    first_name: str
    last_name: str
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ── Profile ───────────────────────────────────────────────────────────────────
class EducationItem(BaseModel):
    institution: str
    degree: str
    field: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None


class ExperienceItem(BaseModel):
    company: str
    role: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None
    is_internship: bool = False


class ProjectItem(BaseModel):
    name: str
    description: Optional[str] = None
    tech_stack: list[str] = []
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    url: Optional[str] = None


class AchievementItem(BaseModel):
    title: str
    description: Optional[str] = None
    date: Optional[str] = None


class ProfileData(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    summary: Optional[str] = None
    skills: list[str] = []
    education: list[EducationItem] = []
    experience: list[ExperienceItem] = []
    projects: list[ProjectItem] = []
    achievements: list[AchievementItem] = []


# ── Jobs ──────────────────────────────────────────────────────────────────────
class JobSearchRequest(BaseModel):
    query: str
    location: Optional[str] = None
    sources: list[str] = []  # e.g. ["Workable", "Greenhouse", "Wellfound", "Lever"]


class JobCard(BaseModel):
    job_hash: str
    title: str
    company: str
    description: Optional[str] = None
    salary: Optional[str] = None
    location: Optional[str] = None
    posted_date: Optional[str] = None
    url: str
    source: Optional[str] = None
    relevance_score: Optional[int] = None
    areas_to_improve: list[str] = []


class SaveJobRequest(BaseModel):
    job_hash: str
    title: str
    company: str
    description: Optional[str] = None
    salary: Optional[str] = None
    location: Optional[str] = None
    posted_date: Optional[str] = None
    url: str
    source: Optional[str] = None


class VisitJobRequest(BaseModel):
    job_hash: str
    title: str
    company: str
    description: Optional[str] = None
    salary: Optional[str] = None
    location: Optional[str] = None
    posted_date: Optional[str] = None
    url: str
    source: Optional[str] = None
