"""
Jobs router — /api/v1/jobs
Search → validate via AI → Brave API → cache in Redis (6h) → score via AI.
"""
import hashlib
import json
import re
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import (
    BRAVE_API_KEY, BRAVE_SEARCH_URL, BRAVE_RESULTS_COUNT, CACHE_TTL_SECONDS
)
from app.core.database import get_db, get_redis
from app.core.security import get_current_user
from app.models.models import Profile, SavedJob, VisitedJob
from app.schemas.schemas import SaveJobRequest, VisitJobRequest
from app.services.ai_service import validate_job_query, score_jobs_against_profile

router = APIRouter(prefix="/jobs", tags=["Jobs"])

# Platform → site: filter mapping
SOURCE_SITE_MAP = {
    "workable": "site:apply.workable.com",
    "greenhouse": "site:boards.greenhouse.io",
    "wellfound": "site:wellfound.com",
    "lever": "site:jobs.lever.co",
}


def _make_job_hash(title: str, company: str, url: str) -> str:
    """Stable SHA-256 hash for deduplication across cache TTL boundaries."""
    raw = f"{title.lower().strip()}|{company.lower().strip()}|{url.strip()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _detect_source(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if "workable" in host:
        return "Workable"
    if "greenhouse" in host:
        return "Greenhouse"
    if "wellfound" in host:
        return "Wellfound"
    if "lever" in host:
        return "Lever"
    return "Other"


def _extract_salary(text: str) -> str | None:
    """Crude salary extraction from description/snippet."""
    pattern = r"\$[\d,]+(?:\s*[-–]\s*\$[\d,]+)?(?:\s*(?:k|K|\/yr|\/year|\/mo|\/month|per year|per month))?"
    match = re.search(pattern, text or "")
    return match.group(0) if match else None


def _parse_brave_results(results: list[dict]) -> list[dict]:
    jobs = []
    for r in results:
        title = r.get("title", "")
        url = r.get("url", "")
        description = r.get("description", "")
        extra = r.get("extra_snippets", [])
        full_desc = description + " " + " ".join(extra)

        # Extract company from title or meta info
        meta_url = r.get("meta_url", {})
        company = meta_url.get("hostname", "").replace("www.", "").split(".")[0].capitalize()
        if " at " in title:
            parts = title.split(" at ", 1)
            company = parts[-1].strip()
            title = parts[0].strip()
        elif " - " in title:
            parts = title.split(" - ", 1)
            title = parts[0].strip()
            if company == "":
                company = parts[1].strip()

        job_hash = _make_job_hash(title, company, url)
        jobs.append({
            "job_hash": job_hash,
            "title": title,
            "company": company,
            "description": description[:500] if description else None,
            "salary": _extract_salary(full_desc),
            "location": None,
            "posted_date": r.get("age"),
            "url": url,
            "source": _detect_source(url),
            "relevance_score": None,
            "areas_to_improve": [],
        })
    return jobs


@router.get("/search")
async def search_jobs(
    query: str = Query(..., min_length=1),
    location: str = Query(default=""),
    sources: str = Query(default=""),          # comma-separated e.g. "Workable,Greenhouse"
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # ── 1. AI validation ──────────────────────────────────────────────────────
    is_valid = await validate_job_query(query)
    if not is_valid:
        raise HTTPException(
            status_code=422,
            detail="INVALID_JOB_QUERY"   # Frontend maps this to the popup
        )

    user_id = int(current_user["id"])

    # Fallback to user profile location if location not specified in search query
    search_loc = location.strip()
    if not search_loc:
        prof_res = await db.execute(select(Profile.location).where(Profile.user_id == user_id))
        user_loc = prof_res.scalar_one_or_none()
        if user_loc:
            search_loc = user_loc.strip()

    # ── 2. Build Brave query ──────────────────────────────────────────────────
    selected_sources = [s.strip().lower() for s in sources.split(",") if s.strip()]
    site_filters = " OR ".join(
        SOURCE_SITE_MAP[s] for s in selected_sources if s in SOURCE_SITE_MAP
    )
    # If no sources selected (or "all"), use all four
    if not site_filters:
        site_filters = " OR ".join(SOURCE_SITE_MAP.values())

    brave_query = f"{query} {site_filters}"
    if search_loc:
        brave_query += f" {search_loc}"

    # ── 3. Cache check ────────────────────────────────────────────────────────
    cache_key = f"jobs:{hashlib.md5(brave_query.encode()).hexdigest()}"
    redis = await get_redis()
    cached = await redis.get(cache_key)
    user_id = int(current_user["id"])

    if cached:
        jobs = json.loads(cached)
        profile_result = await db.execute(select(Profile).where(Profile.user_id == user_id))
        profile = profile_result.scalar_one_or_none()
        if profile:
            profile_dict = {
                "skills": profile.skills or [],
                "experience": profile.experience or [],
                "projects": profile.projects or [],
                "education": profile.education or [],
                "achievements": profile.achievements or [],
            }
            jobs = await score_jobs_against_profile(jobs, profile_dict)
        # Re-attach visited/saved status from persistent DB
        jobs = await _attach_user_status(jobs, user_id, db)
        return {"jobs": jobs, "cached": True}

    # ── 4. Brave API call ─────────────────────────────────────────────────────
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_API_KEY,
    }
    params = {"q": brave_query, "count": BRAVE_RESULTS_COUNT}

    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.get(BRAVE_SEARCH_URL, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"Brave API error: {e.response.text}")
        except httpx.RequestError as e:
            raise HTTPException(status_code=504, detail=f"Network error: {str(e)}")

    raw_results = data.get("web", {}).get("results", [])
    jobs = _parse_brave_results(raw_results)

    # ── 5. AI relevance scoring ───────────────────────────────────────────────
    user_id = int(current_user["id"])
    profile_result = await db.execute(select(Profile).where(Profile.user_id == user_id))
    profile = profile_result.scalar_one_or_none()
    if profile:
        profile_dict = {
            "skills": profile.skills or [],
            "experience": profile.experience or [],
            "projects": profile.projects or [],
            "education": profile.education or [],
            "achievements": profile.achievements or [],
        }
        jobs = await score_jobs_against_profile(jobs, profile_dict)

    # ── 6. Cache processed results (6h) ───────────────────────────────────────
    await redis.setex(cache_key, CACHE_TTL_SECONDS, json.dumps(jobs))

    # ── 7. Re-attach visited/saved status ─────────────────────────────────────
    jobs = await _attach_user_status(jobs, user_id, db)

    return {"jobs": jobs, "cached": False}


async def _attach_user_status(jobs: list[dict], user_id: int, db: AsyncSession) -> list[dict]:
    """Mark each job with is_saved / is_visited from persistent DB."""
    hashes = [j["job_hash"] for j in jobs]
    if not hashes:
        return jobs

    saved_result = await db.execute(
        select(SavedJob.job_hash).where(SavedJob.user_id == user_id, SavedJob.job_hash.in_(hashes))
    )
    saved_hashes = {r[0] for r in saved_result.fetchall()}

    visited_result = await db.execute(
        select(VisitedJob.job_hash).where(VisitedJob.user_id == user_id, VisitedJob.job_hash.in_(hashes))
    )
    visited_hashes = {r[0] for r in visited_result.fetchall()}

    for job in jobs:
        job["is_saved"] = job["job_hash"] in saved_hashes
        job["is_visited"] = job["job_hash"] in visited_hashes
    return jobs


# ── Save / Visit endpoints ────────────────────────────────────────────────────

@router.post("/save")
async def save_job(
    body: SaveJobRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user_id = int(current_user["id"])
    existing = await db.execute(
        select(SavedJob).where(SavedJob.user_id == user_id, SavedJob.job_hash == body.job_hash)
    )
    if existing.scalar_one_or_none():
        return {"message": "Already saved"}

    saved = SavedJob(
        user_id=user_id,
        job_hash=body.job_hash,
        title=body.title,
        company=body.company,
        description=body.description,
        salary=body.salary,
        location=body.location,
        posted_date=body.posted_date,
        url=body.url,
        source=body.source,
    )
    db.add(saved)
    await db.commit()
    return {"message": "Job saved"}


@router.delete("/save/{job_hash}")
async def unsave_job(
    job_hash: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user_id = int(current_user["id"])
    result = await db.execute(
        select(SavedJob).where(SavedJob.user_id == user_id, SavedJob.job_hash == job_hash)
    )
    job = result.scalar_one_or_none()
    if job:
        await db.delete(job)
        await db.commit()
    return {"message": "Removed from saved"}


@router.post("/visit")
async def visit_job(
    body: VisitJobRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user_id = int(current_user["id"])
    existing = await db.execute(
        select(VisitedJob).where(VisitedJob.user_id == user_id, VisitedJob.job_hash == body.job_hash)
    )
    if existing.scalar_one_or_none():
        return {"message": "Already visited"}

    visited = VisitedJob(
        user_id=user_id,
        job_hash=body.job_hash,
        title=body.title,
        company=body.company,
        description=body.description,
        salary=body.salary,
        location=body.location,
        posted_date=body.posted_date,
        url=body.url,
        source=body.source,
    )
    db.add(visited)
    await db.commit()
    return {"message": "Job marked as visited"}


@router.get("/saved")
async def get_saved_jobs(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user_id = int(current_user["id"])
    result = await db.execute(
        select(SavedJob).where(SavedJob.user_id == user_id).order_by(SavedJob.saved_at.desc())
    )
    jobs = result.scalars().all()
    return {"jobs": [_saved_to_dict(j) for j in jobs]}


@router.get("/visited")
async def get_visited_jobs(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user_id = int(current_user["id"])
    result = await db.execute(
        select(VisitedJob).where(VisitedJob.user_id == user_id).order_by(VisitedJob.visited_at.desc())
    )
    jobs = result.scalars().all()
    return {"jobs": [_visited_to_dict(j) for j in jobs]}


def _saved_to_dict(j: SavedJob) -> dict:
    return {
        "job_hash": j.job_hash, "title": j.title, "company": j.company,
        "description": j.description, "salary": j.salary, "location": j.location,
        "posted_date": j.posted_date, "url": j.url, "source": j.source,
        "saved_at": j.saved_at.isoformat() if j.saved_at else None,
        "is_saved": True, "is_visited": False,
        "relevance_score": None, "areas_to_improve": [],
    }


def _visited_to_dict(j: VisitedJob) -> dict:
    return {
        "job_hash": j.job_hash, "title": j.title, "company": j.company,
        "description": j.description, "salary": j.salary, "location": j.location,
        "posted_date": j.posted_date, "url": j.url, "source": j.source,
        "visited_at": j.visited_at.isoformat() if j.visited_at else None,
        "is_saved": False, "is_visited": True,
        "relevance_score": None, "areas_to_improve": [],
    }
