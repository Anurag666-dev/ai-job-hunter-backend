"""
Jobs router — /api/v1/jobs
Search → validate via AI → Brave API → cache in Redis (6h) → score via AI.
"""
import asyncio
import hashlib
import json
import re
from typing import Optional
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

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
    "linkedin": "(site:in.linkedin.com/jobs/view OR site:linkedin.com/jobs/view)",
    "indeed": "(site:in.indeed.com/viewjob OR site:in.indeed.com/m/viewjob)",
    "naukri": "site:naukri.com",
    "workable": "site:apply.workable.com",
    "greenhouse": "site:boards.greenhouse.io",
    "wellfound": "site:wellfound.com",
    "lever": "site:jobs.lever.co",
}


def _clean_canonical_url(url: str) -> str:
    """Strip tracking and referral query parameters (utm_*, ref, trk, etc.) to canonicalize URL."""
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        query_params = parse_qsl(parsed.query, keep_blank_values=False)
        clean_params = [
            (k, v) for k, v in query_params
            if not k.lower().startswith("utm_")
            and k.lower() not in {"ref", "trk", "trackingid", "currentjobid", "originalreferer", "sessionid", "fbclid", "gclid"}
        ]
        clean_query = urlencode(clean_params)
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, clean_query, ""))
    except Exception:
        return url.split("?")[0] if "?" in url else url


def _extract_salary(text: str) -> Optional[str]:
    """Extract salary mentions from description or snippet (e.g. ₹12,00,000 - ₹18,00,000 or 12-18 LPA or $120k-$150k)."""
    if not text:
        return None
    # 1. LPA / Lacs / Lakhs pattern: e.g. "12-18 LPA", "10 to 15 Lakhs"
    lpa_m = re.search(r"(\d+(?:\.\d+)?\s*(?:-|to)\s*\d+(?:\.\d+)?\s*(?:LPA|Lacs|Lakhs|Lac|Lakh))", text, re.IGNORECASE)
    if lpa_m:
        return lpa_m.group(1).strip()

    single_lpa = re.search(r"(\d+(?:\.\d+)?\s*(?:LPA|Lacs|Lakhs|Lac|Lakh))", text, re.IGNORECASE)
    if single_lpa:
        return single_lpa.group(1).strip()

    # 2. Rupee pattern: e.g. "₹8,00,000 - ₹12,00,000 a year"
    inr_m = re.search(r"((?:₹|Rs\.?|INR)\s*[\d,]+(?:\.\d+)?(?:\s*(?:-|to)\s*(?:₹|Rs\.?|INR)?\s*[\d,]+(?:\.\d+)?)?(?:\s*(?:a\s+year|per\s+annum|per\s+month|\/month|\/yr))?)", text, re.IGNORECASE)
    if inr_m:
        return inr_m.group(1).strip()

    # 3. USD / EUR / GBP pattern: e.g. "$120,000 - $160,000" or "$120k - $150k"
    usd_m = re.search(r"((?:\$|€|£)\s*[\d,]+k?(?:\s*(?:-|to)\s*(?:\$|€|£)?\s*[\d,]+k?)?(?:\s*(?:a\s+year|per\s+annum|per\s+month|\/month|\/yr|\/year))?)", text, re.IGNORECASE)
    if usd_m:
        return usd_m.group(1).strip()

    return None


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
    if "linkedin" in host:
        return "LinkedIn"
    if "naukri" in host:
        return "Naukri"
    if "indeed" in host:
        return "Indeed"
    return "Portal"


def _clean_company_name(raw: str) -> str:
    """Normalize and clean messy company strings scraped from search engines."""
    name = raw.strip()
    # Strip common legal entity suffixes
    name = re.sub(r"(?i)\s+(ltd\.?|limited|inc\.?|incorporated|llc|pvt\.?\s*ltd\.?|private\s+limited|corp\.?|corporation|technologies|solutions|services|group)\b.*", "", name).strip()
    # Strip pipes or dashes
    name = re.split(r"[\-|–|—|•|:]", name)[0].strip()
    # Avoid placeholder words as company name
    portal_noise = {
        "naukri", "jobs", "in", "apply", "careers", "career", "hiring", "view",
        "portal", "indeed", "linkedin", "wellfound", "lever", "greenhouse", "workable",
        "boards", "job", "hiring organization"
    }
    if name.lower() in portal_noise:
        return ""
    return name if len(name) > 1 else ""


def _extract_company_from_job(title: str, url: str, meta_url: dict) -> tuple[str, str]:
    """Smartly extract real company name and clean job title from search result metadata and URL patterns."""
    clean_title = title.strip()
    company = ""

    # 1. Pattern: "Cybage Software hiring Python Developer in Pune... | LinkedIn"
    hiring_match = re.search(r"^(.*?)\s+hiring\s+(.*?)(?:\s+in\s+.*)?(?:\s*\|.*)?$", clean_title, re.IGNORECASE)
    if hiring_match:
        company = hiring_match.group(1).strip()
        clean_title = hiring_match.group(2).strip()

    # 2. LinkedIn job URL: ...-at-company-name-12345
    if not company and "linkedin.com/jobs/view" in url:
        at_m = re.search(r"-at-([a-zA-Z0-9-]+?)(?:-\d+)?$", url)
        if at_m:
            company = at_m.group(1).replace("-", " ").title()

    # 3. Workable URL: apply.workable.com/company-name/...
    if not company and "apply.workable.com/" in url:
        try:
            company = url.split("apply.workable.com/")[1].split("/")[0].replace("-", " ").title()
        except Exception:
            pass

    # 4. Greenhouse URL: boards.greenhouse.io/company-name/...
    if not company and "boards.greenhouse.io/" in url:
        try:
            company = url.split("boards.greenhouse.io/")[1].split("/")[0].replace("-", " ").title()
        except Exception:
            pass

    # 5. Lever URL: jobs.lever.co/company-name/...
    if not company and "jobs.lever.co/" in url:
        try:
            company = url.split("jobs.lever.co/")[1].split("/")[0].replace("-", " ").title()
        except Exception:
            pass

    # 6. Title separator: "Job Title at Company" or "Job Title - Company"
    if not company:
        if " at " in clean_title:
            parts = clean_title.split(" at ", 1)
            company = parts[1].strip()
            clean_title = parts[0].strip()
        elif " - " in clean_title:
            parts = clean_title.split(" - ", 1)
            # If second part is a company rather than location/years
            sec = parts[1].strip()
            if not any(k in sec.lower() for k in ["pune", "mumbai", "years", "yrs", "remote", "india"]):
                company = sec
            clean_title = parts[0].strip()

    # Clean suffixes
    clean_title = re.sub(r"\s*\|\s*(?:LinkedIn|Naukri|Indeed|Workable|Wellfound).*$", "", clean_title, flags=re.IGNORECASE).strip()
    clean_title = re.sub(r"\s*-\s*(?:Indeed\.com|LinkedIn|Naukri).*$", "", clean_title, flags=re.IGNORECASE).strip()

    company = _clean_company_name(company)
    if not company:
        company = "Confidential Employer"

    return clean_title, company


def is_description_visible(desc: Optional[str]) -> bool:
    """Checks whether the external job listing provided a visible, readable description snippet."""
    if not desc:
        return False
    d = desc.strip().lower()
    if len(d) < 30:
        return False
    blocked_phrases = [
        "cannot provide a description",
        "would like to show you a description here",
        "site won't allow us",
        "description is not available",
        "no description provided",
        "please click on the link to view the complete job description",
        "verify you are human",
        "enable javascript",
        "access denied",
    ]
    if any(phrase in d for phrase in blocked_phrases):
        return False
    return True


def _parse_brave_results(results: list[dict]) -> list[dict]:
    jobs = []
    seen_hashes = set()

    for r in results:
        raw_title = r.get("title", "")
        raw_url = r.get("url", "")
        clean_url = _clean_canonical_url(raw_url)
        description = r.get("description", "") or ""
        extra = r.get("extra_snippets", [])
        full_desc = description + " " + " ".join(extra)

        if not raw_title:
            continue

        meta_url = r.get("meta_url", {})
        title, company = _extract_company_from_job(raw_title, clean_url, meta_url)

        job_hash = _make_job_hash(title, company, clean_url)
        if job_hash in seen_hashes:
            continue
        seen_hashes.add(job_hash)

        desc_visible = is_description_visible(description)

        jobs.append({
            "job_hash": job_hash,
            "title": title,
            "company": company,
            "description": description[:600] if desc_visible else None,
            "salary": _extract_salary(full_desc),
            "location": None,
            "posted_date": r.get("age"),
            "url": clean_url,
            "source": _detect_source(clean_url),
            "relevance_score": None,
            "score_note": None if desc_visible else "Cannot score: description not visible",
            "matched_keywords": [],
            "missing_keywords": [],
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

    # Clean location to primary city name to avoid over-filtering (e.g. "Pune, Maharashtra 416312" -> "Pune")
    clean_city = ""
    if search_loc:
        # Strip postal codes/digits and take primary city before comma
        parts = search_loc.split(",")
        first_part = parts[0].strip()
        first_part = re.sub(r"\d+", "", first_part).strip()
        clean_city = first_part if len(first_part) > 1 else search_loc

    # ── 2. Build Brave queries (separate per platform to maximize results up to 20 each) ──
    selected_keys = [s.strip().lower() for s in sources.split(",") if s.strip() and s.strip().lower() in SOURCE_SITE_MAP]
    if not selected_keys:
        selected_keys = list(SOURCE_SITE_MAP.keys())

    loc_part = f" {clean_city}" if clean_city else ""

    queries_to_run = []
    for key in selected_keys:
        site_filter = SOURCE_SITE_MAP[key]
        queries_to_run.append((key, f"{query}{loc_part} {site_filter}"))

    cache_signature = f"{query}|{search_loc}|{','.join(sorted(selected_keys))}"
    cache_key = f"jobs_v4:{hashlib.md5(cache_signature.encode()).hexdigest()}"
    redis = await get_redis()
    cached = await redis.get(cache_key)

    if cached:
        jobs = json.loads(cached)
        if jobs:
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
            jobs = await _attach_user_status(jobs, user_id, db)
            return {"jobs": jobs, "cached": True}

    # ── 4. Call Brave API concurrently across platforms ───────────────────────
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_API_KEY,
    }

    all_raw_results = []
    seen_urls = set()

    async def _fetch_single_platform(client: httpx.AsyncClient, p_query: str) -> list[dict]:
        params = {"q": p_query, "count": 20}
        try:
            resp = await client.get(BRAVE_SEARCH_URL, headers=headers, params=params)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("web", {}).get("results", [])
        except Exception:
            return []
        return []

    async with httpx.AsyncClient(timeout=10) as client:
        batch_tasks = [_fetch_single_platform(client, p_q) for _, p_q in queries_to_run]
        batch_responses = await asyncio.gather(*batch_tasks, return_exceptions=True)

        for res in batch_responses:
            if isinstance(res, list):
                for r in res:
                    u = r.get("url")
                    if u and u not in seen_urls:
                        seen_urls.add(u)
                        all_raw_results.append(r)

    jobs = _parse_brave_results(all_raw_results)

    # ── Geographic relevance filtering ────────────────────────────────────────
    # If the user searched for an Indian city (e.g. Mumbai, Pune, Bangalore), filter out
    # results from other countries like Pakistan, Bulgaria, Mexico, Poland, etc.
    if clean_city:
        foreign_indicators = [
            "pakistan", "islamabad", "lahore", "karachi", "bulgaria", "sofia",
            "mexico", "poland", "warsaw", "nigeria", "lagos", "kenya", "nairobi",
            "romania", "bucharest", "ukraine", "kyiv", "russia"
        ]
        filtered_jobs = []
        for j in jobs:
            text = f"{j.get('title', '')} {j.get('description', '')} {j.get('url', '')}".lower()
            # If the job explicitly points to a foreign country and NOT the selected city, skip it
            is_foreign = any(f" {ind}" in text or f"/{ind}" in text or f"({ind}" in text for ind in foreign_indicators)
            if is_foreign and clean_city.lower() not in text:
                continue
            filtered_jobs.append(j)
        jobs = filtered_jobs

    # ── 5. AI relevance scoring ───────────────────────────────────────────────
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
