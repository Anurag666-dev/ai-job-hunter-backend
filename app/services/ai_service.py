"""
OpenRouter AI service helper — text-only (PDF text pre-extracted by pdfplumber).
"""
import httpx
import json
from app.core.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_TEXT_MODEL


async def _call_openrouter(messages: list[dict], max_tokens: int = 2000) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://ai-job-hunter.local",
        "X-Title": "AI Job Hunter",
    }
    payload = {
        "model": OPENROUTER_TEXT_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(OPENROUTER_BASE_URL, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


async def extract_profile_from_text(resume_text: str) -> dict:
    """
    Uses OpenRouter to extract structured profile data from resume plain text.
    Returns a dict matching ProfileData schema.
    """
    prompt = f"""You are a precise resume parser. Extract all information from the resume text below and return ONLY a valid JSON object (no markdown, no explanation) with this exact structure:

{{
  "full_name": "string or null",
  "email": "string or null",
  "phone": "string or null",
  "location": "string or null",
  "summary": "string or null",
  "skills": ["skill1", "skill2"],
  "education": [
    {{
      "institution": "string",
      "degree": "string",
      "field": "string or null",
      "start_date": "string or null",
      "end_date": "string or null",
      "description": "string or null"
    }}
  ],
  "experience": [
    {{
      "company": "string",
      "role": "string",
      "start_date": "string or null",
      "end_date": "string or null",
      "description": "string or null",
      "is_internship": false
    }}
  ],
  "projects": [
    {{
      "name": "string",
      "description": "string or null",
      "tech_stack": ["tech1"],
      "start_date": "string or null",
      "end_date": "string or null",
      "url": "string or null"
    }}
  ],
  "achievements": [
    {{
      "title": "string",
      "description": "string or null",
      "date": "string or null"
    }}
  ]
}}

For is_internship: set true if the experience entry is clearly an internship.
For achievements: include competitions, hackathons, awards, certifications.
Extract ALL dates as they appear in the resume (e.g. "Jan 2023 - Mar 2024" or "2022 - Present").

RESUME TEXT:
{resume_text[:8000]}

Return ONLY the JSON object."""

    content = await _call_openrouter([{"role": "user", "content": prompt}])

    # Strip markdown code fences if the model wraps the JSON
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1]) if lines[-1].startswith("```") else "\n".join(lines[1:])

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Best effort: return empty profile
        return {}


async def validate_job_query(query: str) -> bool:
    """
    Returns True if query is a job/career-related search, False otherwise.
    """
    prompt = f"""You are a classifier. Determine if the following search query is related to job hunting, career opportunities, roles, internships, or companies hiring. 
Reply with ONLY a single word: YES or NO.

Query: "{query}" """

    result = await _call_openrouter([{"role": "user", "content": prompt}], max_tokens=5)
    return result.strip().upper().startswith("Y")


async def score_jobs_against_profile(jobs: list[dict], profile: dict) -> list[dict]:
    """
    For each job, adds relevance_score (0-100) and areas_to_improve list.
    Processes in a single API call to avoid rate limits.
    """
    if not profile or not jobs:
        return jobs

    profile_summary = json.dumps({
        "skills": profile.get("skills", []),
        "experience": [
            {"role": e.get("role", ""), "company": e.get("company", ""), "description": e.get("description", "")}
            for e in profile.get("experience", [])[:5]
        ],
        "projects": [
            {"name": p.get("name", ""), "tech_stack": p.get("tech_stack", []), "description": p.get("description", "")}
            for p in profile.get("projects", [])[:5]
        ],
        "education": [
            {"degree": e.get("degree", ""), "field": e.get("field", "")}
            for e in profile.get("education", [])[:3]
        ],
        "achievements": [a.get("title", "") for a in profile.get("achievements", [])[:5]],
    }, ensure_ascii=False)

    jobs_summary = json.dumps([
        {"index": i, "title": j.get("title", ""), "company": j.get("company", ""), "description": (j.get("description") or "")[:300]}
        for i, j in enumerate(jobs[:15])
    ], ensure_ascii=False)

    prompt = f"""You are a career coach AI. Score how well a candidate's profile matches each job listing.

CANDIDATE PROFILE:
{profile_summary}

JOB LISTINGS (array of objects with index):
{jobs_summary}

For each job, return a JSON array (NO markdown, NO explanation) like this:
[
  {{
    "index": 0,
    "relevance_score": 72,
    "areas_to_improve": ["Add TypeScript to skills", "Mention team leadership experience"]
  }},
  ...
]

Rules:
- relevance_score: integer 0-100
- areas_to_improve: 2-3 short, specific, actionable suggestions
- Return array with same count as input jobs
- Return ONLY the JSON array"""

    content = await _call_openrouter([{"role": "user", "content": prompt}], max_tokens=2000)

    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1]) if lines[-1].startswith("```") else "\n".join(lines[1:])

    try:
        scores = json.loads(content)
        score_map = {item["index"]: item for item in scores}
        for i, job in enumerate(jobs):
            entry = score_map.get(i, {})
            job["relevance_score"] = entry.get("relevance_score", None)
            job["areas_to_improve"] = entry.get("areas_to_improve", [])
        return jobs
    except (json.JSONDecodeError, KeyError):
        return jobs
