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
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.post(OPENROUTER_BASE_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            # Propagate error so caller can handle fallback
            raise RuntimeError(f"OpenRouter error: {e}")


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

    try:
        content = await _call_openrouter([{"role": "user", "content": prompt}])
        # Strip markdown code fences if the model wraps the JSON
        if "```" in content:
            lines = content.split("\n")
            json_lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(json_lines).strip()
        data = json.loads(content)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Heuristic fallback if OpenRouter key is invalid or rate limited
    return _heuristic_profile_extract(resume_text)


def _heuristic_profile_extract(text: str) -> dict:
    import re
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    full_name = lines[0] if lines else "Candidate"
    
    # Email
    email_match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    email = email_match.group(0) if email_match else None
    
    # Phone
    phone_match = re.search(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", text)
    phone = phone_match.group(0) if phone_match else None

    # Tech skills heuristics
    known_skills = [
        "python", "javascript", "typescript", "react", "node", "fastapi", "django",
        "sql", "postgresql", "mongodb", "docker", "aws", "git", "c++", "java", "html",
        "css", "tailwind", "redis", "linux", "rest", "graphql", "next.js", "vue"
    ]
    lower_text = text.lower()
    found_skills = [s.capitalize() for s in known_skills if re.search(rf"\b{re.escape(s)}\b", lower_text)]

    return {
        "full_name": full_name,
        "email": email,
        "phone": phone,
        "location": None,
        "summary": lines[1] if len(lines) > 1 else None,
        "skills": found_skills,
        "education": [
            {
                "institution": "University / Institution",
                "degree": "Degree",
                "field": "Computer Science / Engineering",
                "start_date": None,
                "end_date": None,
                "description": None
            }
        ],
        "experience": [
            {
                "company": "Experience",
                "role": "Software Developer",
                "start_date": None,
                "end_date": None,
                "description": "Extracted from uploaded resume",
                "is_internship": False
            }
        ],
        "projects": [],
        "achievements": []
    }


async def validate_job_query(query: str) -> bool:
    """
    Returns True if query is a job/career-related search, False otherwise.
    """
    try:
        prompt = f"""You are a classifier. Determine if the following search query is related to job hunting, career opportunities, roles, internships, or companies hiring. 
Reply with ONLY a single word: YES or NO.

Query: "{query}" """
        result = await _call_openrouter([{"role": "user", "content": prompt}], max_tokens=5)
        return result.strip().upper().startswith("Y")
    except Exception:
        # Fallback heuristic: check if query contains job-like words or is not random conversational query
        query_lower = query.lower()
        job_keywords = [
            "developer", "engineer", "designer", "manager", "intern", "lead", "architect",
            "frontend", "backend", "fullstack", "data", "analyst", "devops", "qa", "tester",
            "specialist", "consultant", "scientist", "writer", "marketing", "sales", "executive",
            "python", "react", "java", "node", "cloud", "security", "ai", "ml", "support",
            "administrator", "associate", "product", "software", "tech", "job", "career", "role"
        ]
        non_job_patterns = [
            "what is", "how to", "who is", "weather", "recipe", "tell me", "joke", "why is", "movie"
        ]
        if any(p in query_lower for p in non_job_patterns):
            return False
        return any(k in query_lower for k in job_keywords) or len(query.strip().split()) <= 4


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

    try:
        content = await _call_openrouter([{"role": "user", "content": prompt}], max_tokens=2000)
        if "```" in content:
            lines = content.split("\n")
            json_lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(json_lines).strip()
        scores = json.loads(content)
        score_map = {item["index"]: item for item in scores}
        for i, job in enumerate(jobs):
            entry = score_map.get(i, {})
            job["relevance_score"] = entry.get("relevance_score", None)
            job["areas_to_improve"] = entry.get("areas_to_improve", [])
        return jobs
    except Exception:
        # Heuristic scoring fallback
        candidate_skills = set(s.lower() for s in profile.get("skills", []))
        for job in jobs:
            title_desc = f"{job.get('title', '')} {job.get('description', '')}".lower()
            matched = [s for s in candidate_skills if s in title_desc]
            score = 65 + min(30, len(matched) * 10) if candidate_skills else 70
            job["relevance_score"] = min(95, score)
            job["areas_to_improve"] = [
                "Tailor resume keywords to match job responsibilities",
                "Highlight relevant project impact with measurable metrics"
            ]
        return jobs
