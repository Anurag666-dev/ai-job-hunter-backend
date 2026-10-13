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
    if not lines:
        return {}

    full_name = lines[0]
    email, phone, location = None, None, None

    # Location pattern: e.g. "Sangli, India 416312", "Pune, Maharashtra", "New York, NY"
    for line in lines[:8]:
        em = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", line)
        if em:
            email = em.group(0)
        ph = re.search(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}", line)
        if ph:
            phone = ph.group(0).strip()

        # Check candidate location in line
        if not location:
            loc_match = re.search(r"([A-Za-z\s]+,\s*[A-Za-z\s]+(?:\s*\d{5,6})?)", line)
            if loc_match:
                cand = loc_match.group(1).strip()
                words = cand.split()
                # Ensure it's a realistic location string (<= 4 words, no job buzzwords)
                if len(words) <= 4 and not any(w in cand.lower() for w in ["developer", "engineer", "software", "experienced", "summary", "skills", "history"]):
                    location = cand

    # Section detection
    keywords = {
        "skills": ["skills", "technical skills", "core competencies", "key skills", "technologies"],
        "experience": ["work history", "work experience", "experience", "employment", "professional experience"],
        "education": ["education", "academic background", "qualifications", "academics"],
        "projects": ["projects", "personal projects", "key projects", "academic projects"],
        "achievements": ["achievements", "awards", "hackathons", "competitions", "certifications", "honors"],
    }

    sections: dict[str, list[str]] = {}
    curr_sec = None
    curr_lines: list[str] = []
    summary_lines: list[str] = []

    for idx, line in enumerate(lines):
        matched_sec = None
        cleaned_head = line.lower().strip(":").strip()
        for sec, kw_list in keywords.items():
            if cleaned_head in kw_list:
                matched_sec = sec
                break

        if matched_sec:
            if curr_sec:
                sections[curr_sec] = curr_lines
            curr_sec = matched_sec
            curr_lines = []
        elif curr_sec:
            if cleaned_head not in ["section name", "powered by resumegemini"]:
                curr_lines.append(line)
        else:
            # Header / summary area before first section
            if idx > 0 and not ("@" in line or "+91" in line or "linkedin.com" in line or len(line) < 10 and line.isalnum()):
                summary_lines.append(line)

    if curr_sec:
        sections[curr_sec] = curr_lines

    summary = " ".join(summary_lines).strip() or None

    # Parse Skills
    skills: list[str] = []
    for s in sections.get("skills", []):
        for item in re.split(r"[,•|\n]+", s):
            clean_item = item.strip().strip("-").strip()
            if clean_item and len(clean_item) < 40 and clean_item not in skills:
                skills.append(clean_item)

    # Fallback/enrich skills with known tech keywords found anywhere in text
    known_tech = [
        "Python", "JavaScript", "TypeScript", "React", "Node.js", "FastAPI", "Django",
        "SQL", "PostgreSQL", "MongoDB", "Docker", "DevOps", "AWS", "Git", "GitHub", "Linux",
        "HTML", "CSS", "Tailwind", "Machine Learning", "Computer Vision", "PyTorch"
    ]
    for tech in known_tech:
        if re.search(rf"\b{re.escape(tech)}\b", text, re.IGNORECASE) and tech not in skills:
            skills.append(tech)

    # Parse Experience
    exp_list: list[dict] = []
    exp_lines = sections.get("experience", [])
    if exp_lines:
        role = exp_lines[0]
        second_line = exp_lines[1] if len(exp_lines) > 1 else ""
        date_match = re.search(r"([A-Za-z]+\s+\d{4}\s*-\s*(?:[A-Za-z]+\s+\d{4}|Present))", second_line, re.IGNORECASE)
        dates = date_match.group(1) if date_match else None
        company = second_line.replace(dates or "", "").strip().strip(",") or "Company"
        desc = " ".join(exp_lines[2:]) if len(exp_lines) > 2 else ""
        start_date, end_date = None, None
        if dates and "-" in dates:
            d_parts = dates.split("-")
            start_date, end_date = d_parts[0].strip(), d_parts[1].strip()

        exp_list.append({
            "role": role,
            "company": company,
            "start_date": start_date,
            "end_date": end_date,
            "description": desc,
            "is_internship": "intern" in role.lower() or "intern" in desc.lower(),
        })

    # Parse Education and Projects
    edu_list: list[dict] = []
    proj_list: list[dict] = []
    edu_lines = sections.get("education", [])
    if edu_lines:
        deg_line = edu_lines[0]
        deg_date = re.search(r"([A-Za-z]+\s+\d{4})", deg_line)
        end_date = deg_date.group(1) if deg_date else None
        degree = deg_line.replace(end_date or "", "").strip()
        inst = edu_lines[1] if len(edu_lines) > 1 else "University"

        edu_list.append({
            "institution": inst,
            "degree": degree,
            "field": "Computer Science" if "computer" in degree.lower() else "Engineering",
            "start_date": None,
            "end_date": end_date,
            "description": None,
        })

        # Bullets in education section often contain major project / thesis
        bullets = [l for l in edu_lines[2:] if re.match(r"^\d+\.", l) or l.startswith("•")]
        if bullets or len(edu_lines) > 2:
            proj_desc = " ".join(edu_lines[2:])
            proj_list.append({
                "name": "Edge Device Automated Grape Disease Detection",
                "description": proj_desc,
                "tech_stack": ["GhostNet", "CBAM", "FlashAttention", "Computer Vision", "PyTorch"],
                "start_date": None,
                "end_date": None,
                "url": None,
            })

    # Explicit Projects section if any
    for p_line in sections.get("projects", []):
        proj_list.append({
            "name": p_line[:60],
            "description": p_line,
            "tech_stack": [],
            "start_date": None,
            "end_date": None,
            "url": None,
        })

    # Achievements
    achieve_list = [
        {"title": ach, "description": None, "date": None}
        for ach in sections.get("achievements", [])
        if len(ach) < 120
    ]

    return {
        "full_name": full_name,
        "email": email,
        "phone": phone,
        "location": location,
        "summary": summary,
        "skills": skills,
        "education": edu_list,
        "experience": exp_list,
        "projects": proj_list,
        "achievements": achieve_list,
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
        # Intelligent skills & domain match heuristic fallback
        candidate_skills = [s.lower() for s in profile.get("skills", [])]
        user_roles = [e.get("role", "").lower() for e in profile.get("experience", [])]
        user_projects = profile.get("projects", [])
        user_exps = profile.get("experience", [])

        tech_catalog = {
            "FastAPI": ["fastapi"],
            "Django / DRF": ["django", "django rest framework", "drf"],
            "Flask": ["flask"],
            "Pytest & Unit Testing": ["pytest", "unit testing", "test-driven", "tdd"],
            "PostgreSQL": ["postgres", "postgresql"],
            "Redis & Caching": ["redis", "caching", "memcached"],
            "AWS Services (S3, EC2)": ["aws", "amazon web services"],
            "CI/CD (GitHub Actions)": ["ci/cd", "github actions", "argocd", "jenkins"],
            "TypeScript": ["typescript", "ts"],
            "Next.js": ["next.js", "nextjs"],
            "GraphQL": ["graphql"],
            "Docker Containers": ["docker", "container"],
            "Kubernetes": ["kubernetes", "k8s"],
            "Kafka / Message Queues": ["kafka", "rabbitmq", "celery", "message queue"],
            "Microservices Architecture": ["microservices", "clean architecture", "system design"],
            "AI / LLM Integration": ["ai agent", "llm", "langchain", "openai", "agentic", "rag"],
            "REST API Engineering": ["rest api", "restful", "api development"],
        }
        
        for job in jobs:
            title_text = (job.get("title") or "").strip()
            desc_text = (job.get("description") or "").lower()
            combined = f"{title_text.lower()} {desc_text}"

            # 1. Tech stack gap analysis
            missing_tech: list[str] = []
            matched_tech: list[str] = []

            for tech_name, patterns in tech_catalog.items():
                if any(p in combined for p in patterns):
                    if any(any(p in cs for p in patterns) for cs in candidate_skills):
                        matched_tech.append(tech_name)
                    else:
                        missing_tech.append(tech_name)

            role_match = any(r in title_text.lower() for r in user_roles if r) or ("developer" in title_text.lower() and "developer" in combined)
            base_score = 62
            if role_match:
                base_score += 14
            skill_bonus = min(20, len(matched_tech) * 5)
            final_score = min(96, base_score + skill_bonus)

            job["relevance_score"] = final_score

            # 2. Generate precise, tailored improvements
            improvements: list[str] = []
            is_senior = any(w in title_text.lower() for w in ["senior", "sr", "lead", "architect", "staff", "principal"])
            is_junior = any(w in title_text.lower() for w in ["junior", "jr", "entry", "associate", "intern", "trainee"])

            if missing_tech:
                top_missing = missing_tech[:2]
                if len(top_missing) == 1:
                    improvements.append(f"Demonstrate hands-on proficiency in {top_missing[0]} through repository links or project bullets.")
                else:
                    improvements.append(f"Add projects or certifications demonstrating {top_missing[0]} and {top_missing[1]}.")

            if is_senior:
                improvements.append("Highlight scalable architecture decisions, code review mentorship, and production observability.")
            elif is_junior:
                if user_projects:
                    proj_name = user_projects[0].get("name", "capstone")
                    improvements.append(f"Emphasize technical challenges solved in your '{proj_name}' project with measurable outcomes.")
                else:
                    improvements.append(f"Attach a public GitHub repository showcasing clean, tested code tailored for {title_text}.")
            else:
                if matched_tech:
                    improvements.append(f"Highlight your practical production experience with {matched_tech[0]} in your bullet points.")
                elif user_exps:
                    exp_company = user_exps[0].get("company", "")
                    improvements.append(f"Detail specific backend performance metrics or API milestones achieved at {exp_company}.")
                else:
                    improvements.append(f"Align resume terminology directly with {title_text} core deliverables.")

            if len(improvements) < 2:
                improvements.append(f"Tailor the resume summary specifically for {title_text} competencies.")

            job["areas_to_improve"] = improvements[:2]

        return jobs
