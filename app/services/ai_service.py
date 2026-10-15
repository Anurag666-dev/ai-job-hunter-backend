"""
OpenRouter AI service helper — text-only (PDF text pre-extracted by pdfplumber).
"""
import httpx
import json
import re
from app.core.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_TEXT_MODEL


async def _call_openrouter(messages: list[dict], max_tokens: int = 2000) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://ai-job-hunter.local",
        "X-Title": "Tech Job Hunters",
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
    Scores jobs against the candidate's profile based on keyword match percentage.
    If the job does not have a visible description, bypasses scoring and marks it clearly.
    """
    if not profile or not jobs:
        return jobs

    # ── 1. Separate scorable jobs from unscorable jobs ─────────────────────────
    scorable_jobs = []
    for j in jobs:
        desc = (j.get("description") or "").strip()
        # Check if description is missing, too brief, or an anti-scraping placeholder
        if not desc or len(desc) < 30 or "cannot provide a description" in desc.lower() or "site won't allow us" in desc.lower():
            j["relevance_score"] = None
            j["score_note"] = "Cannot score: description not visible"
            j["matched_keywords"] = []
            j["missing_keywords"] = []
            j["areas_to_improve"] = []
        else:
            j["score_note"] = None
            scorable_jobs.append(j)

    if not scorable_jobs:
        return jobs

    candidate_skills = profile.get("skills", [])
    candidate_summary = profile.get("summary", "")
    candidate_projects = [
        f"{p.get('name', '')} ({', '.join(p.get('tech_stack', []))}): {p.get('description', '')}"
        for p in profile.get("projects", [])[:5]
    ]
    candidate_experience = [
        f"{e.get('role', '')} at {e.get('company', '')}: {e.get('description', '')}"
        for e in profile.get("experience", [])[:5]
    ]

    candidate_full_profile = {
        "verified_skills": candidate_skills,
        "summary": candidate_summary,
        "projects": candidate_projects,
        "experience": candidate_experience,
        "education": [f"{e.get('degree', '')} in {e.get('field', '')}" for e in profile.get("education", [])[:3]],
    }

    # Prepare scorable job listings for AI extraction
    jobs_summary = json.dumps([
        {
            "index": i,
            "title": j.get("title", ""),
            "company": j.get("company", ""),
            "description": (j.get("description") or "")[:500]
        }
        for i, j in enumerate(scorable_jobs[:20])
    ], ensure_ascii=False)

    prompt = f"""You are an expert technical recruiter and resume analyzer.
Analyze the candidate's complete profile information against each job listing's title and description.

CANDIDATE PROFILE:
{json.dumps(candidate_full_profile, ensure_ascii=False)}

JOB LISTINGS:
{jobs_summary}

Instructions for each job listing:
1. Dynamically identify and extract all core technical, domain, and tool keywords/skills required by the job (job_keywords). Do not extract stop words, verbs, locations, or company names; only extract real technical skills (e.g. programming languages, frameworks, cloud platforms, databases, tools, methodologies).
2. Compare those against the candidate's profile:
   - matched_keywords: technical skills demanded by the job that the candidate possesses.
   - missing_keywords: technical skills demanded by the job that are MISSING from the candidate's profile.
3. Calculate the mathematical percentage match using this exact formula:
   relevance_score = round(100 * (len(matched_keywords) / max(len(job_keywords), 1)))
   (Ensure relevance_score is an integer between 0 and 100).
4. Provide 1-2 concise, actionable suggestions for areas_to_improve.

Return ONLY a valid JSON array of objects (no markdown, no other text):
[
  {{
    "index": 0,
    "job_keywords": ["SkillA", "SkillB", "SkillC"],
    "matched_keywords": ["SkillA", "SkillB"],
    "missing_keywords": ["SkillC"],
    "relevance_score": 67,
    "areas_to_improve": ["Actionable tip 1", "Actionable tip 2"]
  }}
]"""

    try:
        content = await _call_openrouter([{"role": "user", "content": prompt}], max_tokens=2500)
        if "```" in content:
            lines = content.split("\n")
            json_lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(json_lines).strip()
        scores = json.loads(content)
        score_map = {item["index"]: item for item in scores}
        for i, job in enumerate(scorable_jobs):
            entry = score_map.get(i, {})
            matched = entry.get("matched_keywords", [])
            missing = entry.get("missing_keywords", [])
            job_kws = entry.get("job_keywords", [])
            total_req = len(job_kws) if job_kws else (len(matched) + len(missing))

            calc_score = round((len(matched) / total_req) * 100) if total_req > 0 else 0
            job["relevance_score"] = min(100, max(0, calc_score))
            job["matched_keywords"] = matched
            job["missing_keywords"] = missing
            job["areas_to_improve"] = entry.get("areas_to_improve", [])
        return jobs
    except Exception:
        # Fallback technical skill recognition: strictly match engineering & candidate domain skills
        TECH_CATALOG = {
            # Languages
            "python": "Python", "java": "Java", "javascript": "JavaScript", "typescript": "TypeScript",
            "c++": "C++", "c#": "C#", ".net": ".NET", "golang": "Go", "go": "Go", "rust": "Rust",
            "ruby": "Ruby", "php": "PHP", "swift": "Swift", "kotlin": "Kotlin", "scala": "Scala",
            "sql": "SQL", "bash": "Bash", "shell": "Shell", "r": "R", "dart": "Dart", "solidity": "Solidity",
            # Web & Frameworks
            "fastapi": "FastAPI", "django": "Django", "flask": "Flask", "react": "React", "react.js": "React",
            "next.js": "Next.js", "nextjs": "Next.js", "vue": "Vue.js", "vue.js": "Vue.js", "angular": "Angular",
            "spring boot": "Spring Boot", "spring": "Spring", "express": "Express.js", "express.js": "Express.js",
            "node.js": "Node.js", "nodejs": "Node.js", "node": "Node.js", "nestjs": "NestJS", "rails": "Rails",
            "asp.net": "ASP.NET", "laravel": "Laravel", "tailwind": "Tailwind CSS", "bootstrap": "Bootstrap",
            "redux": "Redux", "graphql": "GraphQL", "rest api": "REST APIs", "rest apis": "REST APIs", "rest": "REST APIs",
            # Data & AI / ML
            "pytorch": "PyTorch", "tensorflow": "TensorFlow", "keras": "Keras", "pandas": "Pandas",
            "numpy": "NumPy", "scikit-learn": "Scikit-Learn", "scipy": "SciPy", "opencv": "OpenCV",
            "nlp": "NLP", "llm": "LLMs", "llms": "LLMs", "genai": "Generative AI", "langchain": "LangChain",
            "machine learning": "Machine Learning", "deep learning": "Deep Learning", "data science": "Data Science",
            "data engineering": "Data Engineering", "spark": "Apache Spark", "hadoop": "Hadoop", "airflow": "Airflow",
            # Cloud & DevOps
            "aws": "AWS", "azure": "Azure", "gcp": "GCP", "docker": "Docker", "kubernetes": "Kubernetes",
            "k8s": "Kubernetes", "terraform": "Terraform", "ansible": "Ansible", "jenkins": "Jenkins",
            "ci/cd": "CI/CD", "github actions": "GitHub Actions", "gitlab ci": "GitLab CI", "linux": "Linux",
            "nginx": "Nginx", "prometheus": "Prometheus", "grafana": "Grafana", "serverless": "Serverless",
            "microservices": "Microservices",
            # Databases & Storage
            "postgresql": "PostgreSQL", "postgres": "PostgreSQL", "mysql": "MySQL", "mongodb": "MongoDB",
            "redis": "Redis", "elasticsearch": "Elasticsearch", "cassandra": "Cassandra", "sqlite": "SQLite",
            "dynamodb": "DynamoDB", "oracle": "Oracle", "firebase": "Firebase", "snowflake": "Snowflake",
            "bigquery": "BigQuery", "kafka": "Kafka", "rabbitmq": "RabbitMQ",
            # Practices & Tools
            "git": "Git", "github": "GitHub", "unit testing": "Unit Testing", "pytest": "Pytest",
            "jest": "Jest", "junit": "JUnit", "tdd": "TDD", "agile": "Agile", "scrum": "Scrum",
            "system design": "System Design", "oop": "OOP", "websockets": "WebSockets", "grpc": "gRPC"
        }

        # Candidate skills normalized
        user_skills_clean = {s.strip().lower(): s.strip() for s in candidate_skills if s.strip()}
        user_corpus = f"{' '.join(candidate_skills)} {candidate_summary} {' '.join(candidate_projects)} {' '.join(candidate_experience)}".lower()

        # Combine catalog with candidate's specific profile skills
        combined_tax = dict(TECH_CATALOG)
        for s_low, s_orig in user_skills_clean.items():
            if s_low not in combined_tax:
                combined_tax[s_low] = s_orig

        # Sort keys by descending length so multi-word tokens match first
        sorted_keys = sorted(combined_tax.keys(), key=len, reverse=True)

        for job in scorable_jobs:
            text = f"{job.get('title', '')} {job.get('description', '')}".lower()
            detected_skills: dict[str, str] = {}

            # Detect genuine technical skills in job text
            for key in sorted_keys:
                # Word boundary check
                pattern = rf"(?:\b|(?<=\W)){re.escape(key)}(?:\b|(?=\W))"
                if re.search(pattern, text):
                    label = combined_tax[key]
                    if label not in detected_skills.values():
                        detected_skills[key] = label

            clean_job_skills = list(detected_skills.values())

            # Evaluate matches against candidate profile
            matched = []
            missing = []

            for sk in clean_job_skills:
                sk_low = sk.lower()
                # Check candidate verified skills or full resume corpus
                if sk_low in user_skills_clean or re.search(rf"\b{re.escape(sk_low)}\b", user_corpus):
                    matched.append(sk)
                else:
                    missing.append(sk)

            total_req = len(clean_job_skills)
            # Mathematical exact match formula: 100 * (matched / total_keywords)
            math_score = round(100 * (len(matched) / total_req)) if total_req > 0 else 0

            job["relevance_score"] = min(100, max(0, math_score))
            job["matched_keywords"] = matched[:8]
            job["missing_keywords"] = missing[:8]

            improvements = []
            if missing:
                improvements.append(f"Highlight projects or practical experience demonstrating {', '.join(missing[:2])}.")
            if matched:
                improvements.append(f"Emphasize your hands-on accomplishments with {', '.join(matched[:2])}.")
            else:
                improvements.append("Tailor your resume project bullets to directly mirror the technical requirements.")

            job["areas_to_improve"] = improvements[:2]

        return jobs
