# 🚀 Tech Job Hunters — Backend API

A high-performance, asynchronous FastAPI backend powering multi-board tech job aggregation, mathematical skill-gap analysis, candidate resume parsing, and market intelligence analytics.

---

## 🏛️ System Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Tech Job Hunters Client (React)                       │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │  HTTPS / REST / JSON
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            FastAPI Backend (Port 1776)                          │
│                                                                                 │
│   ┌─────────────────────┐   ┌────────────────────────┐   ┌──────────────────┐   │
│   │   Auth & Security   │   │   Profile & CV Router  │   │   Jobs Router    │   │
│   │   (JWT + Bcrypt)    │   │  (pdfplumber + Parser) │   │  (Multi-Source)  │   │
│   └──────────┬──────────┘   └───────────┬────────────┘   └─────────┬────────┘   │
└──────────────┼──────────────────────────┼──────────────────────────┼────────────┘
               │                          │                          │
               ▼                          │                          ▼
┌───────────────────────────────┐         │         ┌─────────────────────────────┐
│      PostgreSQL (Asyncpg)     │         │         │      Brave Search API       │
│                               │         │         │  (Parallel Multi-Query)     │
│  • users                      │         │         │                             │
│  • profiles (skills, exp)     │         │         │  • LinkedIn   • Workable    │
│  • saved_jobs                 │         │         │  • Indeed     • Greenhouse  │
│  • visited_jobs               │         │         │  • Naukri     • Wellfound   │
└───────────────────────────────┘         │         │  • Lever                    │
                                          │         └──────────────┬──────────────┘
                                          │                        │
                                          ▼                        ▼
               ┌───────────────────────────────────────────────────────────────┐
               │              Data Ingestion & Normalization Pipe              │
               │                                                               │
               │  • Canonical URL Stripper (removes ?utm_*, ?ref, ?trk)        │
               │  • Compensation Parser (LPA / INR ₹ / USD $ extraction)       │
               │  • Employer Slug Extraction & Platform Filtering              │
               │  • Description Visibility Checker (<30c / anti-bot bypass)    │
               └──────────────────────────────┬────────────────────────────────┘
                                              │
                                              ▼
               ┌───────────────────────────────────────────────────────────────┐
               │             Intelligent Skill & Scoring Engine                │
               │                                                               │
               │  1. Primary: OpenRouter LLM (Structured JSON Analysis)        │
               │  2. Deterministic Fallback: Engineering Taxonomy Matcher      │
               │  3. Formula: Score = 100 * (Matched Skills / Required Skills) │
               │  4. Unscorable Handler: "Cannot score: description not visible"│
               └──────────────────────────────┬────────────────────────────────┘
                                              │
                                              ▼
                               ┌─────────────────────────────┐
                               │     Redis / Memory Cache    │
                               │   (TTL 6h Key Invalidation) │
                               └─────────────────────────────┘
```

---

## ✨ Core Features

1. **Concurrent Multi-Platform Search Orchestration**:
   - Queries 7 top hiring platforms in parallel using `asyncio.gather` for sub-second retrieval.
   - Intelligent domain routing for Workable, Greenhouse, Lever, LinkedIn, Indeed, Naukri, and Wellfound.
2. **Robust Data Ingestion & Sanitization**:
   - **Canonical URL Stripper**: Strips marketing tracking parameters (`utm_*`, `ref`, `trk`, `sessionid`, `fbclid`).
   - **Salary Parser**: Normalizes Indian Lakhs/LPA (`12-18 LPA`), INR (`₹15,00,000`), and international currencies (`$120k-$150k`).
   - **Strict Description Checker**: Detects anti-scraping blocks or missing snippets and labels them gracefully as unscorable without wasting LLM tokens.
3. **Mathematical Match Scoring**:
   - Uses strict mathematical ratio: $\text{Relevance} = 100 \times \frac{|\text{Matched}|}{|\text{Demanded}|}$.
   - Zero hallucinated numbers or fake defaults.
   - Strict stopword suppression to ensure non-skill conversational words (`in`, `to`, `Pune`, `hiring`, `seekers`) never pollute skill counts.
4. **Dual-Engine Architecture (Resilient LLM + Deterministic Fallback)**:
   - Evaluates jobs via OpenRouter LLM when API connectivity is live.
   - Automatically and gracefully falls back to an in-memory engineering catalog matching verified resume skills if the external AI API is rate-limited or unavailable.
5. **Caching & Fast Invalidation**:
   - Structured caching across query, location, and platform signatures with TTL boundaries.
   - Transparent fallback to an asynchronous memory cache when Redis is not running locally.

---

## 🛠️ Tech Stack

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Python 3.12+ / 3.14)
- **Database ORM**: [SQLAlchemy 2.0 (asyncio)](https://docs.sqlalchemy.org/) + [Asyncpg](https://github.com/MagicStack/asyncpg)
- **Database**: PostgreSQL
- **Caching**: Redis / In-Memory TTL Cache
- **Search Provider**: [Brave Search API](https://brave.com/search/api/)
- **AI / LLM Integration**: [OpenRouter API](https://openrouter.ai/) (`openai/gpt-4o-mini`)
- **PDF Extraction**: [pdfplumber](https://github.com/jsvine/pdfplumber)
- **Authentication**: JWT (`pyjwt`) + `passlib` (Bcrypt)

---

## 🚀 Getting Started (Step-by-Step)

### 1. Prerequisites
Ensure you have the following installed on your system:
- **Python**: `>= 3.12` (recommended: Python 3.12 or 3.14)
- **PostgreSQL**: Running locally or on a remote server
- **Redis** *(optional)*: In-memory fallback is automatically used if Redis is not running

---

### 2. Clone the Repository
```bash
git clone https://github.com/Anurag666-dev/ai-job-hunter-backend.git
cd ai-job-hunter-backend
```

---

### 3. Setup Virtual Environment & Install Dependencies

Using `uv` (recommended for blazing fast installs):
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

Or using standard `pip`:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r <(python3 -c "import tomllib; [print(d) for d in tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']]")
```

---

### 4. Setup PostgreSQL Database

Create a database in PostgreSQL:
```bash
psql -U postgres -c "CREATE DATABASE tech_job_hunters;"
```

---

### 5. Configure Environment Variables

Copy the example environment file and fill in your API keys:
```bash
cp .env.example .env
```

Edit `.env`:
```env
# Required: Search API
BRAVE_API_KEY=your_brave_search_api_key

# Optional: OpenRouter AI key (falls back to deterministic skill matcher if omitted)
OPENROUTER_API_KEY=your_openrouter_api_key

# Database Connection
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/tech_job_hunters

# JWT Security Key (32+ characters)
SECRET_KEY=tech-job-hunters-production-secure-key-32-bytes

# Server & CORS
BACKEND_PORT=1776
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:5174
```

---

### 6. Start the Server

```bash
uvicorn main:app --host 0.0.0.0 --port 1776 --reload
```

The interactive API documentation (Swagger UI) is available immediately at:
👉 **[http://localhost:1776/docs](http://localhost:1776/docs)**

---

## 📡 API Endpoints Reference

| Method | Endpoint | Description | Auth Required |
| :--- | :--- | :--- | :---: |
| `POST` | `/api/v1/auth/signup` | Register a new user account | No |
| `POST` | `/api/v1/auth/signin` | Authenticate user & return JWT token | No |
| `GET` | `/api/v1/auth/me` | Retrieve current authenticated user | Yes |
| `GET` | `/api/v1/profile/me` | Retrieve candidate parsed profile & skills | Yes |
| `POST` | `/api/v1/profile/upload-cv` | Upload PDF resume, extract text & parse profile | Yes |
| `PUT` | `/api/v1/profile/me` | Update candidate profile manually | Yes |
| `GET` | `/api/v1/jobs/search` | Search jobs across platforms with real-time scoring | Yes |
| `POST` | `/api/v1/jobs/save` | Save a job to persistent bookmarks | Yes |
| `DELETE`| `/api/v1/jobs/saved/{hash}`| Remove a saved job | Yes |
| `GET` | `/api/v1/jobs/saved` | List all saved jobs for current user | Yes |
| `POST` | `/api/v1/jobs/visit` | Record a visited / applied job | Yes |
| `GET` | `/api/v1/jobs/visited` | List all visited jobs | Yes |

---

## 🔒 Security Best Practices

- Real `.env` files and `docker-compose.yml` are strictly excluded from git tracking.
- Passwords are encrypted with salt using Bcrypt.
- All database operations utilize asynchronous parameterized queries to eliminate SQL injection.
- Canonical URL sanitization removes tracking session tokens and browser referral footprints.
