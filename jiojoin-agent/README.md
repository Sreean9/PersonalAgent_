# JioJoin Personal AI Agent

A production-ready AI assistant backend for Jio users — built on FastAPI, Groq (Llama 3.3-70b), and SQLAlchemy. Supports to-do management, smart reminders, Jio content discovery, and multi-turn conversation with JWT authentication.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Quick Start (Local Dev)](#2-quick-start-local-dev)
3. [Environment Variables Reference](#3-environment-variables-reference)
4. [Deploy on Railway (Step-by-Step)](#4-deploy-on-railway-step-by-step)
5. [Switch to PostgreSQL](#5-switch-to-postgresql)
6. [Docker Deployment](#6-docker-deployment)
7. [API Reference](#7-api-reference)
8. [Frontend Integration Guide](#8-frontend-integration-guide)
9. [Mobile App Integration](#9-mobile-app-integration)
10. [CI/CD with GitHub Actions](#10-cicd-with-github-actions)
11. [Scaling & Production Checklist](#11-scaling--production-checklist)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Layer                              │
│   Web (index.html)  ·  iOS App  ·  Android App  ·  JioPhone    │
└────────────────────────────┬────────────────────────────────────┘
                             │  HTTPS + JWT Bearer Token
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (Railway)                      │
│                                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────────┐  │
│  │  /auth   │  │  /chat   │  │  /todos  │  │  /reminders    │  │
│  │ register │  │  POST    │  │ GET/POST │  │  GET/POST/PUT  │  │
│  │ login    │  │          │  │ PUT/DEL  │  │  DELETE        │  │
│  └──────────┘  └────┬─────┘  └──────────┘  └────────────────┘  │
│                     │                                             │
│              ┌──────▼──────┐                                     │
│              │  Agent Core  │  ← Groq LLM (Llama 3.3-70b)       │
│              │  tool-loop   │  ← Tool dispatcher                  │
│              └──────┬──────┘                                     │
│                     │                                             │
│        ┌────────────┼────────────┐                               │
│        ▼            ▼            ▼                               │
│   todo_tools  utility_tools  discovery_tools                     │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  SQLAlchemy Async ORM → SQLite (dev) / PostgreSQL (prod) │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Key Files

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app, all route handlers, CORS setup |
| `agent.py` | Groq LLM call + tool-calling loop |
| `auth.py` | JWT creation, validation, password hashing |
| `database.py` | SQLAlchemy async engine, session factory |
| `models.py` | ORM models: User, Todo, Reminder, ConversationMessage |
| `config.py` | Pydantic settings — all config from env vars |
| `tools/todo_tools.py` | Create/list/update/delete todos via DB |
| `tools/utility_tools.py` | Calculator, time/timezone, reminders |
| `tools/discovery_tools.py` | JioStar/JioCinema content suggestions |
| `tools/tool_registry.py` | Groq-format tool schema definitions |
| `memory/conversation.py` | Per-session conversation history management |
| `index.html` | Production web frontend (single-file, mobile-first) |

---

## 2. Quick Start (Local Dev)

### Prerequisites

- Python 3.11+
- A free Groq API key from [console.groq.com](https://console.groq.com)

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/Sreean9/PersonalAgent.git
cd PersonalAgent/jiojoin-agent

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env — at minimum set GROQ_API_KEY and JWT_SECRET_KEY

# 5. Start the server
uvicorn main:app --reload --port 8000

# 6. Open the web UI
# Open index.html in your browser (update BACKEND_URL to http://localhost:8000)

# API docs available at:
# http://localhost:8000/docs      <- Swagger UI
# http://localhost:8000/redoc     <- ReDoc
```

---

## 3. Environment Variables Reference

Copy `.env.example` to `.env` and fill in values. **Never commit `.env` to git.**

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | Yes | — | Groq API key from console.groq.com |
| `JWT_SECRET_KEY` | Yes | `change_me` | Random 32+ char string. Generate: `openssl rand -hex 32` |
| `DATABASE_URL` | No | SQLite | Full DB URL (see Section 5 for PostgreSQL) |
| `GROQ_MODEL` | No | `llama-3.3-70b-versatile` | Groq model ID |
| `AGENT_TEMPERATURE` | No | `0.6` | LLM temperature (0.0–1.0) |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | No | `10080` (7 days) | Token TTL in minutes |
| `MAX_CONVERSATION_HISTORY` | No | `20` | Messages retained per session |
| `APP_ENV` | No | `development` | Set to `production` in prod |

### Generate a secure JWT secret

```bash
openssl rand -hex 32
# Example output: a3f8c2d1e4b5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1
```

---

## 4. Deploy on Railway (Step-by-Step)

Railway is the recommended hosting platform. The free Starter plan ($5 credit/month) is enough for a personal or demo deployment.

### Step 1 — Sign up and connect GitHub

1. Go to [railway.app](https://railway.app) → **Start a New Project**
2. Click **Deploy from GitHub repo**
3. Authorize Railway to access your GitHub account
4. Select the repo: `Sreean9/PersonalAgent`

### Step 2 — Set the root directory

Railway will detect the root of the repo. Since the backend lives in a subfolder:

1. In the Railway service settings → **Source** tab
2. Set **Root Directory** to: `jiojoin-agent`
3. Railway's Nixpacks builder will auto-detect Python and use `requirements.txt`

### Step 3 — Set environment variables

In Railway → your service → **Variables** tab, add:

```
GROQ_API_KEY        =  gsk_xxxxxxxxxxxxxxxxxxxx
JWT_SECRET_KEY      =  <output of: openssl rand -hex 32>
APP_ENV             =  production
DATABASE_URL        =  sqlite+aiosqlite:////data/jiojoin.db
```

> For PostgreSQL (recommended for production): add a PostgreSQL plugin in Railway, then set `DATABASE_URL` to the Railway-provided `${{Postgres.DATABASE_URL}}` variable.

### Step 4 — Deploy

1. Click **Deploy** — Railway builds and starts the container
2. Wait ~2 minutes for the build to complete
3. Go to **Settings → Networking → Generate Domain**
4. Copy the URL — e.g. `https://jiojoin-agent-production.up.railway.app`

### Step 5 — Update the frontend

Open `index.html` and update line 1 of the `<script>` block:

```javascript
// Change this:
const BACKEND_URL = 'https://YOUR-APP.railway.app';

// To your actual Railway URL:
const BACKEND_URL = 'https://jiojoin-agent-production.up.railway.app';
```

Then drag-and-drop `index.html` (alone, or with the Netlify Functions folder) to Netlify.

### Step 6 — Verify

```bash
curl https://jiojoin-agent-production.up.railway.app/health
# Expected: {"status":"ok","version":"1.0.0","model":"llama-3.3-70b-versatile"}
```

---

## 5. Switch to PostgreSQL

### Local dev with Docker

```bash
# Start PostgreSQL + Adminer DB UI
docker compose --profile postgres up -d

# Set DATABASE_URL in .env
DATABASE_URL=postgresql+asyncpg://jiojoin:jiojoin_secret@localhost:5432/jiojoin
```

### Production on Railway

1. In Railway project → **New** → **Database** → **PostgreSQL**
2. Railway provisions a managed Postgres instance
3. In your API service → Variables → add:
   ```
   DATABASE_URL = ${{Postgres.DATABASE_URL}}
   ```
   Railway auto-injects the connection string.

4. Add async PostgreSQL driver to `requirements.txt`:
   ```
   asyncpg==0.29.0
   ```

### Why PostgreSQL for production?

SQLite is single-writer only — fine for personal use, breaks under concurrent users. PostgreSQL handles thousands of concurrent connections and Railway's managed Postgres includes automated backups.

---

## 6. Docker Deployment

### Local development (SQLite, with hot-reload)

```bash
docker compose up
# API at http://localhost:8000
# Swagger at http://localhost:8000/docs
```

### Local development with PostgreSQL

```bash
docker compose --profile postgres up
# API at http://localhost:8000
# PostgreSQL at localhost:5432
# Adminer DB UI at http://localhost:8080
```

### Production image build

```bash
# Build the production image
docker build -t jiojoin-agent:latest .

# Run with environment variables
docker run -d \
  --name jiojoin-api \
  -p 8000:8000 \
  -e GROQ_API_KEY=gsk_xxx \
  -e JWT_SECRET_KEY=your_32_char_secret \
  -e DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/jiojoin \
  -e APP_ENV=production \
  jiojoin-agent:latest
```

### Push to a container registry

```bash
# Docker Hub
docker tag jiojoin-agent:latest yourdockerhub/jiojoin-agent:latest
docker push yourdockerhub/jiojoin-agent:latest

# Google Artifact Registry / AWS ECR — same tag and push flow
```

---

## 7. API Reference

All endpoints except `/health`, `/auth/register`, and `/auth/login` require:
```
Authorization: Bearer <jwt_token>
```

### Authentication

**POST /auth/register**
```json
// Request
{ "username": "sridevi", "password": "SecurePass123" }

// Response 201
{ "access_token": "eyJ...", "token_type": "bearer", "username": "sridevi" }
```

**POST /auth/login**
```json
// Request
{ "username": "sridevi", "password": "SecurePass123" }

// Response 200
{ "access_token": "eyJ...", "token_type": "bearer", "username": "sridevi" }
```

### Chat

**POST /chat**
```json
// Request
{
  "message": "Add a reminder to call Rahul tomorrow at 3pm",
  "session_id": null
}
// Pass null for new sessions; send back the returned session_id for continuity.

// Response 200
{
  "reply": "Done! I've set a reminder to call Rahul tomorrow at 3:00 PM.",
  "session_id": "a1b2c3d4",
  "tools_used": ["set_reminder"]
}
```

### To-Dos

**GET /todos** — Returns all todos for the authenticated user.

**POST /todos**
```json
{ "text": "Buy groceries", "priority": "medium" }
```

**PUT /todos/{todo_id}**
```json
{ "done": true }
```

**DELETE /todos/{todo_id}** — Returns `204 No Content`.

### Reminders

**GET /reminders** — Returns upcoming reminders.

**POST /reminders**
```json
{ "text": "Call Rahul", "remind_at": "2026-05-14T15:00:00Z" }
```

**PUT /reminders/{id}** / **DELETE /reminders/{id}**

### Discovery

**GET /whats-new** — Returns personalised Jio content suggestions.
```json
{
  "items": [
    { "title": "Shark Tank India S4", "platform": "JioHotstar", "genre": "Reality", "url": "#" }
  ]
}
```

### Health

**GET /health**
```json
{ "status": "ok", "version": "1.0.0", "model": "llama-3.3-70b-versatile" }
```

---

## 8. Frontend Integration Guide

The production frontend (`index.html`) is a self-contained, mobile-first single-page app.

### Required: One-line backend URL update

```javascript
// Line 1 of the <script> block in index.html
const BACKEND_URL = 'https://YOUR-RAILWAY-URL.railway.app';
```

### Auth flow

```javascript
// 1. Register or login to get JWT token
const res = await fetch(BACKEND_URL + '/auth/login', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ username, password })
});
const { access_token } = await res.json();

// 2. Store token in memory (NOT localStorage — security best practice)
let jwtToken = access_token;

// 3. Include on every subsequent request
headers: { 'Authorization': 'Bearer ' + jwtToken }
```

### CORS — restrict for production

In `main.py`, replace `allow_origins=["*"]` with your actual frontend domain:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-netlify-site.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 9. Mobile App Integration

### React Native / Expo

```javascript
// api/client.js
import * as SecureStore from 'expo-secure-store';

const BASE_URL = 'https://jiojoin-agent-production.up.railway.app';

export async function login(username, password) {
  const res = await fetch(`${BASE_URL}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  });
  const data = await res.json();
  await SecureStore.setItemAsync('jwt_token', data.access_token);
  return data;
}

export async function chat(message, sessionId) {
  const token = await SecureStore.getItemAsync('jwt_token');
  const res = await fetch(`${BASE_URL}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`,
    },
    body: JSON.stringify({ message, session_id: sessionId }),
  });
  return res.json();
}
```

> Use `expo-secure-store` (not AsyncStorage) for JWT tokens — it uses the OS keychain.

### Flutter / Dart

```dart
// lib/services/api_service.dart
import 'dart:convert';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:http/http.dart' as http;

class ApiService {
  static const _base = 'https://jiojoin-agent-production.up.railway.app';
  final _storage = const FlutterSecureStorage();

  Future<void> login(String username, String password) async {
    final res = await http.post(
      Uri.parse('$_base/auth/login'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'username': username, 'password': password}),
    );
    final data = jsonDecode(res.body);
    await _storage.write(key: 'jwt', value: data['access_token']);
  }

  Future<Map<String, dynamic>> chat(String message, String? sessionId) async {
    final token = await _storage.read(key: 'jwt');
    final res = await http.post(
      Uri.parse('$_base/chat'),
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer $token',
      },
      body: jsonEncode({'message': message, 'session_id': sessionId}),
    );
    return jsonDecode(res.body);
  }
}
```

### Push Notifications for Reminders

To deliver reminders as push notifications, add a background scheduler to `main.py`:

```bash
pip install apscheduler firebase-admin
```

```python
# main.py — startup event
from apscheduler.schedulers.asyncio import AsyncIOScheduler

scheduler = AsyncIOScheduler()

@app.on_event("startup")
async def start_scheduler():
    scheduler.add_job(check_and_send_reminders, 'interval', minutes=1)
    scheduler.start()

async def check_and_send_reminders():
    # 1. Query DB for reminders due in the next minute
    # 2. Send FCM push to user's registered device token
    pass
```

---

## 10. CI/CD with GitHub Actions

Create `.github/workflows/deploy.yml` in your repo:

```yaml
name: Deploy to Railway

on:
  push:
    branches: [main]
    paths:
      - 'jiojoin-agent/**'

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        working-directory: jiojoin-agent
        run: pip install -r requirements.txt

      - name: Lint with ruff
        working-directory: jiojoin-agent
        run: |
          pip install ruff
          ruff check .

      - name: Run tests
        working-directory: jiojoin-agent
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
          JWT_SECRET_KEY: ${{ secrets.JWT_SECRET_KEY }}
        run: |
          pip install pytest pytest-asyncio httpx
          pytest tests/ -v

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4

      - name: Install Railway CLI
        run: npm install -g @railway/cli

      - name: Deploy to Railway
        working-directory: jiojoin-agent
        env:
          RAILWAY_TOKEN: ${{ secrets.RAILWAY_TOKEN }}
        run: railway up --service jiojoin-agent
```

### GitHub Secrets to configure

Go to **GitHub repo → Settings → Secrets and variables → Actions**:

| Secret | Value |
|--------|-------|
| `GROQ_API_KEY` | Your Groq API key |
| `JWT_SECRET_KEY` | Your production JWT secret |
| `RAILWAY_TOKEN` | From Railway → Account Settings → Tokens |

---

## 11. Scaling & Production Checklist

### Before going live

- [ ] Replace SQLite with PostgreSQL (`DATABASE_URL` pointing to Railway Postgres)
- [ ] Set `APP_ENV=production` in Railway Variables
- [ ] Generate a strong `JWT_SECRET_KEY` (`openssl rand -hex 32`)
- [ ] Lock CORS `allow_origins` to your actual frontend domain in `main.py`
- [ ] Enable Railway custom domain + HTTPS (automatic with Railway)
- [ ] Test all endpoints via Swagger UI at `/docs`
- [ ] Test login, chat, todos, and reminders end-to-end on mobile

### Monitoring

- Railway's built-in **Metrics** tab shows CPU, memory, and request logs
- Add Sentry for error tracking (free tier available):
  ```python
  import sentry_sdk
  sentry_sdk.init(dsn="https://xxx@sentry.io/yyy", traces_sample_rate=0.1)
  ```

### Rate limiting

```bash
pip install slowapi
```
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/chat")
@limiter.limit("30/minute")
async def chat_endpoint(request: Request, ...):
    ...
```

### Estimated costs at scale

| Users | Railway Plan | PostgreSQL | Estimated Cost |
|-------|-------------|------------|----------------|
| 1–50 | Hobby ($5/mo credit) | SQLite | Free |
| 50–500 | Hobby | Railway Postgres ($5/mo) | ~$10/mo |
| 500–5,000 | Pro ($20/mo) | Railway Postgres ($25/mo) | ~$45/mo |
| 5,000+ | Enterprise | Managed RDS / Cloud SQL | Contact Jio infra |

---

## Support & Contacts

- **Product Owner**: Sridevi Nune — sridevi.nune@ril.com
- **Groq API Docs**: https://console.groq.com/docs
- **Railway Docs**: https://docs.railway.app
- **FastAPI Docs**: https://fastapi.tiangolo.com
- **SQLAlchemy Async**: https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html
