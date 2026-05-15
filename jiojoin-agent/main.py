"""
main.py – JioJoin Personal AI Agent API

REST endpoints:
  POST /auth/register          – create account
  POST /auth/login             – get JWT token
  GET  /auth/me                – current user profile
  POST /auth/push-token        – register FCM push token

  POST /chat                   – agent chat (REST, returns full reply)
  WS   /ws/chat                – agent chat (WebSocket, streams tokens)

  GET  /todos                  – list to-dos
  POST /todos                  – create a to-do
  PUT  /todos/{id}             – update a to-do
  DELETE /todos/{id}           – delete a to-do

  GET  /reminders              – list reminders
  POST /reminders              – create a reminder
  DELETE /reminders/{id}       – cancel a reminder

  GET  /interests              – get user interests
  PUT  /interests              – update user interests

  GET  /whats-new              – latest announcements

  GET  /coins/balance          – user coin balance
  GET  /coins/history          – coin ledger history

  GET  /streak                 – user streak info

  GET  /health                 – service health check
"""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

import pathlib

from pydantic import BaseModel

from fastapi import (
    FastAPI, Depends, HTTPException, Query, status,
    WebSocket, WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from config import get_settings
from database import get_db, init_db
from redis_client import get_redis, close_redis
from auth import (
    hash_password, verify_password,
    create_access_token, get_current_user, decode_token,
)
from models import (
    User, Todo, Reminder, UserInterest, CoinLedger, Streak,
    UserRegister, UserLogin, TokenResponse, UserOut,
    TodoCreate, TodoUpdate, TodoOut,
    ReminderCreate, ReminderOut,
    InterestUpdate, InterestOut,
    AnnouncementOut,
    ChatRequest, ChatResponse,
    CoinBalanceOut, CoinLedgerOut, StreakOut,
    PushTokenUpdate,
    TodoStatus, TodoPriority, ReminderStatus,
)
from agent import agent
from memory import conversation as conv_memory
from tools.todo_tools import (
    add_todo as tool_add_todo,
    list_todos as tool_list_todos,
    update_todo as tool_update_todo,
    delete_todo as tool_delete_todo,
)
from tools.utility_tools import (
    set_reminder as tool_set_reminder,
    list_reminders as tool_list_reminders,
    cancel_reminder as tool_cancel_reminder,
)
from tools.discovery_tools import (
    get_whats_new as tool_get_whats_new,
    get_user_interests as tool_get_interests,
    update_user_interests as tool_update_interests,
)

# ─────────────────────────────────────────────────────────────────────────────
#  App setup
# ─────────────────────────────────────────────────────────────────────────────

settings = get_settings()
logging.basicConfig(
    level=logging.DEBUG if not settings.is_production else logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting JioJoin Agent API…")
    await init_db()
    logger.info("Database initialised.")
    try:
        await get_redis()
        logger.info("Redis connected.")
    except Exception as exc:
        logger.warning("Redis unavailable at startup (%s). Falling back to in-memory.", exc)
    yield
    await close_redis()
    logger.info("JioJoin Agent API shut down.")


app = FastAPI(
    title="JioJoin Personal AI Agent",
    description="Personal AI assistant for the JioJoin platform.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
#  Health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui():
    """Serve the single-page chat UI."""
    html_path = pathlib.Path(__file__).parent / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "service": "JioJoin Agent API", "version": "2.0.0"}


# ─────────────────────────────────────────────────────────────────────────────
#  Auth
# ─────────────────────────────────────────────────────────────────────────────

class JioJoinSSO(BaseModel):
    jio_user_id: str
    name: str = "JioJoin User"
    preferred_language: str = "en"


@app.post("/auth/jiojoin-sso", response_model=TokenResponse, tags=["Auth"])
async def jiojoin_sso(payload: JioJoinSSO, db: AsyncSession = Depends(get_db)):
    """
    Silent SSO called by the JioJoin WebView on load.
    Creates a user account on first visit, then returns a JWT every time.
    No password required — trust comes from the JioJoin app itself.
    """
    try:
        synthetic_email = f"{payload.jio_user_id}@jiojoin.internal"
        result = await db.execute(select(User).where(User.email == synthetic_email))
        user = result.scalar_one_or_none()

        if not user:
            user = User(
                name=payload.name,
                email=synthetic_email,
                hashed_password="",
                preferred_language=payload.preferred_language,
            )
            db.add(user)
            await db.flush()  # assigns user.id without committing

            # Streak creation is best-effort — won't block login if table is missing
            try:
                db.add(Streak(user_id=user.id))
                await db.commit()
            except Exception:
                await db.rollback()
                # Re-insert user without streak so they can still log in
                db.add(user)
                await db.commit()

            await db.refresh(user)
        elif user.name != payload.name:
            user.name = payload.name
            await db.commit()

        token = create_access_token(user.id, user.name)
        return TokenResponse(access_token=token, user_id=user.id, name=user.name)

    except Exception as exc:
        logger.exception("SSO error: %s", exc)
        raise HTTPException(status_code=500, detail=f"SSO error: {exc}")


@app.post("/auth/register", response_model=TokenResponse, status_code=201, tags=["Auth"])
async def register(payload: UserRegister, db: AsyncSession = Depends(get_db)):
    existing = await db.execute(select(User).where(User.email == payload.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered.")

    user = User(
        name=payload.name,
        email=payload.email,
        hashed_password=hash_password(payload.password),
        preferred_language=payload.preferred_language,
    )
    db.add(user)

    # Initialise streak row for new user
    streak = Streak(user_id=user.id)
    db.add(streak)

    await db.commit()
    await db.refresh(user)

    token = create_access_token(user.id, user.name)
    return TokenResponse(access_token=token, user_id=user.id, name=user.name)


@app.post("/auth/login", response_model=TokenResponse, tags=["Auth"])
async def login(payload: UserLogin, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == payload.email))
    user = result.scalar_one_or_none()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    token = create_access_token(user.id, user.name)
    return TokenResponse(access_token=token, user_id=user.id, name=user.name)


@app.get("/auth/me", response_model=UserOut, tags=["Auth"])
async def me(current_user: User = Depends(get_current_user)):
    return current_user


@app.post("/auth/push-token", status_code=204, tags=["Auth"])
async def update_push_token(
    payload: PushTokenUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Register or update the FCM push token for this device."""
    current_user.push_token = payload.token
    # Also cache in Redis for quick lookup by notification workers
    try:
        from redis_client import push_token_key
        r = await get_redis()
        await r.set(push_token_key(current_user.id), payload.token)
    except Exception:
        pass
    await db.commit()


# ─────────────────────────────────────────────────────────────────────────────
#  Chat – REST endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    payload: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Send a message to the JioJoin AI agent and receive a full response."""
    session_id = payload.session_id or str(uuid.uuid4())

    # Load history — fall back to empty list if conversation table is missing
    try:
        history = await conv_memory.get_history(db, current_user.id, session_id)
    except Exception as exc:
        logger.warning("Could not load history (table missing?): %s", exc)
        history = []

    try:
        reply, tools_used, detected_lang = await agent.run(
            user_message=payload.message,
            history=history,
            db=db,
            user_id=current_user.id,
        )
    except Exception as exc:
        logger.exception("Agent error for user %s: %s", current_user.id, exc)
        # Include real error in detail so we can diagnose from the browser
        raise HTTPException(status_code=500, detail=f"Agent error: {type(exc).__name__}: {exc}")

    # Persist history — best-effort, never block the response
    try:
        await conv_memory.add_message(db, current_user.id, session_id, "user", payload.message)
        await conv_memory.add_message(db, current_user.id, session_id, "assistant", reply)
    except Exception as exc:
        logger.warning("Could not save history (table missing?): %s", exc)

    return ChatResponse(
        reply=reply,
        session_id=session_id,
        tools_used=tools_used,
        detected_language=detected_lang,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Chat – WebSocket endpoint (streaming)
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket, token: str = Query(...)):
    """
    WebSocket chat endpoint for the mobile WebView.

    Protocol:
      Client → Server: {"message": "...", "session_id": "..."}   (session_id optional)
      Server → Client: {"type": "reply", "content": "...", "session_id": "...",
                        "tools_used": [...], "detected_language": "en"}
      Server → Client: {"type": "error", "detail": "..."}
    """
    # Authenticate via JWT in query param (WebSocket can't send Bearer headers)
    payload = decode_token(token)
    if not payload:
        await websocket.close(code=4001, reason="Unauthorized")
        return

    user_id: str = payload.get("sub", "")
    if not user_id:
        await websocket.close(code=4001, reason="Unauthorized")
        return

    await websocket.accept()
    logger.info("WebSocket connected for user %s", user_id)

    async for db in get_db():
        # Verify user still exists and is active
        result = await db.execute(
            select(User).where(User.id == user_id, User.is_active == True)  # noqa: E712
        )
        user = result.scalar_one_or_none()
        if not user:
            await websocket.send_json({"type": "error", "detail": "User not found."})
            await websocket.close(code=4003, reason="Forbidden")
            return

        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    data = json.loads(raw)
                    message = data.get("message", "").strip()
                    session_id = data.get("session_id") or str(uuid.uuid4())
                except (json.JSONDecodeError, AttributeError):
                    await websocket.send_json({"type": "error", "detail": "Invalid JSON."})
                    continue

                if not message:
                    continue

                history = await conv_memory.get_history(db, user_id, session_id)

                try:
                    reply, tools_used, detected_lang = await agent.run(
                        user_message=message,
                        history=history,
                        db=db,
                        user_id=user_id,
                    )
                except Exception as exc:
                    logger.exception("Agent WS error for user %s: %s", user_id, exc)
                    await websocket.send_json({
                        "type": "error",
                        "detail": "The agent encountered an error. Please try again.",
                    })
                    continue

                await conv_memory.add_message(db, user_id, session_id, "user", message)
                await conv_memory.add_message(db, user_id, session_id, "assistant", reply)

                await websocket.send_json({
                    "type": "reply",
                    "content": reply,
                    "session_id": session_id,
                    "tools_used": tools_used,
                    "detected_language": detected_lang,
                })

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected for user %s", user_id)
        except Exception as exc:
            logger.exception("WebSocket error for user %s: %s", user_id, exc)

        break  # exit the async for loop


# ─────────────────────────────────────────────────────────────────────────────
#  To-Do REST endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/todos", response_model=List[TodoOut], tags=["To-Do"])
async def get_todos(
    status: Optional[TodoStatus] = Query(None),
    priority: Optional[TodoPriority] = Query(None),
    limit: int = Query(20, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await tool_list_todos(
        db, current_user.id,
        status=status.value if status else None,
        priority=priority.value if priority else None,
        limit=limit,
    )
    return result["todos"]


@app.post("/todos", response_model=TodoOut, status_code=201, tags=["To-Do"])
async def create_todo(
    payload: TodoCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await tool_add_todo(
        db, current_user.id,
        title=payload.title,
        description=payload.description,
        priority=payload.priority.value,
        due_date=payload.due_date.isoformat() if payload.due_date else None,
        tags=payload.tags,
    )
    return result["todo"]


@app.put("/todos/{todo_id}", response_model=TodoOut, tags=["To-Do"])
async def update_todo_endpoint(
    todo_id: str,
    payload: TodoUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await tool_update_todo(
        db, current_user.id, todo_id,
        title=payload.title,
        description=payload.description,
        status=payload.status.value if payload.status else None,
        priority=payload.priority.value if payload.priority else None,
        due_date=payload.due_date.isoformat() if payload.due_date else None,
        tags=payload.tags,
    )
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error", "Not found."))
    return result["todo"]


@app.delete("/todos/{todo_id}", status_code=204, tags=["To-Do"])
async def delete_todo_endpoint(
    todo_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await tool_delete_todo(db, current_user.id, todo_id)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error", "Not found."))


# ─────────────────────────────────────────────────────────────────────────────
#  Reminders REST endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/reminders", response_model=List[ReminderOut], tags=["Reminders"])
async def get_reminders(
    status: Optional[ReminderStatus] = Query(ReminderStatus.ACTIVE),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await tool_list_reminders(db, current_user.id, status=status.value if status else "active")
    return result["reminders"]


@app.post("/reminders", response_model=ReminderOut, status_code=201, tags=["Reminders"])
async def create_reminder(
    payload: ReminderCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await tool_set_reminder(
        db, current_user.id,
        title=payload.title,
        remind_at=payload.remind_at.isoformat(),
    )
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Could not create reminder."))
    return result["reminder"]


@app.delete("/reminders/{reminder_id}", status_code=204, tags=["Reminders"])
async def cancel_reminder_endpoint(
    reminder_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await tool_cancel_reminder(db, current_user.id, reminder_id)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error", "Not found."))


# ─────────────────────────────────────────────────────────────────────────────
#  Interests REST endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/interests", response_model=List[InterestOut], tags=["Interests"])
async def get_interests(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await tool_get_interests(db, current_user.id)
    return result["interests"]


@app.put("/interests", response_model=List[InterestOut], tags=["Interests"])
async def update_interests(
    payload: InterestUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    await tool_update_interests(db, current_user.id, payload.topics)
    result = await tool_get_interests(db, current_user.id)
    return result["interests"]


# ─────────────────────────────────────────────────────────────────────────────
#  What's New
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/whats-new", tags=["Discovery"])
async def whats_new(
    category: str = Query("india"),
    current_user: User = Depends(get_current_user),
):
    """Return real-time news headlines via NewsAPI, grouped by category."""
    from tools.news_tools import fetch_news
    return await fetch_news(category=category, page_size=8)


# ─────────────────────────────────────────────────────────────────────────────
#  Coins
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/coins/balance", response_model=CoinBalanceOut, tags=["Coins"])
async def get_coin_balance(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return current coin balance (earned - spent) and total ever earned."""
    result = await db.execute(
        select(func.sum(CoinLedger.delta)).where(CoinLedger.user_id == current_user.id)
    )
    balance = result.scalar_one() or 0

    result_earned = await db.execute(
        select(func.sum(CoinLedger.delta)).where(
            CoinLedger.user_id == current_user.id,
            CoinLedger.delta > 0,
        )
    )
    total_earned = result_earned.scalar_one() or 0

    return CoinBalanceOut(
        user_id=current_user.id,
        balance=int(balance),
        total_earned=int(total_earned),
    )


@app.get("/coins/history", response_model=List[CoinLedgerOut], tags=["Coins"])
async def get_coin_history(
    limit: int = Query(20, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(CoinLedger)
        .where(CoinLedger.user_id == current_user.id)
        .order_by(CoinLedger.created_at.desc())
        .limit(limit)
    )
    return result.scalars().all()


# ─────────────────────────────────────────────────────────────────────────────
#  Streak
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/streak", response_model=StreakOut, tags=["Engage"])
async def get_streak(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Streak).where(Streak.user_id == current_user.id)
    )
    streak = result.scalar_one_or_none()
    if not streak:
        streak = Streak(user_id=current_user.id)
        db.add(streak)
        await db.commit()
        await db.refresh(streak)
    return streak


# ─────────────────────────────────────────────────────────────────────────────
#  Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    # Railway injects PORT env var — use it, fall back to app_port for local dev
    port = settings.port or settings.app_port
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=port,
        reload=not settings.is_production,
        log_level="debug" if not settings.is_production else "info",
    )
