"""
memory/conversation.py – Conversation history management.

Architecture:
  • Redis (primary): session history stored as a JSON list with TTL.
    Fast reads, survives within the TTL window, handles multi-instance deployments.
  • PostgreSQL (secondary): permanent write-through so history survives beyond TTL
    and is available for audit / analytics.
  • Fallback: if Redis is unavailable, falls back to a process-local in-memory dict
    (single-instance only, lost on restart — fine for local dev without Redis).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from typing import Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import get_settings
from models import ConversationMessage

logger = logging.getLogger(__name__)
settings = get_settings()

# Process-local fallback (used only when Redis is down)
_fallback_memory: Dict[str, List[dict]] = defaultdict(list)
_fallback_loaded: set[str] = set()
_lock = asyncio.Lock()


# ─────────────────────────────────────────────────────────────────────────────
#  Redis helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _redis_get_history(session_key: str) -> List[dict] | None:
    try:
        from redis_client import get_redis
        r = await get_redis()
        raw = await r.get(session_key)
        return json.loads(raw) if raw else None
    except Exception:
        return None


async def _redis_set_history(session_key: str, history: List[dict]) -> None:
    try:
        from redis_client import get_redis
        r = await get_redis()
        await r.set(session_key, json.dumps(history), ex=settings.session_ttl_seconds)
    except Exception:
        pass  # Redis write failure is non-fatal; DB is the source of truth


def _make_key(user_id: str, session_id: str) -> str:
    return f"session:{user_id}:{session_id}"


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

async def get_history(
    db: AsyncSession,
    user_id: str,
    session_id: str,
) -> List[dict]:
    """Return conversation history as a list of LLM-format message dicts."""
    key = _make_key(user_id, session_id)

    # 1. Try Redis
    cached = await _redis_get_history(key)
    if cached is not None:
        return cached

    # 2. Try in-memory fallback
    async with _lock:
        if session_id in _fallback_loaded:
            return list(_fallback_memory[session_id])

    # 3. Load from DB (first access or after Redis eviction)
    history = await _load_from_db(db, user_id, session_id)
    await _redis_set_history(key, history)

    async with _lock:
        _fallback_memory[session_id] = list(history)
        _fallback_loaded.add(session_id)

    return history


async def add_message(
    db: AsyncSession,
    user_id: str,
    session_id: str,
    role: str,
    content: str,
    tool_name: Optional[str] = None,
) -> None:
    """Append a message to Redis + in-memory fallback + DB."""
    msg: dict = {"role": role, "content": content}
    key = _make_key(user_id, session_id)
    max_msgs = settings.max_conversation_history

    # Update Redis
    cached = await _redis_get_history(key) or []
    cached.append(msg)
    if len(cached) > max_msgs:
        cached = cached[-max_msgs:]
    await _redis_set_history(key, cached)

    # Update in-memory fallback
    async with _lock:
        _fallback_memory[session_id].append(msg)
        if len(_fallback_memory[session_id]) > max_msgs:
            _fallback_memory[session_id] = _fallback_memory[session_id][-max_msgs:]

    # Persist to DB
    db_msg = ConversationMessage(
        user_id=user_id,
        session_id=session_id,
        role=role,
        content=content,
        tool_name=tool_name,
    )
    db.add(db_msg)
    await db.commit()


async def clear_session(user_id: str, session_id: str) -> None:
    """Evict session from Redis and in-memory cache (does not delete DB records)."""
    key = _make_key(user_id, session_id)
    try:
        from redis_client import get_redis
        r = await get_redis()
        await r.delete(key)
    except Exception:
        pass

    async with _lock:
        _fallback_memory.pop(session_id, None)
        _fallback_loaded.discard(session_id)


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _load_from_db(
    db: AsyncSession,
    user_id: str,
    session_id: str,
) -> List[dict]:
    max_msgs = settings.max_conversation_history
    stmt = (
        select(ConversationMessage)
        .where(
            ConversationMessage.user_id == user_id,
            ConversationMessage.session_id == session_id,
        )
        .order_by(ConversationMessage.created_at.asc())
        .limit(max_msgs)
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()
    return [{"role": row.role, "content": row.content} for row in rows]
