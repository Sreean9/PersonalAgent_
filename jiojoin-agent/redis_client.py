"""
redis_client.py – Async Redis connection management.

Usage:
    from redis_client import get_redis, close_redis

    redis = await get_redis()
    await redis.set("key", "value", ex=3600)
    val = await redis.get("key")
"""

from __future__ import annotations

import logging

import redis.asyncio as aioredis

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_client: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    """Return the shared async Redis client, creating it on first call."""
    global _client
    if _client is None:
        _client = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
        )
        # Verify connection
        try:
            await _client.ping()
            logger.info("Redis connected: %s", settings.redis_url)
        except Exception as exc:
            logger.warning("Redis unavailable (%s) — falling back to in-memory store.", exc)
            _client = None
            raise
    return _client


async def close_redis() -> None:
    global _client
    if _client:
        await _client.aclose()
        _client = None
        logger.info("Redis connection closed.")


# ── Key helpers ───────────────────────────────────────────────────────────────

def session_key(user_id: str, session_id: str) -> str:
    return f"session:{user_id}:{session_id}"


def coins_key(user_id: str) -> str:
    return f"coins:{user_id}"


def streak_key(user_id: str) -> str:
    return f"streak:{user_id}"


def puzzle_key(user_id: str, date_str: str) -> str:
    """date_str format: YYYY-MM-DD"""
    return f"puzzle:{user_id}:{date_str}"


def rate_key(user_id: str) -> str:
    return f"rate:{user_id}"


def push_token_key(user_id: str) -> str:
    return f"push_token:{user_id}"
