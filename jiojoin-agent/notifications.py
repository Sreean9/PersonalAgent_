"""
notifications.py – Firebase Cloud Messaging push notifications.

Initialise Firebase once at startup, then call send_push_notification()
wherever a notification needs to be dispatched.

The background loop check_and_send_due_reminders() is called every 60 s
from the FastAPI lifespan task to fire reminder alerts.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_firebase_ready = False


# ─────────────────────────────────────────────────────────────────────────────
#  Initialisation
# ─────────────────────────────────────────────────────────────────────────────

def init_firebase() -> bool:
    """
    Initialise Firebase Admin SDK.
    Safe to call multiple times — idempotent.
    Returns True if Firebase is ready, False if credentials are missing.
    """
    global _firebase_ready
    if _firebase_ready:
        return True
    if not settings.firebase_credentials_path or not settings.firebase_project_id:
        logger.info("Firebase not configured — push notifications disabled. "
                    "Set FIREBASE_CREDENTIALS_PATH and FIREBASE_PROJECT_ID in Railway.")
        return False
    try:
        import firebase_admin
        from firebase_admin import credentials
        if not firebase_admin._apps:
            cred = credentials.Certificate(settings.firebase_credentials_path)
            firebase_admin.initialize_app(cred)
        _firebase_ready = True
        logger.info("Firebase ready — project: %s", settings.firebase_project_id)
        return True
    except Exception as exc:
        logger.warning("Firebase init failed: %s", exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  Send
# ─────────────────────────────────────────────────────────────────────────────

async def send_push_notification(token: str, title: str, body: str) -> bool:
    """Send an FCM push notification to a single device token."""
    if not _firebase_ready or not token:
        return False
    try:
        from firebase_admin import messaging
        message = messaging.Message(
            notification=messaging.Notification(title=title, body=body),
            android=messaging.AndroidConfig(priority="high"),
            apns=messaging.APNSConfig(
                payload=messaging.APNSPayload(
                    aps=messaging.Aps(sound="default", badge=1)
                )
            ),
            token=token,
        )
        messaging.send(message)
        logger.info("Push sent to token ...%s: %s", token[-6:], title)
        return True
    except Exception as exc:
        logger.error("FCM send failed: %s", exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  Reminder checker
# ─────────────────────────────────────────────────────────────────────────────

async def check_and_send_due_reminders(db: AsyncSession) -> None:
    """
    Find active reminders that became due in the last 2 minutes,
    send push notifications, and mark them as triggered.
    Called every 60 s from the background loop in main.py.
    """
    from models import Reminder, User

    now = datetime.now(timezone.utc)
    window_start = now - timedelta(minutes=2)

    result = await db.execute(
        select(Reminder).where(
            and_(
                Reminder.status == "active",
                Reminder.remind_at >= window_start,
                Reminder.remind_at <= now,
            )
        )
    )
    due = result.scalars().all()
    if not due:
        return

    for reminder in due:
        user_result = await db.execute(
            select(User).where(User.id == reminder.user_id)
        )
        user = user_result.scalar_one_or_none()
        if user and user.push_token:
            await send_push_notification(
                token=user.push_token,
                title="⏰ Reminder",
                body=reminder.title,
            )
        reminder.status = "triggered"

    await db.commit()
    logger.info("Processed %d due reminder(s)", len(due))
