"""
database.py – Async SQLAlchemy engine, session factory, and DB initialisation.
"""

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import text

from config import get_settings
from models import Base, Announcement

settings = get_settings()

engine = create_async_engine(
    settings.database_url,
    echo=not settings.is_production,
    future=True,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    async with engine.begin() as conn:
        await _run_migrations(conn)
        await conn.run_sync(Base.metadata.create_all)
    await _seed_announcements()


async def _run_migrations(conn) -> None:
    if "postgresql" not in settings.database_url:
        return
    migrations = [
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS preferred_language VARCHAR(10) DEFAULT 'en'",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS push_token VARCHAR(500)",
    ]
    for sql in migrations:
        try:
            await conn.execute(text(sql))
        except Exception:
            pass


async def _seed_announcements() -> None:
    async with AsyncSessionLocal() as session:
        result = await session.execute(text("SELECT COUNT(*) FROM announcements"))
        count = result.scalar_one()
        if count > 0:
            return
        sample = [
            Announcement(title="Welcome to JioJoin AI Agent!", body="Meet your new personal assistant inside JioJoin. Ask me to manage your tasks, set reminders, do quick calculations, or discover something new today!", category="feature"),
            Announcement(title="Hindi Support is Here 🎉", body="आप अब हिंदी में बात कर सकते हैं! Just switch to Hindi and your AI agent will respond accordingly.", category="feature"),
            Announcement(title="To-Do Lists – Now Smarter", body="Set priorities, due dates, and tags on your tasks. Just say 'Add a high-priority task: Submit report by Friday' and we'll handle the rest.", category="feature"),
            Announcement(title="Tip: Explore New Interests", body="Tell your agent what you're curious about — cooking, fitness, finance, cricket — and get personalised content recommendations every day.", category="general"),
        ]
        session.add_all(sample)
        await session.commit()
