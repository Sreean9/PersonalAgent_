"""
models.py – SQLAlchemy ORM models + Pydantic request/response schemas.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from enum import Enum
from typing import Optional, List

from sqlalchemy import (
    String, Text, Boolean, DateTime, Date, ForeignKey,
    Enum as SAEnum, Integer, Float,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
#  SQLAlchemy Base & helpers
# ─────────────────────────────────────────────────────────────────────────────

def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def today() -> date:
    return datetime.now(timezone.utc).date()


def new_uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────────────────────────────────────
#  Enums
# ─────────────────────────────────────────────────────────────────────────────

class TodoStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TodoPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ReminderStatus(str, Enum):
    ACTIVE = "active"
    TRIGGERED = "triggered"
    CANCELLED = "cancelled"


class PlanStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class AlertStatus(str, Enum):
    ACTIVE = "active"
    TRIGGERED = "triggered"
    CANCELLED = "cancelled"


class PuzzleType(str, Enum):
    WORD_SCRAMBLE = "word_scramble"
    TRIVIA = "trivia"
    FILL_BLANK = "fill_blank"
    EMOJI_RIDDLE = "emoji_riddle"
    MATH = "math"


class PuzzleDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ─────────────────────────────────────────────────────────────────────────────
#  ORM Models — existing
# ─────────────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    preferred_language: Mapped[str] = mapped_column(String(10), default="en")
    push_token: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    todos: Mapped[List["Todo"]] = relationship("Todo", back_populates="user", cascade="all, delete-orphan")
    reminders: Mapped[List["Reminder"]] = relationship("Reminder", back_populates="user", cascade="all, delete-orphan")
    interests: Mapped[List["UserInterest"]] = relationship("UserInterest", back_populates="user", cascade="all, delete-orphan")
    conversations: Mapped[List["ConversationMessage"]] = relationship("ConversationMessage", back_populates="user", cascade="all, delete-orphan")
    coin_ledger: Mapped[List["CoinLedger"]] = relationship("CoinLedger", back_populates="user", cascade="all, delete-orphan")
    streak: Mapped[Optional["Streak"]] = relationship("Streak", back_populates="user", uselist=False, cascade="all, delete-orphan")
    plans: Mapped[List["Plan"]] = relationship("Plan", back_populates="user", cascade="all, delete-orphan")
    alerts: Mapped[List["Alert"]] = relationship("Alert", back_populates="user", cascade="all, delete-orphan")
    puzzle_attempts: Mapped[List["UserPuzzleAttempt"]] = relationship("UserPuzzleAttempt", back_populates="user", cascade="all, delete-orphan")


class Todo(Base):
    __tablename__ = "todos"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[TodoStatus] = mapped_column(SAEnum(TodoStatus, native_enum=False), default=TodoStatus.PENDING)
    priority: Mapped[TodoPriority] = mapped_column(SAEnum(TodoPriority, native_enum=False), default=TodoPriority.MEDIUM)
    due_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    tags: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    user: Mapped["User"] = relationship("User", back_populates="todos")


class Reminder(Base):
    __tablename__ = "reminders"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    remind_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    is_recurring: Mapped[bool] = mapped_column(Boolean, default=False)
    recurrence: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # daily/weekly/monthly
    status: Mapped[ReminderStatus] = mapped_column(SAEnum(ReminderStatus, native_enum=False), default=ReminderStatus.ACTIVE)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    user: Mapped["User"] = relationship("User", back_populates="reminders")


class UserInterest(Base):
    __tablename__ = "user_interests"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    topic: Mapped[str] = mapped_column(String(255), nullable=False)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    user: Mapped["User"] = relationship("User", back_populates="interests")


class Announcement(Base):
    __tablename__ = "announcements"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(100), default="general")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tool_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    user: Mapped["User"] = relationship("User", back_populates="conversations")


# ─────────────────────────────────────────────────────────────────────────────
#  ORM Models — new (Phase 0)
# ─────────────────────────────────────────────────────────────────────────────

class CoinLedger(Base):
    """Append-only ledger — never UPDATE rows, only INSERT. Balance = SUM(delta)."""
    __tablename__ = "coin_ledger"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    delta: Mapped[int] = mapped_column(Integer, nullable=False)          # +earn / -spend
    reason: Mapped[str] = mapped_column(String(100), nullable=False)     # "daily_login", "puzzle_win", etc.
    ref_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)  # e.g. puzzle_id, todo_id
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)

    user: Mapped["User"] = relationship("User", back_populates="coin_ledger")


class Streak(Base):
    """One row per user — updated on each daily login."""
    __tablename__ = "streaks"

    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), primary_key=True)
    current_streak: Mapped[int] = mapped_column(Integer, default=0)
    longest_streak: Mapped[int] = mapped_column(Integer, default=0)
    last_active_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    user: Mapped["User"] = relationship("User", back_populates="streak")


class Plan(Base):
    """Multi-step plans: travel, meal, study, routine, event, bill tracker."""
    __tablename__ = "plans"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    plan_type: Mapped[str] = mapped_column(String(50), nullable=False)  # travel/meal/study/routine/event/bill
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    steps: Mapped[Optional[str]] = mapped_column(Text, nullable=True)   # JSON array of step dicts
    start_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    end_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    status: Mapped[PlanStatus] = mapped_column(SAEnum(PlanStatus, native_enum=False), default=PlanStatus.ACTIVE)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    user: Mapped["User"] = relationship("User", back_populates="plans")


class Alert(Base):
    """User-defined custom alerts (time-based in Phase 1, event-based in Phase 2)."""
    __tablename__ = "alerts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    alert_type: Mapped[str] = mapped_column(String(30), default="time_based")  # time_based | event_based
    trigger_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    is_recurring: Mapped[bool] = mapped_column(Boolean, default=False)
    recurrence: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # daily/weekly/monthly
    status: Mapped[AlertStatus] = mapped_column(SAEnum(AlertStatus, native_enum=False), default=AlertStatus.ACTIVE)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    user: Mapped["User"] = relationship("User", back_populates="alerts")


class Puzzle(Base):
    """Pre-generated puzzle bank served daily to users."""
    __tablename__ = "puzzles"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    puzzle_type: Mapped[PuzzleType] = mapped_column(SAEnum(PuzzleType, native_enum=False), nullable=False)
    topic: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)  # linked to interest topics
    question: Mapped[str] = mapped_column(Text, nullable=False)
    options: Mapped[Optional[str]] = mapped_column(Text, nullable=True)   # JSON array (for MCQ)
    answer: Mapped[str] = mapped_column(String(500), nullable=False)
    hint: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    difficulty: Mapped[PuzzleDifficulty] = mapped_column(SAEnum(PuzzleDifficulty, native_enum=False), default=PuzzleDifficulty.MEDIUM)
    language: Mapped[str] = mapped_column(String(10), default="en")
    coins_reward: Mapped[int] = mapped_column(Integer, default=10)
    serve_count: Mapped[int] = mapped_column(Integer, default=0)  # how many times served
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    attempts: Mapped[List["UserPuzzleAttempt"]] = relationship("UserPuzzleAttempt", back_populates="puzzle")


class UserPuzzleAttempt(Base):
    """One row per user per puzzle day — tracks attempt and coin award."""
    __tablename__ = "user_puzzle_attempts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    puzzle_id: Mapped[str] = mapped_column(String(36), ForeignKey("puzzles.id"), nullable=False)
    puzzle_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)  # YYYY-MM-DD served
    solved: Mapped[bool] = mapped_column(Boolean, default=False)
    attempts_count: Mapped[int] = mapped_column(Integer, default=0)
    coins_earned: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    user: Mapped["User"] = relationship("User", back_populates="puzzle_attempts")
    puzzle: Mapped["Puzzle"] = relationship("Puzzle", back_populates="attempts")


# ─────────────────────────────────────────────────────────────────────────────
#  Pydantic Schemas
# ─────────────────────────────────────────────────────────────────────────────

# ── Auth ─────────────────────────────────────────────────────────────────────

class UserRegister(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    email: str = Field(..., pattern=r"^[\w\.-]+@[\w\.-]+\.\w{2,}$")
    password: str = Field(..., min_length=6)
    preferred_language: str = Field(default="en")


class UserLogin(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    name: str


class UserOut(BaseModel):
    id: str
    name: str
    email: str
    preferred_language: str
    created_at: datetime

    class Config:
        from_attributes = True


# ── Todo ─────────────────────────────────────────────────────────────────────

class TodoCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    description: Optional[str] = None
    priority: TodoPriority = TodoPriority.MEDIUM
    due_date: Optional[datetime] = None
    tags: Optional[str] = None


class TodoUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=500)
    description: Optional[str] = None
    status: Optional[TodoStatus] = None
    priority: Optional[TodoPriority] = None
    due_date: Optional[datetime] = None
    tags: Optional[str] = None


class TodoOut(BaseModel):
    id: str
    title: str
    description: Optional[str]
    status: TodoStatus
    priority: TodoPriority
    due_date: Optional[datetime]
    tags: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ── Reminder ─────────────────────────────────────────────────────────────────

class ReminderCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    remind_at: datetime
    is_recurring: bool = False
    recurrence: Optional[str] = Field(None, pattern=r"^(daily|weekly|monthly)$")


class ReminderOut(BaseModel):
    id: str
    title: str
    remind_at: datetime
    is_recurring: bool
    recurrence: Optional[str]
    status: ReminderStatus
    created_at: datetime

    class Config:
        from_attributes = True


# ── Interests ────────────────────────────────────────────────────────────────

class InterestUpdate(BaseModel):
    topics: List[str] = Field(..., min_length=1)


class InterestOut(BaseModel):
    id: str
    topic: str
    weight: float

    class Config:
        from_attributes = True


# ── Chat ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = None

    @field_validator("message")
    @classmethod
    def strip_message(cls, v: str) -> str:
        return v.strip()


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    tools_used: List[str] = Field(default_factory=list)
    detected_language: str = "en"


# ── Announcement ─────────────────────────────────────────────────────────────

class AnnouncementOut(BaseModel):
    id: str
    title: str
    body: str
    category: str
    created_at: datetime

    class Config:
        from_attributes = True


# ── Coins ─────────────────────────────────────────────────────────────────────

class CoinBalanceOut(BaseModel):
    user_id: str
    balance: int
    total_earned: int


class CoinLedgerOut(BaseModel):
    id: int
    delta: int
    reason: str
    created_at: datetime

    class Config:
        from_attributes = True


# ── Streak ───────────────────────────────────────────────────────────────────

class StreakOut(BaseModel):
    current_streak: int
    longest_streak: int
    last_active_date: Optional[date]

    class Config:
        from_attributes = True


# ── Plan ─────────────────────────────────────────────────────────────────────

class PlanCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    plan_type: str = Field(..., pattern=r"^(travel|meal|study|routine|event|bill)$")
    description: Optional[str] = None
    steps: Optional[str] = None   # JSON string
    start_date: Optional[date] = None
    end_date: Optional[date] = None


class PlanOut(BaseModel):
    id: str
    title: str
    plan_type: str
    description: Optional[str]
    steps: Optional[str]
    start_date: Optional[date]
    end_date: Optional[date]
    status: PlanStatus
    created_at: datetime

    class Config:
        from_attributes = True


# ── Alert ─────────────────────────────────────────────────────────────────────

class AlertCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    description: Optional[str] = None
    trigger_at: Optional[datetime] = None
    is_recurring: bool = False
    recurrence: Optional[str] = Field(None, pattern=r"^(daily|weekly|monthly)$")


class AlertOut(BaseModel):
    id: str
    title: str
    description: Optional[str]
    alert_type: str
    trigger_at: Optional[datetime]
    is_recurring: bool
    recurrence: Optional[str]
    status: AlertStatus
    created_at: datetime

    class Config:
        from_attributes = True


# ── Push token ────────────────────────────────────────────────────────────────

class PushTokenUpdate(BaseModel):
    token: str = Field(..., min_length=10)
