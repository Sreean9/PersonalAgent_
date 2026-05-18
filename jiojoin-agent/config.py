"""
config.py – Centralised settings for JioJoin Agent.
All values are read from environment variables (or a .env file).
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Groq / Primary LLM ──────────────────────────────────────────────────
    groq_api_key: str = "your_groq_api_key_here"
    groq_model: str = "llama-3.3-70b-versatile"
    agent_temperature: float = 0.6

    # ── Sarvam AI (Indian regional language translation) ─────────────────────
    sarvam_api_key: str = ""
    sarvam_base_url: str = "https://api.sarvam.ai"
    # "auto" = detect language and route; "groq" = always use Groq
    llm_provider: str = "auto"

    # ── Database ─────────────────────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///./jiojoin.db"

    # ── Redis ────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    # TTL for conversation sessions in Redis (seconds) — 2 hours
    session_ttl_seconds: int = 7200

    # ── JWT Auth ─────────────────────────────────────────────────────────────
    jwt_secret_key: str = "change_me_in_production"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 10080  # 7 days

    # ── Firebase (push notifications) ────────────────────────────────────────
    firebase_credentials_json: str = ""   # full service-account JSON as string (Railway)
    firebase_credentials_path: str = ""   # path to JSON file (local dev fallback)
    firebase_project_id: str = ""

    # ── NewsAPI (trending topics / alerts) ───────────────────────────────────
    news_api_key: str = ""
    news_api_base_url: str = "https://newsapi.org/v2"

    # ── OpenWeatherMap ────────────────────────────────────────────────────────
    openweather_api_key: str = ""

    # ── Server ───────────────────────────────────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 8000          # Railway overrides this via PORT env var
    port: int = 8000              # Railway injects PORT automatically
    app_env: str = "development"

    # ── Agent behaviour ──────────────────────────────────────────────────────
    max_conversation_history: int = 20
    max_tool_rounds: int = 5

    # ── Engagement ───────────────────────────────────────────────────────────
    # Coins awarded for each action
    coins_daily_login: int = 5
    coins_todo_complete: int = 2
    coins_puzzle_win_easy: int = 10
    coins_puzzle_win_medium: int = 15
    coins_puzzle_win_hard: int = 20
    coins_plan_created: int = 3
    coins_streak_7day_bonus: int = 50

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"

    @property
    def sarvam_enabled(self) -> bool:
        return bool(self.sarvam_api_key) and self.llm_provider == "auto"


@lru_cache
def get_settings() -> Settings:
    return Settings()
