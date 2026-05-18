"""
agent.py – JioJoin Personal AI Agent

Core loop:
  1. Receives user message + conversation history.
  2. Routes through LLMRouter (language detection → Groq directly, or Sarvam translate → Groq).
  3. If the model requests tool calls, executes them and feeds results back.
  4. Loops until a final text reply is produced (max MAX_TOOL_ROUNDS).
  5. Returns (reply, tools_used, detected_language).
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from config import get_settings
from llm.router import LLMRouter
from llm.sarvam_provider import detect_language
from tools.tool_registry import TOOLS
from tools.todo_tools import (
    add_todo, list_todos, update_todo, delete_todo, search_todos,
)
from tools.utility_tools import (
    calculate, convert_units,
    set_reminder, list_reminders, cancel_reminder,
    get_weather,
)
from tools.news_tools import fetch_news

logger = logging.getLogger(__name__)
settings = get_settings()

SYSTEM_PROMPT = """You are Jio, a friendly and intelligent personal AI assistant built into the JioJoin app.

You help users with these core areas:
1. **To-Do & Tasks** – add, view, update, search, and delete tasks
2. **Plans** – help create and track travel, meal, study, routine, event, and bill-payment plans
3. **Reminders & Alerts** – set, list, and manage reminders and custom alerts
4. **Utility** – perform calculations, convert units, check live weather
5. **Latest News** – fetch real-time news across India, sports, world, business, tech, and entertainment
6. **General Knowledge** – answer any question the user has from your own knowledge
7. **Daily Puzzles & Coins** – guide users to play today's puzzle and earn coins

Guidelines:
- Always use the available tools to act; never make up data.
- For ANY news-related query (India news, sports, cricket, world, business, tech, entertainment, health, science), ALWAYS call the fetch_news tool immediately with the appropriate category. Do not explain or apologize first — just call the tool.
- For ANY weather query ("what is the weather in X", "temperature in Y", "climate in Z", "is it raining in X"), ALWAYS call the get_weather tool immediately with the city name. Never say you cannot help with weather.
- Only call tools that are explicitly listed in your tools schema. Never invent or guess tool names.
- Do not repeat an action already confirmed earlier in the same conversation.
- Be concise, warm, and helpful. Avoid long monologues unless the user asks for detail.
- When listing tasks or reminders, present them in a clean, easy-to-read format.
- For reminders and alerts, confirm the time back to the user in a human-readable format.
- When a tool returns an error, explain it simply and suggest what the user can do.
- Never expose internal IDs unless the user specifically asks for them.
- For plans, ask clarifying questions to build a complete, structured plan.
- Occasionally remind users about their daily puzzle if they haven't played today (don't be pushy).

**LANGUAGE RULE — follow this exactly, no exceptions:**
Every user message starts with a language tag. Obey the tag:
- [EN] at the start → your ENTIRE reply must be in English only. Indian place names do NOT change this.
- [HI] at the start → your ENTIRE reply must be in Devanagari Hindi only (never Roman/transliteration).
- No tag → match the script the user wrote in; default to English if unclear.
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Tool dispatcher
# ─────────────────────────────────────────────────────────────────────────────

async def _dispatch_tool(
    name: str,
    args: dict,
    db: AsyncSession,
    user_id: str,
) -> str:
    try:
        result: dict

        if name == "add_todo":
            result = await add_todo(db, user_id, **args)
        elif name == "list_todos":
            result = await list_todos(db, user_id, **args)
        elif name == "update_todo":
            result = await update_todo(db, user_id, **args)
        elif name == "delete_todo":
            result = await delete_todo(db, user_id, **args)
        elif name == "search_todos":
            result = await search_todos(db, user_id, **args)
        elif name == "calculate":
            result = calculate(**args)
        elif name == "convert_units":
            result = convert_units(**args)
        elif name == "set_reminder":
            result = await set_reminder(db, user_id, **args)
        elif name == "list_reminders":
            result = await list_reminders(db, user_id, **args)
        elif name == "cancel_reminder":
            result = await cancel_reminder(db, user_id, **args)
        elif name == "get_weather":
            result = await get_weather(city=args.get("city", ""))
        elif name == "fetch_news":
            result = await fetch_news(category=args.get("category", "india"))
        else:
            result = {"error": f"Unknown tool: {name}"}

    except Exception as exc:
        logger.exception("Tool '%s' raised: %s", name, exc)
        result = {"error": str(exc)}

    return json.dumps(result, default=str)


# ─────────────────────────────────────────────────────────────────────────────
#  Agent
# ─────────────────────────────────────────────────────────────────────────────

class JioJoinAgent:
    """
    Stateless agent wrapping LLMRouter with a tool-calling loop.

    Usage:
        reply, tools_used, lang = await agent.run(
            user_message="Add task: Submit Q2 report by Friday",
            history=[...],
            db=db_session,
            user_id="abc-123",
        )
    """

    def __init__(self) -> None:
        self._router = LLMRouter()

    async def run(
        self,
        user_message: str,
        history: list[dict],
        db: AsyncSession,
        user_id: str,
    ) -> tuple[str, list[str], str]:
        """
        Returns (reply_text, tools_used, detected_language_code).
        """
        tools_used: list[str] = []

        # Detect language via script analysis before building messages.
        # This is reliable (Unicode-based) unlike langdetect which misidentifies
        # English text containing Indian city/place names as Hindi.
        detected_lang = detect_language(user_message)
        if detected_lang == "en":
            tagged_message = f"[EN] {user_message}"
        elif detected_lang == "hi":
            tagged_message = f"[HI] {user_message}"
        else:
            tagged_message = user_message  # regional langs go through Sarvam

        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": tagged_message},
        ]

        for round_num in range(settings.max_tool_rounds):
            logger.debug("Agent round %d – %d messages", round_num + 1, len(messages))

            raw = user_message if round_num == 0 else ""
            try:
                response, lang = await self._router.chat(
                    messages=messages,
                    tools=TOOLS,
                    temperature=settings.agent_temperature,
                    max_tokens=2048,
                    user_message_raw=raw,
                )
            except Exception as exc:
                if getattr(exc, "status_code", None) == 400:
                    logger.warning("400 tool_use_failed in agent loop round %d — retrying without tools. %s", round_num + 1, exc)
                    response, lang = await self._router.chat(
                        messages=messages,
                        tools=None,
                        temperature=0.3,
                        max_tokens=512,
                        user_message_raw=raw,
                    )
                else:
                    raise

            # Update detected_lang if router disagrees (e.g. Sarvam regional lang)
            if round_num == 0 and lang not in ("en", "hi"):
                detected_lang = lang

            # Final answer — no tool calls
            if not response.tool_calls:
                logger.debug("Agent done after %d round(s). Tools: %s", round_num + 1, tools_used)
                return response.content, tools_used, detected_lang

            # Append assistant message with tool_calls for the next round
            messages.append({
                "role": "assistant",
                "content": response.content,
                "tool_calls": response.tool_calls,
            })

            # Execute each requested tool
            for tc in response.tool_calls:
                tool_name = tc["function"]["name"]
                tools_used.append(tool_name)

                try:
                    args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {}

                logger.debug("Tool '%s' args: %s", tool_name, args)
                tool_result = await _dispatch_tool(tool_name, args, db, user_id)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": tool_name,
                    "content": tool_result,
                })

        logger.warning("Agent hit max rounds for user %s", user_id)
        return (
            "I'm having a bit of trouble completing that right now. Please try again.",
            tools_used,
            detected_lang,
        )


# Singleton — created once at startup, shared across requests
agent = JioJoinAgent()
