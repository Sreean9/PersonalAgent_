"""llm/groq_provider.py – Groq (Llama 3.3-70b) LLM provider."""

from __future__ import annotations

import json
import logging
import re

from groq import AsyncGroq, BadRequestError

from config import get_settings
from llm.base import BaseLLMProvider, LLMResponse

logger = logging.getLogger(__name__)
settings = get_settings()


def _extract_failed_generation(exc: Exception) -> str:
    """
    Extract failed_generation from a Groq 400 error.
    Tries three independent strategies so SDK version differences don't matter.
    """
    # Strategy 1: exc.body (may be dict or JSON string)
    try:
        body = getattr(exc, "body", None)
        if isinstance(body, str):
            body = json.loads(body)
        if isinstance(body, dict):
            fg = (body.get("error") or {}).get("failed_generation", "")
            if fg:
                logger.debug("failed_generation extracted via exc.body")
                return fg
    except Exception:
        pass

    # Strategy 2: exc.response.json()
    try:
        rb = getattr(exc, "response", None)
        if rb is not None:
            data = rb.json()
            fg = (data.get("error") or {}).get("failed_generation", "")
            if fg:
                logger.debug("failed_generation extracted via exc.response.json()")
                return fg
    except Exception:
        pass

    # Strategy 3: regex over str(exc) — works regardless of SDK internals
    try:
        s = str(exc)
        for pat in [
            r"'failed_generation':\s*'(.*?)'(?=\s*[,}])",
            r'"failed_generation":\s*"(.*?)"(?=\s*[,}])',
        ]:
            m = re.search(pat, s, re.DOTALL)
            if m:
                logger.debug("failed_generation extracted via str(exc) regex")
                return m.group(1)
    except Exception:
        pass

    logger.error("Could not extract failed_generation from exc. str(exc)=%.300s", str(exc))
    return ""


def _parse_xml_tool_calls(text: str) -> list[dict]:
    """
    Parse all known Llama XML tool-call variants into standard tool_calls format.

    Known formats:
      A: <function=name({"k":"v"})</function>
      B: <function=name>{"k":"v"}            ← most common failing case
      C: <function=name>{"k":"v"}</function>

    Single unified regex: after <function=NAME allow optional ( or >, then JSON.
    Uses [^<] instead of . to avoid greedily crossing into a second tool call.
    """
    pattern = re.compile(r'<function=(\w+)[>(]?\s*(\{[^<]*?\})', re.DOTALL)

    tool_calls: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for m in pattern.finditer(text):
        name = m.group(1)
        args_str = m.group(2).strip()
        key = (name, args_str)
        if key in seen:
            continue
        seen.add(key)
        try:
            json.loads(args_str)  # validate before using
            tool_calls.append({
                "id": f"call_recovered_{len(tool_calls)}",
                "type": "function",
                "function": {"name": name, "arguments": args_str},
            })
            logger.info("XML recovery: parsed tool call %s(%s)", name, args_str[:120])
        except json.JSONDecodeError:
            logger.warning("XML recovery: bad JSON for tool %s: %.80s", name, args_str)

    return tool_calls


class GroqProvider(BaseLLMProvider):
    def __init__(self) -> None:
        self._client = AsyncGroq(api_key=settings.groq_api_key)

    async def _call(self, kwargs: dict) -> object:
        """Single API call; raises on error."""
        return await self._client.chat.completions.create(**kwargs)

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.6,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        kwargs: dict = {
            "model": settings.groq_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            response = await self._call(kwargs)

        except Exception as exc:
            status = getattr(exc, "status_code", None)

            # ── Rate limit (429): retry with smaller fallback model ────────────
            if status == 429:
                fallback = settings.groq_fallback_model
                logger.warning("Rate-limited on %s. Retrying with %s.", kwargs["model"], fallback)
                kwargs["model"] = fallback
                response = await self._call(kwargs)

            # ── Bad tool call (400): three-level recovery ─────────────────────
            elif status == 400 or isinstance(exc, BadRequestError):

                # Level 1: extract and parse the XML tool call Groq rejected
                failed_gen = _extract_failed_generation(exc)
                if failed_gen:
                    recovered = _parse_xml_tool_calls(failed_gen)
                    if recovered:
                        logger.info("Level-1 recovery OK: %d tool call(s) parsed.", len(recovered))
                        return LLMResponse(
                            content="", tool_calls=recovered, finish_reason="tool_calls"
                        )
                    logger.warning("Level-1 recovery failed (no parseable calls in failed_gen).")
                else:
                    logger.warning("Level-1 recovery failed (could not extract failed_gen).")

                # Level 2: retry the same request without tools → graceful text response
                if "tools" in kwargs:
                    logger.warning("Level-2 fallback: retrying without tools.")
                    kwargs_plain = {k: v for k, v in kwargs.items()
                                    if k not in ("tools", "tool_choice")}
                    response = await self._call(kwargs_plain)
                else:
                    raise  # no tools were passed; nothing more we can do

            else:
                raise

        choice = response.choices[0]
        msg = choice.message

        tool_calls: list[dict] = []
        if msg.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]

        return LLMResponse(
            content=msg.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
        )
