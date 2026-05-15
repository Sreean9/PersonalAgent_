"""llm/groq_provider.py – Groq (Llama 3.3-70b) LLM provider."""

from __future__ import annotations

import json
import logging
import re

from groq import AsyncGroq, BadRequestError, RateLimitError

from config import get_settings
from llm.base import BaseLLMProvider, LLMResponse

logger = logging.getLogger(__name__)
settings = get_settings()


def _extract_failed_generation(exc: BadRequestError) -> str:
    """
    Extract the failed_generation string from a Groq 400 error.
    Tries structured body first, then falls back to parsing the exception string.
    """
    # Strategy 1: structured body dict
    try:
        body = getattr(exc, "body", None) or {}
        if isinstance(body, dict):
            fg = (body.get("error") or {}).get("failed_generation", "")
            if fg:
                return fg
            fg = body.get("failed_generation", "")
            if fg:
                return fg
    except Exception:
        pass

    # Strategy 2: regex over the exception's string representation
    # e.g. "...  'failed_generation': '<function=...'}"
    try:
        s = str(exc)
        m = re.search(r"'failed_generation':\s*'(.*?)'(?=\s*[,}])", s, re.DOTALL)
        if m:
            return m.group(1)
    except Exception:
        pass

    return ""


def _parse_xml_tool_calls(text: str) -> list[dict]:
    """
    Llama emits XML-style tool calls in several formats that Groq rejects.
    Known variants:
      A: <function=name({"k":"v"})</function>
      B: <function=name>{"k":"v"}          (no parens, no closing tag)
      C: <function=name>{"k":"v"}</function>

    Single unified regex handles all three: after <function=NAME we allow
    an optional ( or >, optional whitespace, then capture the JSON object.
    """
    # Unified pattern: optional open-paren or >, then JSON object
    pattern = re.compile(
        r'<function=(\w+)[>(]?\s*(\{.*?\})',
        re.DOTALL,
    )

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
            json.loads(args_str)  # validate — don't accept malformed JSON
            tool_calls.append({
                "id": f"call_recovered_{len(tool_calls)}",
                "type": "function",
                "function": {"name": name, "arguments": args_str},
            })
            logger.info("Recovered XML tool call: %s args=%s", name, args_str[:120])
        except json.JSONDecodeError:
            logger.warning("XML recovery: could not parse JSON for tool %s: %.80s", name, args_str)

    return tool_calls


class GroqProvider(BaseLLMProvider):
    def __init__(self) -> None:
        self._client = AsyncGroq(api_key=settings.groq_api_key)

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
            response = await self._client.chat.completions.create(**kwargs)

        except RateLimitError:
            # Primary model (70b) daily token limit hit — fall back to 8b instantly.
            fallback = settings.groq_fallback_model
            logger.warning("Primary model rate-limited. Retrying with fallback: %s", fallback)
            kwargs["model"] = fallback
            response = await self._client.chat.completions.create(**kwargs)

        except BadRequestError as exc:
            # Groq rejects malformed XML-style tool calls the model generates.
            # Recover by extracting and parsing the failed_generation field.
            failed_gen = _extract_failed_generation(exc)
            if failed_gen:
                logger.warning(
                    "Groq tool_use_failed — attempting XML recovery. Raw: %.200s", failed_gen
                )
                recovered = _parse_xml_tool_calls(failed_gen)
                if recovered:
                    logger.info("XML recovery succeeded: %d tool call(s)", len(recovered))
                    return LLMResponse(
                        content="", tool_calls=recovered, finish_reason="tool_calls"
                    )
                logger.error("XML recovery failed — no parseable tool calls found in: %.200s", failed_gen)
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
