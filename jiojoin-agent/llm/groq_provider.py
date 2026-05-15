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


def _get_failed_generation(exc: BadRequestError) -> str:
    """Extract failed_generation from a Groq 400 error."""
    try:
        body = getattr(exc, "body", None)
        if isinstance(body, str):
            body = json.loads(body)
        if isinstance(body, dict):
            fg = (body.get("error") or {}).get("failed_generation", "")
            if fg:
                return fg
    except Exception:
        pass
    # Fallback: parse from string representation
    m = re.search(r"'failed_generation':\s*'(.*?)'(?=\s*[,}])", str(exc), re.DOTALL)
    return m.group(1) if m else ""


def _parse_xml_tool_calls(text: str) -> list[dict]:
    """
    Parse Llama XML-style tool calls into standard format.
    Handles: <function=name({"k":"v"})</function>  and  <function=name>{"k":"v"}
    """
    tool_calls: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for m in re.finditer(r'<function=(\w+)[>(]?\s*(\{[^<]*?\})', text, re.DOTALL):
        name, args = m.group(1), m.group(2).strip()
        if (name, args) in seen:
            continue
        seen.add((name, args))
        try:
            json.loads(args)
            tool_calls.append({
                "id": f"call_r{len(tool_calls)}",
                "type": "function",
                "function": {"name": name, "arguments": args},
            })
            logger.info("Recovered XML tool call: %s %s", name, args[:80])
        except json.JSONDecodeError:
            logger.warning("Could not parse args for tool %s: %.60s", name, args)
    return tool_calls


class GroqProvider(BaseLLMProvider):
    def __init__(self) -> None:
        # 20-second per-call timeout prevents hanging when Groq is slow or rate-limited
        self._client = AsyncGroq(api_key=settings.groq_api_key, timeout=20.0)

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

        except BadRequestError as exc:
            # Groq rejected a malformed tool call — try to recover the XML format
            failed_gen = _get_failed_generation(exc)
            if failed_gen:
                recovered = _parse_xml_tool_calls(failed_gen)
                if recovered:
                    logger.info("XML recovery succeeded: %d tool call(s)", len(recovered))
                    return LLMResponse(content="", tool_calls=recovered, finish_reason="tool_calls")
            # Recovery failed — retry without tools for a plain text response
            if "tools" in kwargs:
                logger.warning("XML recovery failed — retrying without tools (max_tokens=512)")
                kwargs.pop("tools", None)
                kwargs.pop("tool_choice", None)
                kwargs["max_tokens"] = 512
                response = await self._client.chat.completions.create(**kwargs)
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
