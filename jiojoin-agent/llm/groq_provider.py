"""llm/groq_provider.py - Groq LLM provider with XML tool-call recovery."""

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
    m = re.search(r"'failed_generation':\s*'(.*?)'(?=\s*[,}])", str(exc), re.DOTALL)
    return m.group(1) if m else ""


def _parse_xml_tool_calls(text: str) -> list[dict]:
    """
    Parse XML-style tool calls that llama-3.1-8b-instant sometimes emits
    as plain text instead of using the proper tool_calls JSON structure.

    Handles two formats the 8b model produces:
      Format A: <function=name({"k":"v"})</function>
      Format B: <toolname>{"k":"v"}</toolname>
    """
    tool_calls: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def _add(name: str, args: str) -> None:
        args = args.strip()
        if (name, args) in seen:
            return
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

    # Format A: <function=name({"k":"v"})  or  <function=name>{"k":"v"}
    for m in re.finditer(r'<function=(\w+)[>(]?\s*(\{[^<]*?\})', text, re.DOTALL):
        _add(m.group(1), m.group(2))

    # Format B: <toolname>{"k":"v"}</toolname>  (most common with 8b-instant)
    for m in re.finditer(r'<(\w+)>\s*(\{[^<]+?\})\s*</\1>', text, re.DOTALL):
        _add(m.group(1), m.group(2))

    return tool_calls


class GroqProvider(BaseLLMProvider):
    def __init__(self) -> None:
        self._client = AsyncGroq(api_key=settings.groq_api_key, timeout=20.0)

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.6,
        max_tokens: int = 1024,
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
            # Groq rejected a malformed tool call — try to recover from the
            # failed_generation field which contains the raw LLM output
            failed_gen = _get_failed_generation(exc)
            if failed_gen:
                recovered = _parse_xml_tool_calls(failed_gen)
                if recovered:
                    logger.info("Error-path XML recovery: %d tool call(s)", len(recovered))
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

        # Success path: model may still emit XML tool calls as plain text.
        # This is common with llama-3.1-8b-instant — Groq accepts the response
        # but returns the XML in content instead of tool_calls.
        if not tool_calls and msg.content and tools:
            xml_calls = _parse_xml_tool_calls(msg.content)
            if xml_calls:
                logger.info("Success-path XML recovery: %d tool call(s) found in content", len(xml_calls))
                return LLMResponse(content="", tool_calls=xml_calls, finish_reason="tool_calls")

        return LLMResponse(
            content=msg.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
        )
