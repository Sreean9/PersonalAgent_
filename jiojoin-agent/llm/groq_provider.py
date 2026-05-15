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


def _parse_xml_tool_calls(text: str) -> list[dict]:
    """
    Llama occasionally emits XML-style tool calls that Groq rejects with 400:
      <function=tool_name({"key": "val"})</function>
    Parse them and return the standard tool_calls list format so the agent
    loop can execute them normally.
    """
    pattern = r'<function=(\w+)\((.*?)\)(?:</function>|>)'
    matches = re.findall(pattern, text, re.DOTALL)
    tool_calls = []
    for i, (name, args_str) in enumerate(matches):
        args_str = args_str.strip() or "{}"
        try:
            json.loads(args_str)  # validate JSON before using it
            tool_calls.append({
                "id": f"call_recovered_{i}",
                "type": "function",
                "function": {"name": name, "arguments": args_str},
            })
            logger.info("Recovered XML tool call: %s(%s)", name, args_str[:100])
        except json.JSONDecodeError:
            logger.warning("Could not parse args for recovered tool %s: %s", name, args_str)
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
            # Extract the raw failed_generation and parse the tool call ourselves.
            body = exc.body or {}
            failed_gen = (body.get("error", {}).get("failed_generation", "")
                          if isinstance(body, dict) else "")
            if failed_gen:
                logger.warning("Groq 400 tool_use_failed. Attempting XML recovery. Raw: %.200s", failed_gen)
                recovered = _parse_xml_tool_calls(failed_gen)
                if recovered:
                    return LLMResponse(content="", tool_calls=recovered, finish_reason="tool_calls")
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
