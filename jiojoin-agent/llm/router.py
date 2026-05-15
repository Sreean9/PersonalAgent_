"""
llm/router.py – Language-aware LLM routing.

Flow:
  1. Detect the language of the user's message.
  2. If it's a regional Indian language (and Sarvam is enabled):
       a. Translate input → English via Sarvam.
       b. Run the agent with Groq (English).
       c. Translate reply → original language via Sarvam.
  3. Otherwise run Groq directly (handles English + Hindi natively).

The router exposes a single `chat()` method that mirrors BaseLLMProvider
so agent.py doesn't need to know about translation at all.
"""

from __future__ import annotations

import logging

from llm.base import LLMResponse
from llm.groq_provider import GroqProvider
from llm.sarvam_provider import SarvamTranslator, detect_language, needs_translation

logger = logging.getLogger(__name__)


class LLMRouter:
    """
    Drop-in replacement for GroqProvider that transparently handles
    regional Indian languages via Sarvam translation.
    """

    def __init__(self) -> None:
        self._groq = GroqProvider()
        self._sarvam = SarvamTranslator()

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.6,
        max_tokens: int = 2048,
        user_message_raw: str = "",
    ) -> tuple[LLMResponse, str]:
        """
        Run the LLM, routing through Sarvam if the user's language warrants it.

        Returns:
            (LLMResponse, detected_lang_code)
            detected_lang_code is 'en' for English, 'hi' for Hindi, 'ta' for Tamil, etc.
        """
        lang = detect_language(user_message_raw) if user_message_raw else "en"
        logger.debug("Detected language: %s", lang)

        if needs_translation(lang):
            return await self._routed_chat(messages, tools, temperature, max_tokens, lang)

        response = await self._groq.chat(messages, tools, temperature, max_tokens)
        return response, lang

    async def _routed_chat(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        temperature: float,
        max_tokens: int,
        lang: str,
    ) -> tuple[LLMResponse, str]:
        """Translate the last user message to English, run Groq, translate reply back."""
        translated_messages = list(messages)

        # Translate the last user message (the one just added)
        for i in range(len(translated_messages) - 1, -1, -1):
            if translated_messages[i].get("role") == "user":
                original = translated_messages[i]["content"]
                translated = await self._sarvam.to_english(original, lang)
                translated_messages[i] = {**translated_messages[i], "content": translated}
                logger.debug("Translated '%s' → '%s' (lang=%s)", original[:60], translated[:60], lang)
                break

        response = await self._groq.chat(translated_messages, tools, temperature, max_tokens)

        # Translate the final reply back to user's language (only if no tool calls pending)
        if response.content and not response.tool_calls:
            response.content = await self._sarvam.from_english(response.content, lang)

        return response, lang
