"""
llm/sarvam_provider.py – Sarvam AI translation bridge for Indian regional languages.

Strategy: translate regional-language input → English, run agent (Groq), translate reply → back.
Sarvam natively supports 10 Indian languages: bn, gu, hi, kn, ml, mr, od, pa, ta, te.
"""

from __future__ import annotations

import logging

import httpx

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# langdetect ISO 639-1 codes → Sarvam BCP-47 codes
_LANG_TO_SARVAM: dict[str, str] = {
    "ta": "ta-IN",   # Tamil
    "te": "te-IN",   # Telugu
    "bn": "bn-IN",   # Bengali
    "mr": "mr-IN",   # Marathi
    "gu": "gu-IN",   # Gujarati
    "kn": "kn-IN",   # Kannada
    "ml": "ml-IN",   # Malayalam
    "pa": "pa-IN",   # Punjabi
    "or": "od-IN",   # Odia
}

# Languages Groq handles well enough without Sarvam
_GROQ_NATIVE = {"en", "hi"}


# Common Roman-script Hindi words that don't appear in English
_ROMAN_HINDI_MARKERS = frozenset([
    "kaise", "kya", "mujhe", "aapko", "tumhe", "hain", "nahi",
    "bahut", "karo", "bata", "batao", "mera", "meri", "humara",
    "sunao", "dekho", "chahiye", "theek", "accha", "shukriya",
    "namaste", "yaar", "dost", "bhai", "acha", "hoga", "karega",
    "karein", "batao", "dikhao", "likhao", "sunao",
])


def detect_language(text: str) -> str:
    """
    Return ISO 639-1 language code using Unicode script detection.

    Uses Unicode block ranges — deterministic and never misidentifies
    English text with Indian city/place names as Hindi.
    """
    if not text or not text.strip():
        return "en"
    t = text.strip()
    # Devanagari script → Hindi
    if any('ऀ' <= c <= 'ॿ' for c in t):
        return "hi"
    # Tamil
    if any('஀' <= c <= '௿' for c in t):
        return "ta"
    # Telugu
    if any('ఀ' <= c <= '౿' for c in t):
        return "te"
    # Bengali
    if any('ঀ' <= c <= '৿' for c in t):
        return "bn"
    # Malayalam
    if any('ഀ' <= c <= 'ൿ' for c in t):
        return "ml"
    # Gujarati
    if any('઀' <= c <= '૿' for c in t):
        return "gu"
    # Kannada
    if any('ಀ' <= c <= '೿' for c in t):
        return "kn"
    # Punjabi (Gurmukhi)
    if any('਀' <= c <= '੿' for c in t):
        return "pa"
    # Odia
    if any('଀' <= c <= '୿' for c in t):
        return "or"
    # Latin-script only — check for Roman transliteration Hindi markers
    words = set(t.lower().split())
    if len(words & _ROMAN_HINDI_MARKERS) >= 2:
        return "hi"
    # Default: Latin/ASCII text is English
    return "en"


def needs_translation(lang_code: str) -> bool:
    """True when Sarvam translation is required and configured."""
    return (
        settings.sarvam_enabled
        and lang_code not in _GROQ_NATIVE
        and lang_code in _LANG_TO_SARVAM
    )


def sarvam_code(lang_code: str) -> str | None:
    return _LANG_TO_SARVAM.get(lang_code)


class SarvamTranslator:
    """Thin async wrapper around the Sarvam AI /translate endpoint."""

    def __init__(self) -> None:
        self._headers = {
            "api-subscription-key": settings.sarvam_api_key,
            "Content-Type": "application/json",
        }

    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """
        Translate text between Sarvam BCP-47 language codes.

        Args:
            text: Input text.
            source_lang: e.g. "ta-IN"
            target_lang: e.g. "en-IN"

        Returns:
            Translated text, or the original on failure.
        """
        url = f"{settings.sarvam_base_url}/translate"
        payload = {
            "input": text,
            "source_language_code": source_lang,
            "target_language_code": target_lang,
            "speaker_gender": "Female",
            "mode": "formal",
            "model": "mayura:v1",
            "enable_preprocessing": True,
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, json=payload, headers=self._headers)
                resp.raise_for_status()
                data = resp.json()
                return data.get("translated_text", text)
        except Exception as exc:
            logger.warning("Sarvam translation failed (%s): returning original text.", exc)
            return text

    async def to_english(self, text: str, source_lang_code: str) -> str:
        sarvam_src = sarvam_code(source_lang_code)
        if not sarvam_src:
            return text
        return await self.translate(text, source_lang=sarvam_src, target_lang="en-IN")

    async def from_english(self, text: str, target_lang_code: str) -> str:
        sarvam_tgt = sarvam_code(target_lang_code)
        if not sarvam_tgt:
            return text
        return await self.translate(text, source_lang="en-IN", target_lang=sarvam_tgt)
