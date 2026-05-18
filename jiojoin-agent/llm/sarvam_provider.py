"""
llm/sarvam_provider.py - Sarvam AI translation bridge for Indian regional languages.

Strategy: translate regional-language input -> English, run agent (Groq), translate reply -> back.
Sarvam natively supports 10 Indian languages: bn, gu, hi, kn, ml, mr, od, pa, ta, te.
"""

from __future__ import annotations

import logging

import httpx

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# langdetect ISO 639-1 codes -> Sarvam BCP-47 codes
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

# Strong Roman Hindi markers: ONE match is enough to conclude Hindi
_ROMAN_HINDI_STRONG = frozenset([
    "kidhar", "kahan", "kaise", "kyun", "karega", "karein",
    "batao", "dikhao", "sunao", "dekho", "chahiye", "shukriya",
    "nahi", "hain", "mujhe", "aapko", "tumhe", "humara",
    "namaste", "bahut", "theek", "accha", "acha",
    "bata", "karo", "mera", "meri", "yaar", "bhai", "dost",
    "hoga", "likhao", "samjho", "bolna", "bolte", "chahta",
    "chahte", "chahti", "milna", "milte", "khana", "pina",
])

# Weak Roman Hindi markers: need TWO or more matches to conclude Hindi
_ROMAN_HINDI_WEAK = frozenset([
    "hai", "hoon", "kya", "tum", "aap", "bhi", "aur",
    "yeh", "woh", "ek", "ab", "kal", "aaj", "kab",
])


def detect_language(text: str) -> str:
    """
    Return ISO 639-1 language code using Unicode block detection.

    Checks Unicode code-point ranges directly -- deterministic and never
    misidentifies English text containing Indian city/place names as Hindi.
    No external library required.
    """
    if not text or not text.strip():
        return "en"
    t = text.strip()

    def _has_block(lo: int, hi: int) -> bool:
        return any(lo <= ord(c) <= hi for c in t)

    if _has_block(0x0900, 0x097F):   # Devanagari -> Hindi
        return "hi"
    if _has_block(0x0980, 0x09FF):   # Bengali
        return "bn"
    if _has_block(0x0A00, 0x0A7F):   # Gurmukhi (Punjabi)
        return "pa"
    if _has_block(0x0A80, 0x0AFF):   # Gujarati
        return "gu"
    if _has_block(0x0B00, 0x0B7F):   # Odia
        return "or"
    if _has_block(0x0B80, 0x0BFF):   # Tamil
        return "ta"
    if _has_block(0x0C00, 0x0C7F):   # Telugu
        return "te"
    if _has_block(0x0C80, 0x0CFF):   # Kannada
        return "kn"
    if _has_block(0x0D00, 0x0D7F):   # Malayalam
        return "ml"

    # Latin-script: tiered Roman-Hindi detection
    words = set(t.lower().split())
    if words & _ROMAN_HINDI_STRONG:          # 1 strong marker = Hindi
        return "hi"
    if len(words & _ROMAN_HINDI_WEAK) >= 2:  # 2+ weak markers = Hindi
        return "hi"

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
