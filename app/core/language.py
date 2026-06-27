"""Odia language detection and translation layer.

Detects whether user input is in Odia (or another non-English language),
translates it to English for agent processing, and translates the response
back to the original language. Uses the same OpenAI model configured for
the project to keep dependencies minimal.
"""
from __future__ import annotations

import re
from functools import lru_cache

from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

_ODIA_RANGE = re.compile(r"[଀-୿]")

_DEVANAGARI_RANGE = re.compile(r"[ऀ-ॿ]")


def detect_language(text: str) -> str:
    """Detect input language using Unicode script analysis.

    Returns 'odia', 'hindi', or 'english'. Falls back to 'english' for
    ambiguous inputs so the agent processes them without translation.
    """
    odia_chars = len(_ODIA_RANGE.findall(text))
    devanagari_chars = len(_DEVANAGARI_RANGE.findall(text))
    alpha_chars = sum(1 for c in text if c.isalpha())

    if alpha_chars == 0:
        return "english"

    if odia_chars / alpha_chars > 0.3:
        return "odia"
    if devanagari_chars / alpha_chars > 0.3:
        return "hindi"

    return "english"


@lru_cache
def _translation_llm() -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=0.0,
        api_key=settings.openai_api_key,
    )


def translate_to_english(text: str, source_lang: str) -> str:
    """Translate non-English input to English for agent processing."""
    if source_lang == "english":
        return text

    lang_name = {"odia": "Odia", "hindi": "Hindi"}.get(source_lang, source_lang)
    prompt = (
        f"Translate the following {lang_name} text to English. "
        "Return ONLY the English translation, nothing else.\n\n"
        f"{text}"
    )
    result = _translation_llm().invoke([("system", prompt)])
    translated = str(result.content).strip()
    logger.info("Translated %s -> English: '%s' -> '%s'", lang_name, text[:50], translated[:50])
    return translated


def translate_from_english(text: str, target_lang: str) -> str:
    """Translate English response back to the user's language."""
    if target_lang == "english":
        return text

    lang_name = {"odia": "Odia", "hindi": "Hindi"}.get(target_lang, target_lang)
    prompt = (
        f"Translate the following English text to {lang_name}. "
        "Return ONLY the translation, nothing else. "
        "Keep technical terms, numbers, and proper nouns as-is.\n\n"
        f"{text}"
    )
    result = _translation_llm().invoke([("system", prompt)])
    translated = str(result.content).strip()
    logger.info("Translated English -> %s", lang_name)
    return translated
