"""
src/utils/gemini.py — Shared Gemini client using google-genai SDK.

Uses gemini-flash-lite-latest — non-thinking, ~0.5s/call, free tier.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

MODEL_NAME = "gemini-flash-lite-latest"   # non-thinking, ~0.5s/call, free tier


def _get_client():
    """Return a configured google.genai Client (lazy import)."""
    from google import genai
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY not set. Add it to your .env file.\n"
            "Get a free key at: https://aistudio.google.com/app/apikey"
        )
    return genai.Client(api_key=api_key)


def generate_json(
    system_prompt: str,
    user_content: str,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """
    Prompt Gemini and return a parsed JSON dict.
    Returns {} on any error so callers always get a dict back.
    """
    try:
        from google import genai
        from google.genai import types

        client = _get_client()
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                response_mime_type="application/json",
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        log.error(f"Gemini generate_json error: {e}")
        return {}


def generate_text(
    system_prompt: str,
    user_content: str,
    temperature: float = 0.7,
) -> str:
    """
    Prompt Gemini and return plain text.
    Returns empty string on error.
    """
    try:
        from google import genai
        from google.genai import types

        client = _get_client()
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
            ),
        )
        return response.text or ""
    except Exception as e:
        log.error(f"Gemini generate_text error: {e}")
        return ""
