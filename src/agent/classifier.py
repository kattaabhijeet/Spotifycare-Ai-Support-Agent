"""
classifier.py — Intent classifier for SpotifyCare customer messages.

Method: Few-shot prompt with Gemini Flash Lite (`gemini-flash-lite-latest`, JSON mode).

Intents (9 total — derived from cluster analysis):
  playback_issue | account_billing | app_device_bug | search_discovery
  download_offline | social_playlist | content_availability | compliment | other
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Intent taxonomy
# ─────────────────────────────────────────────────────────────────────────────
INTENTS: dict[str, str] = {
    "playback_issue":       "Songs, podcasts, or audio won't play; stuttering, skipping, or buffering.",
    "account_billing":      "Subscription problems, unexpected charges, premium/family plan issues, login credentials.",
    "app_device_bug":       "App crashes, freezes, won't open, black screen, login loop, sync errors.",
    "search_discovery":     "Can't find a specific song/artist/playlist; wrong search results; missing recommendations.",
    "download_offline":     "Offline mode not working; downloads fail, disappear, or are greyed out.",
    "social_playlist":      "Collaborative playlists, following friends, seeing others' activity, share features.",
    "content_availability": "A song, album, or podcast is missing/unavailable; region-locked content.",
    "compliment":           "Positive feedback, praise, or expressions of satisfaction with Spotify.",
    "other":                "Off-topic, unclear request, spam, or cannot be categorised into the above.",
}

# Few-shot examples (one per intent)
FEW_SHOT_EXAMPLES: list[dict[str, str]] = [
    {"message": "Hey Spotify my songs keep skipping on their own and it's really annoying",
     "intent": "playback_issue", "confidence": "0.95"},
    {"message": "I was charged twice this month and I only have one account what's going on",
     "intent": "account_billing", "confidence": "0.93"},
    {"message": "Spotify just crashes every time I open it on my Android 14 phone",
     "intent": "app_device_bug", "confidence": "0.94"},
    {"message": "Why can't I find the new Taylor Swift album in search? It's not showing up",
     "intent": "search_discovery", "confidence": "0.91"},
    {"message": "All my downloaded songs disappeared after I updated the app last night",
     "intent": "download_offline", "confidence": "0.90"},
    {"message": "My friend's playlist isn't showing up as collaborative anymore after the update",
     "intent": "social_playlist", "confidence": "0.88"},
    {"message": "The song Blinding Lights is not available in my country anymore why",
     "intent": "content_availability", "confidence": "0.89"},
    {"message": "Just wanted to say Spotify's new AI DJ feature is incredible, love it!",
     "intent": "compliment", "confidence": "0.97"},
    {"message": "hi can you follow me back on twitter lol",
     "intent": "other", "confidence": "0.99"},
]


def _build_system_prompt() -> str:
    intent_list = "\n".join(
        f"  - {name}: {desc}" for name, desc in INTENTS.items()
    )
    examples_str = "\n".join(
        f'  Message: "{ex["message"]}"\n'
        f'  → intent: {ex["intent"]}, confidence: {ex["confidence"]}'
        for ex in FEW_SHOT_EXAMPLES
    )
    return f"""You are an expert intent classifier for SpotifyCare customer support.

## Intent taxonomy
{intent_list}

## Few-shot examples
{examples_str}

## Instructions
- Analyse the customer message and optional thread context.
- Pick the SINGLE best-matching intent from the taxonomy.
- Return valid JSON only, with this exact schema:
  {{"intent": "<intent_name>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}}
- Do NOT include any text outside the JSON object.
"""


_SYSTEM_PROMPT = _build_system_prompt()


class IntentClassifier:
    """
    Classifies customer messages into one of 9 SpotifyCare intents.
    Uses Gemini Flash Lite (`gemini-flash-lite-latest`) with JSON-mode output.
    """

    def classify(self, message: str, thread_context: str = "") -> dict[str, Any]:
        """
        Classify a single customer message.

        Returns:
            dict with keys: intent (str), confidence (float), reasoning (str)
        """
        user_content = f"Customer message: {message}"
        if thread_context:
            user_content = f"Thread context:\n{thread_context}\n\n{user_content}"

        from src.utils.gemini import generate_json
        result = generate_json(_SYSTEM_PROMPT, user_content, temperature=0.0)

        # Validate intent is in taxonomy
        if result.get("intent") not in INTENTS:
            log.warning(f"Unknown intent '{result.get('intent')}'; defaulting to 'other'.")
            result["intent"] = "other"
            result["confidence"] = 0.5

        return {
            "intent": result.get("intent", "other"),
            "confidence": float(result.get("confidence", 0.5)),
            "reasoning": result.get("reasoning", ""),
        }

    def classify_batch(
        self, messages: list[str], contexts: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Classify a list of messages."""
        if contexts is None:
            contexts = [""] * len(messages)
        return [self.classify(msg, ctx) for msg, ctx in zip(messages, contexts)]


if __name__ == "__main__":
    msg = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "My Spotify keeps crashing on iPhone"
    clf = IntentClassifier()
    out = clf.classify(msg)
    print(json.dumps(out, indent=2))
