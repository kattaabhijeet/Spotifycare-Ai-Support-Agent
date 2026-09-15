"""
escalation.py — Decide whether to auto-handle or escalate to a human agent.

Two-layer architecture:
  Layer 1 — Hard rules (deterministic, no API call):
    legal keywords, account compromise, financial disputes, profanity, low confidence

  Layer 2 — Gemini Flash Lite (only runs if Layer 1 passes):
    sentiment severity, ambiguity, high-stakes nuance

Output: {"decision": "auto" | "escalate", "reason": str, "layer": "rules" | "llm"}
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Hard-rule keyword lists
# ─────────────────────────────────────────────────────────────────────────────
LEGAL_KEYWORDS = [
    r"\blawsuit\b", r"\bsue\b", r"\bsuing\b", r"\battorney\b", r"\blegal\s+action\b",
    r"\bfraud\b", r"\bchargebacks?\b", r"\bdisputing?\b", r"\bbank\b.*\bcharge\b",
    r"\bconsumer\s+protection\b", r"\bftc\b", r"\bclass\s+action\b",
]

ACCOUNT_COMPROMISE_KEYWORDS = [
    r"\bhacked\b", r"\bunauthorized\b", r"\bsomeone\s+else\s+(is\s+)?using\b",
    r"\bstolen\b.*\baccount\b", r"\bidentity\s+theft\b", r"\bpassword\s+(was\s+)?changed\b",
    r"\bcannot\s+(log\s+in|access)\b",
]

PROFANITY_KEYWORDS = [
    r"\bkill\b.*\byou\b", r"\bthreaten\b", r"\bscam\b", r"\bstupid\b.*\bcompany\b",
    r"\bworst\s+company\b", r"\brefund\s+or\s+I\b",
]

FINANCIAL_HIGH_STAKES = [
    r"\brefund\b", r"\bcharged\b.*\bmultiple\b", r"\bdouble\s+charged\b",
    r"\bunauthorized\s+charge\b", r"\bpayment\s+failed\b.*\bstill\b.*\bcharged\b",
]

CONFIDENCE_THRESHOLD = 0.60


def _matches_any(text: str, patterns: list[str]) -> Optional[str]:
    """Return the first matching pattern, or None."""
    text_lower = text.lower()
    for pattern in patterns:
        if re.search(pattern, text_lower):
            return pattern
    return None


def _apply_hard_rules(
    message: str, intent: str, confidence: float
) -> Optional[dict]:
    """Apply deterministic rules. Returns escalation dict if triggered, else None."""

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "decision": "escalate",
            "reason": f"Classifier confidence {confidence:.2f} below threshold {CONFIDENCE_THRESHOLD}.",
            "layer": "rules",
        }
    if _matches_any(message, LEGAL_KEYWORDS):
        return {
            "decision": "escalate",
            "reason": "Legal/dispute language detected — requires human judgment.",
            "layer": "rules",
        }
    if _matches_any(message, ACCOUNT_COMPROMISE_KEYWORDS):
        return {
            "decision": "escalate",
            "reason": "Possible account compromise or security issue.",
            "layer": "rules",
        }
    if _matches_any(message, FINANCIAL_HIGH_STAKES):
        return {
            "decision": "escalate",
            "reason": "Financial dispute — refunds/billing corrections require a human agent.",
            "layer": "rules",
        }
    if _matches_any(message, PROFANITY_KEYWORDS):
        return {
            "decision": "escalate",
            "reason": "Aggressive/distressed language detected — human de-escalation recommended.",
            "layer": "rules",
        }
    return None


_LLM_SYSTEM = """You are a triage specialist for SpotifyCare customer support.

Decide if the message should be AUTO-HANDLED by the AI agent or ESCALATED to a human.

Escalate if:
  - High emotional distress or urgency (e.g., "waiting 2 weeks")
  - Financial risk not caught by keyword rules
  - Ambiguous enough that a wrong automated reply could cause brand damage
  - Multiple compounding issues simultaneously

Auto-handle if:
  - Standard technical issue with a clear fix
  - Compliment or informational query
  - Low stakes — getting it slightly wrong causes only minor inconvenience

Return ONLY valid JSON: {"decision": "auto" | "escalate", "reason": "<one sentence>"}
"""


class EscalationDecider:
    """Two-layer escalation decision engine using Gemini Flash Lite."""

    def __init__(self, confidence_threshold: float = CONFIDENCE_THRESHOLD):
        self.confidence_threshold = confidence_threshold

    def decide(
        self,
        message: str,
        intent: str = "",
        confidence: float = 1.0,
        thread_context: str = "",
    ) -> dict[str, Any]:
        """
        Returns dict with: decision (str), reason (str), layer (str)
        """
        # Layer 1: Hard rules (free)
        hard_result = _apply_hard_rules(message, intent, confidence)
        if hard_result:
            return hard_result

        # Layer 2: Gemini Flash Lite judgment
        from src.utils.gemini import generate_json

        user_content = (
            f"Intent: {intent or 'unknown'}\n"
            f"Confidence: {confidence:.2f}\n"
            f"{'Thread context: ' + thread_context + chr(10) if thread_context else ''}"
            f"Customer message: {message}"
        )

        result = generate_json(_LLM_SYSTEM, user_content, temperature=0.0)
        if not result:
            return {
                "decision": "escalate",
                "reason": "Gemini error during triage — defaulting to escalate.",
                "layer": "error",
            }
        return {
            "decision": result.get("decision", "auto"),
            "reason": result.get("reason", ""),
            "layer": "llm",
        }


if __name__ == "__main__":
    import sys, json
    msg = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "You guys charged me twice and I'm calling my bank"
    decider = EscalationDecider()
    out = decider.decide(msg, intent="account_billing", confidence=0.88)
    print(json.dumps(out, indent=2))
