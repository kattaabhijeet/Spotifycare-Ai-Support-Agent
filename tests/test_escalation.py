"""
tests/test_escalation.py — Unit tests for escalation logic.

These tests require no API calls when testing the hard-rule layer.
LLM-layer tests are skipped unless OPENAI_API_KEY is set.
"""

import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent.escalation import _apply_hard_rules, CONFIDENCE_THRESHOLD


class TestHardRules:
    def test_low_confidence_escalates(self):
        result = _apply_hard_rules("my app is slow", "app_device_bug", 0.3)
        assert result is not None
        assert result["decision"] == "escalate"
        assert result["layer"] == "rules"

    def test_legal_keyword_escalates(self):
        result = _apply_hard_rules(
            "I'm going to file a lawsuit against Spotify", "account_billing", 0.9
        )
        assert result is not None
        assert result["decision"] == "escalate"

    def test_chargeback_escalates(self):
        result = _apply_hard_rules(
            "I am disputing this charge with my bank", "account_billing", 0.85
        )
        assert result is not None
        assert result["decision"] == "escalate"

    def test_account_compromise_escalates(self):
        result = _apply_hard_rules(
            "Someone hacked my Spotify account", "account_billing", 0.92
        )
        assert result is not None
        assert result["decision"] == "escalate"

    def test_refund_escalates(self):
        result = _apply_hard_rules(
            "I want a refund for this month", "account_billing", 0.9
        )
        assert result is not None
        assert result["decision"] == "escalate"

    def test_normal_bug_does_not_escalate(self):
        result = _apply_hard_rules(
            "My Spotify keeps buffering on WiFi", "playback_issue", 0.88
        )
        assert result is None   # passes through to LLM layer

    def test_compliment_does_not_escalate(self):
        result = _apply_hard_rules(
            "Love the new AI DJ feature!", "compliment", 0.97
        )
        assert result is None

    def test_confidence_threshold_boundary(self):
        # Exactly at threshold should NOT escalate
        result = _apply_hard_rules("my app crashes", "app_device_bug", CONFIDENCE_THRESHOLD)
        assert result is None

        # Just below threshold should escalate
        result = _apply_hard_rules("my app crashes", "app_device_bug", CONFIDENCE_THRESHOLD - 0.01)
        assert result is not None


class TestClassifierIntegration:
    """Test that the intent taxonomy is valid (no API calls)."""

    def test_intents_are_non_empty(self):
        import importlib, sys
        # Import only the data constants — avoid triggering gemini DLL at collection
        spec = importlib.util.spec_from_file_location(
            "classifier_consts",
            str(ROOT / "src" / "agent" / "classifier.py")
        )
        # Simpler: just check the dict is defined and non-empty by importing directly
        from src.agent.classifier import INTENTS
        assert len(INTENTS) >= 5

    def test_few_shot_examples_reference_valid_intents(self):
        from src.agent.classifier import INTENTS, FEW_SHOT_EXAMPLES
        for ex in FEW_SHOT_EXAMPLES:
            assert ex["intent"] in INTENTS, f"Example intent '{ex['intent']}' not in taxonomy"
