"""
tests/test_pipeline.py — Unit tests for the data pipeline.

These tests run WITHOUT API calls and WITHOUT the real dataset.
They use synthetic data to test the logic in ingest.py and preprocess.py.

Run with: pytest tests/
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline.preprocess import clean_text, clean_dataframe, add_metadata
from src.pipeline.ingest import filter_brand, subsample, build_threads


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_raw_df():
    """Synthetic raw tweet DataFrame mimicking twcs.csv structure."""
    return pd.DataFrame({
        "tweet_id":                ["1", "2", "3", "4", "5", "6"],
        "author_id":               ["userA", "SpotifyCares", "userB", "SpotifyCares", "userC", "AmazonHelp"],
        "inbound":                 [True, False, True, False, True, False],
        "created_at":              ["2023-01-01"] * 6,
        "text":                    [
            "@SpotifyCares my app keeps crashing!",
            "@userA Hi! Let's fix that — try reinstalling the app.",
            "@SpotifyCares I was charged twice this month",
            "@userB We're sorry! Please DM us for billing help.",
            "@AmazonHelp where is my order?",
            "@userC Your order ships tomorrow!",
        ],
        "response_tweet_id":       [None, "1", None, "3", None, "5"],
        "in_response_to_tweet_id": [None, "1", None, "3", None, "5"],
    })


# ─────────────────────────────────────────────────────────────────────────────
# preprocess.py tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanText:
    def test_strips_mentions(self):
        assert "@SpotifyCares" not in clean_text("@SpotifyCares help me!")

    def test_normalises_url(self):
        result = clean_text("Check https://support.spotify.com for help")
        assert "<URL>" in result
        assert "https://" not in result

    def test_removes_html_entities(self):
        result = clean_text("That&amp;s great &amp; works")
        assert "&amp;" not in result

    def test_strips_hashtag_symbol(self):
        result = clean_text("#Spotify is great")
        assert "#" not in result
        assert "Spotify" in result

    def test_collapses_whitespace(self):
        result = clean_text("too   many    spaces")
        assert "  " not in result

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_non_string(self):
        assert clean_text(None) == ""
        assert clean_text(123) == ""


class TestCleanDataframe:
    def test_drops_short_messages(self):
        df = pd.DataFrame({
            "customer_message": ["hi", "my spotify keeps crashing every time I open it"],
            "brand_reply": ["ok", "let us help you!"],
            "thread_context": ["", ""],
        })
        result = clean_dataframe(df)
        assert len(result) == 1
        assert "crashing" in result.iloc[0]["customer_message_clean"]

    def test_drops_retweets(self):
        df = pd.DataFrame({
            "customer_message": ["RT @SpotifyCares: Check this out", "my app is broken"],
            "brand_reply": ["", "we'll help!"],
            "thread_context": ["", ""],
        })
        result = clean_dataframe(df)
        assert len(result) == 1

    def test_drops_duplicates(self):
        df = pd.DataFrame({
            "customer_message": ["My Spotify crashes every day"] * 3,
            "brand_reply": ["ok"] * 3,
            "thread_context": [""] * 3,
        })
        result = clean_dataframe(df)
        assert len(result) == 1


class TestAddMetadata:
    def test_adds_message_len(self):
        df = pd.DataFrame({"customer_message_clean": ["hello world"], "brand_reply_clean": ["hi"], "has_reply": [True]})
        result = add_metadata(df)
        assert "message_len" in result.columns
        assert result.iloc[0]["message_len"] == 11

    def test_has_url_flag(self):
        df = pd.DataFrame({"customer_message_clean": ["check <URL>"], "brand_reply_clean": ["ok"], "has_reply": [True]})
        result = add_metadata(df)
        assert result.iloc[0]["has_url"] == True


# ─────────────────────────────────────────────────────────────────────────────
# ingest.py tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFilterBrand:
    def test_keeps_brand_tweets(self, sample_raw_df):
        filtered = filter_brand(sample_raw_df, brand="SpotifyCares")
        # Tweets 1,2,3,4 are in-scope; 5,6 are Amazon-only
        assert "5" not in filtered["tweet_id"].values
        assert "6" not in filtered["tweet_id"].values

    def test_keeps_customer_tweets_in_brand_threads(self, sample_raw_df):
        filtered = filter_brand(sample_raw_df, brand="SpotifyCares")
        assert "1" in filtered["tweet_id"].values
        assert "3" in filtered["tweet_id"].values

    def test_empty_brand_returns_empty(self, sample_raw_df):
        filtered = filter_brand(sample_raw_df, brand="NonExistentBrand")
        assert len(filtered) == 0


class TestSubsample:
    def test_respects_n_limit(self, sample_raw_df):
        filtered = filter_brand(sample_raw_df, brand="SpotifyCares")
        result = subsample(filtered, n=1)
        # We sample 1 customer message + its brand reply = at most 2 rows
        assert len(result) <= 3

    def test_no_subsample_if_already_small(self, sample_raw_df):
        filtered = filter_brand(sample_raw_df, brand="SpotifyCares")
        result = subsample(filtered, n=10_000)
        assert len(result) == len(filtered)


class TestBuildThreads:
    def test_builds_correct_thread_count(self, sample_raw_df):
        filtered = filter_brand(sample_raw_df, brand="SpotifyCares")
        threads = build_threads(filtered)
        # We have 2 customer messages (tweet_id 1 and 3)
        assert len(threads) == 2

    def test_attaches_brand_reply(self, sample_raw_df):
        filtered = filter_brand(sample_raw_df, brand="SpotifyCares")
        threads = build_threads(filtered)
        # Both should have brand replies
        assert threads["has_brand_reply"].all()

    def test_schema(self, sample_raw_df):
        filtered = filter_brand(sample_raw_df, brand="SpotifyCares")
        threads = build_threads(filtered)
        expected_cols = {"tweet_id", "customer_message", "brand_reply", "thread_context", "has_brand_reply"}
        assert expected_cols.issubset(set(threads.columns))
