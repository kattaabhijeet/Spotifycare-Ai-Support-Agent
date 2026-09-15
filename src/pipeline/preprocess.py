"""
preprocess.py — Clean tweets and produce the final feature dataset.

Input:  data/processed/threads.csv   (output of ingest.py)
Output: data/processed/clean.csv     (ready for embedding + agent)

Cleaning steps:
  1. Strip @mentions (but keep the message meaning)
  2. Normalise URLs → <URL>
  3. Remove HTML entities
  4. Strip leading/trailing whitespace + collapse multiple spaces
  5. Drop duplicate / near-empty messages
"""

import logging
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
INPUT_FILE = PROCESSED_DIR / "threads.csv"
OUTPUT_FILE = PROCESSED_DIR / "clean.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Cleaning helpers
# ─────────────────────────────────────────────────────────────────────────────

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HTML_ENTITY_RE = re.compile(r"&\w+;|&#\d+;")
_MULTI_SPACE_RE = re.compile(r" {2,}")
_HASHTAG_RE = re.compile(r"#(\w+)")   # keep the word, drop the #


def clean_text(text: str) -> str:
    """Apply all cleaning transformations to a single tweet string."""
    if not isinstance(text, str):
        return ""
    text = _HTML_ENTITY_RE.sub(" ", text)
    text = _MENTION_RE.sub("", text)          # strip @mentions entirely
    text = _URL_RE.sub("<URL>", text)
    text = _HASHTAG_RE.sub(r"\1", text)       # #Spotify → Spotify
    text = text.replace("\n", " ").replace("\r", " ")
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean all text columns and drop bad rows."""
    df = df.copy()

    df["customer_message_clean"] = df["customer_message"].apply(clean_text)
    df["brand_reply_clean"] = df["brand_reply"].apply(clean_text)
    df["thread_context_clean"] = df["thread_context"].apply(clean_text)

    # Drop rows where the customer message is essentially empty after cleaning
    min_len = 5
    before = len(df)
    df = df[df["customer_message_clean"].str.len() >= min_len]
    log.info(f"  Dropped {before - len(df):,} rows with near-empty customer messages.")

    # Drop rows that are just retweet artifacts (starts with "RT")
    before = len(df)
    df = df[~df["customer_message_clean"].str.startswith("RT")]
    log.info(f"  Dropped {before - len(df):,} retweet rows.")

    # Drop duplicates on cleaned message
    before = len(df)
    df = df.drop_duplicates(subset=["customer_message_clean"])
    log.info(f"  Dropped {before - len(df):,} duplicate customer messages.")

    return df.reset_index(drop=True)


def add_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Add useful derived columns for the agent."""
    df = df.copy()
    df["message_len"] = df["customer_message_clean"].str.len()
    df["has_url"] = df["customer_message_clean"].str.contains("<URL>")
    df["has_reply"] = df["brand_reply_clean"].str.len() > 5
    return df


def main():
    log.info(f"Loading threads from {INPUT_FILE} …")
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"{INPUT_FILE} not found. Run `python -m src.pipeline.ingest` first."
        )

    df = pd.read_csv(INPUT_FILE, dtype=str)
    log.info(f"  Loaded {len(df):,} threads.")

    df = clean_dataframe(df)
    df = add_metadata(df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    log.info(f"Saved clean dataset to {OUTPUT_FILE}  ({len(df):,} rows).")


if __name__ == "__main__":
    main()
