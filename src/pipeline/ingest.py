"""
ingest.py — Load, filter, and subsample the Kaggle Twitter dataset.

Dataset: twcs.csv  (Customer Support on Twitter, thoughtvector/customer-support-on-twitter)
Brand:   SpotifyCare  (@SpotifyCares)

Usage:
    python -m src.pipeline.ingest                      # default: 10k sample
    python -m src.pipeline.ingest --sample 5000        # custom sample size
    python -m src.pipeline.ingest --csv path/to/twcs.csv
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = ROOT / "data" / "raw" / "twcs.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
BRAND_HANDLE = "SpotifyCares"   # exact author_id string in the dataset

# All handles that are NOT customers (support agents / brand accounts)
BRAND_HANDLES = {
    "SpotifyCares",
}


def load_raw(csv_path: Path) -> pd.DataFrame:
    """Load the raw twcs.csv. Only relevant columns are kept."""
    log.info(f"Loading CSV from {csv_path} …")
    df = pd.read_csv(
        csv_path,
        usecols=["tweet_id", "author_id", "inbound", "created_at",
                  "text", "response_tweet_id", "in_response_to_tweet_id"],
        dtype={"tweet_id": str, "in_response_to_tweet_id": str,
               "response_tweet_id": str},
        low_memory=False,
    )
    log.info(f"  Loaded {len(df):,} rows total.")
    return df


def filter_brand(df: pd.DataFrame, brand: str = BRAND_HANDLE) -> pd.DataFrame:
    """
    Keep only tweets that are either:
      - FROM the brand (outbound support replies), or
      - IN A THREAD that the brand replied to (customer inbound messages)

    Strategy:
      1. Find all tweet_ids that the brand responded to.
      2. Find all tweet_ids the brand authored.
      3. Keep rows whose tweet_id appears in either set.
    """
    log.info(f"Filtering for brand: @{brand} …")

    # Tweets authored by the brand
    brand_tweets = df[df["author_id"] == brand]["tweet_id"].tolist()
    brand_tweet_set = set(brand_tweets)

    # Tweets the brand replied to (customer messages)
    brand_replies = df[df["author_id"] == brand]["in_response_to_tweet_id"].dropna().tolist()
    brand_reply_set = set(brand_replies)

    # All relevant tweet_ids
    relevant_ids = brand_tweet_set | brand_reply_set

    filtered = df[df["tweet_id"].isin(relevant_ids)].copy()
    log.info(f"  {len(filtered):,} rows after brand filter "
             f"({len(brand_tweet_set):,} brand tweets, "
             f"{len(brand_reply_set):,} customer tweets in-scope).")
    return filtered


def subsample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """
    Subsample to at most n rows. We subsample *customer* messages first
    (inbound=True) so we don't break thread structure, then pull in all
    brand replies to those messages.
    """
    if len(df) <= n:
        log.info(f"  Dataset already ≤ {n} rows; no subsampling needed.")
        return df

    # Separate inbound (customer) and outbound (brand)
    inbound = df[df["inbound"] == True]
    outbound = df[df["inbound"] == False]

    # Sample customer messages
    sampled_inbound = inbound.sample(n=min(n, len(inbound)), random_state=seed)
    sampled_ids = set(sampled_inbound["tweet_id"].tolist())

    # Pull brand replies to sampled messages
    relevant_outbound = outbound[
        outbound["in_response_to_tweet_id"].isin(sampled_ids)
    ]

    result = pd.concat([sampled_inbound, relevant_outbound]).drop_duplicates("tweet_id")
    log.info(f"  Subsampled to {len(result):,} rows "
             f"({len(sampled_inbound):,} customer, {len(relevant_outbound):,} brand).")
    return result


def build_threads(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct conversation threads.

    For each customer message, attach:
      - The brand's direct reply (if it exists in df)
      - The preceding customer context (up to 2 turns)

    Returns a flat DataFrame of (customer_msg, brand_reply, thread_context) triples.
    """
    log.info("Building conversation threads …")

    tweet_by_id = df.set_index("tweet_id")["text"].to_dict()
    author_by_id = df.set_index("tweet_id")["author_id"].to_dict()

    # Index brand replies by which tweet they're responding to
    brand_replies = df[df["inbound"] == False].copy()
    reply_map: dict[str, list[str]] = {}
    for _, row in brand_replies.iterrows():
        parent = str(row["in_response_to_tweet_id"])
        if parent not in reply_map:
            reply_map[parent] = []
        reply_map[parent].append(row["tweet_id"])

    records = []
    customer_msgs = df[df["inbound"] == True]

    for _, row in customer_msgs.iterrows():
        tid = str(row["tweet_id"])
        customer_text = str(row["text"])

        # Direct brand reply
        brand_reply_ids = reply_map.get(tid, [])
        brand_reply_text = tweet_by_id.get(brand_reply_ids[0], "") if brand_reply_ids else ""

        # Build context: walk up the reply chain (at most 2 prior turns)
        context_parts = []
        current_parent = str(row.get("in_response_to_tweet_id", ""))
        for _ in range(2):
            if not current_parent or current_parent == "nan":
                break
            parent_text = tweet_by_id.get(current_parent, "")
            parent_author = author_by_id.get(current_parent, "")
            if parent_text:
                role = "Brand" if parent_author in BRAND_HANDLES else "Customer"
                context_parts.insert(0, f"[{role}]: {parent_text}")
            parent_row = df[df["tweet_id"] == current_parent]
            if parent_row.empty:
                break
            current_parent = str(parent_row.iloc[0].get("in_response_to_tweet_id", ""))

        thread_context = "\n".join(context_parts)

        records.append({
            "tweet_id": tid,
            "customer_message": customer_text,
            "brand_reply": brand_reply_text,
            "thread_context": thread_context,
            "created_at": row.get("created_at", ""),
            "has_brand_reply": bool(brand_reply_text),
        })

    threads = pd.DataFrame(records)
    log.info(f"  Built {len(threads):,} threads; "
             f"{threads['has_brand_reply'].sum():,} have brand replies.")
    return threads


def main():
    parser = argparse.ArgumentParser(description="Ingest Spotify support tweets.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="Path to twcs.csv")
    parser.add_argument("--sample", type=int, default=10_000,
                        help="Max customer messages to sample (default: 10000)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        log.error(f"CSV not found at {csv_path}. "
                  f"Download it from https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter "
                  f"and place twcs.csv in data/raw/")
        sys.exit(1)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw(csv_path)
    df = filter_brand(df)
    df = subsample(df, n=args.sample, seed=args.seed)
    threads = build_threads(df)

    # Save
    out_raw = PROCESSED_DIR / "spotify_raw.csv"
    out_threads = PROCESSED_DIR / "threads.csv"
    df.to_csv(out_raw, index=False)
    threads.to_csv(out_threads, index=False)
    log.info(f"Saved filtered raw to   {out_raw}")
    log.info(f"Saved threads to        {out_threads}")


if __name__ == "__main__":
    main()
