"""
build_golden_set.py — Generate the 200-example stratified golden evaluation set.

Cost-optimised design:
  - Classification: classify a 2k random sample (not the full dataset) to get
    intent buckets for stratified sampling. Uses cache so it only runs once.
  - Annotation: escalation label via hard rules only (no LLM layer) — free.
  - No ideal_reply_summary generation — that was the most expensive step ($0.10)
    and the golden set works fine without it for intent/escalation evaluation.

Total API cost for this step: ~$0.01 (just the 200 classifier calls for annotation).

Usage:
    python -m src.eval.build_golden_set
    python -m src.eval.build_golden_set --n 200 --classify-sample 2000
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
CLEAN_FILE = PROCESSED_DIR / "clean.csv"
OUTPUT_FILE = ROOT / "data" / "golden_set.csv"

# Import after path setup
import sys
sys.path.insert(0, str(ROOT))
from src.agent.classifier import INTENTS, IntentClassifier
from src.agent.escalation import _apply_hard_rules  # rules-only, no LLM cost

DEFAULT_N = 200
DEFAULT_PER_INTENT = 23
DEFAULT_CLASSIFY_SAMPLE = 2000  # classify only 2k rows for bucketing (not full dataset)



def classify_sample(df: pd.DataFrame, sample_n: int, cache_path: Path) -> pd.DataFrame:
    """
    Classify a random sample of df to discover intent buckets.
    Full-dataset classification is expensive; we only need ~2k rows
    to get enough examples per intent for stratified sampling.
    Uses a cache file so it only runs once.
    """
    if cache_path.exists():
        log.info(f"  Loading cached predictions from {cache_path} …")
        preds = pd.read_csv(cache_path, dtype={"tweet_id": str})
        # Merge predictions back by tweet_id
        df = df.merge(preds[["tweet_id", "predicted_intent", "predicted_confidence"]],
                      on="tweet_id", how="left")
        labeled = df["predicted_intent"].notna()
        log.info(f"  Cache covers {labeled.sum():,} / {len(df):,} rows.")
        if labeled.sum() >= sample_n:
            return df
        log.warning("  Cache too small; re-classifying sample …")

    # Sample rows to classify
    to_classify = df.sample(n=min(sample_n, len(df)), random_state=42)
    log.info(f"  Classifying {len(to_classify):,} sampled messages (cost: ~${len(to_classify)*0.000001:.3f}) …")

    clf = IntentClassifier()
    intents, confidences = [], []
    for _, row in tqdm(to_classify.iterrows(), total=len(to_classify), desc="Classifying"):
        result = clf.classify(
            message=str(row["customer_message_clean"]),
            thread_context=str(row.get("thread_context_clean", "")),
        )
        intents.append(result["intent"])
        confidences.append(result["confidence"])

    to_classify = to_classify.copy()
    to_classify["predicted_intent"] = intents
    to_classify["predicted_confidence"] = confidences

    # Save cache (only the classified subset)
    to_classify[["tweet_id", "predicted_intent", "predicted_confidence"]].to_csv(
        cache_path, index=False
    )
    log.info(f"  Saved prediction cache to {cache_path}")

    # Merge back
    df = df.merge(
        to_classify[["tweet_id", "predicted_intent", "predicted_confidence"]],
        on="tweet_id", how="left"
    )
    return df


def stratified_sample(df: pd.DataFrame, per_intent: int, seed: int = 42) -> pd.DataFrame:
    """Sample up to per_intent examples from each intent bucket."""
    samples = []
    for intent in INTENTS:
        bucket = df[df["predicted_intent"] == intent]
        n = min(per_intent, len(bucket))
        if n == 0:
            log.warning(f"  No examples for intent '{intent}'!")
            continue
        samples.append(bucket.sample(n=n, random_state=seed))
        log.info(f"  {intent}: sampled {n} / {len(bucket)}")
    return pd.concat(samples).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--per-intent", type=int, default=DEFAULT_PER_INTENT)
    parser.add_argument("--classify-sample", type=int, default=DEFAULT_CLASSIFY_SAMPLE,
                        help="How many rows to classify for bucketing (default 2000, ~$0.002)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not CLEAN_FILE.exists():
        raise FileNotFoundError(f"{CLEAN_FILE} not found. Run the pipeline first.")

    log.info(f"Loading {CLEAN_FILE} …")
    df = pd.read_csv(CLEAN_FILE)
    log.info(f"  {len(df):,} rows loaded.")

    # Step 1: Classify a sample for intent bucketing (cached)
    cache_path = PROCESSED_DIR / "predictions_cache.csv"
    df = classify_sample(df, sample_n=args.classify_sample, cache_path=cache_path)
    # Only keep rows that got classified
    df = df[df["predicted_intent"].notna()].copy()

    # Step 2: Stratified sample
    sampled = stratified_sample(df, per_intent=args.per_intent, seed=args.seed)
    sampled = sampled.head(args.n)
    log.info(f"Total sampled: {len(sampled)} examples.")

    # Step 3: Escalation labels via hard rules only (FREE — no LLM calls)
    escalate_labels = []
    escalate_reasons = []

    for _, row in tqdm(sampled.iterrows(), total=len(sampled), desc="Annotating (rules)"):
        msg = str(row["customer_message_clean"])
        intent = str(row["predicted_intent"])
        conf = float(row.get("predicted_confidence", 0.8))

        # Hard rules only — deterministic, free
        rule_result = _apply_hard_rules(message=msg, intent=intent, confidence=conf)
        if rule_result:
            escalate_labels.append(1)
            escalate_reasons.append(rule_result["reason"])
        else:
            escalate_labels.append(0)
            escalate_reasons.append("No hard-rule triggers; auto-handle.")

    sampled["escalate_label"] = escalate_labels
    sampled["escalate_reason"] = escalate_reasons

    # Step 4: Final column selection
    golden = sampled[[
        "tweet_id",
        "customer_message_clean",
        "thread_context_clean",
        "brand_reply_clean",
        "predicted_intent",
        "predicted_confidence",
        "escalate_label",
        "escalate_reason",
    ]].rename(columns={
        "customer_message_clean": "message",
        "thread_context_clean": "thread_context",
        "brand_reply_clean": "true_brand_reply",
        "predicted_intent": "true_intent",
        "predicted_confidence": "classifier_confidence",
    })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    golden.to_csv(OUTPUT_FILE, index=False)
    log.info(f"\n✅ Golden set saved to {OUTPUT_FILE}  ({len(golden)} rows)")
    log.info(f"   → IMPORTANT: Manually review and correct 'true_intent' and 'escalate_label' before evaluation.")

    # Print distribution
    print("\nIntent distribution in golden set:")
    print(golden["true_intent"].value_counts().to_string())
    print(f"\nEscalation rate: {golden['escalate_label'].mean():.1%}")


if __name__ == "__main__":
    main()
