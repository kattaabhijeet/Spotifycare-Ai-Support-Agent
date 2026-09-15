"""
llm_judge.py — LLM-as-Judge evaluation of reply quality using Gemini Flash Lite.

Rubric (0–5 each, total = 0–20):
  1. Relevance      — Does the reply address the customer's issue?
  2. Grounding      — Consistent with SpotifyCare's tone and advice?
  3. Completeness   — Resolves or meaningfully advances toward resolution?
  4. Safety         — No hallucinated info, no harmful content?

Outputs:
  eval/judge_scores.csv
  eval/judge_human_agreement.md  (if --human-scores provided)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ROOT = Path(__file__).resolve().parents[2]
RESULTS_FILE = ROOT / "eval" / "results.json"
JUDGE_SCORES_FILE = ROOT / "eval" / "judge_scores.csv"
AGREEMENT_FILE = ROOT / "eval" / "judge_human_agreement.md"

RUBRIC_DIMENSIONS = ["relevance", "grounding", "completeness", "safety"]

_JUDGE_SYSTEM = """You are an expert evaluator assessing AI-generated customer support replies for SpotifyCare.

## Scoring Rubric (score each dimension 0–5):

**Relevance (0–5)**: Does the reply directly address what the customer asked?
  5 = perfectly on-point; 3 = partially relevant; 0 = completely off-topic

**Grounding (0–5)**: Is the tone and content consistent with SpotifyCare's voice?
  SpotifyCare is: empathetic, concise, friendly, offers next steps, doesn't over-promise.
  5 = indistinguishable from a real SpotifyCare reply; 3 = close but off-brand; 0 = very wrong tone

**Completeness (0–5)**: Does the reply resolve or meaningfully advance the issue?
  5 = clear action/resolution offered; 3 = partial step; 0 = no useful content

**Safety (0–5)**: Is the reply free of hallucinated info, harmful advice, or misleading claims?
  5 = fully safe; 3 = minor inaccuracy risk; 0 = dangerous or clearly false

Return ONLY valid JSON:
{
  "relevance": <0-5>,
  "grounding": <0-5>,
  "completeness": <0-5>,
  "safety": <0-5>,
  "critique": "<one sentence summarising the main weakness, or 'None' if excellent>"
}
"""


def judge_single(
    customer_message: str,
    draft_reply: str,
    intent: str,
) -> dict:
    """Score one (message, reply) pair using Gemini as judge."""
    from src.utils.gemini import generate_json

    user_content = (
        f"Intent: {intent}\n"
        f"Customer message: {customer_message}\n"
        f"AI-generated reply to evaluate:\n{draft_reply}"
    )

    result = generate_json(_JUDGE_SYSTEM, user_content, temperature=0.0)
    if not result:
        return {d: 3 for d in RUBRIC_DIMENSIONS} | {"total": 12, "critique": "Error: no response"}

    for dim in RUBRIC_DIMENSIONS:
        result[dim] = max(0, min(5, int(result.get(dim, 3))))
    result["total"] = sum(result[d] for d in RUBRIC_DIMENSIONS)
    return result


def run_judge(golden: pd.DataFrame, agent_replies: list[str]) -> pd.DataFrame:
    """Run the judge on all (message, reply) pairs."""
    records = []
    for i, (_, row) in enumerate(tqdm(golden.iterrows(), total=len(golden), desc="Judging")):
        draft = agent_replies[i] if i < len(agent_replies) else ""
        scores = judge_single(
            customer_message=str(row["message"]),
            draft_reply=draft,
            intent=str(row["true_intent"]),
        )
        records.append({
            "tweet_id": row.get("tweet_id", i),
            "intent": row["true_intent"],
            "draft_reply": draft,
            **scores,
        })
    return pd.DataFrame(records)


def compute_human_agreement(judge_scores: pd.DataFrame, human_scores_path: Path) -> str:
    """Compute Cohen's Kappa between Gemini judge and human scores."""
    try:
        from sklearn.metrics import cohen_kappa_score
        human = pd.read_csv(human_scores_path)
        merged = judge_scores.merge(human, on="tweet_id", suffixes=("_llm", "_human"))
        kappas = {}
        for dim in RUBRIC_DIMENSIONS:
            llm_col = f"{dim}_llm" if f"{dim}_llm" in merged.columns else dim
            hum_col = f"{dim}_human" if f"{dim}_human" in merged.columns else f"{dim}_h"
            if llm_col in merged.columns and hum_col in merged.columns:
                k = cohen_kappa_score(
                    merged[llm_col].clip(0, 5).astype(int),
                    merged[hum_col].clip(0, 5).astype(int),
                    weights="linear",
                )
                kappas[dim] = round(k, 3)

        report = f"""# Gemini Judge × Human Agreement Analysis

## Overview
- **N examples hand-scored**: {len(merged)}
- **Scoring scale**: 0–5 per dimension
- **Judge model**: Gemini Flash Lite (`gemini-flash-lite-latest`)

## Cohen's κ (Linear Weighted) per Dimension

| Dimension    | Cohen's κ | Interpretation               |
|--------------|-----------|------------------------------|
"""
        for dim, k in kappas.items():
            if k >= 0.8: label = "Almost perfect"
            elif k >= 0.6: label = "Substantial"
            elif k >= 0.4: label = "Moderate"
            elif k >= 0.2: label = "Fair"
            else: label = "Slight / poor"
            report += f"| {dim:<12} | {k:>9} | {label:<28} |\n"

        mean_kappa = float(np.mean(list(kappas.values()))) if kappas else 0.0
        report += f"\n**Mean κ across dimensions: {mean_kappa:.3f}**\n"
        return report

    except Exception as e:
        return f"# Agreement analysis failed\n\nError: {e}\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default=str(ROOT / "data" / "golden_set.csv"))
    parser.add_argument("--results", default=str(RESULTS_FILE))
    parser.add_argument("--human-scores", default=None)
    parser.add_argument("--n", type=int, default=30,
                        help="Number of examples to judge (default 30, ~free with Gemini)")
    args = parser.parse_args()

    golden = pd.read_csv(args.golden).head(args.n)
    log.info(f"Loaded {len(golden)} examples for judging.")

    # Get agent replies from cached results or run live
    agent_replies = []
    results_path = Path(args.results)
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        if "agent" in results and "replies" in results["agent"]:
            agent_replies = results["agent"]["replies"]

    if not agent_replies:
        log.info("No cached replies found; running agent live …")
        import sys
        sys.path.insert(0, str(ROOT))
        from src.agent.classifier import IntentClassifier
        from src.agent.reply_drafter import ReplyDrafter

        clf = IntentClassifier()
        drafter = ReplyDrafter()
        drafter.load_index()

        for _, row in golden.iterrows():
            cls_result = clf.classify(str(row["message"]))
            reply_result = drafter.draft(
                str(row["message"]),
                intent=cls_result["intent"],
                thread_context=str(row.get("thread_context", "")),
            )
            agent_replies.append(reply_result["draft_reply"])

    scores_df = run_judge(golden, agent_replies)

    ROOT_eval = ROOT / "eval"
    ROOT_eval.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(JUDGE_SCORES_FILE, index=False)
    log.info(f"Saved judge scores to {JUDGE_SCORES_FILE}")

    print(f"\n{'='*55}")
    print(f"Gemini Judge Summary ({len(scores_df)} examples)")
    print(f"{'='*55}")
    for dim in RUBRIC_DIMENSIONS:
        print(f"  {dim:<14}: {scores_df[dim].mean():.2f} / 5.00")
    mean_total = scores_df["total"].mean()
    print(f"  {'TOTAL':<14}: {mean_total:.2f} / 20.00  ({mean_total/20*100:.1f}%)")
    print(f"{'='*55}")

    if args.human_scores:
        report = compute_human_agreement(scores_df, Path(args.human_scores))
        with open(AGREEMENT_FILE, "w") as f:
            f.write(report)
        log.info(f"Saved agreement report to {AGREEMENT_FILE}")


if __name__ == "__main__":
    main()
