"""
metrics.py — Evaluation harness for the SpotifyCare AI agent.

Computes:
  Intent Classification:
    - Accuracy, Macro-F1, Per-class F1, Confusion Matrix

  Reply Quality:
    - ROUGE-L (lexical overlap)
    - BERTScore (semantic similarity)

  Escalation Decision:
    - Precision, Recall, F1 on escalation label

  Baselines compared:
    - Trivial: always predict most-common intent / mode reply / always "auto"
    - Simple:  TF-IDF nearest-neighbour retrieval

Usage:
    python -m src.eval.metrics
    python -m src.eval.metrics --golden data/golden_set.csv
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

GOLDEN_SET_FILE = ROOT / "data" / "golden_set.csv"
RESULTS_FILE = ROOT / "eval" / "results.json"


# ─────────────────────────────────────────────────────────────────────────────
# Intent classification metrics
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_intent(y_true: list[str], y_pred: list[str]) -> dict:
    from sklearn.metrics import (
        accuracy_score, f1_score, confusion_matrix, classification_report
    )
    labels = sorted(set(y_true) | set(y_pred))
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels,
                                   zero_division=0, output_dict=True)
    log.info(f"Intent Accuracy:  {acc:.4f}")
    log.info(f"Intent Macro-F1:  {macro_f1:.4f}")
    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class_f1": {k: round(v["f1-score"], 4)
                         for k, v in report.items()
                         if isinstance(v, dict) and "f1-score" in v},
        "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reply quality metrics
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_reply_rouge(predictions: list[str], references: list[str]) -> dict:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(ref, pred)["rougeL"].fmeasure
              for pred, ref in zip(predictions, references)
              if pred and ref]
    mean = float(np.mean(scores)) if scores else 0.0
    log.info(f"ROUGE-L (mean):   {mean:.4f}")
    return {"rouge_l_mean": round(mean, 4), "n_scored": len(scores)}


def evaluate_reply_bertscore(predictions: list[str], references: list[str]) -> dict:
    try:
        from bert_score import score as bert_score
        valid = [(p, r) for p, r in zip(predictions, references) if p and r]
        if not valid:
            return {"bertscore_f1_mean": 0.0, "n_scored": 0}
        preds_v, refs_v = zip(*valid)
        P, R, F1 = bert_score(list(preds_v), list(refs_v), lang="en",
                               model_type="distilbert-base-uncased", verbose=False)
        mean_f1 = float(F1.mean())
        log.info(f"BERTScore F1:     {mean_f1:.4f}")
        return {"bertscore_f1_mean": round(mean_f1, 4), "n_scored": len(valid)}
    except Exception as e:
        log.warning(f"BERTScore failed: {e}")
        return {"bertscore_f1_mean": None, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Escalation metrics
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_escalation(y_true: list[int], y_pred: list[int]) -> dict:
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    log.info(f"Escalation Acc:   {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Baselines
# ─────────────────────────────────────────────────────────────────────────────

def trivial_baseline(golden: pd.DataFrame) -> dict:
    """
    Trivial baseline:
      - Intent: always predict the most common class
      - Reply:  always return the modal historical reply (empty if not available)
      - Escalation: always "auto"
    """
    log.info("Running trivial baseline …")
    mode_intent = golden["true_intent"].mode()[0]
    y_true_intent = golden["true_intent"].tolist()
    y_pred_intent = [mode_intent] * len(golden)

    true_replies = golden.get("true_brand_reply", pd.Series([""] * len(golden))).fillna("").tolist()
    # Use the most common brand reply as the "mode" reply
    mode_reply = golden["true_brand_reply"].dropna().mode()
    mode_reply_text = mode_reply[0] if len(mode_reply) > 0 else ""
    pred_replies = [mode_reply_text] * len(golden)

    true_esc = golden["escalate_label"].astype(int).tolist()
    pred_esc = [0] * len(golden)   # always "auto"

    return {
        "name": "Trivial (mode)",
        "intent": evaluate_intent(y_true_intent, y_pred_intent),
        "reply_rouge": evaluate_reply_rouge(pred_replies, true_replies),
        "escalation": evaluate_escalation(true_esc, pred_esc),
    }


def tfidf_baseline(golden: pd.DataFrame) -> dict:
    """
    Simple baseline:
      - Intent: TF-IDF + nearest-neighbour (self-retrieval on golden set)
      - Reply:  return verbatim nearest historical brand reply
      - Escalation: rule-only (no LLM layer)
    """
    log.info("Running TF-IDF baseline …")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import LabelEncoder

    messages = golden["message"].fillna("").tolist()
    true_intents = golden["true_intent"].tolist()

    # TF-IDF vectors
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=10_000)
    X = vec.fit_transform(messages)

    # Nearest neighbour (leave-one-out)
    pred_intents = []
    pred_replies = []
    for i in range(len(messages)):
        sim = cosine_similarity(X[i], X).flatten()
        sim[i] = -1  # exclude self
        nn = int(np.argmax(sim))
        pred_intents.append(true_intents[nn])
        pred_replies.append(golden.iloc[nn].get("true_brand_reply", ""))

    true_replies = golden.get("true_brand_reply", pd.Series([""] * len(golden))).fillna("").tolist()
    true_esc = golden["escalate_label"].astype(int).tolist()

    # Rule-only escalation
    from src.agent.escalation import _apply_hard_rules
    pred_esc = []
    for _, row in golden.iterrows():
        result = _apply_hard_rules(
            message=str(row["message"]),
            intent=str(row["true_intent"]),
            confidence=0.8,  # assume high confidence for baseline
        )
        pred_esc.append(1 if result else 0)

    return {
        "name": "TF-IDF nearest neighbour",
        "intent": evaluate_intent(true_intents, pred_intents),
        "reply_rouge": evaluate_reply_rouge(pred_replies, true_replies),
        "escalation": evaluate_escalation(true_esc, pred_esc),
    }


def agent_evaluation(golden: pd.DataFrame, sample_n: int = 50) -> dict:
    """Run the full AI agent on the golden set and compute metrics."""
    log.info(f"Running AI agent evaluation on {sample_n} examples …")
    # Subsample for cost efficiency; set --eval-sample 200 to run on full set
    if sample_n < len(golden):
        golden = golden.sample(n=sample_n, random_state=42).reset_index(drop=True)
        log.info(f"  Sampled {len(golden)} examples (use --eval-sample {len(golden)} to change).")
    from src.agent.classifier import IntentClassifier
    from src.agent.reply_drafter import ReplyDrafter
    from src.agent.escalation import EscalationDecider

    clf = IntentClassifier()
    drafter = ReplyDrafter()
    drafter.load_index()
    decider = EscalationDecider()

    pred_intents, pred_replies, pred_esc = [], [], []
    from tqdm import tqdm

    for _, row in tqdm(golden.iterrows(), total=len(golden), desc="Agent"):
        msg = str(row["message"])
        ctx = str(row.get("thread_context", ""))

        cls_result = clf.classify(msg, ctx)
        intent = cls_result["intent"]
        conf = cls_result["confidence"]

        reply_result = drafter.draft(msg, intent=intent, thread_context=ctx)
        draft = reply_result["draft_reply"]

        esc_result = decider.decide(msg, intent=intent, confidence=conf, thread_context=ctx)

        pred_intents.append(intent)
        pred_replies.append(draft)
        pred_esc.append(1 if esc_result["decision"] == "escalate" else 0)

    true_intents = golden["true_intent"].tolist()
    true_replies = golden.get("true_brand_reply", pd.Series([""] * len(golden))).fillna("").tolist()
    true_esc = golden["escalate_label"].astype(int).tolist()

    return {
        "name": "AI Agent (Gemini Flash Lite + RAG)",
        "intent": evaluate_intent(true_intents, pred_intents),
        "reply_rouge": evaluate_reply_rouge(pred_replies, true_replies),
        "reply_bertscore": evaluate_reply_bertscore(pred_replies, true_replies),
        "escalation": evaluate_escalation(true_esc, pred_esc),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default=str(GOLDEN_SET_FILE))
    parser.add_argument("--skip-agent", action="store_true",
                        help="Only run baselines (no API calls)")
    parser.add_argument("--skip-bertscore", action="store_true")
    parser.add_argument(
        "--eval-sample", type=int, default=50,
        help="Number of golden-set examples to run the agent on (default 50 ≈$0.05). Use 200 for full eval."
    )
    args = parser.parse_args()

    golden_path = Path(args.golden)
    if not golden_path.exists():
        raise FileNotFoundError(
            f"Golden set not found at {golden_path}. "
            "Run `python -m src.eval.build_golden_set` first."
        )

    golden = pd.read_csv(golden_path)
    log.info(f"Loaded golden set: {len(golden)} examples.")

    results = {}
    results["trivial_baseline"] = trivial_baseline(golden)
    results["tfidf_baseline"] = tfidf_baseline(golden)

    if not args.skip_agent:
        results["agent"] = agent_evaluation(golden, sample_n=args.eval_sample)

    # Save results
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\n✅ Results saved to {RESULTS_FILE}")

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'System':<35} {'Intent Acc':>10} {'Intent F1':>10} {'ROUGE-L':>9} {'Esc F1':>8}")
    print("=" * 70)
    for key, res in results.items():
        name = res.get("name", key)[:34]
        iacc = res.get("intent", {}).get("accuracy", "-")
        if1  = res.get("intent", {}).get("macro_f1", "-")
        rl   = res.get("reply_rouge", {}).get("rouge_l_mean", "-")
        ef1  = res.get("escalation", {}).get("f1", "-")
        print(f"{name:<35} {str(iacc):>10} {str(if1):>10} {str(rl):>9} {str(ef1):>8}")
    print("=" * 70)


if __name__ == "__main__":
    main()
