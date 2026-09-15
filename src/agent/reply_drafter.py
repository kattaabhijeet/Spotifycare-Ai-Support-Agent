"""
reply_drafter.py — RAG-based reply generator for SpotifyCare.

Pipeline:
  1. Embed incoming message (local all-MiniLM-L6-v2, FREE)
  2. Retrieve top-K most similar historical brand replies (cosine similarity)
  3. Prompt Gemini Flash Lite to draft a reply in SpotifyCare's brand voice
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
EMBED_FILE = PROCESSED_DIR / "embeddings.npy"
INDEX_FILE = PROCESSED_DIR / "embed_index.csv"
CLEAN_FILE = PROCESSED_DIR / "clean.csv"

TOP_K = 5
LOCAL_EMBED_MODEL = "all-MiniLM-L6-v2"

# ─────────────────────────────────────────────────────────────────────────────
# Brand voice + system prompt
# ─────────────────────────────────────────────────────────────────────────────
BRAND_VOICE = """
SpotifyCare tone of voice:
- Friendly, empathetic, and concise (Twitter character limits in mind).
- Acknowledge the frustration before jumping to solutions.
- Use "we" to represent the team; never blame the customer.
- Offer a concrete next step (troubleshooting link, DM for details, etc.).
- Avoid jargon; keep it conversational.
- Never promise refunds or account changes in a public tweet — ask to DM.
- End with a question or invitation to follow up if not fully resolved.
""".strip()

_SYSTEM_PROMPT = f"""You are a SpotifyCare support agent writing replies on Twitter.

{BRAND_VOICE}

You will receive:
  1. The customer's message and intent.
  2. Up to {TOP_K} real examples of how SpotifyCare has handled similar issues.

Your task:
  - Draft ONE new reply (max 280 characters) that addresses the customer's specific issue.
  - Ground your reply in the style and content of the provided examples.
  - Do NOT copy an example verbatim — synthesise a fresh, personalised response.
  - Output valid JSON only: {{"draft_reply": "<your reply here>"}}
"""


def cosine_similarity_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = query / (np.linalg.norm(query) + 1e-9)
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
    return matrix_norm @ query_norm


class ReplyDrafter:
    """RAG-based reply drafter. Call load_index() before draft()."""

    def __init__(self, top_k: int = TOP_K):
        self.top_k = top_k
        self._embeddings: np.ndarray | None = None
        self._df: pd.DataFrame | None = None
        self._embed_model = None   # lazy-load sentence-transformers

    def load_index(self) -> None:
        if not EMBED_FILE.exists():
            raise FileNotFoundError(
                f"Embeddings not found at {EMBED_FILE}. "
                "Run `python -m src.pipeline.embed` first."
            )
        self._embeddings = np.load(EMBED_FILE)
        self._df = pd.read_csv(CLEAN_FILE)
        log.info(f"Loaded {len(self._df):,} examples, embeddings shape={self._embeddings.shape}.")

    def _embed_query(self, text: str) -> np.ndarray:
        """Embed a single query using the local sentence-transformers model (FREE)."""
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer(LOCAL_EMBED_MODEL)
        vec = self._embed_model.encode([text[:500]], normalize_embeddings=True)
        return vec[0].astype(np.float32)

    def retrieve(self, message: str, intent: str | None = None) -> list[dict[str, str]]:
        """Retrieve top-K most similar historical brand replies."""
        if self._embeddings is None or self._df is None:
            raise RuntimeError("Index not loaded. Call load_index() first.")

        query_vec = self._embed_query(message)
        sims = cosine_similarity_matrix(query_vec, self._embeddings)

        has_reply = self._df["has_reply"].astype(str).str.lower() == "true"

        if intent and "predicted_intent" in self._df.columns:
            same_intent = self._df["predicted_intent"] == intent
            sims = sims + (same_intent.values * 0.10)

        sims[~has_reply.values] = -1

        top_indices = np.argsort(sims)[::-1][: self.top_k * 2]
        examples = []
        for idx in top_indices:
            row = self._df.iloc[idx]
            reply = str(row.get("brand_reply_clean", ""))
            if len(reply) > 5:
                examples.append({
                    "customer_message": str(row.get("customer_message_clean", "")),
                    "brand_reply": reply,
                    "similarity": float(sims[idx]),
                })
            if len(examples) >= self.top_k:
                break
        return examples

    def draft(
        self,
        message: str,
        intent: str = "",
        thread_context: str = "",
    ) -> dict[str, Any]:
        """
        Draft a reply for a customer message.

        Returns dict with: draft_reply (str), retrieved_examples (list)
        """
        from src.utils.gemini import generate_json

        examples = self.retrieve(message, intent)
        examples_block = "\n\n".join(
            f"EXAMPLE {i+1}:\n  Customer: {ex['customer_message']}\n  SpotifyCare: {ex['brand_reply']}"
            for i, ex in enumerate(examples)
        )

        user_content = (
            f"Intent: {intent or 'unknown'}\n"
            f"{'Thread context: ' + thread_context + chr(10) if thread_context else ''}"
            f"Customer message: {message}\n\n"
            f"--- Historical Examples ---\n{examples_block}"
        )

        result = generate_json(_SYSTEM_PROMPT, user_content, temperature=0.7)
        return {
            "draft_reply": result.get("draft_reply", ""),
            "retrieved_examples": examples,
        }


if __name__ == "__main__":
    import sys
    msg = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "My Spotify won't play any songs on Bluetooth speaker"
    drafter = ReplyDrafter()
    drafter.load_index()
    out = drafter.draft(msg, intent="playback_issue")
    print(f"\nDraft reply:\n  {out['draft_reply']}\n")
    print("Retrieved examples:")
    for i, ex in enumerate(out["retrieved_examples"], 1):
        print(f"  [{i}] (sim={ex['similarity']:.3f}) {ex['brand_reply'][:100]}")
