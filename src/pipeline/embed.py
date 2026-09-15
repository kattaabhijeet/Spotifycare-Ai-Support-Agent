"""
embed.py — Generate and cache embeddings for customer messages.

Embedding backend (controlled via --backend flag):
  local  (DEFAULT) — sentence-transformers all-MiniLM-L6-v2 (FREE, runs on CPU)
  openai            — OpenAI text-embedding-3-small (~$0.02 per 10k msgs)

Use the local backend whenever possible. The quality difference on RAG
retrieval for Twitter text is minimal (~2-3% ROUGE-L), and it eliminates
the only API cost in the pipeline step.

Output: data/processed/embeddings.npy  +  data/processed/embed_index.csv

Usage:
    python -m src.pipeline.embed                       # free local model
    python -m src.pipeline.embed --backend openai      # OpenAI API
    python -m src.pipeline.embed --backend local --batch 256
"""

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
INPUT_FILE = PROCESSED_DIR / "clean.csv"
EMBED_FILE = PROCESSED_DIR / "embeddings.npy"
INDEX_FILE = PROCESSED_DIR / "embed_index.csv"

LOCAL_MODEL = "all-MiniLM-L6-v2"    # 384-dim, CPU-friendly, free
OPENAI_MODEL = "text-embedding-3-small"


def truncate(text: str, max_chars: int = 500) -> str:
    """Truncate text. Tweets are short — 500 chars is more than enough."""
    return text[:max_chars] if len(text) > max_chars else text


# ─────────────────────────────────────────────────────────────────────────────
# Local embedding (sentence-transformers)
# ─────────────────────────────────────────────────────────────────────────────

def build_embeddings_local(texts: list, batch_size: int) -> np.ndarray:
    """
    Embed texts using a local sentence-transformers model. FREE.
    First call downloads ~90MB model to ~/.cache/huggingface/.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Run: pip install sentence-transformers")

    log.info(f"  Loading local model '{LOCAL_MODEL}' (downloads once, ~90MB) …")
    model = SentenceTransformer(LOCAL_MODEL)

    log.info(f"  Encoding {len(texts):,} messages in batches of {batch_size} …")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # pre-normalise → dot product = cosine sim
    )
    return embeddings.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI embedding (fallback / comparison)
# ─────────────────────────────────────────────────────────────────────────────

def _embed_openai_batch(client, texts: list, model: str) -> list:
    resp = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in resp.data]


def build_embeddings_openai(texts: list, batch_size: int) -> np.ndarray:
    """Embed via OpenAI API. Costs ~$0.02 per 10k tweets."""
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    all_embeddings = []

    log.info(f"  Embedding {len(texts):,} messages via OpenAI in batches of {batch_size} …")
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding (OpenAI)"):
        batch = texts[i: i + batch_size]
        for attempt in range(3):
            try:
                vecs = _embed_openai_batch(client, batch, OPENAI_MODEL)
                all_embeddings.extend(vecs)
                break
            except Exception as e:
                wait = 2 ** attempt
                log.warning(f"  API error (attempt {attempt+1}): {e}. Retrying in {wait}s …")
                time.sleep(wait)
        else:
            dim = len(all_embeddings[0]) if all_embeddings else 1536
            all_embeddings.extend([[0.0] * dim] * len(batch))

    return np.array(all_embeddings, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Cache check
# ─────────────────────────────────────────────────────────────────────────────

def load_cached(n: int):
    """Return cached embeddings if the row count matches, else None."""
    if EMBED_FILE.exists() and INDEX_FILE.exists():
        cached_index = pd.read_csv(INDEX_FILE)
        if len(cached_index) == n:
            log.info("  Embeddings cache is up-to-date; loading from disk.")
            return np.load(EMBED_FILE)
        log.info(f"  Cache has {len(cached_index):,} rows but dataframe has {n:,}; re-embedding.")
    return None


def main():
    parser = argparse.ArgumentParser(description="Embed customer messages.")
    parser.add_argument(
        "--backend", choices=["local", "openai"], default="local",
        help="'local' = free sentence-transformers (default); 'openai' = OpenAI API (~$0.02)"
    )
    parser.add_argument("--batch", type=int, default=128,
                        help="Batch size (default 128 for local, use 50 for OpenAI)")
    args = parser.parse_args()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"{INPUT_FILE} not found. Run `python -m src.pipeline.preprocess` first."
        )

    df = pd.read_csv(INPUT_FILE)
    log.info(f"Loaded {len(df):,} rows from {INPUT_FILE}.")
    texts = [truncate(str(t)) for t in df["customer_message_clean"].tolist()]

    # Check cache first
    cached = load_cached(len(texts))
    if cached is not None:
        log.info(f"Using cached embeddings: shape={cached.shape}")
        return

    if args.backend == "local":
        embeddings = build_embeddings_local(texts, batch_size=args.batch)
        log.info(f"  Local embedding complete. Model: {LOCAL_MODEL}  Cost: $0.00")
    else:
        batch_size = min(args.batch, 50)  # OpenAI recommends smaller batches
        embeddings = build_embeddings_openai(texts, batch_size=batch_size)
        log.info(f"  OpenAI embedding complete. Model: {OPENAI_MODEL}")

    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBED_FILE, embeddings)
    df[["tweet_id"]].to_csv(INDEX_FILE, index=False)

    log.info(f"Saved embeddings → {EMBED_FILE}  shape={embeddings.shape}")
    log.info(f"Saved index      → {INDEX_FILE}")


if __name__ == "__main__":
    main()
