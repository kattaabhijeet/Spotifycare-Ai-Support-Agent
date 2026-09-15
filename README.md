# SpotifyCare AI Support Agent

> **Hiver SDE Intern Take-Home** — Turn a noisy real-world dataset into a working AI support system and prove it works.

**Brand**: SpotifyCare (`@SpotifyCares`)  
**LLM**: Google Gemini Flash Lite (`gemini-flash-lite-latest`)  
**Dataset**: Customer Support on Twitter (Kaggle)  

---

## Quick-start — Reproduce headline results in < 15 minutes

### Prerequisites

```bash
python --version      # 3.10+
pip --version         # 23+
```

**You need**:
- An OpenAI API key (set in `.env`)  
- The Kaggle dataset file `twcs.csv` (~600 MB)  

---

### Step 1 — Clone & install

```bash
git clone <your-repo-url>
cd hiver-support-agent
pip install -r requirements.txt
```

---

### Step 2 — Set your OpenAI API key

```bash
cp .env.example .env
# Edit .env and replace the placeholder with your real key:
# OPENAI_API_KEY=sk-...
```

---

### Step 3 — Download the dataset

**Option A — Kaggle CLI** (recommended if you have `kaggle.json`):
```bash
pip install kaggle
# Place kaggle.json in ~/.kaggle/kaggle.json (Linux/Mac) or %USERPROFILE%\.kaggle\kaggle.json (Windows)
kaggle datasets download -d thoughtvector/customer-support-on-twitter -p data/raw/ --unzip
```

**Option B — Manual download**:
1. Go to https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
2. Click **Download** → `customer-support-on-twitter.zip`
3. Extract and place `twcs.csv` in `data/raw/twcs.csv`

---

### Step 4 — Run the full pipeline

```bash
# 4a. Ingest + filter SpotifyCare data (subsample 10k tweets)
python -m src.pipeline.ingest

# 4b. Clean tweets, build threads
python -m src.pipeline.preprocess

# 4c. Generate embeddings — FREE with local model (downloads ~90MB once)
python -m src.pipeline.embed

# Optional: use OpenAI instead (~$0.02 for 10k messages)
# python -m src.pipeline.embed --backend openai
```

After these 3 steps, `data/processed/` will contain:
- `spotify_raw.csv` — filtered raw tweets  
- `threads.csv` — reconstructed conversation threads  
- `clean.csv` — cleaned, ready-to-use dataset  
- `embeddings.npy` — pre-computed embeddings (cached)  

---

### Step 5 — Build the golden evaluation set

```bash
python -m src.eval.build_golden_set
```

Generates `data/golden_set.csv` — 200 stratified examples with:
- `true_intent` labels (LLM-classified from a 2k sample, manually reviewed)
- `escalate_label` (0 = auto, 1 = escalate) — determined by hard rules, free
- `true_brand_reply` — the actual SpotifyCare reply from the dataset

> **Note on labelling**: The initial intent labels are LLM-generated from a 2k sample (~$0.01). Manually review and correct `true_intent` in `data/golden_set.csv` before final evaluation. See `DECISION_LOG.md` D9 for the methodology.

---

### Step 6 — Run evaluation

```bash
# Full metrics: intent accuracy, ROUGE-L, escalation F1
# Compares AI agent vs. 2 baselines
python -m src.eval.metrics

# LLM-as-judge (30 examples, free with Gemini)
python -m src.eval.llm_judge --n 30

# Unit tests (no API calls needed)
pytest tests/ -v
```

Results are saved to `eval/results.json` and `eval/judge_scores.csv`.  
Human-agreement analysis (Cohen's κ) is pre-computed in `eval/judge_human_agreement.md`.

---

### Step 7 — Try the agent interactively

```bash
# Classify a message
python -m src.agent.classifier "My Spotify keeps crashing on my iPhone 15"

# Draft a reply
python -m src.agent.reply_drafter "All my downloaded songs disappeared after the update"

# Make an escalation decision
python -m src.agent.escalation "You charged me twice and I'm calling my bank"
```

---

## Architecture

```
Raw Twitter CSV (twcs.csv, ~3M tweets)
         │
         ▼
[1. Ingest]  filter → @SpotifyCares threads → subsample 10k
         │
         ▼
[2. Preprocess]  clean text → build threads → add metadata
         │
         ▼
[3. Embed]  all-MiniLM-L6-v2 (local, free) → embeddings.npy (cached)
         │
         ▼
[4. AI Agent]
  ├── Classifier     — few-shot Gemini Flash Lite → {intent, confidence}
  ├── Reply Drafter  — cosine-similarity RAG → prompt → {draft_reply}
  └── Escalation     — hard rules → LLM judgment → {auto | escalate, reason}
         │
         ▼
[5. Evaluation]
  ├── Golden Set (200 hand-labelled examples, stratified)
  ├── Auto metrics: Accuracy, Macro-F1, ROUGE-L, BERTScore, Escalation P/R/F1
  └── LLM Judge: 4-dimension rubric (0-20) + Cohen's κ human agreement
```

---

## Intent Taxonomy

| Intent | Description |
|--------|-------------|
| `playback_issue` | Songs/podcasts won't play; skipping, stuttering, buffering |
| `account_billing` | Subscriptions, charges, premium/family plan issues |
| `app_device_bug` | App crashes, freezes, login loops, sync errors |
| `search_discovery` | Can't find songs/artists; wrong search results |
| `download_offline` | Offline downloads failing or disappearing |
| `social_playlist` | Collaborative playlists, following, share features |
| `content_availability` | Songs/albums missing or region-locked |
| `compliment` | Positive feedback and praise |
| `other` | Off-topic, spam, or unclear |

---

## File Structure

```
hiver-support-agent/
├── README.md
├── requirements.txt
├── DECISION_LOG.md
├── .env.example
├── data/
│   ├── raw/                     # twcs.csv (place here — gitignored)
│   ├── processed/               # pipeline outputs (gitignored)
│   └── golden_set.csv           # 200 hand-labelled examples
├── notebooks/
│   ├── 01_eda.ipynb             # EDA + brand selection
│   └── 02_intent_discovery.ipynb  # clustering → intent labels
├── src/
│   ├── pipeline/
│   │   ├── ingest.py            # load + filter SpotifyCare data
│   │   ├── preprocess.py        # clean tweets, build threads
│   │   └── embed.py             # OpenAI embeddings + cache
│   ├── agent/
│   │   ├── classifier.py        # intent classifier (few-shot GPT-4o-mini)
│   │   ├── reply_drafter.py     # RAG-based reply generation
│   │   └── escalation.py        # rules + LLM escalation engine
│   └── eval/
│       ├── build_golden_set.py  # stratified sampling + LLM annotation
│       ├── metrics.py           # accuracy, F1, ROUGE, BERTScore
│       └── llm_judge.py         # LLM-as-judge rubric
├── eval/
│   ├── results.json             # evaluation outputs
│   ├── judge_scores.csv         # per-example judge scores
│   └── judge_human_agreement.md # Cohen's κ analysis
├── report/
│   └── report.md               # 6-page evaluation report
└── tests/
    ├── test_pipeline.py         # pipeline unit tests (no API needed)
    └── test_escalation.py       # escalation rule tests (no API needed)
```

---

## Estimated Costs

| Step | Old cost | New cost | Notes |
|------|----------|----------|-------|
| Embeddings (`all-MiniLM-L6-v2`, local) | **$0.00** | One-time ~90MB model download |
| Build golden set (2k classify + rules) | **~$0.01** | Classify 2k sample; escalation = rules only |
| Agent evaluation (50 examples) | **~$0.05** | Use `--eval-sample 200` for full run |
| LLM judge (30 examples) | **~$0.03** | Use `--n 50` for broader coverage |
| **Total** | **~$0.09** | **~83% reduction** |

> **Want to spend more for better coverage?**  
> `python -m src.eval.metrics --eval-sample 200` — full 200-example agent eval (~$0.30)  
> `python -m src.eval.llm_judge --n 50` — judge 50 examples (~$0.05)

---

## Key design decisions

See [`DECISION_LOG.md`](DECISION_LOG.md) for the full list of 15 non-obvious decisions.  
See [`report/report.md`](report/report.md) for results, failure analysis, and next steps.
