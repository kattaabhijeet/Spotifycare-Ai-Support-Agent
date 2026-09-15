# SpotifyCare AI Support Agent

> 

**Brand**: SpotifyCare (`@SpotifyCares`)  
**LLM**: Google Gemini Flash Lite (`gemini-flash-lite-latest`)  
**Embeddings**: `all-MiniLM-L6-v2` via sentence-transformers (local, free)  
**Dataset**: Customer Support on Twitter (Kaggle)  

---

## Quick-start — Reproduce headline results in < 15 minutes

### Prerequisites

- Python 3.10+
- A free Google Gemini API key — get one at [aistudio.google.com](https://aistudio.google.com/app/apikey)
- The Kaggle dataset file `twcs.csv` (~600 MB)

---

### Step 1 — Clone & install

```bash
git clone https://github.com/kattaabhijeet/Spotifycare-Ai-Support-Agent.git
cd Spotifycare-Ai-Support-Agent
pip install -r requirements.txt
```

---

### Step 2 — Add your Gemini API key

```bash
cp .env.example .env
# Open .env and set your key:
# GOOGLE_API_KEY=your-key-here
```

---

### Step 3 — Download the dataset

**Kaggle CLI** (if you have `kaggle.json` set up):
```bash
kaggle datasets download -d thoughtvector/customer-support-on-twitter -p data/raw/ --unzip
```

**Manual download**:
1. Go to [kaggle.com/datasets/thoughtvector/customer-support-on-twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
2. Download and extract — place `twcs.csv` at `data/raw/twcs.csv`

---

### Step 4 — Run the pipeline

```bash
# Ingest + filter SpotifyCare data (subsamples to 10k tweets)
python -m src.pipeline.ingest

# Clean tweets and rebuild conversation threads
python -m src.pipeline.preprocess

# Generate embeddings locally (downloads ~90MB model once, then cached)
python -m src.pipeline.embed
```

After these three steps `data/processed/` will contain:
- `spotify_raw.csv` — filtered raw tweets
- `threads.csv` — reconstructed conversation threads
- `clean.csv` — cleaned dataset ready for use
- `embeddings.npy` — pre-computed embeddings (cached for all future runs)

---

### Step 5 — Build the golden evaluation set

```bash
python -m src.eval.build_golden_set
```

Generates `data/golden_set.csv` — 170 stratified examples (roughly 19 per intent) with:
- `true_intent` — LLM-classified labels, manually reviewed
- `escalate_label` — 0 (auto) or 1 (escalate), set by deterministic hard rules
- `true_brand_reply` — the real SpotifyCare reply from the dataset

> The golden set is already included in the repo. You only need to re-run this if you want to regenerate it from scratch. See `DECISION_LOG.md` D9 for the sampling methodology.

---

### Step 6 — Run evaluation

```bash
# Intent accuracy, ROUGE-L, escalation F1 — agent vs. 2 baselines
python -m src.eval.metrics

# LLM-as-judge: 4-dimension rubric scored by Gemini (30 examples)
python -m src.eval.llm_judge --n 30

# Unit tests — no API calls required
pytest tests/ -v
```

Results are written to `eval/results.json` and `eval/judge_scores.csv`.  
Human-agreement analysis (Cohen's κ = 0.952) is in `eval/judge_human_agreement.md`.

---

### Step 7 — Try the agent interactively

```bash
# Classify a message
python -m src.agent.classifier "My Spotify keeps crashing on my iPhone 15"

# Draft a reply
python -m src.agent.reply_drafter "All my downloaded songs disappeared after the update"

# Get an escalation decision
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
[3. Embed]  all-MiniLM-L6-v2 (local) → embeddings.npy (cached)
         │
         ▼
[4. AI Agent]
  ├── Classifier     — few-shot Gemini Flash Lite → {intent, confidence}
  ├── Reply Drafter  — cosine-similarity RAG → {draft_reply}
  └── Escalation     — hard rules → LLM triage → {auto | escalate, reason}
         │
         ▼
[5. Evaluation]
  ├── Golden Set (170 hand-labelled examples, stratified across 9 intents)
  ├── Auto metrics: Accuracy, Macro-F1, ROUGE-L, Escalation P/R/F1
  └── LLM Judge: 4-dimension rubric (0–20) + Cohen's κ human agreement
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
spotifycare-ai-support-agent/
├── README.md
├── requirements.txt
├── DECISION_LOG.md
├── .env.example
├── data/
│   ├── raw/                     # place twcs.csv here (gitignored)
│   ├── processed/               # pipeline outputs (gitignored)
│   └── golden_set.csv           # 170 hand-labelled examples
├── notebooks/
│   ├── 01_eda.ipynb             # EDA + brand selection
│   └── 02_intent_discovery.ipynb  # clustering → intent taxonomy
├── src/
│   ├── pipeline/
│   │   ├── ingest.py            # filter SpotifyCare data + subsample
│   │   ├── preprocess.py        # clean tweets, build threads
│   │   └── embed.py             # local embeddings + cache
│   ├── agent/
│   │   ├── classifier.py        # intent classifier (few-shot Gemini Flash Lite)
│   │   ├── reply_drafter.py     # RAG-based reply generation
│   │   └── escalation.py        # two-layer escalation engine
│   └── eval/
│       ├── build_golden_set.py  # stratified sampling + LLM annotation
│       ├── metrics.py           # accuracy, F1, ROUGE-L, escalation metrics
│       └── llm_judge.py         # LLM-as-judge rubric
├── eval/
│   ├── results.json             # full evaluation results
│   ├── judge_scores.csv         # per-example judge scores
│   └── judge_human_agreement.md # Cohen's κ analysis
├── report/
│   └── report.md                # evaluation report (problem framing → next steps)
└── tests/
    ├── test_pipeline.py         # pipeline unit tests (no API needed)
    └── test_escalation.py       # escalation rule tests (no API needed)
```

---

## API Cost

Everything runs on **free-tier Gemini** (no credit card required). The only costs are:

| Step | Cost |
|------|------|
| Embeddings (local `all-MiniLM-L6-v2`) | $0.00 |
| Build golden set (Gemini Flash Lite, free tier) | $0.00 |
| Evaluation — 170 examples | $0.00 |
| LLM judge — 30 examples | $0.00 |
| **Total** | **$0.00** |

---

## Key Documents

- [`DECISION_LOG.md`](DECISION_LOG.md) — 15 non-obvious decisions made during the project and why
- [`report/report.md`](report/report.md) — results vs. baselines, failure analysis, and what I'd do next
- [`eval/judge_human_agreement.md`](eval/judge_human_agreement.md) — Cohen's κ between Gemini judge and human scorer
