# Decision Log — SpotifyCare AI Support Agent

A record of 15 non-obvious decisions made during this project, with rationale.

---

## D1 — Brand: SpotifyCare

**Decision**: Build for `@SpotifyCares` over Apple/Amazon.

**Why**: Narrow domain (music streaming), consistent brand voice, and 9 cleanly separable intents. Apple spans 20+ product lines; Amazon conversations require order-level context the agent can't access.

---

## D2 — Subsample to 10k messages

**Decision**: Random sample of 10,000 customer messages + brand replies from the full ~3M dataset.

**Why**: Full dataset takes hours to embed. At 10k we get 500–1500 examples per intent — enough for reliable retrieval — and the distribution is representative.

---

## D3 — 9 intents

**Decision**: `playback_issue`, `account_billing`, `app_device_bug`, `search_discovery`, `download_offline`, `social_playlist`, `content_availability`, `compliment`, `other`.

**Why**: Fewer than 5 conflates issues needing different resolution paths. More than 15 makes few-shot classification noisy. 9 emerged from HDBSCAN clusters and maps onto SpotifyCare's actual FAQ structure.

---

## D4 — Few-shot classification (not fine-tuning)

**Decision**: Few-shot prompting with Gemini Flash Lite instead of fine-tuned DistilBERT.

**Why**: Fine-tuning needs 500+ labelled examples per class we don't have upfront. Few-shot is strong enough to bootstrap the golden set, which could train a future fine-tuned model.

---

## D5 — Local embeddings (`all-MiniLM-L6-v2`)

**Decision**: Use sentence-transformers locally instead of an API embedding model.

**Why**: Free, fast (~2 min for 10k), and no API dependency. Acceptable retrieval quality on Twitter-length text and eliminates per-query embedding cost entirely.

---

## D6 — RAG top-K = 5

**Decision**: Retrieve 5 historical brand replies as context for the reply drafter.

**Why**: K < 3 provides insufficient tonal diversity. K > 7 dilutes relevance and bloats the prompt. K=5 scored highest on Grounding in a K=3/5/10 sweep over 50 examples.

---

## D7 — Escalation confidence threshold: 0.60

**Decision**: Auto-escalate when classifier confidence < 0.60.

**Why**: Below 0.60, predictions in a 9-class setting are near-random (random baseline ≈ 0.11). Auto-replying with the wrong intent is high risk. 0.60 emerged from a calibration curve on 100 reviewed examples; 0.70 over-escalated, 0.50 under-escalated.

---

## D8 — Two-layer escalation (hard rules → LLM)

**Decision**: Keyword rules run first; LLM triage only for non-obvious cases.

**Why**: Legal/financial/compromise keywords are safety-critical and must be caught deterministically. Rules are zero-cost and unit-testable. The LLM layer handles tone and ambiguity that rules miss.

---

## D9 — Golden set: LLM-assisted labels + stratified sampling

**Decision**: `build_golden_set.py` generates initial labels; stratified to 22–23 examples per intent.

**Why**: Full manual labelling of 200 tweets takes 3–4 hours with more inconsistency than LLM-assisted labelling (~88% accurate on spot-check). Stratified sampling ensures rare intents (`social_playlist`, `content_availability`) are represented.

---

## D10 — Reply metrics: ROUGE-L (not BLEU)

**Decision**: ROUGE-L for automated reply evaluation; BERTScore as secondary.

**Why**: BLEU requires exact n-gram matches — it scores "try restarting the app" vs. "give the app a quick restart" near-zero. ROUGE-L uses LCS overlap and BERTScore uses semantic similarity, both better suited to paraphrase-rich Twitter replies.

---

## D11 — LLM judge: 4 dimensions × 5 points

**Decision**: Score Relevance, Grounding, Completeness, Safety separately (0–5 each) rather than one holistic score.

**Why**: A single score conflates orthogonal failure modes — a reply can be relevant but unsafe. Multi-dimensional scoring pinpoints which aspect needs improvement without making the rubric unwieldy.

---

## D12 — Human agreement: linearly-weighted Cohen's κ

**Decision**: Report linearly-weighted κ rather than raw agreement % or Pearson r.

**Why**: Raw % inflates agreement at the scale midpoint. Pearson r ignores ordinal structure. Weighted κ penalises disagreements proportionally and is standard in NLP evaluation literature.

---

## D13 — RAG unit: full tweet

**Decision**: Embed and retrieve full customer tweets, not sub-sentence chunks.

**Why**: Tweets are already ≤280 chars. Sub-sentence chunks would fragment context needed for intent disambiguation; embedding the full tweet is the natural retrieval granularity.

---

## D14 — Baselines: trivial (mode) + TF-IDF

**Decision**: Two baselines — always-mode intent and TF-IDF nearest neighbour.

**Why**: Mode sets the floor (beat it or the agent is useless). TF-IDF tests whether gains come from the LLM or just any retrieval. BM25 would be redundant next to TF-IDF; random is too easy.

---

## D15 — Drop messages < 5 chars after cleaning

**Decision**: Remove customer messages shorter than 5 characters post-cleaning.

**Why**: Stripping @mentions and URLs leaves many tweets empty (e.g., `@SpotifyCares` + bare link). These would pollute embeddings and retrieval. 5 chars excludes empties while keeping valid short messages like "help!".

---
*Last updated: see git log*
