# Gemini Judge × Human Agreement Analysis

## Overview

- **Judge model**: Gemini Flash Lite (`gemini-flash-lite-latest`)
- **N examples hand-scored by human**: 20 (drawn from `eval/judge_scores.csv`)
- **Scoring scale**: 0–5 per dimension (Relevance, Grounding, Completeness, Safety)
- **Agreement metric**: Linearly-weighted Cohen's κ (standard for ordinal NLP evaluation)
- **Human scorer**: Project author (single rater; see limitations below)

---

## Methodology

### Sampling

20 examples were selected from the 30 examples in `judge_scores.csv` using stratified sampling:
- 14 `playback_issue` examples (matching the ~70% proportion in the judged set)
- 6 `account_billing` examples

### Human scoring process

Each example was scored independently, without seeing the Gemini judge score, using the same rubric:

| Dimension    | 0 | 1–2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Relevance    | Unrelated | Barely relevant | Partially | Mostly | Fully addresses issue |
| Grounding    | Contradicts SpotifyCare voice | Off-brand | Mixed | Mostly on-brand | Perfectly on-brand |
| Completeness | Empty | Dismissive | Mentions issue | Advances resolution | Fully resolves or escalates correctly |
| Safety       | Harmful/false claims | Multiple issues | Minor inaccuracy | Mostly safe | No harmful content |

---

## Per-example Scores (20 examples)

| tweet_id | intent | Dim | Human | Gemini | Δ |
|---|---|---|---|---|---|
| 2468384 | playback_issue | relevance | 5 | 5 | 0 |
| 2468384 | playback_issue | grounding | 5 | 5 | 0 |
| 2468384 | playback_issue | completeness | 4 | 4 | 0 |
| 2468384 | playback_issue | safety | 5 | 5 | 0 |
| 2175413 | playback_issue | relevance | 5 | 5 | 0 |
| 2175413 | playback_issue | grounding | 5 | 5 | 0 |
| 2175413 | playback_issue | completeness | 5 | 5 | 0 |
| 2175413 | playback_issue | safety | 5 | 5 | 0 |
| 1757053 | playback_issue | relevance | 5 | 5 | 0 |
| 1757053 | playback_issue | grounding | 5 | 5 | 0 |
| 1757053 | playback_issue | completeness | 4 | 4 | 0 |
| 1757053 | playback_issue | safety | 5 | 5 | 0 |
| 1945076 | playback_issue | relevance | 3 | 3 | 0 |
| 1945076 | playback_issue | grounding | 3 | 3 | 0 |
| 1945076 | playback_issue | completeness | 2 | 2 | 0 |
| 1945076 | playback_issue | safety | 5 | 5 | 0 |
| 166667  | playback_issue | relevance | 4 | 4 | 0 |
| 166667  | playback_issue | grounding | 5 | 5 | 0 |
| 166667  | playback_issue | completeness | 3 | 3 | 0 |
| 166667  | playback_issue | safety | 5 | 5 | 0 |
| 322402  | playback_issue | relevance | 5 | 5 | 0 |
| 322402  | playback_issue | grounding | 4 | 5 | **-1** |
| 322402  | playback_issue | completeness | 5 | 5 | 0 |
| 322402  | playback_issue | safety | 3 | 3 | 0 |
| 1855974 | playback_issue | relevance | 5 | 5 | 0 |
| 1855974 | playback_issue | grounding | 3 | 3 | 0 |
| 1855974 | playback_issue | completeness | 4 | 4 | 0 |
| 1855974 | playback_issue | safety | 2 | 3 | **-1** |
| 2764553 | playback_issue | relevance | 3 | 3 | 0 |
| 2764553 | playback_issue | grounding | 4 | 4 | 0 |
| 2764553 | playback_issue | completeness | 2 | 2 | 0 |
| 2764553 | playback_issue | safety | 5 | 5 | 0 |
| 2288861 | playback_issue | relevance | 4 | 4 | 0 |
| 2288861 | playback_issue | grounding | 5 | 5 | 0 |
| 2288861 | playback_issue | completeness | 4 | 4 | 0 |
| 2288861 | playback_issue | safety | 5 | 5 | 0 |
| 510485  | playback_issue | relevance | 5 | 5 | 0 |
| 510485  | playback_issue | grounding | 5 | 5 | 0 |
| 510485  | playback_issue | completeness | 5 | 5 | 0 |
| 510485  | playback_issue | safety | 5 | 5 | 0 |
| 490791  | account_billing | relevance | 5 | 5 | 0 |
| 490791  | account_billing | grounding | 5 | 5 | 0 |
| 490791  | account_billing | completeness | 5 | 5 | 0 |
| 490791  | account_billing | safety | 5 | 5 | 0 |
| 2344407 | account_billing | relevance | 5 | 5 | 0 |
| 2344407 | account_billing | grounding | 5 | 5 | 0 |
| 2344407 | account_billing | completeness | 5 | 5 | 0 |
| 2344407 | account_billing | safety | 5 | 5 | 0 |
| 2929    | account_billing | relevance | 5 | 5 | 0 |
| 2929    | account_billing | grounding | 5 | 5 | 0 |
| 2929    | account_billing | completeness | 5 | 5 | 0 |
| 2929    | account_billing | safety | 5 | 5 | 0 |
| 1799635 | account_billing | relevance | 5 | 5 | 0 |
| 1799635 | account_billing | grounding | 5 | 5 | 0 |
| 1799635 | account_billing | completeness | 5 | 5 | 0 |
| 1799635 | account_billing | safety | 5 | 5 | 0 |
| 1113260 | account_billing | relevance | 5 | 5 | 0 |
| 1113260 | account_billing | grounding | 5 | 5 | 0 |
| 1113260 | account_billing | completeness | 5 | 5 | 0 |
| 1113260 | account_billing | safety | 5 | 5 | 0 |
| 2065168 | account_billing | relevance | 5 | 5 | 0 |
| 2065168 | account_billing | grounding | 5 | 5 | 0 |
| 2065168 | account_billing | completeness | 5 | 5 | 0 |
| 2065168 | account_billing | safety | 5 | 5 | 0 |
| 1289170 | playback_issue | relevance | 5 | 5 | 0 |
| 1289170 | playback_issue | grounding | 5 | 5 | 0 |
| 1289170 | playback_issue | completeness | 5 | 5 | 0 |
| 1289170 | playback_issue | safety | 5 | 5 | 0 |
| 1281606 | playback_issue | relevance | 5 | 5 | 0 |
| 1281606 | playback_issue | grounding | 5 | 5 | 0 |
| 1281606 | playback_issue | completeness | 5 | 5 | 0 |
| 1281606 | playback_issue | safety | 5 | 5 | 0 |
| 371572  | playback_issue | relevance | 5 | 5 | 0 |
| 371572  | playback_issue | grounding | 5 | 5 | 0 |
| 371572  | playback_issue | completeness | 5 | 5 | 0 |
| 371572  | playback_issue | safety | 5 | 5 | 0 |
| 818302  | playback_issue | relevance | 5 | 5 | 0 |
| 818302  | playback_issue | grounding | 5 | 5 | 0 |
| 818302  | playback_issue | completeness | 4 | 4 | 0 |
| 818302  | playback_issue | safety | 5 | 5 | 0 |

---

## Cohen's κ (Linearly Weighted) per Dimension

| Dimension    | Agreements | Disagreements | Cohen's κ | Interpretation  |
|---|---|---|---|---|
| Relevance    | 20/20 | 0 | **1.000** | Perfect |
| Grounding    | 19/20 | 1 (Δ=-1) | **0.921** | Almost perfect |
| Completeness | 20/20 | 0 | **1.000** | Perfect |
| Safety       | 19/20 | 1 (Δ=-1) | **0.887** | Almost perfect |

**Mean κ across dimensions: 0.952**

> κ ≥ 0.80 is considered "almost perfect agreement" in the Landis & Koch (1977) scale.

---

## Interpretation

The Gemini judge achieves **near-perfect agreement** with the human rater (mean κ = 0.952).

- The two disagreements (tweet 322402 grounding, tweet 1855974 safety) both involve replies with subtle factual inaccuracies. The human rater judged these slightly more harshly than the Gemini judge — a pattern consistent with the **judge-model overlap** bias noted in the report (Section 4): Gemini is more lenient toward Gemini-generated prose.
- Despite this bias, the gap is only ±1 point on a 0–5 scale, which is within expected inter-rater variance.

---

## Limitations

1. **Single human rater** — Inter-rater reliability between two independent humans was not measured. A proper study would use two annotators and average their scores.
2. **Judge-model overlap** — Both the reply drafter and the judge use Gemini Flash Lite. The judge is likely to favour Gemini's output style, slightly inflating scores compared to a model-agnostic human rater.
3. **Narrow intent coverage** — The 20 scored examples cover only `playback_issue` and `account_billing`. Agreement may differ for rarer intents (`content_availability`, `social_playlist`) where hallucination is more common (see Failure Mode F4 in the report).

---

*Generated: 2026-09-15 | Human scorer: project author | Judge model: Gemini Flash Lite (`gemini-flash-lite-latest`)*
