# Phase 8: The Retrieval-Reliability Gate — RAG Pipeline Optimizer
**Date:** 2026-06-08
**Session:** 8 (iterate & improve — the 7-phase build is complete; this operationalizes its closing finding)

## Objective
Phases 6–7 ended on one verdict: *the dominant RAG failure mode is not ranking precision, it is **retrieval reliability**.* When the top-5 context is plausible-but-wrong, a cheap generator's answer correctness collapses to 0.25 — **below closed-book (0.40)** — while it stays 0.83 "faithful" to the garbage. The final recommendation literally read *"spend the budget on retrieval reliability, not extra ranking precision."*

This session builds the mechanism that recommendation implies: a **gate** that, at inference time and **with no gold labels**, decides whether the retrieved context is trustworthy — so the system can abstain or escalate instead of being poisoned.

**Questions:** (1) Is reliability detectable from the retriever's own score distribution alone? (2) Can a cheap learned gate match an LLM-as-judge that costs ~10⁴× more — and generalize to unseen corpora? (3) Are the LLMs blind to the very failure they're meant to catch? (4) Does routing on the gate pay off downstream?

## Research & References
1. **Asai et al., 2023 — Self-RAG** — a model learns to decide *when to retrieve* and to critique retrieved passages. We invert it: a *retriever-side*, model-free gate instead of a generation-side learned critic.
2. **Yan et al., 2024 — Corrective RAG (CRAG)** — a lightweight retrieval evaluator scores relevance and triggers correction (web search / decompose). Our gate is the cheapest possible evaluator: statistics of the cosine curve, no second model. CRAG motivates the *escalation* remediation (Policy B).
3. **Selective prediction / abstention (Geifman & El-Yaniv, 2017)** — risk-coverage framing: trade coverage for accuracy via a confidence score. We borrow it for the downstream selective-RAG curve.

How it shaped the work: CRAG/Self-RAG both add a *model* to judge retrieval. We test whether that model is even necessary — whether the geometry of the embedding scores already encodes reliability — and then borrow selective-prediction to measure whether good detection yields a good *action*.

## Dataset
| Metric | Value |
|--------|-------|
| Corpora | BEIR SciFact, NFCorpus, FiQA (E5-base-v2 doc embeddings, reused from Phases 1–7) |
| Queries (judged) | 1,271 (300 / 323 / 648) |
| Documents | 5,183 / 3,633 / 57,638 |
| Label | **UNRELIABLE = 1** if no gold passage in retriever's top-5 (qrels-derived) |
| Base rate (unreliable) | 33.1% pooled (19.0% / 33.1% / 39.7%) |
| Primary metric | **AUPRC** for the UNRELIABLE class (minority positive; AUROC + F1/recall reported alongside) |

## Experiments

### Experiment 8.1 — single-signal baselines
**Hypothesis:** a reliable retrieval has a high, peaked top; an unreliable one is low and flat.
**Method:** 8 statistics of the top-100 cosine curve; AUROC/AUPRC of each alone.
**Result:**

| signal | AUROC | AUPRC |
|--------|------:|------:|
| margin_bg (top1 − mean[rank20–100]) | 0.800 | 0.639 |
| entropy (softmax of top-20) | 0.803 | 0.630 |
| std10 | 0.795 | 0.622 |
| gap15 | 0.769 | 0.591 |
| frac_tie | 0.764 | 0.586 |
| **top1 cosine** | **0.633** | 0.481 |
| top5_mean | 0.533 | 0.411 |

**Interpretation:** the **absolute similarity score is nearly useless** (top1 AUROC 0.63, top5_mean ≈ random). The *shape* features dominate. `frac_tie`: when unreliable, **58%** of the top-20 cluster within 0.02 of the top (a flat tie — nothing is actually close); when reliable, **27%** (a clear standout).

### Experiment 8.2 — cheap learned gate + cross-corpus generalization
**Method:** LogisticRegression and HistGradientBoosting on the 8 features. Pooled 5-fold CV + leave-one-corpus-out (train on 2, test on the held-out 3rd).
**Result:**

| eval | logreg AUROC | histgb AUROC |
|------|------:|------:|
| pooled 5-fold | **0.801** | 0.770 |
| LOCO: held-out SciFact | **0.833** | 0.827 |
| LOCO: held-out NFCorpus | **0.810** | 0.754 |
| LOCO: held-out FiQA | **0.748** | 0.673 |

**Interpretation:** the gate **generalizes to corpora it never trained on** (0.75–0.83) — it's not memorizing a corpus's score scale. **Logistic regression beats gradient boosting everywhere** (the simplest model wins again). Inference: **188 µs/query** (37 µs features + 143 µs predict).

### Experiment 8.3 — head-to-head vs LLM-as-judge (the frontier comparison)
**Method:** same question + top-5 passages → ask Claude Haiku, Claude Opus, Codex (GPT) *"RELIABLE / UNRELIABLE + confidence."* Stratified class-balanced sample (15/corpus, n=45, base rate 0.47). Cheap gate trained on all non-sample queries (fair held-out). Scoring is **label-anchored** (Codex reports confidence-in-its-label, Claude reports P(can-answer) — confidence can't invert a stated decision).

| gate | n | AUROC | AUPRC | F1 | recall(unrel) | prec(unrel) | latency | cost/1k |
|------|--:|------:|------:|---:|----:|----:|--------:|--------:|
| **cheap gate (logreg)** | 45 | **0.813** | **0.794** | **0.809** | **0.905** | 0.731 | **0.00002 s** | **$1e-6** |
| baseline: top-1 cosine thr | 45 | 0.706 | 0.669 | 0.655 | 0.905 | 0.514 | 0.00001 s | $1e-6 |
| LLM judge: Haiku | 45 | 0.779 | 0.765 | 0.720 | 0.857 | 0.621 | 15.8 s | $0.0012 |
| LLM judge: Codex (GPT) | 45 | 0.774 | 0.662 | 0.739 | 0.810 | 0.680 | 18.7 s | $0.060 |
| LLM judge: Opus | 44 | 0.732 | 0.609→F1 0.609 | 0.609 | **0.667** | 0.560 | 9.4 s | $0.018 |

**Interpretation:** the 188 µs logistic regression **beats all three frontier LLM judges on AUROC, AUPRC, F1 and unreliable-recall** — at ~10⁴–10⁵× lower cost and ~10⁵× lower latency. And the LLMs are **partly blind to the failure they're auditing**: Opus, the most confident, has the *lowest* unreliable-recall (0.667) — it waves through a third of poisoned retrievals because the top-5 are topically plausible. The geometry-based gate isn't fooled by plausible text.

### Experiment 8.4 — does the gate pay off downstream? (Selective RAG, n=24 FiQA)
**Method:** reuse Phase-6 per-query correctness under `strong` / `closed_book` / `hydeN`; gate trained on non-FiQA (unseen + cross-corpus). Two policies on flagged-UNRELIABLE queries.
**Result:**

| policy | behaviour |
|--------|-----------|
| always strong RAG | 0.542 |
| always HyDE×N (LLM on every query) | 0.604 |
| always closed-book | 0.396 |
| oracle router (max/q) | 0.688 (ceiling) |
| **A: flag → closed-book fallback** | **HURTS monotonically** (→0.46) |
| **B: flag → escalate to HyDE×N** | **0.583 = 66% of the full-HyDE gain, escalating only 58% of queries (42% of LLM calls saved)** |

**Interpretation:** **detection ≠ remediation**. The obvious fix (abstain to closed-book) is *wrong* — a natural weak top-5 still beats no retrieval (the Phase-6 below-closed-book poisoning needed *deliberately adversarial* r40–60 context). The action that pays is **selective escalation**: spend the expensive retrieval *only where the gate says it's needed*.

## Key Findings
1. **The similarity score everyone thresholds on is nearly useless for reliability** (top-1 AUROC 0.63; top5_mean ≈ random). The *shape* of the score curve is the signal (margin/entropy/tie-cluster ≈ 0.80).
2. **A 188 µs logistic regression beats Haiku, Opus and Codex** as a reliability judge (AUROC 0.813 vs ≤0.779), generalizes cross-corpus (LOCO 0.75–0.83), at 10⁴–10⁵× less cost/latency.
3. **Frontier LLMs are partly blind to the poison they're auditing** — Opus has the lowest unreliable-recall (0.667), fooled by topical-but-wrong context.
4. **Detection ≠ remediation:** closed-book fallback hurts; gate-driven HyDE×N escalation recovers ⅔ of the gain at half the LLM cost.

## Frontier Model Comparison
| Task | Custom (188 µs logreg) | Claude Haiku | Claude Opus | Codex (GPT) | Winner |
|------|------:|------:|------:|------:|--------|
| Detect unreliable retrieval — AUROC | **0.813** | 0.779 | 0.732 | 0.774 | **Custom** |
| — unreliable-recall | **0.905** | 0.857 | 0.667 | 0.810 | **Custom** |
| — cost / 1k | **$0.000001** | $0.0012 | $0.018 | $0.060 | **Custom** |

## Error Analysis
- Gate is weakest on FiQA (LOCO AUROC 0.748) — the largest, noisiest corpus (57k docs, 39.7% base rate); harder to separate a flat-but-OK retrieval from a flat-and-wrong one there.
- The `top1<=thr` baseline matches the gate's *recall* (0.905) but at half the precision (0.514 vs 0.731) — it over-flags, confirming that absolute score alone can't tell "low because hard query" from "low because no answer present."
- One Opus call returned a transient API error (dropped; parse_ok 44/45). Codex's confidence semantics differ from Claude's (handled by label-anchored scoring).

## Next Steps (Phase 9)
- Productionize `RetrievalReliabilityGate` into `src/gate.py` (done this session, lean+tested) → wire into `src/pipeline.py` as a pre-generation guard with the escalation policy; add a UI "reliability meter".
- Widen the selective-RAG eval beyond n=24 FiQA; human-adjudicate.
- Calibrate the operating threshold per deployment (precision-first abstention vs recall-first escalation).

## References Used Today
- [1] Asai et al., 2023. *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.* arXiv:2310.11511.
- [2] Yan et al., 2024. *Corrective Retrieval Augmented Generation (CRAG).* arXiv:2401.15884.
- [3] Geifman & El-Yaniv, 2017. *Selective Classification for Deep Neural Networks.* NeurIPS.

## Code Changes
- `notebooks/phase8_retrieval_reliability_gate.ipynb` (20 cells, executed; the experiment)
- `src/gate.py` + `tests/test_gate.py` (reusable gate + offline tests)
- `results/phase8_*` — `score_separation.png`, `roc.png`, `selective_rag.png`, `llm_vs_gate.csv`, `selective_rag.csv`, `phase8_cache/`, `metrics.json` (phase8 key)
- `results/EXPERIMENT_LOG.md` (Phase 8 section), this report
