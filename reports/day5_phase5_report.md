# Phase 5: Query Transformation Techniques — RAG Pipeline Optimizer
**Date:** 2026-06-05
**Session:** 5 of 7

## Objective
Phase 4 hit a wall: a cross-encoder re-ranker could not push nDCG@10 past the candidate set the first stage handed it (FiQA oracle@100 = 0.77, every re-ranker landed ~0.38). The bottleneck is **what the first stage retrieves**, not how it's ordered. This phase attacks the first stage from the *query* side — rewrite the query before retrieval — and answers one framing question:

> On a **strong** dense retriever (E5-base-v2), does expensive LLM query expansion (HyDE, multi-query, step-back) actually beat a **free, no-LLM, embedding-space pseudo-relevance feedback**?

Primary metric: **nDCG@10** (BEIR standard, used every prior phase). Secondaries: Recall@10/@100, MRR@10.

## Research & References
1. **Gao et al., 2022 — "Precise Zero-Shot Dense Retrieval without Relevance Labels" (HyDE).** Generate a hypothetical answer document with an LLM, embed *that* instead of the query; averaging N generations + the query is the published recipe. Their gains were largest on *weak/zero-shot* encoders — a warning flag for a strong E5 retriever.
2. **Zheng et al., 2023 — "Take a Step Back" (step-back prompting).** Abstract the query to a more general question to retrieve background knowledge. Designed for reasoning QA, not short-passage retrieval — we test whether it transfers.
3. **Rocchio (1971) / classic pseudo-relevance feedback.** Pull the query toward the centroid of its own top-k retrieved docs. We run it entirely in embedding space on the already-cached vectors — zero model calls.

**How research shaped the experiments:** the HyDE paper's "helps weak encoders" caveat plus Phase 4's "LLM machinery drags a strong retriever toward a mediocre middle" predicted that LLM query tricks would be domain-dependent and possibly harmful on E5. We therefore (a) added the free PRF bar that any LLM technique must clear to justify cost, and (b) added a generator-strength ablation (Haiku vs Opus HyDE) to separate *technique* failure from *model* failure.

## Dataset
Three BEIR corpora, identical to Phases 2–4 (corpus embeddings reused from cache — nothing re-embedded on the doc side, so first-stage numbers are directly comparable).

| Corpus | Domain | Docs | Queries | Full-set naive nDCG@10 |
|--------|--------|-----:|--------:|----------------------:|
| SciFact | scientific claim verification | 5,183 | 300 | 0.7274 |
| NFCorpus | medical / health | 3,633 | 323 | 0.3529 |
| FiQA-2018 | personal finance / investing | 57,638 | 648 | 0.3987 |

**Sampling for the LLM arm.** LLM techniques were scored on a **stratified sample of 40 queries per corpus**, bucketed by naive per-query nDCG@10 (failed / mid / solved) so the sample over-represents the hard queries where expansion can actually help. Consequently the **sample-naive baseline is lower than the full-set naive** (SciFact 0.428, NFCorpus 0.362, FiQA 0.338). **All deltas below are sample-relative and measure the technique effect on hard queries — they are not claims of beating the full-corpus baseline.** PRF, being free, was additionally swept on the **full** query set.

## Experiments

### Experiment 5.1: PRF / Rocchio — the free, no-LLM bar (full query set)
**Hypothesis:** the cheapest expansion needs no LLM — pull q toward the centroid of its own top-k docs.
**Method:** `q' = normalize(q + β·mean(top-k docs))`, swept k∈{3,5,10}, β∈{0.3,0.5,0.7,1.0}, on cached vectors.
**Result (best config, full set):**

| Corpus | best k / β | nDCG@10 | Δ vs naive | Recall@100 | Δ R@100 |
|--------|-----------|--------:|-----------:|-----------:|--------:|
| NFCorpus | 3 / 0.5 | 0.3663 | **+0.0133** | 0.3379 | +0.0183 |
| FiQA | 3 / 0.3 | 0.4032 | +0.0045 | 0.7399 | +0.0079 |
| SciFact | 3 / 0.3 | 0.7268 | −0.0007 | 0.9633 | +0.0006 |

**Interpretation:** PRF is a reliable, free **recall** booster (Recall@100 up on all three) but its nDCG@10 gain is tiny and largest exactly where the retriever is weakest (noisy medical NFCorpus). Shallow feedback wins (k=3 every time); deeper feedback drifts the query off-target. It *slightly hurts* clean SciFact, where the raw query is already near-optimal.

### Experiment 5.2: HyDE (single hypothetical passage)
**Hypothesis:** embedding a generated ideal-answer passage beats embedding the raw query.
**Result (sample, ΔnDCG@10 vs sample-naive):** SciFact **+0.1227**, NFCorpus −0.0026, FiQA **−0.0745**.
**Interpretation:** HyDE is a **domain gamble**. On SciFact — where the corpus is literally written in the form HyDE hallucinates (scientific abstracts) — it is the single best technique (+0.12, and Recall@100 0.85→0.95). On FiQA — messy financial-forum Q&A — the clean, over-specific hypothetical answer *misleads* the retriever and costs 0.075. Same technique, opposite sign, decided by domain.

### Experiment 5.3: HyDE×N (mean of 2 hypothetical passages + the original query)
**Result (sample, ΔnDCG@10):** SciFact **+0.1157**, NFCorpus **+0.0306**, FiQA **+0.0403**.
**Interpretation:** the published averaging recipe is the **only transform that helps on all three corpora.** Folding the real query back in converts single-HyDE's volatility into a consistent win — it keeps ~95% of HyDE's SciFact gain while turning FiQA's −0.075 *loss* into a +0.040 *gain*. The original query is the safety net.

### Experiment 5.4: Multi-query (paraphrase RRF) and Step-back
**Result (sample, ΔnDCG@10):**

| Technique | SciFact | NFCorpus | FiQA |
|-----------|--------:|---------:|-----:|
| Multi-query (RRF of 3 paraphrases + orig) | +0.0177 | −0.0060 | +0.0348 |
| Step-back (RRF of orig + broader question) | −0.0321 | −0.0887 | −0.0453 |

**Interpretation:** multi-query is mildly positive on the keyword-heavy corpora (FiQA +0.035) — paraphrase diversity recovers a few lexical misses — but neutral-to-negative on NFCorpus. **Step-back hurts on all three** (worst −0.089 on NFCorpus): abstracting the query to a broader question is exactly the wrong move for a precision-sensitive dense retriever; it dilutes the specificity the embedder relies on.

### Experiment 5.5: Generator-strength ablation (Haiku vs Opus HyDE, FiQA)
**Hypothesis (from Phase 4's "bigger was worse"):** if HyDE fails on FiQA, does a stronger generator fix it?
**Result (FiQA sample, n=40):** naive 0.3377 · HyDE-Haiku 0.2632 (**−0.0745**) · HyDE-Opus 0.2990 (**−0.0387**).
**Interpretation:** Opus narrows the loss by half but **both still lose to the raw query.** A better LLM does not flip the verdict — the failure is the *technique-on-domain* (single HyDE on financial text), not the model. Clean echo of Phase 4: throwing a bigger model at the wrong lever doesn't rescue it.

## Head-to-Head Comparison (sample, nDCG@10; best per corpus in bold)
| Technique | LLM? | SciFact | NFCorpus | FiQA |
|-----------|:---:|--------:|---------:|-----:|
| naive (sample baseline) | no | 0.4279 | 0.3624 | 0.3377 |
| PRF / Rocchio | no | 0.4235 | 0.3651 | 0.3385 |
| HyDE (single) | yes | **0.5506** | 0.3598 | 0.2632 |
| HyDE passage-prefix | yes | 0.5211 | 0.3827 | 0.2557 |
| **HyDE×N (mean)** | yes | 0.5436 | **0.3929** | **0.3779** |
| Multi-query (RRF) | yes | 0.4456 | 0.3564 | 0.3724 |
| Step-back | yes | 0.3958 | 0.2737 | 0.2924 |
| *oracle@100 (ceiling)* | — | *0.8500* | *0.6169* | *0.7522* |

## Key Findings
1. **HyDE×N is the only universally-safe query transform** (+0.116 / +0.031 / +0.040). Single-HyDE is a domain bet; averaging the hypotheticals back with the real query is what makes it robust.
2. **Domain decides HyDE's sign.** +0.123 on scientific abstracts, −0.075 on financial Q&A — the corpus's writing style, not the LLM, determines whether a hallucinated ideal answer helps or misleads.
3. **Step-back prompting is actively harmful for dense retrieval** (−0.03 to −0.09). A reasoning-QA technique that does not transfer to short-passage retrieval.
4. **A bigger generator narrows but does not flip a bad technique** (Opus-HyDE −0.039 vs Haiku-HyDE −0.075 on FiQA; both < naive). Same lesson as Phase 4's "bigger re-ranker was worse."
5. **Free PRF buys recall, not ranking.** Zero-LLM PRF lifts Recall@100 on all three corpora; when you can afford one LLM call, HyDE×N dominates it on nDCG@10. The two are complementary, not competing.
6. **Recall-vs-nDCG split reframes Phase 4.** Several transforms raise Recall@100 while lowering nDCG@10 (HyDE on NFCorpus: R@100 0.350→0.431, nDCG −0.003) — they pull more gold into the pool but rank it worse. That is precisely the input a re-ranker needs. Phase 4's re-rankers may have failed partly because naive first-stage recall was already near-exhausted; query expansion that lifts R@100 is what gives a re-ranker something to fix.

## Error Analysis (FiQA, per-query ΔnDCG@10 vs naive — see results/phase5_error_analysis.png)
- **HyDE (single):** net negative — breaks more FiQA queries than it rescues; the over-specific hypothetical answer drags gold passages down.
- **HyDE×N:** net positive — averaging with the original query caps the downside, so rescues outweigh breaks.
- **Multi-query:** broadly positive small deltas — paraphrase RRF rarely hurts badly, recovering lexical misses without large regressions.

## Frontier-Model Note
The "frontier" comparison this phase is structural, not a classifier head-to-head: the *expensive LLM techniques* (HyDE/multi-query/step-back via Haiku & Opus) vs the *free no-LLM techniques* (raw E5, PRF). Verdict — on a strong dense retriever, the LLM only wins when used in the **robust HyDE×N form**; the naive single-HyDE recommended by most tutorials loses outright on 2 of 3 domains, and a stronger generator does not save it.

## Next Steps (Phase 6)
- **Generation & faithfulness (RAGAS):** does the LLM actually use retrieved context? Faithfulness / answer-relevancy / context-precision, citation grounding.
- **Feed the recall, not the ranking:** combine HyDE×N (lifts R@100) → cross-encoder re-rank (Phase 4) and test whether the expansion finally gives the re-ranker room to help — the backward link this phase opened.
- Per-query *router*: pick HyDE×N vs naive vs multi-query by predicted domain/difficulty rather than applying one technique globally.

## References Used Today
- [1] Gao, Ma, Lin, Callan (2022). *Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE).* arXiv:2212.10496
- [2] Zheng et al. (2023). *Take a Step Back: Evoking Reasoning via Abstraction in LLMs.* arXiv:2310.06117
- [3] Rocchio (1971), *Relevance Feedback in Information Retrieval*; BEIR (Thakur et al., 2021) for datasets & metric conventions.

## Code Changes
- `notebooks/phase5_query_techniques.ipynb` — 16 cells, all executed (11 real-work cells, 0 errors): naive baseline, PRF sweep (full set), Haiku generation harness (119/120 parsed), 5 techniques + passage-prefix ablation, error analysis, Opus generator ablation, consolidated metrics.
- `results/phase5_prf_sweep.csv`, `phase5_query_techniques.csv`
- `results/phase5_prf_heatmap.png`, `phase5_query_techniques.png`, `phase5_error_analysis.png`
- `results/phase5_llm_cache/{gen_haiku.json, gen_opus_fiqa.json, gen_emb.npz, sample_*.json}` (cached, idempotent)
- `results/metrics.json` → `phase5` block
