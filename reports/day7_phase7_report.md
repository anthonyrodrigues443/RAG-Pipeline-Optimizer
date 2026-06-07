# Phase 7: End-to-End Optimal Pipeline + Production + UI + Tests — RAG Pipeline Optimizer
**Date:** 2026-06-07
**Session:** 7 of 7 (closing session)

## Objective
Two questions to close the 7-day project:
1. **Does the optimal pipeline beat naive RAG end-to-end** — on *both* the retrieval number six
   phases chased *and* the generated answer that actually reaches a user?
2. **Can the six phases' findings be productionized** as clean, importable, tested code with an
   interactive demo — not just notebooks?

## Research & References
1. **Gao et al., 2022 (HyDE / "Precise Zero-Shot Dense Retrieval without Relevance Labels")** —
   the hypothetical-document recipe; Phase 5 found the *averaged* N-variant is what generalised,
   carried into `E5Retriever.hyde_vectors`.
2. **Es et al., 2023 (RAGAS)** — faithfulness / answer-relevancy / context-precision/recall as
   reference-free RAG metrics; re-implemented with a fixed Claude judge in Phase 6 and replayed
   deterministically here.
3. **Google / Hugging Face Model Card standard** — structure for `models/model_card.md`
   (intended use, evaluation, limitations, ethical considerations).
4. **Streamlit production-app patterns (2024-25)** — `st.cache_data`/`cache_resource`, tabbed
   layout, cached-vs-live mode so the demo runs on a clean checkout with no model.

How it shaped the session: the pipeline is the *consolidated verdict*, so the new code adds no
new technique — it freezes Phases 1-6 into `src/pipeline.py` and the end-to-end test re-uses the
Phase-5 HyDE cache + the Phase-6 judge cache so the comparison is deterministic (zero new LLM
calls), with one live run to prove the assembled pipeline actually answers a question.

## Dataset
| Metric | Value |
|--------|-------|
| Corpora | SciFact (5.2k docs), NFCorpus (3.6k), FiQA-2018 (57.6k) |
| Retrieval eval | n=40 sampled queries per corpus (Phase-5 cached sample) |
| Generation eval | FiQA, n=24 (balanced across naive-difficulty thirds) |
| Encoder | E5-base-v2, whole-doc, 512-token window, 768d (CPU) |
| Generator / Judge | Claude Haiku (fixed judge sees the gold reference) |
| Primary metric | nDCG@10 (retrieval); answer-correctness-vs-gold (generation) |

## Experiments

### Experiment 7.1: Optimal vs naive — retrieval axis (3 corpora)
**Hypothesis:** HyDE×N (the Phase-5 winner) lifts nDCG@10 over a naive E5 first stage everywhere.
**Method:** recompute both runs live from the cached E5 vectors + the Phase-5 cached
hypotheticals; score with the validated TREC harness (`src/evaluate.build_runs`).
**Result:**
| Corpus | Naive E5 | Optimal (HyDE×N) | Δ nDCG@10 | Δ R@100 |
|--------|---------:|-----------------:|----------:|--------:|
| SciFact | 0.4279 | **0.5436** | +0.116 | +0.050 |
| NFCorpus | 0.3624 | **0.3929** | +0.031 | +0.085 |
| FiQA | 0.3377 | **0.3779** | +0.040 | +0.051 |
**Interpretation:** HyDE×N wins on all three, biggest on the abstract-like SciFact claims —
consistent with Phase 5. Reproduces the Phase-6 Exp-E baselines exactly (a self-check).

### Experiment 7.2: Optimal vs naive — generation axis (does it reach the answer?)
**Hypothesis:** better retrieval should produce a better *answer*, not just a better ranking.
**Method:** replay the Phase-6 FiQA generation+judge cache (24 queries × 5 conditions) through
`src/evaluate.score_generation_records` — deterministic, no new calls.
**Result:**
| Context | Correctness | Faithfulness | Citation | Refused (heuristic) |
|---------|------------:|-------------:|---------:|--------------------:|
| Oracle (gold) | 0.604 | 0.94 | 1.00 | 0.25 |
| **Optimal (HyDE×N)** | **0.604** | 0.99 | 1.00 | 0.17 |
| Naive (E5) | 0.542 | 0.97 | 1.00 | 0.25 |
| Closed-book | 0.396 | 0.00 | — | 0.00 |
| Adversarial | 0.250 | 0.83 | 1.00 | 0.79 |
**Interpretation:** the optimal pipeline (+0.063 over naive) **ties the unreachable gold oracle**
— a cheap generator on a reliable retriever is as good as perfect retrieval. The "refused" column
is an independent heuristic recomputation (phrase-match), so it differs slightly from Phase 6's
explicitly-tracked rate; the headline correctness/faithfulness/citation reproduce exactly.

### Experiment 7.3: Live end-to-end smoke test (does the assembled pipeline run?)
**Hypothesis:** `src/pipeline.py` answers an unseen free-text query end to end.
**Method:** `python -m src.predict "How is freelance income taxed for a sole proprietor?"` —
HyDE×N path: live Haiku HyDE generation → embed+average → E5 retrieve → cited Haiku answer.
**Result:** 4 hypotheticals generated → 5 FiQA passages retrieved (all on-topic 1099/Schedule-C
docs) → a 3-sentence answer citing `[d1][d2][d4][d5]`, **citation grounding 1.00**, 36.8 s.
**Interpretation:** the production pipeline works; the answer is grounded and well-cited.
(`results/phase7_live_smoke.txt`.)

## Head-to-Head Comparison (end-to-end, the project's bottom line)
| Pipeline | SciFact nDCG@10 | NFCorpus | FiQA | FiQA answer-correctness |
|----------|----------------:|---------:|-----:|------------------------:|
| Naive RAG (E5 top-k → Haiku) | 0.428 | 0.362 | 0.338 | 0.542 |
| **Optimal (E5+HyDE×N, no rerank → Haiku+cite)** | **0.544** | **0.393** | **0.378** | **0.604** |
| Gold-oracle ceiling (generation) | — | — | — | 0.604 |

## Key Findings
1. **The optimal pipeline ties the gold oracle on answer correctness (0.604)** while beating
   naive E5 (+0.063) — a cheap generator + reliable retrieval is as good as a perfect retriever.
2. **Retrieval lift is real but downstream-bounded:** HyDE×N adds +0.03-0.12 nDCG@10, yet that
   only converts to +0.06 answer correctness — the six-phase chase has diminishing *downstream*
   returns past "good enough."
3. **Context poisoning is the deployment risk, not low recall:** adversarial context cut
   correctness to 0.250 (below closed-book) at 0.83 faithfulness — so the production lesson is
   *retrieval reliability*, encoded as the UI's poisoning toggle.

## Frontier Model Comparison (carried from Phase 6, same fixed judge)
| Generator | FiQA correctness (hard tail, n=12) | Cost/1k | Winner |
|-----------|-----------------------------------:|--------:|--------|
| Haiku (shipped) | 0.250 | $0.60 | cheap, ties frontier on easy/mid |
| Opus | 0.375 | $9.00 | hard tail only (+50%) |
| Codex (GPT) | 0.375 | $50.00 | hard tail only (+50%) |

## Error Analysis
- **Adversarial highest refusal (0.79 heuristic):** when handed plausible-wrong context the model
  often *correctly* says "the context does not contain the answer" — but still scores 0.25
  correctness because the remaining answers confidently restate the wrong context. Refusal is a
  partial, not complete, safeguard.
- **FiQA correctness ceiling ≈0.60 even with gold context:** FiQA gold passages are short forum
  answers; the judge penalises answers that are correct-but-differently-phrased from the single
  gold reference. This caps the absolute number — the *relative* ordering is the signal.

## Next Steps
- Project complete (7/7). Natural Phase 8+ extensions: human adjudication of the n=24 generation
  set; a learned retrieval-reliability gate (refuse/escalate when context-precision is low); a
  direct-API path to cut the 37 s CLI latency; widen generation eval beyond FiQA.

## References Used Today
- [1] Gao et al., 2022 — *Precise Zero-Shot Dense Retrieval without Relevance Labels* (HyDE) — arXiv:2212.10496
- [2] Es et al., 2023 — *RAGAS: Automated Evaluation of Retrieval Augmented Generation* — arXiv:2309.15217
- [3] Hugging Face / Mitchell et al., 2019 — *Model Cards for Model Reporting* — arXiv:1810.03993
- [4] Streamlit docs — caching & multipage app patterns — https://docs.streamlit.io

## Code Changes
- `src/pipeline.py` — the optimal RAG pipeline (E5Retriever, RAGCorpus, RAGPipeline, HyDE×N, conditions)
- `src/predict.py` — single-query inference CLI
- `src/evaluate.py` — eval suite + `--phase7` end-to-end driver + `--demo` JSON builder
- `src/llm.py` — Claude/Codex CLI harness + RAGAS judge/citation parsers (factored out of Phase-6 notebook)
- `app.py` — Streamlit demo (Explore / Poisoning / Live tabs)
- `tests/` — `conftest.py` + 45 offline tests (chunking, metrics, llm parsers, pipeline conditions, eval aggregation)
- `results/phase7_{retrieval,end_to_end}.csv`, `results/phase7_end_to_end.png`, `results/ui_screenshot.png`,
  `results/phase7_demo.json`, `results/phase7_live_smoke.txt`, `results/metrics.json` (phase7 block)
- `models/model_card.md`, `reports/final_report.md`, `README.md` (mini-paper rewrite), `requirements.txt`
