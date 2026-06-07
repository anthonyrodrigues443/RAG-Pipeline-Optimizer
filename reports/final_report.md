# RAG Pipeline Optimizer — Final Report
**A measurement-first study of what actually moves RAG quality.**
Project window: 2026-06-01 → 2026-06-07 (7 phases, Mon–Sun). Author: Anthony Rodrigues.

---

## Abstract

RAG tutorials are a pile of folk remedies — "use 512-token chunks," "add a re-ranker," "use
HyDE," "put it in HNSW." This project isolated each of those levers and benchmarked it on real
**BEIR** corpora (SciFact, NFCorpus, FiQA-2018) with an evaluation harness validated against
published BM25 numbers (SciFact nDCG@10 0.652 vs ≈0.665). Four of the most-repeated pieces of
advice turned out to be **wrong, conditional, or actively harmful** at typical RAG scale. The
consolidated pipeline — E5-base-v2 over whole documents, exact NumPy top-k, HyDE×N query
transform, **no re-ranker**, cheap generator with citation enforcement — ties an unreachable
gold-oracle on answer correctness (0.604) while a *wrong* retrieval poisons the answer below
closed-book. The headline for practitioners: **spend your budget on retrieval reliability, not
ranking precision.**

Primary metric: **nDCG@10** (BEIR leaderboard standard, rank- and grade-aware) for retrieval;
**answer-correctness-vs-gold** (LLM-judged) for generation. Every number is from an executed
notebook or a deterministic eval script.

---

## The optimal pipeline (one-paragraph recommendation)

> Encode with **E5-base-v2 over whole documents** (don't chunk below the encoder window).
> Retrieve with **exact cosine top-k** (a NumPy matmul) below ~40k vectors; switch to HNSW
> `ef=128` only above that. **Skip the cross-encoder re-ranker entirely.** Apply **HyDE×N**
> (average the query vector with N hypothetical-answer vectors) when you have an LLM budget;
> use free **PRF** otherwise. **Generate with a cheap model + citation enforcement** for the
> easy/mid majority, and escalate to a frontier generator only on the hard tail.

---

## Phase-by-phase findings

### Phase 1 — Chunking & the eval harness (2026-06-01)
*Q: which chunking strategy maximises retrieval quality?*
- Validated the harness first (SciFact BM25 0.652 vs ≈0.665; NFCorpus 0.307 vs ≈0.325) so every
  later number is trustworthy.
- 7-way chunking ablation on NFCorpus: best chunker (`sentence`) won **+3.6% nDCG@10** — but
  fixed 256/512/1024 spanned just **0.004 nDCG@10**, and `recursive_256`/`fixed_512` *underperformed*
  doing nothing.
- **Finding:** chunking is **encoder-bounded**, not free lift. "Use 512-1024 token chunks" is
  contingent on a long-context embedder, not a law. 78.6% of docs overflowed the 256-tok window
  yet chunking barely helped.

### Phase 2 — Embedding head-to-head & the chunking-vs-window law (2026-06-02)
*Q: which encoder, and does chunking still matter as the window grows?*
- 4-encoder leaderboard: **E5-base-v2** won both datasets (SciFact 0.727 / NFCorpus 0.353). The
  MiniLM→E5 jump is **+12.2% / +10.7%** — ~3× any chunking gain.
- **THE LAW:** best-chunker lift over whole-doc falls monotonically as the encoder window grows
  (+2.9%@256 → +2.1%@512 → +0.4%@8192) — Phase 1's falsifiable prediction, confirmed.
- **Surprise:** hybrid BM25+E5 RRF *lowered* nDCG@10 on both datasets (fusing a weaker lexical
  ranker pollutes the top ranks); it only helped deep recall. RRF pays only when both retrievers
  are comparably strong.
- **Finding:** the encoder is the dominant lever — pick it before touching chunk size.

### Phase 3 — Index structure & the ANN crossover (2026-06-03)
*Q: which FAISS index, and at what corpus size does ANN stop being a liability?*
- Traced Flat/IVF/HNSW/IVFPQ across 3.6k→57k vectors (added FiQA-2018, 57,638 docs).
- Below ~10k vectors a **2-line NumPy matmul wins on quality, latency, build time AND RAM**. HNSW
  only earns its keep past the crossover (~40k+); at 57k it's 5.6× faster than Flat for −0.004 nDCG.
- **Surprise:** IVF is **Pareto-dominated by HNSW at every N** — exact-quality IVF is slower than
  brute force even at 57k, and `nprobe=1` throws away 47% nDCG@10 to save 0.36 ms.
- **Finding:** most RAG knowledge bases (10³–10⁵ vectors) sit *below* the point where an ANN index
  helps. The "HNSW is 70× faster" claims are measured at 10M vectors and don't transfer.

### Phase 4 — Re-ranking, the lever that made retrieval worse (2026-06-04)
*Q: does a cross-encoder re-ranker improve a strong E5 first stage?*
- Tested 4 re-rankers on 3 corpora. **Every one hurt on all three**, the 278M model was the worst,
  and deeper re-ranking was monotonically worse.
- Re-rankers are **equalisers** — they lift BM25 (+0.10) and drag E5 (−0.01) onto their own ~0.35
  band — so they help only a *weak* first stage.
- Only GPT-5.x re-ranking beat E5, at ~50,000× the cost of a 22M cross-encoder.
- **Finding:** don't add a re-ranker to a strong dense retriever. The single most-recommended RAG
  "quality lever" made quality worse here.

### Phase 5 — Query transformation; only HyDE×N is safe (2026-06-05)
*Q: HyDE vs multi-query vs step-back vs PRF — which improves retrieval?*
- **Single-HyDE was a coin flip:** +0.12 nDCG@10 on scientific papers, **−0.075** on financial Q&A
  — same technique, opposite sign, decided by domain. Step-back hurt everywhere.
- Only the **averaged HyDE×N** (N hypotheticals + the real query, averaged in embedding space)
  helped on all three corpora. A bigger LLM for the hypothesis halved the loss but still lost to
  the raw query.
- **Finding:** HyDE×N is the only query transform worth shipping; free PRF is the budget-free
  fallback. It also opened a "backward link": HyDE×N raises R@100, so *maybe* a re-ranker pays now?

### Phase 6 — Generation faithfulness & closing the loop (2026-06-06)
*Q: does any of the retrieval tuning change the generated answer?*
- Built a real RAG generator (Haiku over FiQA) and swept context quality oracle → strong → HyDE×N
  → adversarial → closed-book, scoring RAGAS faithfulness/relevancy/correctness/citation with a
  *fixed* Haiku judge.
- **Retrieval past "good enough" barely moves the answer:** strong→oracle quadruples context
  precision (0.234→1.0) for just **+6 pts** correctness; HyDE×N ties the oracle.
- **Context poisoning:** wrong-but-plausible context cut correctness **54% (below closed-book)**
  while the model stayed **0.83 faithful** to the garbage. **No retrieval beats bad retrieval.**
- **Backward link half-works:** a cross-encoder on HyDE×N's better candidates flips positive-vs-naive
  on 2/3 corpora, but raw HyDE×N still beats it everywhere — better retrieval makes the re-ranker
  *redundant*, not rescued. The per-query router's oracle ceiling (≤0.046) isn't worth building.
- **Frontier generators** win the hard tail only (+50% correctness at 15-83× cost).

### Phase 7 — End-to-end pipeline, production & UI (2026-06-07)
*Q: does the optimal pipeline beat naive RAG end to end, and can it be productionised?*
- Froze Phases 1-6 into importable code (`src/pipeline.py`, `predict.py`, `evaluate.py`, `llm.py`),
  45 offline tests, and a Streamlit demo.
- **Retrieval:** HyDE×N beats naive E5 on all 3 corpora (+0.116/+0.031/+0.040 nDCG@10).
- **Generation:** optimal HyDE×N (0.604) **ties the gold oracle** and beats naive E5 (0.542). A
  cheap generator on a reliable retriever is as good as perfect retrieval.
- **Live:** the assembled pipeline answered an unseen FiQA query — 4 hypotheticals → 5 passages →
  a cited answer at **citation-grounding 1.00**.
- **Finding:** the project's bottom line — *retrieval reliability beats ranking precision*, made
  tangible in the UI's poisoning toggle.

---

## Consolidated decision table

| Component | Conventional advice | This project's verdict | Why |
|-----------|--------------------|------------------------|-----|
| Chunking | "512-1024 token chunks" | **Whole-doc** (with a 512-tok encoder) | Encoder-bounded; sub-window chunking ≈ no-op |
| Embedding | many options | **E5-base-v2** | +12% nDCG@10 over MiniLM, ~3× any chunking gain |
| Hybrid BM25+dense | "always fuse" | **Skip** unless both are strong | RRF polluted top ranks; helped only deep recall |
| Index | "use HNSW/IVF" | **Exact Flat <40k**, HNSW above | ANN loses nDCG below the crossover; IVF dominated |
| Re-ranker | "the #1 quality lever" | **None** | Every cross-encoder hurt a strong E5 stage |
| Query transform | "add HyDE" | **HyDE×N** (avg), else free PRF | Single-HyDE flips sign by domain; only the average generalised |
| Generation | "use the best model" | **Cheap + cite**, escalate hard tail | Frontier wins hard tail only, at 15-83× cost |
| Where to spend | "improve ranking" | **Retrieval reliability** | Wrong retrieval poisons the answer below closed-book |

## End-to-end numbers (Phase 7)

| Pipeline | SciFact | NFCorpus | FiQA | FiQA answer-correctness |
|----------|--------:|---------:|-----:|------------------------:|
| Naive RAG (E5 top-k → Haiku) | 0.428 | 0.362 | 0.338 | 0.542 |
| **Optimal (E5+HyDE×N, no rerank → Haiku+cite)** | **0.544** | **0.393** | **0.378** | **0.604** |
| Gold-oracle ceiling (generation) | — | — | — | 0.604 |

## Limitations
- Generation evaluated only on FiQA (n=24), judged by a fixed Claude Haiku that sees the gold
  reference — internally consistent and reproducible, but LLM-judged and small-n, not human-adjudicated.
- The FiQA gold passages are short forum answers, capping absolute correctness ≈0.60 even with the
  oracle; the *relative* ordering is the signal.
- Live latency (~37 s/HyDE×N query) is dominated by sequential CLI startup; the cost math reflects
  API pricing, not the CLI wall-clock.
- All encoding on CPU (Apple-Silicon MPS/faiss instability); corpora ≤57k vectors.

## Artifacts
- Code: `src/` (pipeline, predict, evaluate, llm, retrieval_eval, chunking), `app.py`, `tests/` (45 passing)
- Research: `notebooks/phase1…phase6` (executed) · per-phase `reports/dayN_phaseN_report.md`
- Results: `results/metrics.json`, per-phase CSVs/plots, `results/EXPERIMENT_LOG.md`, `results/ui_screenshot.png`
- Model card: `models/model_card.md`
