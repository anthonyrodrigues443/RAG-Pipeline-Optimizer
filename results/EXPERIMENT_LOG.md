# RAG Pipeline Optimizer — Consolidated Experiment Log

Primary metric **nDCG@10** throughout (BEIR standard). Retriever backbone after Phase 2 is **E5-base-v2** (numpy matmul / exact). One header table per phase; full per-config numbers live in the per-phase CSVs under `results/`.

---

## Phase 1 — Chunking strategy (NFCorpus, baselines)
*Question: which chunking strategy maximizes retrieval quality, and what's the floor?*

| Approach | nDCG@10 | Note |
|----------|--------:|------|
| BM25 (lexical baseline) | 0.3071 | sparse floor |
| MiniLM-L6 whole-doc | 0.3189 | dense floor |
| Best chunker (fixed-256, max-pool) | ~0.330 | +3.6% over whole-doc — but encoder-dependent |

**Finding:** chunking helps, but the lift is small and conditional on the encoder's context window — set up "the law" tested in Phase 2.

## Phase 2 — Embedding model + the chunking-vs-window law
*Question: which encoder, and does chunking still matter with a better one?*

| Model (whole-doc) | nDCG@10 SciFact | nDCG@10 NFCorpus |
|-------------------|----------------:|-----------------:|
| **E5-base-v2** | **0.7274** | **0.3529** |
| Nomic-v1.5 (8192-ctx) | 0.7076 | 0.3471 |
| BGE-small (512) | 0.7057 | 0.3444 |
| MiniLM-L6 (256) | 0.6484 | 0.3189 |
| BM25 | 0.6523 | 0.3071 |

**Findings:** (1) the encoder is a bigger lever than chunking — swapping MiniLM→E5 beat the best chunker by ~3×. (2) **THE LAW:** chunking lift over whole-doc falls monotonically with encoder window (+2.9%@256 → +2.1%@512 → +0.4%@8192). (3) Hybrid BM25+dense RRF *hurt* when the dense ranker is far stronger than BM25.

## Phase 3 — FAISS index structure & the ANN crossover
*Question: which index, and at what corpus size does ANN stop being a liability?*

| Corpus | N | exact nDCG@10 | single-q latency | best ANN |
|--------|--:|--------------:|-----------------:|----------|
| SciFact | 5,183 | 0.7274 | 0.25 ms | Flat (ANN not worth it) |
| NFCorpus | 3,633 | 0.3525 | 0.17 ms | Flat (ANN not worth it) |
| FiQA | 57,638 | 0.3987 | 2.08 ms | HNSW ef=128 (0.37 ms, 5.6×) |

**Findings:** below ~10k vectors an ANN index is pure overhead (IVF threw away 47% nDCG@10 to save 0.36 ms); HNSW only *pays* past ~40k vectors. IVF is Pareto-dominated by HNSW at every N. Also falsified the Phase-2 "Nomic long-doc dilution" intuition.

## Phase 4 — Re-ranking (the lever that made retrieval worse)
*Question: does a cross-encoder re-ranker improve nDCG@10 over E5?*

| Approach | FiQA nDCG@10 | Δ vs E5 | Verdict |
|----------|-------------:|--------:|---------|
| E5 (no rerank) | 0.3987 | — | baseline |
| Oracle@100 (ceiling) | 0.7706 | +0.372 | ordering is the gap, candidates are fine |
| MiniLM-L6 (22M) @100 | 0.3862 | −0.012 | hurts |
| BGE-base (278M) @100 | 0.3134 | −0.085 | **bigger = worse** |
| GPT-5.x listwise (n=30 subset) | 0.7203 | beats E5 | wins quality, $50/1k, 15 s/q |

**Findings:** every cross-encoder *reduced* nDCG@10 on a strong retriever; bigger and deeper were monotonically worse. Re-rankers are *equalisers* (lift BM25 +0.10, drag E5 −0.01 → both ~0.35), helping only a weak first stage.

## Phase 5 — Query transformation (HyDE / multi-query / step-back / PRF)
*Question: does LLM query expansion beat a free no-LLM PRF on a strong retriever?*
*(LLM techniques on a stratified 40-query/corpus sample; ΔnDCG@10 vs sample-naive. PRF on full set.)*

| Technique | LLM? | SciFact Δ | NFCorpus Δ | FiQA Δ |
|-----------|:---:|----------:|-----------:|-------:|
| PRF / Rocchio (full set) | no | −0.001 | +0.013 | +0.005 |
| HyDE (single) | yes | **+0.123** | −0.003 | −0.075 |
| **HyDE×N (mean)** | yes | +0.116 | **+0.031** | **+0.040** |
| Multi-query (RRF) | yes | +0.018 | −0.006 | +0.035 |
| Step-back | yes | −0.032 | −0.089 | −0.045 |
| HyDE generator: Haiku→Opus (FiQA) | yes | — | — | −0.075 → −0.039 (both < naive) |

**Findings:** (1) **HyDE×N is the only transform that helps on all three corpora.** (2) Domain decides HyDE's sign (+0.12 on scientific abstracts, −0.075 on financial Q&A). (3) Step-back is actively harmful for dense retrieval. (4) A bigger generator narrows but doesn't flip a bad technique. (5) Free PRF buys *recall* (R@100 up on all three), not ranking. (6) Some transforms raise R@100 while lowering nDCG@10 — the input a re-ranker actually needs, reframing Phase 4.

---

## Phase 6 — Generation faithfulness (RAGAS) + the backward link + frontier generators
*Question: does any of the retrieval tuning change the generated answer, and does HyDE×N's recall finally make a re-ranker pay?*
*(Generation: Haiku over FiQA, n=24, 5 context conditions, fixed Haiku judge. Exp A/E on n=40 × 3 corpora.)*

**Exp B/C/D — FiQA generation by context condition (mean):**
| Context | Correctness | Faithfulness | Citation grounding | Refused |
|---------|------------:|-------------:|-------------------:|--------:|
| oracle (gold) | 0.604 | 0.944 | 1.00 | 0.13 |
| **HyDE×N top-5** | **0.604** | 0.993 | 1.00 | 0.21 |
| strong (E5 top-5) | 0.542 | 0.967 | 1.00 | 0.21 |
| closed-book | 0.396 | 0.000 | — | 0.04 |
| adversarial (E5 r40–60) | **0.250** | 0.831 | 1.00 | 0.58 |

**Exp E — backward link, cross-encoder over naive vs HyDE×N candidates (nDCG@10):**
| Corpus | E5 naive | HyDE×N raw | CE(naive) | CE(HyDE×N) |
|--------|---------:|----------:|----------:|-----------:|
| SciFact | 0.428 | **0.544** | 0.453 | 0.471 |
| NFCorpus | 0.362 | **0.393** | 0.372 | 0.384 |
| FiQA | 0.338 | **0.378** | 0.316 | 0.324 |

**Exp F — frontier generators, same E5 context, fixed Haiku judge (n=12 hard tail, mean naive nDCG@10=0.07):**
| Generator | Correctness | Faithfulness | Cost/1k |
|-----------|------------:|-------------:|--------:|
| Haiku | 0.250 | 0.917 | **$0.60** |
| Opus | 0.375 | 0.988 | $9.00 |
| Codex (GPT) | 0.375 | 0.967 | $50.00 |

**Findings:** (1) **Retrieval quality past "good enough" barely moves the answer** — strong→oracle raises context precision 0.234→1.0 for +6 pts correctness; HyDE×N ties oracle (within n=24 noise). (2) **Context poisoning:** wrong retrieval cuts correctness 54% (below closed-book) while the model stays 0.83 faithful to the garbage. (3) **No retrieval beats bad retrieval** (closed-book 0.40 > adversarial 0.25). (4) **Backward link half-works:** CE on HyDE×N candidates flips positive-vs-naive on 2/3 corpora but raw HyDE×N still wins everywhere → re-ranker redundant, not rescued. (5) **Frontier generators win the hard tail only** (+50% correctness at 15–83× cost); per-query router headroom ≤0.046 → not worth building.

---

### Running production recommendation (through Phase 6)
**E5-base-v2, whole-doc, exact (Flat) search below ~40k vectors / HNSW ef=128 above; skip the cross-encoder re-ranker entirely (redundant even on HyDE×N's better candidates); apply HyDE×N when an LLM-budget exists (especially on abstract-like corpora), else free PRF for a recall bump; generate with a cheap model (Haiku) for the easy/mid majority and reserve a frontier generator for the hard tail.** Retrieval gains past "good enough" barely move the answer, but a *wrong* retrieval actively poisons it — so spend the budget on retrieval *reliability*, not extra ranking precision. Next: Phase 7 — optimal end-to-end pipeline + Streamlit UI + tests.
