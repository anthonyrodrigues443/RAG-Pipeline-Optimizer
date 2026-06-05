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

### Running production recommendation (through Phase 5)
**E5-base-v2, whole-doc, exact (Flat) search below ~40k vectors / HNSW ef=128 above; skip the cross-encoder re-ranker on a strong dense retriever; apply HyDE×N when an LLM-budget exists (especially on abstract-like corpora), else free PRF for a recall bump.** Next: RAGAS faithfulness + test whether HyDE×N's recall gain finally makes a re-ranker pay (Phase 6).
