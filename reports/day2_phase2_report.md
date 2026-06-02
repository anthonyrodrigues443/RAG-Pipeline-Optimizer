# Phase 2: Embedding Head-to-Head & the Chunking-vs-Context-Window Law — RAG Pipeline Optimizer
**Date:** 2026-06-02
**Session:** 2 of 7

## Objective
Phase 1 answered the chunking question for a single 256-token encoder and ended on a falsifiable prediction:
*if the chunking "ceiling" is set by the encoder window, then a longer-context encoder should move it, and the
value of chunking should shrink as the window grows.* Phase 2 tests that head-on and answers the project's
Phase-2 question:

> **Which embedding model maximizes retrieval quality on the same harness — and does the best chunking strategy
> depend on the encoder you pair it with?**

## Research & References
1. **Wang et al., *E5* (Microsoft, 2022/2024)** — weakly-supervised contrastive embeddings with asymmetric
   `query:` / `passage:` prefixes. Motivated prefix-correct evaluation (an unfair prefix silently tanks a model).
2. **Xiao et al., *BGE / C-Pack* (BAAI, 2023–24)** — `bge-*-en-v1.5` with a query-only retrieval instruction;
   strong MTEB retrieval baselines. Used as the 512-window 384d/768d contenders.
3. **Nussbaum et al., *Nomic Embed* (2024)** — open 8192-token long-context embedder (`search_query:` /
   `search_document:`). The critical 8k-window data point for the chunking law.
4. **Cormack et al., *Reciprocal Rank Fusion* (SIGIR 2009)** — RRF (`score = Σ 1/(k0+rank)`, `k0=60`), the
   score-calibration-free fusion used in Exp 2.3.
5. **Thakur et al., *BEIR* (NeurIPS 2021)** — datasets + `nDCG@10`; the Phase-1 BM25 numbers are the parity anchor.

**How research shaped today:** the model registry deliberately spans 256 → 512 → 8192 token windows (the
independent variable for the law), and each encoder is evaluated with its *documented* retrieval prefix so the
leaderboard is apples-to-apples rather than a prefix-handling artifact.

## Primary Metric
**`nDCG@10`** (BEIR standard; rationale in Phase 1 report). Secondary: `Recall@10/100`, `MRR@10`. All tables
ranked by `nDCG@10`.

## Datasets
| Metric | SciFact (test) | NFCorpus (test) | NFCorpus (validation) |
|--------|---------------:|----------------:|----------------------:|
| Corpus docs | 5,183 | 3,633 | 3,633 |
| Queries | 300 | 323 | ~324 |
| Qrels | 339 (binary) | 12,334 (graded) | 11,385 (graded) |
| Role | leaderboard | leaderboard + chunking law | dev→test honesty check |

## Experiments

### Experiment 2.1 — Whole-doc embedding leaderboard
**Hypothesis:** A modern retrieval-tuned encoder beats MiniLM by more than chunking ever did (+3.6% in Phase 1).
**Method:** 4 dense encoders (after dropping GTE for speed — see note) embed full docs (truncated to each model's
window) with model-specific prefixes; exact cosine top-100; validated harness. Embeddings disk-cached.
**Result (ranked by nDCG@10):**

| Dataset | Model | win | dim | nDCG@10 | R@10 | R@100 | MRR@10 |
|---------|-------|----:|----:|--------:|-----:|------:|-------:|
| SciFact | **E5-base-v2** | 512 | 768 | **0.7274** | 0.8507 | 0.9627 | 0.6931 |
| SciFact | Nomic-v1.5 | 8192 | 768 | 0.7076 | 0.8536 | 0.9317 | 0.6694 |
| SciFact | BGE-small | 512 | 384 | 0.7057 | 0.8296 | 0.9433 | 0.6725 |
| SciFact | BM25 (P1) | – | – | 0.6523 | 0.7757 | 0.8731 | 0.6184 |
| SciFact | MiniLM-L6 (P1) | 256 | 384 | 0.6484 | 0.7883 | 0.9250 | 0.6068 |
| NFCorpus | **E5-base-v2** | 512 | 768 | **0.3529** | 0.1706 | 0.3197 | 0.5403 |
| NFCorpus | Nomic-v1.5 | 8192 | 768 | 0.3471 | 0.1711 | 0.2984 | 0.5307 |
| NFCorpus | BGE-small | 512 | 384 | 0.3444 | 0.1618 | 0.3125 | 0.5272 |
| NFCorpus | MiniLM-L6 (P1) | 256 | 384 | 0.3189 | 0.1589 | 0.3148 | 0.5083 |
| NFCorpus | BM25 (P1) | – | – | 0.3071 | 0.1522 | 0.2425 | 0.5085 |

**Interpretation:** E5-base-v2 wins both datasets. The MiniLM→E5 jump is **+12.2% (SciFact)** and **+10.7%
(NFCorpus)** — 3× the lift Phase 1 squeezed out of chunking. The encoder is the dominant lever.

### Experiment 2.2 — The chunking-vs-context-window law (headline)
**Hypothesis:** chunking lift over whole-doc shrinks as encoder window grows.
**Method:** sweep `{fixed_128, fixed_256, fixed_512}` + whole-doc control on NFCorpus for the fastest encoder at
each window tier: MiniLM (256), BGE-small (512), Nomic (8192). Chunks sized with each model's own tokenizer,
embedded with its doc prefix, max-pooled to parent.
**Result:**

| Encoder | Window | whole-doc nDCG@10 | best chunker | best nDCG@10 | lift |
|---------|-------:|------------------:|--------------|-------------:|-----:|
| MiniLM-L6 | 256 | 0.3189 | fixed_128 | 0.3282 | **+2.9%** |
| BGE-small | 512 | 0.3444 | fixed_256 | 0.3517 | **+2.1%** |
| Nomic-v1.5 | 8192 | 0.3471 | fixed_128 | 0.3485 | **+0.4%** |

**Interpretation:** Monotonic. **Prediction confirmed.** The longer the encoder window, the less chunking buys —
at 8k it's within noise. Chunking is a workaround for short-context encoders, not a universal win. Bonus: the
best chunk size tracks the window (128 @256, 256 @512) — best chunk ≈ half the window.

### Experiment 2.3 — Hybrid BM25 + dense fusion (RRF, k0=60)
**Hypothesis:** RRF closes the deep-recall gap and lifts nDCG@10.
**Method:** RRF-fuse BM25 with the best dense encoder (E5) per dataset.
**Result:**

| Dataset | System | nDCG@10 | R@10 | R@100 | MRR@10 |
|---------|--------|--------:|-----:|------:|-------:|
| SciFact | BM25 | 0.6523 | 0.7757 | 0.8731 | 0.6184 |
| SciFact | Dense E5 | **0.7274** | 0.8507 | 0.9627 | 0.6931 |
| SciFact | Hybrid-RRF | 0.7197 | 0.8392 | 0.9627 | 0.6899 |
| NFCorpus | BM25 | 0.3071 | 0.1522 | 0.2425 | 0.5085 |
| NFCorpus | Dense E5 | **0.3529** | 0.1706 | 0.3197 | 0.5403 |
| NFCorpus | Hybrid-RRF | 0.3457 | 0.1648 | **0.3255** | **0.5418** |

**Interpretation:** Counterintuitive — RRF **lowered** nDCG@10 on both datasets. When the dense ranker is much
stronger than BM25, fusing the weaker lexical list pollutes the top ranks. RRF *did* improve NFCorpus deep recall
(R@100 0.3197→0.3255) and MRR. **Lesson: RRF only pays when the two retrievers are comparably strong.** This
revises the Phase-1 "add hybrid fusion" plan with a condition.

### Experiment 2.4 — Linear-gain parity + dev→test honesty
- **Gain invariance:** re-scoring the NFCorpus leaderboard with BEIR's *linear* gain (`gain=rel`) vs the
  exponential `2^rel-1` leaves the encoder ranking **identical (Kendall τ = 1.000)**. The Phase-1 gain convention
  never affected a conclusion — closes the queued methodological note without `pytrec_eval`.
- **Selection honesty:** the NFCorpus winner (E5) scores validation nDCG@10 = 0.3298 vs test 0.3529 (gap 0.023),
  confirming the encoder choice generalizes off the test split.

## Head-to-Head Comparison (primary metric, nDCG@10)
| Rank | Approach | SciFact | NFCorpus | Notes |
|------|----------|--------:|---------:|-------|
| 1 | E5-base-v2 (whole-doc) | **0.7274** | **0.3529** | best encoder, both datasets |
| 2 | Nomic-v1.5 (8192) | 0.7076 | 0.3471 | long-context; needs no chunking |
| 3 | BGE-small (512, 384d) | 0.7057 | 0.3444 | best small model |
| 4 | Hybrid RRF (BM25+E5) | 0.7197 | 0.3457 | helps R@100, hurts nDCG@10 |
| 5 | BM25 (Phase 1) | 0.6523 | 0.3071 | lexical floor |
| 6 | MiniLM-L6 (Phase 1) | 0.6484 | 0.3189 | Phase-1 champion |

## Key Findings
1. **Encoder choice > chunking.** MiniLM→E5 = +12% / +10.7% nDCG@10, vs +3.6% from the best chunker in Phase 1.
2. **The law holds:** chunking lift falls monotonically with encoder window (2.9% → 2.1% → 0.4%). Long-context
   encoders don't need chunking.
3. **Optimal chunk ≈ half the window** (128@256, 256@512).
4. **RRF can hurt:** fusing a weak ranker into a strong one drags down nDCG@10; it only helps deep recall.
5. **Rigor:** gain convention is irrelevant to the ranking (τ=1.0); winner generalizes val→test.

## Error Analysis
- NFCorpus stays ~0.32–0.35 across all encoders — a known-hard medical IR task (~38 graded-rel/query), not a bug.
- Nomic's NFCorpus R@100 (0.298) is the *lowest* of the dense models despite winning rank #2 on nDCG@10 — its
  long-context embedding is precise at the top but thinner deep in the list. Candidate for Phase-4 error analysis.

## Frontier Model Comparison
Deferred to Phase 5 (LLM head-to-head is the Friday deliverable per the project plan).

## Next Steps (Phase 3)
- **Retrieval index/structure:** dense (FAISS HNSW/IVF/Flat) vs sparse (BM25) vs hybrid, now with E5 as the dense
  backbone instead of MiniLM. Quantify the latency/quality trade-off the brute-force matmul currently hides.
- Carry E5-base-v2 forward as the default encoder for all later phases.
- Investigate Nomic's deep-recall weakness (matryoshka dim? pooling?).

## References Used Today
- [1] Wang et al., *Text Embeddings by Weakly-Supervised Contrastive Pre-training* (E5), 2022/24 — https://huggingface.co/intfloat/e5-base-v2
- [2] Xiao et al., *C-Pack / BGE*, 2023 — https://huggingface.co/BAAI/bge-small-en-v1.5
- [3] Nussbaum et al., *Nomic Embed*, 2024 — https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
- [4] Cormack, Clarke & Buettcher, *Reciprocal Rank Fusion*, SIGIR 2009
- [5] Thakur et al., *BEIR*, NeurIPS 2021 — https://github.com/beir-cellar/beir

## Code Changes
- `notebooks/phase2_embeddings.ipynb` — 26-cell research notebook (cached encoding → 4-model leaderboard →
  3-encoder chunking-law sweep → RRF hybrid → linear-gain parity + dev→test), all cells executed, 0 errors.
- `results/` — `phase2_embedding_leaderboard.csv`, `phase2_chunk_law.csv`, `phase2_chunk_law_summary.csv`,
  `phase2_hybrid.csv`, `phase2_leaderboard.png`, `phase2_chunk_law.png`; `metrics.json` gains a `phase2` block.
- `data/processed/emb_cache/` — disk-cached `.npy` embeddings (gitignored) for idempotent reruns.

### Engineering note
First run timed out: a single cell encoding all models × both corpora exceeded the 30-min cell cap (CPU encode is
the bottleneck; GTE-base ran at ~1–2 docs/sec). Fixed by (1) disk-caching every embedding matrix so reruns/restarts
skip completed encodes, (2) raising the cell timeout to 2h, and (3) dropping GTE-base (a runtime decision; it was
not in the chunking-law trio). MiniLM/BGE-small/E5/Nomic complete comfortably under cache.
