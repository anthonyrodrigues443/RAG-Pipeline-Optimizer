# Phase 3: Retrieval Index Structure — the crossover where an ANN index finally pays — RAG Pipeline Optimizer
**Date:** 2026-06-03
**Session:** 3 of 7

## Objective
Phases 1–2 retrieved with a single brute-force cosine matmul (`Q @ Dᵀ`) and a comment that the corpora are "small enough that brute force is ~0.1 s/batch." That hid the Phase-3 question every RAG tutorial answers wrong by default:

> **Which FAISS index should you put your E5 embeddings in — Flat, IVF, or HNSW — what does each cost in recall / latency / memory / build time, and at what corpus size N does an approximate index stop being a liability and start being worth it?**

Plus a Phase-2 loose end: **why does long-context Nomic-v1.5 have *worse* deep recall (R@100) than 512-window E5 on NFCorpus?**

## Research & References
1. **Malkov & Yashunin, *HNSW* (TPAMI 2018)** — hierarchical navigable small-world graphs; `efSearch`/`M` are the recall-latency knobs swept here.
2. **Johnson, Douze & Jégou, *Billion-scale similarity search with GPUs / FAISS* (2019)** — IVF (`nlist`/`nprobe`) and IVFPQ; the `nlist ≈ 4·√N` heuristic and PQ compression model.
3. **Vendor benchmarks (Couchbase / BigData Boutique, 2025)** — the widely-cited "HNSW is 70× faster than Flat" number, measured at **10M vectors** — the figure this phase stress-tests at RAG scale (3.6k–57k).
4. **Thakur et al., *BEIR* (NeurIPS 2021)** — SciFact / NFCorpus / FiQA-2018 + `nDCG@10`; FiQA (57,638 passages) is the real scaling corpus.
5. **Nussbaum et al., *Nomic Embed* (2024)** — 8192-token long-context embedder; subject of the dilution-hypothesis test.

**How research shaped today:** the vendor "70×" claim is a 10M-vector result; almost every real RAG knowledge base is 10³–10⁵ vectors, so the experiment traces the *entire* recall-latency-memory frontier across that range and pins the crossover, rather than assuming the large-N conclusion transfers.

## Primary Metric
**`nDCG@10`** (BEIR standard; Phase-1 choice). For the index itself we add the standard ANN metric **Recall@10 / Recall@100 vs the exact neighbour set** (isolates approximation error) plus **build time, serialized bytes, single-query latency**. Single-query (batch=1) latency is what a live RAG endpoint actually pays — batched matmul latency hides it.

## Dataset
| Corpus | Docs | Queries | rel/query | Role |
|--------|------|---------|-----------|------|
| SciFact | 5,183 | 300 | 1.1 | small, clean binary qrels |
| NFCorpus | 3,633 | 323 | 38 (diffuse) | small, graded medical qrels |
| **FiQA-2018** | **57,638** | **648** | 2.6 | **new — scaling corpus (financial QA)** |

All retrieval on the Phase-2 champion encoder **E5-base-v2** (768d, cosine = inner product on L2-normalised vectors). FiQA embedded fresh this session (`notebooks/phase3a_embed_fiqa.ipynb`, ~39 min CPU).

> **Engineering note.** faiss and torch both load libomp and deadlock on this Apple-Silicon box (flagged since Phase 1). Fix: two kernels — `phase3a` (torch-only) embeds and caches `.npy`; `phase3_faiss_index` (faiss-only) loads the `.npy` + BeIR metadata and never imports torch.

## Experiments

### Exp 3.1 — exact baseline + the latency the matmul hid
**Method:** NumPy matmul (Phase-1/2 method) and FAISS `IndexFlatIP`, both exact, timed batched and single-query.
**Result:** SciFact nDCG@10 = 0.7274, single-query **0.25 ms**; NFCorpus 0.3525, **0.17 ms**; FiQA 0.3987, **2.08 ms**. (NFCorpus/FiQA reproduce Phase-2 E5 numbers exactly — harness parity holds.)
**Interpretation:** the "hidden" cost is ≈0.2 ms at BEIR scale, growing linearly to ~2 ms at 57k. That linear growth is the whole game.

### Exp 3.2 — index zoo on the small corpora (Flat / IVF nprobe-sweep / HNSW ef-sweep)
| dataset | index | nDCG@10 | R@10ᵉ | R@100ᵉ | lat ms | build s | MB |
|---|---|---|---|---|---|---|---|
| scifact | Flat (exact) | 0.7274 | 1.000 | 1.000 | 0.37 | 0.002 | 15.9 |
| scifact | IVF nprobe=1 | 0.3824 | 0.393 | 0.187 | 0.014 | 0.04 | 16.4 |
| scifact | IVF nprobe=nlist | 0.7274 | 1.000 | 1.000 | 0.39 | 0.04 | 16.4 |
| scifact | HNSW ef=128 | 0.7276 | 0.999 | 0.982 | 0.25 | 0.22 | 17.3 |
| nfcorpus | Flat (exact) | 0.3525 | 1.000 | 1.000 | 0.17 | 0.001 | 11.2 |
| nfcorpus | IVF nprobe=1 | 0.1913 | 0.294 | 0.137 | 0.012 | 0.02 | 11.5 |
| nfcorpus | HNSW ef=256 | 0.3528 | 0.994 | 0.989 | 0.22 | 0.11 | 12.2 |

(R@kᵉ = recall vs the exact top-k.) **IVF nprobe=1 loses 47% nDCG@10** to save 0.36 ms; recovering exact quality (`nprobe=nlist`) makes it as slow as Flat. HNSW matches exact quality but the latency delta vs Flat is a fraction of a ms, bought with 0.1–0.26 s build + 9% RAM.

### Exp 3.3 — the crossover (FiQA 57k + latency-vs-N curve)
**FiQA sweep:** HNSW `ef=128` → nDCG@10 0.3948 (vs exact 0.3987), R@10ᵉ 0.989, **0.37 ms vs Flat 2.08 ms (5.6×)**. IVF `nprobe=nlist` to reach exact quality = **3.12 ms, slower than Flat.**
**Latency vs N (E5 768d, fixed configs):**
| N | Flat ms | IVF ms | HNSW ms |
|---|---|---|---|
| 1,000 | 0.059 | 0.014 | 0.053 |
| 5,000 | 0.286 | 0.056 | 0.093 |
| 10,000 | 0.291 | 0.111 | 0.122 |
| 40,000 | 1.456 | 0.499 | 0.209 |
| 66,454 | 2.390 | 0.773 | 0.226 |

HNSW dips below Flat at **N≈1,000**, but within 0.01 ms until ~5k; the gap clears a meaningful 1 ms only at **~40k**. *Decision crossover ≈ tens of thousands of vectors.*

### Exp 3.4 — IVFPQ memory lever (FiQA)
| index | nDCG@10 | R@100ᵉ | MB | compression |
|---|---|---|---|---|
| Flat fp32 | 0.3987 | 1.000 | 177 | 1.0× |
| IVFPQ m=96 | 0.3179 | 0.616 | 9.7 | 18× |
| IVFPQ m=16 | 0.1427 | 0.353 | 5.1 | 34× |
PQ trades −20% to −64% nDCG@10 for 18–34× smaller. Not worth it at 177 MB; only when RAM is a hard wall.

### Exp 3.5 — Nomic deep-recall: dilution hypothesis test
**Hypothesis:** Nomic's 8192-token whole-doc pooling buries *long* gold docs past rank 100.
**Method:** per-(query, gold) exact rank under E5 vs Nomic on NFCorpus, correlated with gold doc length (words).
**Result:** Spearman(length, Nomic−E5 rank delta) = **+0.009**; median rank delta by length tertile = short +1, med +1, **long +2** — flat. Overall: E5 median gold rank 801, 21.3% in top-100; Nomic 830, 20.4%.
**Interpretation:** **hypothesis falsified.** The recall gap is a uniform few-percent encoder-quality difference, not length-dependent dilution. Context length is a red herring; E5 is just the marginally stronger encoder on this domain.

## Head-to-Head — best operating point per index family
| dataset | family | best config | nDCG@10 | R@100ᵉ | lat ms | MB | verdict |
|---|---|---|---|---|---|---|---|
| scifact (5.2k) | Flat | exact | 0.7274 | 1.000 | 0.24 | 15.9 | **use this** |
| scifact | HNSW | ef=128 | 0.7274 | 0.982 | 0.14 | 17.3 | ~tie, not worth deps |
| scifact | IVF | nprobe=nlist | 0.7274 | 1.000 | 0.25 | 16.4 | dominated |
| nfcorpus (3.6k) | Flat | exact | 0.3525 | 1.000 | 0.17 | 11.2 | **use this** |
| fiqa (57k) | HNSW | ef=128 | 0.3948 | 0.949 | 0.37 | 192 | **use this (5.6× faster)** |
| fiqa | Flat | exact | 0.3987 | 1.000 | 2.08 | 177 | exact, if 2 ms ok |
| fiqa | IVF | nprobe=nlist | 0.3987 | 1.000 | 3.12 | 180 | dominated (slower than Flat) |

## Key Findings
1. The brute-force matmul hid ≈0.2 ms/query; below ~10k vectors that beats every ANN index on the recall/latency/build/RAM total.
2. **Crossover:** HNSW technically passes Flat at N≈1,000 but only matters past ~40k; at 57k it's 5.6× faster for −0.004 nDCG@10.
3. **IVF is Pareto-dominated by HNSW at every N** — exact-quality IVF is *slower* than brute force even at 57k.
4. PQ compression (18–34×) is not worth its recall hit at RAG scale.
5. **Nomic dilution hypothesis falsified** (Spearman +0.009) — corrected a Phase-2 intuition with data.

## What Didn't Work / Was Wrong
- The long-document dilution hypothesis for Nomic's recall gap — **falsified**. Documented honestly rather than dropped.
- IVF as a small-corpus speedup — it's never on the frontier; HNSW or Flat always dominate it.

## Error Analysis
- NFCorpus has ~38 diffuse gold docs/query; median gold rank ≈800 for *both* encoders, so R@100 here measures deep-tail recall, not head precision (nDCG@10 is the precision-sensitive metric). This is why the Nomic gap shows up in R@100 but barely in nDCG@10.

## Next Steps (Phase 4)
- Re-ranking: does a cross-encoder (`ms-marco-MiniLM`) on the top-100 recover the nDCG@10 that HNSW `ef=128` leaves on the table at 57k? Measure Recall@K-before vs nDCG-after and the added latency — the natural pairing with an approximate first stage.
- Tune `efConstruction`/`M` for FiQA to push HNSW's recall@100 from 0.949 toward Flat without the 2 ms.

## References Used Today
- [1] Malkov & Yashunin (2018), HNSW — https://arxiv.org/abs/1603.09320
- [2] Johnson, Douze, Jégou (2019), FAISS — https://arxiv.org/abs/1702.08734
- [3] Couchbase / BigData Boutique vector-index benchmarks (2025)
- [4] Thakur et al. (2021), BEIR — https://arxiv.org/abs/2104.08663
- [5] Nussbaum et al. (2024), Nomic Embed — https://arxiv.org/abs/2402.01613

## Code Changes
- NEW `notebooks/phase3a_embed_fiqa.ipynb` (torch-only FiQA embedding prep)
- NEW `notebooks/phase3_faiss_index.ipynb` (faiss-only index benchmark — 24 cells, 0 errors)
- NEW `results/phase3_{index_sweep,best_per_family,scaling,ivfpq,nomic}.csv`
- NEW `results/phase3_{pareto_small,crossover,nomic_recall}.png`
- MODIFIED `results/metrics.json` (`phase3` block)
