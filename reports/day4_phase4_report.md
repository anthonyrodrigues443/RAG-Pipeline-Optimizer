# Phase 4: Re-ranking — the #1 RAG quality lever made retrieval *worse*, and the bigger the re-ranker the worse it got — RAG Pipeline Optimizer
**Date:** 2026-06-04
**Session:** 4 of 7

## Objective
Phases 1–3 settled the first stage: chunk at the encoder window, embed with **E5-base-v2** (Phase-2 champion), store in Flat/HNSW (Phase-3 crossover). Dense retrieval is fast but order-blind. Every RAG tutorial's next move is the same: **bolt on a cross-encoder re-ranker** — re-score the top-K with full query×document cross-attention, "the single biggest off-the-shelf quality lever." This session stress-tests that advice:

> Given the Phase-3 E5 candidate set, which re-ranker (TinyBERT-L2 → MiniLM-L6/L12 → BGE-reranker-base → an LLM listwise re-ranker) maximises **nDCG@10**, how deep should you re-rank, what does it cost, and **does the cross-encoder actually beat a frontier LLM — or even beat doing nothing?**

The answer turned out to be the most counterintuitive result of the project so far.

## Research & References
1. **Nogueira & Cho, *Passage Re-ranking with BERT* (2019)** — the cross-encoder paradigm (first stage proposes top-K, BERT re-scores). The `ms-marco-*` cross-encoders tested here are this lineage, trained on MS MARCO web passages.
2. **Thakur et al., *BEIR* (NeurIPS 2021)** — same SciFact/NFCorpus/FiQA corpora + `nDCG@10`. BEIR's own appendix already hints that MS-MARCO cross-encoders transfer *unevenly* out-of-domain — the thread this phase pulls on.
3. **Xiao et al., *C-Pack / BGE* (2023–24)** — `bge-reranker-base` (278M), the "modern strong" re-ranker; included as the heavy end of the zoo.
4. **Sun et al., *RankGPT* (EMNLP 2023)** — listwise/permutation LLM re-ranking; the protocol for Exp 4.5's Claude/GPT head-to-head.
5. **Pradeep et al., *RankVicuna/RankZephyr* (2023)** — distilling listwise LLM rankers into small models; framing for the cost/latency comparison.

**How research shaped today:** because a re-ranker can only *reorder the candidate set*, Exp 4.1 pins the **oracle ceiling** (best achievable nDCG@10 given E5's top-K) before any model runs. Every later number is read against that ceiling, not against 1.0.

## Primary Metric
**`nDCG@10`** (BEIR standard, carried from Phases 1–3). Secondary: `MRR@10`, `Recall@10`, first-stage `Recall@K`. Cost axis: re-rank **latency (ms/query)** and **USD/1k**.

## Dataset
| Corpus | Docs | Queries | rel/q | Role | E5 nDCG@10 (parity) |
|--------|------|---------|-------|------|---------------------|
| SciFact | 5,183 | 300 | 1.1 | clean, easy | 0.7274 ✓ (P3 = 0.7274) |
| NFCorpus | 3,633 | 323 | 38 | diffuse medical | 0.3532 (P3 = 0.3525)¹ |
| FiQA-2018 | 57,638 | 648 | 2.6 | financial QA, scaling | 0.3987 ✓ (P3 = 0.3987) |

¹NFCorpus differs by 0.0007 — argpartition top-200 tie-breaking vs Phase-3 faiss top-100; alignment confirmed.

All re-ranking re-scores E5's top-100 candidates (top-200 retrieved). Re-rankers need raw **text**, not embeddings, so the BEIR corpora are loaded fresh and aligned 1:1 with the cached E5 vectors.

> **Engineering note.** Re-ranking 4 models × 3 corpora on CPU is hours of work. The first run died in a monolithic cell at a 3 h timeout with nothing saved. Fix: every cross-encoder score array is cached to `results/phase4_cache/` keyed by `(model, corpus, depth)` the moment it's computed — a timed-out run resumes instead of restarting. The 278M BGE model (~33 pairs/s on CPU) was the bottleneck and was cache-warmed out-of-band.

## Experiments

### Exp 4.1 — The recall ceiling (what a *perfect* re-ranker could do)
| dataset | E5 nDCG@10 | R@10 | R@100 | R@200 | oracle@10 | oracle@100 | oracle@200 |
|---|---|---|---|---|---|---|---|
| scifact | 0.7274 | 0.851 | 0.963 | 0.980 | 0.854 | **0.963** | 0.980 |
| nfcorpus | 0.3532 | 0.171 | 0.320 | 0.389 | 0.426 | **0.646** | 0.708 |
| fiqa | 0.3987 | 0.471 | 0.732 | 0.792 | 0.519 | **0.771** | 0.825 |

**Interpretation:** huge headroom exists. On FiQA a perfect re-ranker of the top-100 could reach **0.771** vs E5's 0.399 — the gold docs *are* retrieved, just badly ordered. On SciFact E5 (0.727) is already within 0.001 of oracle@10 (0.854 only by reaching deeper) — almost nothing to gain at shallow depth. The candidate sets are not the bottleneck. The question is whether any *real* re-ranker can convert that headroom.

### Exp 4.2 — The cross-encoder zoo (re-score E5 top-100) — **the headline**
**Hypothesis:** bigger cross-encoder → higher nDCG@10, monotonically.
**Result:** the opposite. **Every re-ranker on every corpus underperformed the E5 baseline**, and the 278M BGE model was the *worst* on 2 of 3.

| reranker | params | scifact | nfcorpus | fiqa | mean Δ vs E5 |
|---|---|---|---|---|---|
| **E5 (no rerank)** | — | **0.7274** | **0.3532** | **0.3987** | — |
| TinyBERT-L2 | 4M | 0.6576 | 0.3350 | 0.3299 | −0.052 |
| MiniLM-L6 | 22M | 0.6858 | 0.3530 | 0.3862 | −0.018 |
| MiniLM-L12 | 33M | 0.6940 | 0.3539 | 0.3893 | −0.013 |
| BGE-base | 278M | 0.6570 | 0.2968 | 0.3134 | **−0.069** |
| *oracle@100* | — | *0.9629* | *0.6459* | *0.7706* | *(ceiling)* |

**Interpretation:** the MS-MARCO cross-encoders are trained on web search; SciFact (scientific), NFCorpus (medical) and FiQA (financial) are **out-of-domain**. A strong modern bi-encoder (E5) has already absorbed these domains via contrastive pre-training, so a web-trained re-ranker is a *weaker* ranker than the retriever it's "correcting." Larger ≠ better: BGE-base, despite 12× the parameters of MiniLM-L6, was the worst — capacity doesn't help when the training distribution is wrong. The oracle row proves the candidates were fine; the models simply rank them worse than E5 does.

### Exp 4.3 — Re-rank depth sweep — **deeper is monotonically worse**
**Hypothesis:** more candidates → more chances to surface a buried gold → better.
**Result:** strictly inverted. MiniLM-L6, nDCG@10 by depth:

| depth K | fiqa | scifact | fiqa ms/q |
|---|---|---|---|
| 10 | **0.4037** | **0.7136** | 39 |
| 25 | 0.4024 | 0.7030 | 99 |
| 50 | 0.3943 | 0.6912 | 203 |
| 100 | 0.3862 | 0.6858 | 472 |
| 200 | 0.3786 | 0.6788 | 860 |

**Interpretation:** at depth 10 the re-ranker is roughly neutral (FiQA 0.4037 actually edges E5's 0.3987 — the *only* configuration in the whole zoo that beat the baseline). Every candidate added past 10 injects a hard negative the bi-encoder had correctly buried, and the weak cross-encoder occasionally over-scores it — so quality falls *and* latency rises 22×. The universal advice "retrieve 100, re-rank to 10" is exactly backwards here: re-rank **10**, or don't re-rank.

### Exp 4.4 — First stage vs re-ranker (the mechanism: re-rankers are *equalisers*)
Re-rank BM25, E5, and hybrid candidate sets on FiQA with the *same* MiniLM-L6:

| first stage | pre nDCG@10 | R@100 | post nDCG@10 | Δ |
|---|---|---|---|---|
| BM25 (sparse) | 0.2175 | 0.475 | 0.3172 | **+0.0997** |
| E5 (dense) | 0.3987 | 0.732 | 0.3862 | **−0.0125** |
| Hybrid RRF | 0.3554 | 0.707 | 0.3709 | +0.0155 |

**Interpretation — this is the whole story.** The re-ranker pulls a *weak* first stage **up** (+0.10 for BM25) and a *strong* one **down** (−0.01 for E5), collapsing all three onto its own ~0.32–0.39 quality band regardless of input. A cross-encoder doesn't add absolute quality — it imposes *its* ranking, which is an upgrade over BM25 and a downgrade over E5. "Re-rankers always help" is an artifact of a literature that benchmarks them on **BM25**. Re-rank a retriever that already clears the re-ranker's ceiling and you've bought latency to *lose* accuracy.

### Exp 4.5 — LLM-as-re-ranker (RankGPT listwise, FiQA E5 top-10, n=30 stratified)
| reranker | nDCG@10 | MRR@10 | latency/q | cost/1k | parse |
|---|---|---|---|---|---|
| Oracle@10 (ceiling) | 0.8079 | 1.000 | — | — | — |
| **GPT-5.x (codex)** | **0.7203** | 0.886 | 15.2 s | $50.00 | 100% |
| MiniLM-L6 (22M) | 0.6807 | 0.817 | **49 ms** | **$0.001** | 100% |
| Claude Haiku | 0.6684 | 0.791 | 22.4 s | $1.35 | 100% |
| E5 (no rerank) | 0.6506 | 0.790 | 2 ms | $0 | — |
| Claude Opus | 0.6443 | 0.732 | 7.2 s | $20.20 | 100% |

**Interpretation (on the winnable subset — queries with ≥1 retrievable gold in top-10):** the frontier LLM **GPT-5.x is the only re-ranker that meaningfully beats both the cross-encoder and no-rerank** (0.720 vs 0.681 vs 0.651). But it costs **$50/1k and 15 s/query**. The 22M cross-encoder captures ~80% of GPT-5.x's lift-over-baseline at **1/50,000th the cost and 300× the speed** ($0.001/1k, 49 ms). And the mirror of Exp 4.2: the *biggest* model loses — **Claude Opus re-ranking was worse than doing nothing** (0.644 < 0.651), while smaller Haiku helped. Listwise re-ranking quality is not monotone in model size for either family.

### Exp 4.6 — Error analysis: re-ranking breaks more than it rescues
BGE-base vs E5 on all 648 FiQA queries: **rescued 140 · broke 267 · unchanged 241.** Mean Δ = **−0.085**; broken queries lose −0.328 on average, rescued gain +0.231 — it breaks nearly 2× as many and the breaks are bigger. Worst case (`qid 4827`, *"Are all financial advisors compensated in the same way?"*): E5 ranked the gold passage **#1 (nDCG 1.000)**, BGE pushed it out of the top-10 entirely (**0.000**). Concrete proof of the equaliser effect — a confident, correct bi-encoder rank, demoted by a re-ranker reasoning past the right answer on surface features.

## Head-to-Head — best re-ranking config per corpus
| corpus | E5 (no rerank) | best CE (config) | best CE Δ | verdict |
|---|---|---|---|---|
| scifact | **0.7274** | 0.6940 (MiniLM-L12) | −0.033 | **don't re-rank** |
| nfcorpus | 0.3532 | 0.3539 (MiniLM-L12) | +0.001 | neutral; not worth latency |
| fiqa | 0.3987 | 0.4037 (MiniLM-L6 @depth-10) | +0.005 | re-rank top-**10** only, or skip |

## Key Findings
1. **The cross-encoder — RAG's most-recommended quality lever — *reduced* nDCG@10 on all 3 BEIR corpora when re-ranking a strong dense retriever.** Mean −0.013 to −0.069. The advice assumes a weak (BM25) first stage.
2. **Bigger re-ranker = worse.** 278M BGE-base was the worst of four (mean −0.069); 33M MiniLM-L12 the least-bad. Same pattern in the LLM family — **Opus < Haiku**, and Opus < no-rerank.
3. **Deeper re-ranking is monotonically worse** (FiQA 0.404 @10 → 0.379 @200). Re-rank shallow or not at all; "retrieve 100 → rerank" is backwards here.
4. **Mechanism: re-rankers are equalisers, not amplifiers.** They lift BM25 +0.10 and drag E5 −0.01 onto their own ~0.35 band. They add value only when the first stage is below that band.
5. **GPT-5.x is the one re-ranker that beats E5** on the winnable subset (0.720 vs 0.651) — but at $50/1k and 15 s/query, where a 22M cross-encoder gets 0.681 at $0.001/1k and 49 ms.

## Frontier Model Comparison (FiQA listwise re-rank, n=30)
| Model | nDCG@10 | Latency/q | Cost/1k | vs MiniLM-L6 |
|---|---|---|---|---|
| MiniLM-L6 (custom, 22M) | 0.6807 | 49 ms | $0.001 | — |
| GPT-5.x codex | 0.7203 | 15.2 s | $50.00 | +0.040 quality / **50,000× cost, 300× latency** |
| Claude Haiku | 0.6684 | 22.4 s | $1.35 | −0.012 |
| Claude Opus | 0.6443 | 7.2 s | $20.20 | −0.036 |

## Error Analysis
- Re-ranking is **net-negative on a strong first stage**: 267/648 FiQA queries degraded vs 140 improved (BGE).
- Failure mode: confident correct bi-encoder ranks (gold @#1) demoted by surface-feature over-scoring → catastrophic single-query drops (1.000 → 0.000).
- The damage grows with re-rank depth and with model size — both add hard-negative exposure / over-reasoning.

## Next Steps (Phase 5 — Query techniques)
- The real lever isn't re-scoring — it's **better candidates**. Phase 5 tests HyDE, multi-query, query decomposition, step-back prompting: do they raise first-stage Recall@K (the thing that *did* cap everything here)?
- Re-visit re-ranking **only on the weak-first-stage path** (BM25-only / keyword fallback), where Exp 4.4 shows +0.10 — a conditional re-ranker, not a default one.
- Consider a **domain-tuned** cross-encoder (fine-tune on FiQA qrels) to test whether the out-of-domain hypothesis (Finding 1's mechanism) is the true cause.

## References Used Today
- [1] Nogueira & Cho (2019), *Passage Re-ranking with BERT*, arXiv:1901.04085
- [2] Thakur et al. (2021), *BEIR*, NeurIPS Datasets & Benchmarks — https://github.com/beir-cellar/beir
- [3] Xiao et al. (2024), *C-Pack: Packed Resources for General Chinese Embeddings (BGE)*, arXiv:2309.07597
- [4] Sun et al. (2023), *Is ChatGPT Good at Search? (RankGPT)*, EMNLP, arXiv:2304.09542
- [5] Pradeep et al. (2023), *RankVicuna / RankZephyr*, arXiv:2309.15088 / 2312.02724

## Code Changes
- `notebooks/phase4_reranking.ipynb` — 32 cells, 0 errors; resumable score-cache architecture (`results/phase4_cache/`)
- `results/phase4_reranker_zoo.{csv,png}`, `phase4_depth_sweep.{csv,png}`, `phase4_firststage.csv`, `phase4_llm_vs_custom.{csv,png}`, `phase4_error_analysis.png`
- `results/metrics.json` — added `phase4` block (zoo, ceiling, depth sweep, first-stage, LLM head-to-head, error analysis)
