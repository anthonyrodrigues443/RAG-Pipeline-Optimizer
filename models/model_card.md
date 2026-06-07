# Model Card — RAG Pipeline Optimizer (the optimal pipeline)

This project ships a **recipe, not a trained checkpoint**. There are no learned weights to
serialize — the "model" is a configuration of off-the-shelf components, and every choice in
that configuration is an empirical finding from a 7-phase benchmark. This card follows the
Hugging Face / Google Model Card format.

## Overview

| | |
|---|---|
| **Name** | RAG Pipeline Optimizer — optimal end-to-end pipeline |
| **Type** | Retrieval-augmented generation pipeline (retriever + query transform + generator) |
| **Task** | Open-domain question answering over a document corpus |
| **Components** | E5-base-v2 encoder · exact NumPy cosine top-k · HyDE×N query transform · Claude Haiku generator with citation enforcement |
| **Trained weights** | None — all components are pre-trained and used as-is |
| **Code** | `src/pipeline.py`, `src/predict.py` |
| **License (code)** | project code MIT; component models under their own licenses (E5: MIT; Claude: Anthropic ToS) |

## Intended use

- **Primary:** answer factual questions grounded in a retrieved document corpus, with inline
  `[dN]` citations to the supporting passages. Demonstrated on **FiQA-2018** (personal-finance
  QA) and validated for retrieval on **SciFact** and **NFCorpus**.
- **Out of scope:** anything requiring the model to *not* be grounded (creative writing);
  high-stakes financial/medical/legal advice (the corpus is a Q&A forum, not authority); any
  setting where a confidently-wrong answer is dangerous and unverified — see *Limitations*.

## The pipeline (and why each piece)

1. **Encoder — E5-base-v2, whole document, 512-token window.** Phase 2: the encoder is the
   dominant retrieval lever (MiniLM→E5 = +12% nDCG@10, ~3× any chunking gain). Phase 1: with a
   512-token encoder, chunking below the window is a near-no-op, so we don't chunk.
2. **Index — exact cosine top-k via one NumPy matmul.** Phase 3: below ~40k vectors an ANN
   index (IVF/HNSW) *loses* nDCG to save sub-millisecond latency. Switch to HNSW `ef=128` only
   above the crossover.
3. **Query transform — HyDE×N (when an LLM budget exists).** Phase 5: average the query vector
   with N hypothetical-answer vectors. The only transform that helped on all three corpora;
   single-HyDE and step-back flipped sign by domain. Free **PRF** is the budget-free fallback.
4. **Re-ranker — none.** Phase 4: every cross-encoder tested *hurt* a strong E5 first stage,
   and Phase 6 showed it stays redundant even on HyDE×N's higher-recall candidates.
5. **Generator — Claude Haiku with citation enforcement.** Phase 6: a cheap generator on good
   retrieval matches a frontier generator for the easy/mid majority; reserve Opus/GPT for the
   hard tail (+50% correctness at 15-83× cost).

## Evaluation

**Retrieval** — nDCG@10, n=40 sampled queries/corpus (`results/phase7_retrieval.csv`):

| Corpus | Naive E5 | Optimal (HyDE×N) | Δ |
|--------|---------:|-----------------:|--:|
| SciFact | 0.428 | 0.544 | +0.116 |
| NFCorpus | 0.362 | 0.393 | +0.031 |
| FiQA | 0.338 | 0.378 | +0.040 |

**Generation** — FiQA, n=24, Haiku generator + fixed Haiku judge (`results/phase7_end_to_end.csv`):

| Context | Correctness | Faithfulness | Citation |
|---------|------------:|-------------:|---------:|
| Oracle (gold) | 0.604 | 0.94 | 1.00 |
| **Optimal (HyDE×N)** | **0.604** | 0.99 | 1.00 |
| Naive (E5) | 0.542 | 0.97 | 1.00 |
| Closed-book | 0.396 | 0.00 | — |
| Adversarial | 0.250 | 0.83 | 1.00 |

**Latency (live, CPU):** ~37 s/query for the HyDE×N path (2 sequential Claude CLI calls + E5
encode). The token-cost math is representative of API usage; the wall-clock includes CLI
startup — expect 5-10× speedup via the direct API.

## Limitations & ethical considerations

- **Context poisoning is the dominant failure mode.** A *wrong* retrieval cut answer
  correctness to 0.250 — below closed-book (0.396) — while the model stayed 0.83 "faithful" to
  the wrong passages and kept citing them. Faithfulness ≠ correctness. Deployments must monitor
  retrieval *reliability*, not just relevance scores.
- **The judge is an LLM.** Correctness/faithfulness are scored by a fixed Claude Haiku judge
  that sees the gold reference. Numbers are internally consistent and reproduce deterministically
  from the cache, but they are LLM-judged, not human-adjudicated; n=24 for generation is small.
- **Domain coverage.** Generation is evaluated only on FiQA (personal finance). Retrieval
  generalizes across SciFact/NFCorpus/FiQA, but the generation findings may not transfer to
  every domain.
- **No safety/PII filtering** in the demo pipeline; not production-hardened for sensitive data.

## Reproduce

```bash
pip install -r requirements.txt
pytest -q                              # 45 offline tests
python -m src.evaluate --phase7        # deterministic end-to-end table + plot
python -m src.predict "How is freelance income taxed for a sole proprietor?"
```
