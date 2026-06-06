# Phase 6: Generation & Faithfulness — does any of the retrieval work change the answer?
**Date:** 2026-06-06
**Session:** 6 of 7

## Objective
Phases 1–5 spent five sessions moving one retrieval number — nDCG@10 — with chunking, encoders,
FAISS indexes, cross-encoder re-rankers and LLM query rewrites. Every gain rested on an unchecked
assumption: that a better-ranked context list produces a better **answer**. Phase 6 closes the loop:

> Does the retrieved context actually reach the generated answer, and does *better* retrieval make the
> answer measurably better — or does the LLM route around it? And does wrong-but-plausible retrieval
> merely fail to help, or does it actively manufacture confident hallucinations?

We build a real RAG generator (Claude Haiku over FiQA financial Q&A), sweep context quality from
oracle → strong (E5) → HyDE×N → adversarial → none, and score four RAGAS-style signals:
**faithfulness**, **answer relevancy**, **answer correctness vs the gold passage**, and **citation
grounding**. Two carry-over questions from Phases 4–5 are also resolved: the cross-encoder **backward
link** and the per-query **router** ceiling.

**Primary metrics:** retrieval = nDCG@10 (every prior phase); generation = faithfulness + correctness.
Generator and judge are both Claude; the judge model is **held fixed (Haiku)** while the generator
varies, so generator differences are not confounded with judge drift.

## Research & References
1. **Es, James, Espinosa-Anke, Schockaert (2023) — "RAGAS: Automated Evaluation of RAG" (arXiv:2309.15217).**
   Source of the four metric definitions. Faithfulness = fraction of answer claims entailed by the
   *retrieved context*; answer relevancy = regenerate questions from the answer, embed, compare to the
   original; context precision/recall from relevance labels. We re-implemented all of them with a Claude
   judge + the E5 encoder rather than the OpenAI-bound `ragas` package, for full offline control.
2. **Gao, Ma, Lin, Callan (2022) — HyDE (arXiv:2212.10496).** Source of the HyDE×N candidates (reused
   verbatim from the Phase-5 cache) used both as a context condition and in the Exp-E backward link.
3. **Cross-encoder re-ranking (Phase 4, in-repo).** The MiniLM-L6 cross-encoder that *hurt* nDCG@10 on
   the naive candidate set; Exp E re-tests it on the higher-recall HyDE×N candidates.

**How research shaped the experiments.** RAGAS separates *grounding in context* (faithfulness) from
*being right* (correctness vs reference). Keeping those two axes distinct is what surfaced the headline:
a wrong retrieval can score high faithfulness and low correctness simultaneously — the signature of a
context-poisoned hallucination.

## Dataset
Generation testbed: **FiQA-2018** (BEIR), genuine personal-finance Q&A so the answer is something a
user would actually read. A balanced **24-query** subset stratified by naive per-query nDCG@10 (8 hard /
8 mid / 8 solved). The no-LLM retrieval metrics (Exp A) and the backward link (Exp E) run on all three
BEIR corpora (SciFact, NFCorpus, FiQA) on the cached 40-query Phase-5 samples. All corpus embeddings are
the exact E5-base-v2 vectors from Phases 2–5 — nothing re-embedded on the document side.

| Corpus | Domain | Docs | Queries | Sample |
|--------|--------|-----:|--------:|-------:|
| SciFact | scientific claim verification | 5,183 | 300 | 40 |
| NFCorpus | medical / health | 3,633 | 323 | 40 |
| FiQA-2018 | personal finance / investing | 57,638 | 648 | 40 (24 for generation) |

## Experiments

### Experiment A — RAGAS context precision & recall, computed *exactly* from qrels (no LLM)
**Hypothesis:** HyDE×N (the Phase-5 winner) improves the *retrieval-quality axis* the generation
results will be plotted against. **Method:** context precision@5 = rank-weighted average precision over
the retrieved list; context recall@k = fraction of judged-relevant docs in the top-k. **Result:**

| Corpus | retriever | ctx_prec@5 | ctx_rec@5 | ctx_rec@10 | nDCG@10 |
|--------|-----------|-----------:|----------:|-----------:|--------:|
| SciFact | naive | 0.324 | 0.491 | 0.653 | 0.428 |
| SciFact | **HyDE×N** | **0.454** | **0.660** | **0.745** | **0.544** |
| NFCorpus | naive | 0.304 | 0.163 | 0.206 | 0.362 |
| NFCorpus | **HyDE×N** | **0.331** | **0.186** | **0.221** | **0.393** |
| FiQA | naive | 0.234 | 0.318 | 0.417 | 0.338 |
| FiQA | **HyDE×N** | **0.275** | **0.394** | **0.458** | **0.378** |

**Interpretation:** HyDE×N lifts both precision and recall on all three corpora — it genuinely puts more
relevant docs higher in the window. This is a *real* improvement on the retrieval axis, which makes the
generation result below all the more striking.

### Experiment B/C/D — RAG generation across 5 context conditions (the headline)
**Hypothesis:** answer correctness rises monotonically with context quality. **Method:** Haiku answers
each FiQA query under five context conditions; a fixed Haiku judge returns claim-level faithfulness,
3 sub-questions (→ E5 relevancy), and correctness vs the gold passage; citation grounding is computed
without the LLM. **Result (FiQA, n=24, mean):**

| Condition | Correctness | Faithfulness | Relevancy | Citation grounding | Refused | mean claims |
|-----------|------------:|-------------:|----------:|-------------------:|--------:|------------:|
| oracle (gold docs) | 0.604 | 0.944 | 0.876 | 1.000 | 0.13 | 4.6 |
| **HyDE×N top-5** | **0.604** | 0.993 | 0.885 | 1.000 | 0.21 | 4.9 |
| strong (E5 top-5) | 0.542 | 0.967 | 0.890 | 1.000 | 0.21 | 4.5 |
| adversarial (E5 r40–60) | **0.250** | 0.831 | 0.877 | 1.000 | 0.58 | 2.3 |
| closed-book (no context) | 0.396 | 0.000 | 0.868 | 0.04 | 0.04 | 6.2 |

**Interpretation — three genuine findings:**

1. **The retrieval→answer curve is flat past "good enough."** Going from the strong E5 retriever to a
   *perfect* oracle raises context precision@5 from 0.234 (E5) to 1.0 (oracle, by construction) but moves
   correctness only 6 points (0.542 → 0.604). **HyDE×N matches the oracle (0.604, within n=24 noise).** The
   LLM needs one relevant passage in the window; E5's top-5 already supplies it often enough. Five phases of
   retrieval tuning bought a large nDCG@10 gain and a small answer gain. *(Caveat: oracle feeds fewer but
   100%-relevant docs — mean 2.46 vs strong's 5 — so the flat oracle-vs-strong gap mixes precision with
   context length; see Limitations.)*
2. **Context poisoning is real and large.** Feeding unjudged, low-ranked passages (E5 ranks 40–60, gold
   removed — presumed off-topic, not labelled non-relevant) cut correctness by more than half (0.542 →
   **0.250**, a 54% drop) — *below* closed-book — while faithfulness stayed at **0.83**. The model dutifully
   grounds its answer in the garbage. Refusal jumps to 58% (the model often correctly says "the context
   doesn't answer this"), but when it doesn't refuse, it is confidently and faithfully wrong.
3. **No retrieval beats bad retrieval.** Closed-book correctness (0.396) exceeds adversarial (0.250):
   the LLM's parametric finance knowledge is better than a wrong retrieval. Closed-book faithfulness is
   **0.0 by construction** — there is no context to be faithful to — which cleanly demonstrates that
   faithfulness measures *grounding*, not *correctness*: a perfectly correct closed-book answer scores 0.

### Experiment E — the backward link: does HyDE×N's recall finally make a cross-encoder pay?
**Hypothesis (opened in Phase 4/5):** the cross-encoder hurt because it could only reorder the naive
candidate set; given HyDE×N's higher-recall candidates it should finally help. **Method:** MiniLM-L6
cross-encoder over the top-100 of the naive vs HyDE×N first stage. **Result (nDCG@10):**

| Corpus | E5 naive | HyDE×N (raw) | CE(naive top100) | CE(HyDE×N top100) |
|--------|---------:|-------------:|-----------------:|------------------:|
| SciFact | 0.428 | **0.544** | 0.453 (+0.025) | 0.471 (+0.043) |
| NFCorpus | 0.362 | **0.393** | 0.372 (+0.009) | 0.384 (+0.021) |
| FiQA | 0.338 | **0.378** | 0.316 (−0.022) | 0.324 (−0.014) |

**Interpretation:** the backward link **half-works**. Re-ranking the higher-recall HyDE×N candidates
does flip the cross-encoder from harmful to *helpful vs naive* on SciFact (+0.043) and NFCorpus (+0.021)
— confirming the Phase-5 hypothesis that better candidates give the re-ranker room. **But raw HyDE×N
still beats CE-on-HyDE×N on all three corpora.** Better retrieval doesn't *rescue* the re-ranker; it
makes it **redundant**. The cheapest correct pipeline is HyDE×N with no re-ranking at all.

### Experiment F — frontier generator head-to-head (same context, judge fixed)
**Method:** hold context at strong E5 top-5, swap only the generator; fixed Haiku judge. n=12 — the
**hardest tail** of the sample (mean naive nDCG@10 = 0.07; 8 of 12 queries are total retrieval failures,
nDCG@10 = 0), chosen to bound Codex's slower calls and to stress the generator where retrieval is weakest.
**Result:**

| Generator | Correctness | Faithfulness | Relevancy | Citation | Latency/call | Cost/1k calls |
|-----------|------------:|-------------:|----------:|---------:|-------------:|--------------:|
| Claude Haiku | 0.250 | 0.917 | 0.890 | 1.000 | 9.6 s | **$0.60** |
| Claude Opus | **0.375** | **0.988** | **0.946** | 1.000 | 10.6 s | $9.00 (15×) |
| Codex (GPT) | **0.375** | 0.967 | 0.937 | 1.000 | 11.7 s | $50.00 (83×) |

**Interpretation:** on the hard subset, both frontier generators convert 50% more answers to correct
(0.375 vs 0.250) and edge Haiku on faithfulness/relevancy — at 15–83× the cost. For the easy/mid
majority, retrieval-augmented Haiku already matches them (see Exp B/D); pay for the frontier generator
only on the hard tail. Codex completed 12/12 with zero errors. *(Latency is CLI-bound — all three are
~10 s/call via the local CLI; direct-API timing is far lower. The token-cost math reflects real API
pricing.)*

### Bonus — per-query router headroom (oracle ceiling)
The Phase-5 router thread: per query, pick the best of {naive, HyDE×N, PRF}. The **oracle** ceiling over
always-HyDE×N is **+0.027 (SciFact), +0.025 (NFCorpus), +0.046 (FiQA)** nDCG@10. HyDE×N is already near
the per-query frontier; a *learned* router would capture only part of an already-small ceiling — not
worth building in Phase 7.

## Head-to-Head Comparison (FiQA generation, n=24)
| Rank | Context condition | Correctness | Faithfulness | Notes |
|------|-------------------|------------:|-------------:|-------|
| 1 | HyDE×N top-5 | 0.604 | 0.993 | ties oracle at 1/4 the context precision |
| 1 | oracle (gold) | 0.604 | 0.944 | upper bound; HyDE×N matches it |
| 3 | strong E5 top-5 | 0.542 | 0.967 | production default; ~within noise of oracle |
| 4 | closed-book | 0.396 | 0.000 | parametric only; beats bad retrieval |
| 5 | adversarial | 0.250 | 0.831 | context poisoning — faithful to wrong docs |

## Key Findings
1. **Retrieval quality past "good enough" barely moves the answer.** Strong→oracle quadruples context
   precision for +6 pts correctness; HyDE×N ties oracle. The five-phase nDCG@10 chase has steeply
   diminishing downstream returns — the loop, closed.
2. **Wrong retrieval manufactures confident hallucinations.** Adversarial context cuts correctness by
   more than half (0.54→0.25, a 54% drop, below closed-book) while faithfulness stays 0.83. Retrieval
   misses don't just fail to help; they actively poison the answer. This is the post. *(n=24, directional;
   the 0.54-vs-0.25 gap is large but not significance-tested.)*
3. **No retrieval beats bad retrieval** (closed-book 0.40 > adversarial 0.25) — and faithfulness=0 for
   closed-book proves faithfulness ≠ correctness.
4. **The cross-encoder backward link half-works:** better candidates flip it positive-vs-naive on 2/3
   corpora, but raw HyDE×N still wins everywhere — the re-ranker is redundant, not rescued (closes the
   Phase-4/5 thread).
5. **Frontier generators win the hard tail only** (+50% correctness at 15–83× cost); cheap model + good
   retrieval suffices for the majority. **A per-query router isn't worth building** (oracle headroom ≤0.046).

## Frontier-Model Comparison
| Task | Metric | Haiku+E5 | Opus+E5 | Codex+E5 | Winner |
|------|--------|---------:|--------:|---------:|--------|
| FiQA answer (hard n=12) | correctness | 0.250 | 0.375 | 0.375 | Opus = Codex |
| FiQA answer | faithfulness | 0.917 | 0.988 | 0.967 | Opus |
| FiQA answer | relevancy | 0.890 | 0.946 | 0.937 | Opus |
| FiQA answer | cost / 1k calls | **$0.60** | $9.00 | $50.00 | Haiku |

## Error Analysis
- **Adversarial refusals (58%)** are the model's *correct* behavior — it detects the context is off-topic.
  The damage is the other 42%, where it produces a faithful answer to the wrong passage.
- **Closed-book over-claims:** highest mean claim count (6.2) and 0 faithfulness — long, confident,
  ungrounded answers. Exactly the failure RAG is supposed to fix.
- **FiQA is hard for everyone** (best correctness 0.60): many FiQA "relevant" passages are opinions from
  a Q&A forum, so a single gold passage is a noisy correctness reference — a known FiQA limitation.

## Limitations
*(Hardened after an adversarial self-review that re-derived every number from the CSVs — all numeric
claims matched source to rounding; the items below are the confounds that survived verification.)*
- **Sample size.** n=24 (generation) / n=12 (frontier) — small; deltas are directional, not
  significance-tested. The retrieval metrics (Exp A/E) run on n=40 × 3 corpora and are more robust.
- **Oracle confounds quality with length.** The `oracle` condition feeds only gold docs (mean 2.46 per
  query) while `strong` feeds a fixed 5. So the flat oracle-vs-strong correctness gap (0.604 vs 0.542)
  mixes a precision improvement (1.0 vs 0.234) with a ~49% shorter context — it is *not* a clean
  precision-only contrast. The headline (diminishing returns past "good enough") still holds because
  HyDE×N — which feeds 5 docs like `strong` — also matches the oracle.
- **Adversarial docs are presumed, not labelled, non-relevant.** They are E5 ranks 40–60 with gold
  removed; BEIR qrels don't judge them, so a few could be unjudged-relevant. The poisoning effect is a
  lower bound on truly adversarial context.
- **Judge sees the gold passage.** The single fixed Haiku judge is shown the gold reference (for the
  correctness field) while scoring faithfulness-against-context; this could marginally and *uniformly*
  inflate absolute faithfulness. Because the judge model and prompt are identical across all conditions
  and generators, *relative* comparisons remain valid.
- **Secondary-metric caveats.** Answer relevancy embeds sub-questions and the query with the same E5
  `query:` prefix (symmetric question-question similarity) and is near-constant (~0.88) across conditions,
  so it drives no headline. Citation grounding is *conditional on the answer emitting `[dN]` markers*; the
  closed-book answers rarely cite, so its citation-grounding is undefined (reported as NaN), not 0.
- **Latency is CLI-bound** (~10 s/call) and not representative of direct-API serving; the cost math is.

## Next Steps (Phase 7 — final session)
- **Build the optimal end-to-end pipeline** from all six phases: E5-base-v2 whole-doc + **HyDE×N**
  query expansion + **no re-ranker** (Exp E) + Haiku generation with citation enforcement. Compare
  end-to-end against the naive RAG baseline.
- **Productionize** (`src/`): config-driven retrieve→generate→cite pipeline, a Streamlit UI exposing the
  context-condition toggle so a user can *see* poisoning live, and the consolidated 6-phase results.
- Skip the per-query router (headroom too small) and skip the cross-encoder (redundant).

## References Used Today
- [1] Es, James, Espinosa-Anke, Schockaert (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation.* arXiv:2309.15217
- [2] Gao, Ma, Lin, Callan (2022). *Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE).* arXiv:2212.10496
- [3] Thakur et al. (2021). *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of IR Models.* (datasets, metric conventions)

## Code Changes
- `notebooks/phase6_generation_ragas.ipynb` — 16 code cells, all executed, 0 errors, 0 fake/display-only
  cells: 3-corpus load, HyDE×N reconstruction from cache, RAGAS context P/R (Exp A), cross-encoder
  backward link (Exp E), 5-condition generation (Exp B, cached), fixed-judge RAGAS scoring (Exp C),
  headline plots (Exp D), frontier head-to-head (Exp F), router headroom, metrics consolidation.
- `results/phase6_context_pr.csv`, `phase6_backward_link.csv`, `phase6_generation_summary.csv`,
  `phase6_generation_perq.csv`, `phase6_frontier.csv`, `phase6_router_headroom.csv`
- `results/phase6_generation_headline.png`, `phase6_backward_link.png`
- `results/phase6_llm_cache/{gen_answers.json, judge.json, frontier.json, frontier_judge.json}` (cached, idempotent)
- `results/metrics.json` → `phase6` block
