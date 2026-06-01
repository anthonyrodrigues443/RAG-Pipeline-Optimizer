"""TREC/BEIR-style retrieval metrics — graded gains (2^rel - 1), matching pytrec_eval.

Validated in notebooks/phase1_foundation_chunking.ipynb against the published BEIR
BM25 baseline (SciFact nDCG@10 ≈ 0.665; ours = 0.652).
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

Run = Dict[str, List[str]]          # qid -> [doc_id ranked best..worst]
Qrels = Dict[str, Dict[str, int]]   # qid -> {doc_id: gain}


def dcg(gains: Sequence[float]) -> float:
    """DCG with **exponential** gain ``2^rel - 1`` (Burges et al. / sklearn convention).

    Note: this coincides with trec_eval/BEIR's *linear* gain on binary qrels (rel∈{0,1}),
    so the SciFact validation is exact. On graded qrels (NFCorpus, rel∈{1,2}) it weights
    rel=2 slightly more than BEIR's linear gain — a monotonic transform applied identically
    to every strategy, so the chunking *ranking* is unaffected. A linear-gain parity pass
    against pytrec_eval is queued for Phase 2 (see reports/day1_phase1_report.md).
    """
    g = np.asarray(gains, dtype=float)
    if g.size == 0:
        return 0.0
    return float(np.sum((2 ** g - 1) / np.log2(np.arange(2, g.size + 2))))


def evaluate(run: Run, qrels: Qrels, ks: Sequence[int] = (1, 3, 5, 10, 20, 100)) -> Dict[str, float]:
    """Mean nDCG@k, Recall@k and MRR@k over **all judged queries**.

    Iterating over ``qrels`` (not ``run``) means a query missing from the run scores 0
    instead of being silently dropped; ranked doc-ids are de-duplicated per query so a
    malformed run cannot push Recall above 1.
    """
    out: Dict[str, List[float]] = {}
    for k in ks:
        out[f"ndcg@{k}"] = []
        out[f"recall@{k}"] = []
        out[f"mrr@{k}"] = []
    for qid, gold in qrels.items():
        if not gold:
            continue
        # de-dup ranked ids, keeping first (best) occurrence
        seen, ranked = set(), []
        for d in run.get(qid, []):
            if d not in seen:
                seen.add(d)
                ranked.append(d)
        n_rel = sum(1 for g in gold.values() if g > 0)
        gains = [gold.get(d, 0) for d in ranked]
        ideal = sorted(gold.values(), reverse=True)
        for k in ks:
            idcg = dcg(ideal[:k])
            out[f"ndcg@{k}"].append(dcg(gains[:k]) / idcg if idcg > 0 else 0.0)
            out[f"recall@{k}"].append(
                sum(1 for g in gains[:k] if g > 0) / n_rel if n_rel else 0.0
            )
            rr = 0.0
            for i, g in enumerate(gains[:k]):
                if g > 0:
                    rr = 1.0 / (i + 1)
                    break
            out[f"mrr@{k}"].append(rr)
    return {m: (float(np.mean(v)) if v else 0.0) for m, v in out.items()}


def topk_search(doc_emb: np.ndarray, query_emb: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Exact cosine top-k via a single matmul on L2-normalized vectors.

    Used instead of FAISS: on Apple Silicon faiss-cpu deadlocks against torch's libomp,
    and corpora here are small enough (<= ~20k vectors) that brute force is ~0.1s/batch.
    Returns (similarities, indices), both shape (n_queries, k).
    """
    k = min(k, doc_emb.shape[0])
    sims = query_emb @ doc_emb.T
    part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    rows = np.arange(query_emb.shape[0])[:, None]
    order = np.argsort(-sims[rows, part], axis=1)
    idx = part[rows, order]
    return sims[rows, idx], idx
