"""Retrieval-Reliability Gate — Phase 8.

A model-free-features + tiny-classifier guard that, given only the retriever's own cosine
*score curve* (no gold labels, no second embedding model, no LLM), predicts whether the
top-k context is reliable enough to ground an answer — so the pipeline can **escalate** a
flagged query to HyDE×N instead of letting a plausible-but-wrong context poison the generator.

The Phase-8 finding this encodes: the *magnitude* of similarity is nearly useless for
reliability (top-1 cosine AUROC ≈ 0.63), but the *shape* of the curve is not (margin from the
rank-20–100 background, peakedness/entropy, and the flat-tie fraction reach ≈ 0.80). A
logistic regression on the eight features below scored AUROC 0.801 pooled / 0.75–0.83
leave-one-corpus-out over BEIR SciFact+NFCorpus+FiQA — beating Claude Haiku/Opus and Codex
as zero-shot judges at ~10⁴–10⁵× lower cost, in ~188 µs/query.

Pure NumPy/scikit-learn; no network or heavy ML deps, so it unit-tests on a clean checkout.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

FEATURE_NAMES: List[str] = [
    "top1", "top5_mean", "gap12", "gap15", "std10", "margin_bg", "entropy", "frac_tie",
]


def gate_features(scores: Sequence[float]) -> List[float]:
    """Eight statistics of a descending cosine score curve. Identical to the Phase-8 notebook.

    ``scores`` is the retriever's similarity list for one query (any length ≥ 2; the canonical
    case is top-100). Indices beyond the available length degrade gracefully — the "background"
    falls back to whatever sits below rank 20, and the top-k windows clip to the array.
    """
    s = np.sort(np.asarray(scores, dtype=float))[::-1]
    n = s.size
    if n < 2:
        raise ValueError("need at least 2 scores to characterise the curve")
    top1 = s[0]
    top5_mean = s[: min(5, n)].mean()
    gap12 = s[0] - s[1]
    gap15 = s[0] - s[min(4, n - 1)]
    std10 = s[: min(10, n)].std()
    bg_slice = s[20:100] if n > 20 else s[max(1, n // 2):]   # rank-20–100 background, or lower half
    bg = bg_slice.mean() if bg_slice.size else s[-1]
    margin_bg = top1 - bg
    top20 = s[: min(20, n)]
    z = (top20 - top20.max()) / 0.05
    p = np.exp(z)
    p /= p.sum()
    entropy = float(-np.sum(p * np.log(p + 1e-12)))
    frac_tie = float(np.mean(top20 >= top1 - 0.02))
    return [top1, top5_mean, gap12, gap15, std10, margin_bg, entropy, frac_tie]


class RetrievalReliabilityGate:
    """StandardScaler → LogisticRegression over :func:`gate_features`.

    Predicts ``P(unreliable)`` — the probability that the top-k retrieval lacks a gold passage.
    Logistic regression (not gradient boosting) by Phase-8 ablation: the linear model generalised
    better to unseen corpora and is trivially cheap.
    """

    def __init__(self, threshold: float = 0.5):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.threshold = threshold
        self._model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        )
        self._fitted = False

    @staticmethod
    def featurize(score_curves: Sequence[Sequence[float]]) -> np.ndarray:
        """Stack :func:`gate_features` over many queries → ``(n_queries, 8)``."""
        return np.asarray([gate_features(s) for s in score_curves], dtype=float)

    def fit(self, score_curves: Sequence[Sequence[float]], unreliable: Sequence[int]) -> "RetrievalReliabilityGate":
        self._model.fit(self.featurize(score_curves), np.asarray(unreliable, dtype=int))
        self._fitted = True
        return self

    def predict_proba(self, score_curves: Sequence[Sequence[float]]) -> np.ndarray:
        """``P(unreliable)`` for each query's score curve."""
        if not self._fitted:
            raise RuntimeError("gate is not fitted")
        return self._model.predict_proba(self.featurize(score_curves))[:, 1]

    def flag(self, scores: Sequence[float], threshold: float | None = None) -> bool:
        """True if this single retrieval should be treated as UNRELIABLE (escalate / abstain)."""
        thr = self.threshold if threshold is None else threshold
        return bool(self.predict_proba([scores])[0] >= thr)
