"""Offline tests for the Phase-8 Retrieval-Reliability Gate.

No data / model / network — synthetic score curves only, so coverage survives a clean checkout.
A 'reliable' retrieval is modelled as a peaked curve (one doc clearly closest); an 'unreliable'
one as a flat tie cluster (nothing actually close) — the empirical signature from Phase 8.
"""
import numpy as np
import pytest

from src.gate import FEATURE_NAMES, RetrievalReliabilityGate, gate_features

rng = np.random.default_rng(0)


def _peaked(n=100):
    """Reliable-looking: a clear standout above a low background."""
    s = rng.uniform(0.55, 0.70, n)
    s[0] = 0.92
    s[1] = 0.84
    return s


def _flat(n=100):
    """Unreliable-looking: a flat tie cluster near the top, no standout."""
    return rng.uniform(0.80, 0.83, n)


def test_features_shape_and_finite():
    f = gate_features(_peaked())
    assert len(f) == len(FEATURE_NAMES) == 8
    assert all(np.isfinite(f))


def test_short_curve_degrades_gracefully():
    f = gate_features([0.9, 0.5])           # only two scores
    assert len(f) == 8 and all(np.isfinite(f))


def test_too_short_raises():
    with pytest.raises(ValueError):
        gate_features([0.9])


def test_nonfinite_raises():
    with pytest.raises(ValueError):
        gate_features([0.9, np.nan, 0.5])
    with pytest.raises(ValueError):
        gate_features([0.9, np.inf, 0.5])


def test_non_1d_raises():
    with pytest.raises(ValueError):
        gate_features([[0.9, 0.8], [0.7, 0.6]])      # (2,2) matrix, not a curve


def test_single_class_fit_raises():
    with pytest.raises(ValueError):
        RetrievalReliabilityGate().fit([_peaked() for _ in range(5)], [0] * 5)


def test_peaked_has_larger_margin_and_smaller_tie_than_flat():
    fp = dict(zip(FEATURE_NAMES, gate_features(_peaked())))
    ff = dict(zip(FEATURE_NAMES, gate_features(_flat())))
    assert fp["margin_bg"] > ff["margin_bg"]        # standout separates from background
    assert fp["gap12"] > ff["gap12"]                # #1 stands out from #2
    assert fp["frac_tie"] < ff["frac_tie"]          # flat curve is one big tie cluster


def test_gate_learns_to_separate():
    curves = [_peaked() for _ in range(40)] + [_flat() for _ in range(40)]
    y = [0] * 40 + [1] * 40                          # 0 reliable, 1 unreliable
    gate = RetrievalReliabilityGate().fit(curves, y)
    p = gate.predict_proba(curves)
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, p) > 0.9
    assert gate.predict_proba([_flat()])[0] > gate.predict_proba([_peaked()])[0]


def test_flag_and_unfitted_guard():
    gate = RetrievalReliabilityGate(threshold=0.5)
    with pytest.raises(RuntimeError):
        gate.predict_proba([_peaked()])
    gate.fit([_peaked() for _ in range(30)] + [_flat() for _ in range(30)], [0] * 30 + [1] * 30)
    assert gate.flag(_flat()) is True
    assert gate.flag(_peaked()) is False
