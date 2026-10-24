"""Lightweight synthetic checks; never fit/evaluate a research dataset locally."""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import check_grad

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from boundary_calibration import BoundaryCalibrator, REV, feature_matrix, normalize, objective

rng = np.random.default_rng(123)
x = np.c_[np.ones(12), rng.normal(size=(12, 4))]
offsets = np.log(normalize(rng.uniform(size=(12, 4))))
y = np.arange(12) % 4
penalty = np.full(x.shape[1], 0.1)
w = rng.normal(size=x.shape[1] * 4)
error = check_grad(lambda z: objective(z, x, offsets, y, penalty)[0],
                   lambda z: objective(z, x, offsets, y, penalty)[1], w)
assert error < 1e-5, error
texts = [("a small dog", "an animal"), ("left", "right"), ("runs", "is running"), ("one", "as one")]
p = normalize(rng.uniform(size=(4, 4)))
evidence = rng.uniform(size=(4, 6))
evidence[:, :3] /= evidence[:, :3].sum(1, keepdims=True)
evidence[:, 3:] /= evidence[:, 3:].sum(1, keepdims=True)
for mode in ("probability", "lexical", "semantic"):
    # Synthetic labels only: exercise the optimizer and the symmetry guarantee.
    ev = evidence if mode == "semantic" else None
    cal = BoundaryCalibrator(mode).fit(texts, p, np.arange(4), ev)
    q = cal.predict_proba(texts, p, ev)
    rev_ev = None if ev is None else ev[:, [3, 4, 5, 0, 1, 2]]
    rev = cal.predict_proba([(b, a) for a, b in texts], p[:, REV], rev_ev)[:, REV]
    np.testing.assert_allclose(q, rev, atol=1e-12)
    np.testing.assert_allclose(q.sum(axis=1), 1)
    assert feature_matrix(texts, p, mode, ev).shape[0] == 4
print(f"Synthetic checks passed: analytic gradient error={error:.2e}; all variants reverse equivariantly.")
