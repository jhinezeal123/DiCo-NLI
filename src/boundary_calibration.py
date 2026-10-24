"""Per-input calibration. Metadata is used solely to join/audit records, never as features."""
import re

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

LABELS = ["BACKWARD_ENTAILMENT", "EQUIVALENCE", "FORWARD_ENTAILMENT", "NEGATIVE_OTHER"]
REV = np.array([2, 1, 0, 3])


def normalize(p):
    p = np.asarray(p, dtype=np.float64)
    assert p.ndim == 2 and p.shape[1] == 4 and np.isfinite(p).all()
    p = np.clip(p, 1e-7, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def lexical(a, b):
    ta, tb = re.findall(r"\w+", a.lower()), re.findall(r"\w+", b.lower())
    sa, sb = set(ta), set(tb)
    la, lb = len(ta), len(tb)
    union, common = len(sa | sb), len(sa & sb)
    neg = {"not", "no", "never", "without", "neither"}
    return [np.log1p(la), np.log1p(lb), (la - lb) / max(la, lb, 1),
            abs(la - lb) / max(la, lb, 1), common / max(union, 1),
            float(sa <= sb), float(sb <= sa), float(not common),
            float(sa == sb), float(min(la, lb) <= 1),
            float(bool(sa & neg) != bool(sb & neg)),
            float(bool(re.search(r"\d", a)) != bool(re.search(r"\d", b))),
            float(any(c in a + b for c in ",.;:!?")),
            float(ta[:1] == tb[:1]), float(ta[-1:] == tb[-1:])]


def feature_matrix(texts, p, mode, evidence=None):
    p = normalize(p)
    lp = np.log(p)
    sorted_p = np.sort(p, axis=1)
    blocks = [lp, -(p * lp).sum(axis=1, keepdims=True),
              (sorted_p[:, -1] - sorted_p[:, -2])[:, None]]
    if mode == "lexical":
        lex = np.asarray([lexical(a, b) for a, b in texts])
        blocks += [lex, (lex[:, :, None] * lp[:, None, :]).reshape(len(p), -1)]
    elif mode == "semantic":
        evidence = np.asarray(evidence, dtype=np.float64)
        assert evidence.shape == (len(p), 6) and np.isfinite(evidence).all()
        ev = np.log(np.clip(evidence, 1e-7, 1.0))
        blocks += [ev, (ev[:, :, None] * lp[:, None, :]).reshape(len(p), -1)]
    elif mode != "probability":
        raise ValueError(mode)
    return np.concatenate(blocks, axis=1)


def objective(flat, x, offsets, y, penalty):
    weights = flat.reshape(x.shape[1], 4)
    logits = offsets + x @ weights
    q = softmax(logits, axis=1)
    loss = np.mean(logsumexp(logits, axis=1) - logits[np.arange(len(y)), y])
    loss += 0.5 * np.sum(penalty[:, None] * weights ** 2)
    q[np.arange(len(y)), y] -= 1
    grad = x.T @ q / len(y) + penalty[:, None] * weights
    return float(loss), grad.ravel()


class BoundaryCalibrator:
    def __init__(self, mode, l2=0.1):
        self.mode, self.l2 = mode, l2

    def fit(self, texts, probs, labels, evidence=None):
        p = normalize(probs)
        reverse_texts = [(b, a) for a, b in texts]
        rev_evidence = None if evidence is None else evidence[:, [3, 4, 5, 0, 1, 2]]
        feats = np.concatenate([feature_matrix(texts, p, self.mode, evidence),
                                feature_matrix(reverse_texts, p[:, REV], self.mode, rev_evidence)])
        self.mean = feats.mean(axis=0)
        self.scale = np.maximum(feats.std(axis=0), 0.01)
        x = np.c_[np.ones(len(feats)), (feats - self.mean) / self.scale]
        offsets = np.log(np.concatenate([p, p[:, REV]]))
        y = np.concatenate([labels, REV[labels]])
        penalty = np.full(x.shape[1], self.l2)
        penalty[0] = self.l2 * 0.01
        result = minimize(objective, np.zeros(x.shape[1] * 4),
                          args=(x, offsets, y, penalty), jac=True, method="L-BFGS-B",
                          options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-7})
        if not result.success:
            raise RuntimeError(f"Calibration optimization failed: {result.message}")
        self.weights = result.x.reshape(x.shape[1], 4)
        self.diagnostics = {"converged": bool(result.success), "iterations": result.nit,
                            "objective": float(result.fun), "features": x.shape[1]}
        return self

    def predict_proba(self, texts, probs, evidence=None):
        p = normalize(probs)
        def forward(texts_, p_, evidence_):
            f = feature_matrix(texts_, p_, self.mode, evidence_)
            x = np.c_[np.ones(len(f)), (f - self.mean) / self.scale]
            return softmax(np.log(p_) + x @ self.weights, axis=1)
        rev_evidence = None if evidence is None else evidence[:, [3, 4, 5, 0, 1, 2]]
        q = 0.5 * (forward(texts, p, evidence) + forward([(b, a) for a, b in texts], p[:, REV], rev_evidence)[:, REV])
        assert np.allclose(q.sum(axis=1), 1.0)
        return q
