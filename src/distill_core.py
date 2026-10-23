"""Distillation targets and group audit, independent of model/tokenizer families."""
import re
import numpy as np

LABELS = ["BACKWARD_ENTAILMENT", "EQUIVALENCE", "FORWARD_ENTAILMENT", "NEGATIVE_OTHER"]
REV = np.array([2, 1, 0, 3])


def text_key(row):
    clean = lambda value: re.sub(r"\s+", " ", value).strip().casefold()
    return tuple(sorted((clean(row["text1"]), clean(row["text2"]))))


def clean_train_groups(train, dev_inputs):
    """Uses dev TEXT only; remove the whole train pair_id and its reversals."""
    dev_keys = {text_key(r) for r in dev_inputs}
    contaminated = {r["pair_id"] for r in train if text_key(r) in dev_keys}
    kept = [r for r in train if r["pair_id"] not in contaminated]
    assert not ({text_key(r) for r in kept} & dev_keys)
    assert not ({r["pair_id"] for r in kept} & {r["pair_id"] for r in dev_inputs})
    return kept, sorted(contaminated)


def softmax(logits, temperature):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    z = np.asarray(logits, dtype=np.float64) / temperature
    if z.ndim != 2 or z.shape[1] != 4 or not np.isfinite(z).all():
        raise ValueError("expected finite N x 4 logits")
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def targets(orig_logits, swap_logits, temperature, reversible, projected):
    p, q = softmax(orig_logits, temperature), softmax(swap_logits, temperature)
    if p.shape != q.shape:
        raise ValueError("original and swapped logits must align")
    mask = np.asarray(reversible, dtype=bool)
    if mask.shape != (len(p),):
        raise ValueError("reversible mask must align with rows")
    if projected:
        p = p.copy()
        p[mask] = 0.5 * (p[mask] + q[mask][:, REV])
        q = q.copy()
        q[mask] = p[mask][:, REV]
    # NEGATIVE_OTHER has no official reversal: no projected target or swapped loss.
    return p.astype(np.float32), q.astype(np.float32)


def per_row_loss(logits, labels, teacher_probs, kd_weight, temperature):
    import torch.nn.functional as F
    ce = F.cross_entropy(logits.float(), labels, reduction="none")
    if kd_weight == 0:
        return ce
    if not 0 < kd_weight <= 1:
        raise ValueError("kd_weight must lie in [0,1]")
    kl = F.kl_div(F.log_softmax(logits.float() / temperature, dim=-1),
                  teacher_probs, reduction="none").sum(-1) * temperature ** 2
    return (1 - kd_weight) * ce + kd_weight * kl
