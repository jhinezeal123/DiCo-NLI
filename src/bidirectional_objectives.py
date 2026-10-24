"""Matched two-order training losses. Candidate sets are supervision, not input features."""
import math

import torch
import torch.nn.functional as F

REV = [2, 1, 0, 3]


def objective(logits, reverse_logits, labels, mode, smoothing=0.1,
              candidates=None, gold_anchor=0.25):
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits, reverse_logits = logits.float(), reverse_logits.float()
    log_p = F.log_softmax(logits, dim=-1)
    log_q = F.log_softmax(reverse_logits, dim=-1)[:, REV]
    index = torch.arange(len(labels), device=labels.device)

    def smoothed_nll(log_prob):
        return -(1 - smoothing) * log_prob[index, labels] - smoothing * log_prob.mean(-1)

    if mode == "paired_ce":
        return (0.5 * (smoothed_nll(log_p) + smoothed_nll(log_q))).mean()
    log_m = torch.logaddexp(log_p, log_q) - math.log(2)
    if mode == "fused_ce":
        return smoothed_nll(log_m).mean()
    if mode != "fused_partial":
        raise ValueError(f"Unknown two-order objective: {mode}")
    if candidates is None:
        candidates = F.one_hot(labels, num_classes=4).bool()
    assert candidates.shape == log_m.shape
    assert bool(candidates[index, labels].all()), "Gold must remain in each candidate set"
    log_mass = torch.logsumexp(log_m.masked_fill(~candidates, -torch.inf), dim=-1)
    nll = gold_anchor * (-log_m[index, labels]) + (1 - gold_anchor) * (-log_mass)
    return ((1 - smoothing) * nll - smoothing * log_m.mean(-1)).mean()


def validate_objectives():
    """Small CPU checks on synthetic tensors; do not alter global model RNG."""
    generator = torch.Generator().manual_seed(20261024)
    a = torch.randn(3, 4, generator=generator, dtype=torch.float64, requires_grad=True)
    b = torch.randn(3, 4, generator=generator, dtype=torch.float64, requires_grad=True)
    y = torch.tensor([0, 1, 3])
    rev_y = torch.tensor(REV)[y]
    singleton = F.one_hot(y, num_classes=4).bool()
    candidates = singleton.clone()
    candidates[0, 1] = candidates[2, 2] = True
    expected = 0.5 * (F.cross_entropy(a, y, label_smoothing=0.1)
                      + F.cross_entropy(b, rev_y, label_smoothing=0.1))
    torch.testing.assert_close(objective(a, b, y, "paired_ce"), expected)
    torch.testing.assert_close(objective(a, b, y, "fused_partial", candidates=singleton),
                               objective(a, b, y, "fused_ce"))
    for mode in ("paired_ce", "fused_ce", "fused_partial"):
        loss = objective(a, b, y, mode, candidates=candidates)
        reverse_loss = objective(b, a, rev_y, mode, candidates=candidates[:, REV])
        torch.testing.assert_close(loss, reverse_loss)
        assert torch.autograd.gradcheck(lambda x, z: objective(x, z, y, mode, candidates=candidates),
                                        (a, b), eps=1e-6, atol=1e-5)
    assert objective(a, b, y, "fused_partial", candidates=candidates) <= objective(a, b, y, "fused_ce")
    print("OBJECTIVE_CHECKS passed paired CE parity, singleton recovery, reversal invariance, "
          "and finite-difference gradients for all3 losses", flush=True)
