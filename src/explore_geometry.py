"""Small, differentiable subspace residual for the four-label classifier.

This is a task-specific pilot inspired by subspace inclusion, not a reproduction
of a published model. The NEGATIVE_OTHER logit is supervised freely together
with the other three logits. No entailment closure or binary-product labels are
imposed. The vector control has precisely the same trainable parameter count.
"""
from contextlib import nullcontext
import math

import torch
from torch import nn
from torch.nn import functional as F


def masked_pool(hidden, mask):
    """Mean over valid phrase tokens; an empty phrase maps to a zero vector."""
    weights = mask.to(device=hidden.device, dtype=torch.float32).unsqueeze(-1)
    return (hidden.float() * weights).sum(1) / weights.sum(1).clamp_min(1.0)


def soft_projection(basis, ridge=0.1):
    """Return B(B^T B + ridge I)^-1 B^T, computed in float32.

    A positive ridge makes the solve well-defined for empty/rank-deficient bases.
    Supports arbitrary leading batch dimensions and retains autograd.
    """
    if ridge <= 0:
        raise ValueError("geometry_ridge must be strictly positive")
    basis = basis.float()
    gram = basis.transpose(-1, -2) @ basis
    eye = torch.eye(gram.shape[-1], device=basis.device, dtype=basis.dtype)
    solved = torch.linalg.solve(gram + float(ridge) * eye, basis.transpose(-1, -2))
    return basis @ solved


def subspace_features(basis1, basis2, ridge=0.1):
    """Eight ordered features; indices 0/1 are S1-in-S2 / S2-in-S1."""
    p1, p2 = soft_projection(basis1, ridge), soft_projection(basis2, ridge)
    trace1 = p1.diagonal(dim1=-2, dim2=-1).sum(-1)
    trace2 = p2.diagonal(dim1=-2, dim2=-1).sum(-1)
    overlap = (p1 * p2).sum((-2, -1))
    ambient = p1.shape[-1]
    incl12 = overlap / trace1.clamp_min(1e-6)
    incl21 = overlap / trace2.clamp_min(1e-6)
    # Clamp before sqrt: sqrt(0) has an infinite derivative and can create NaN
    # gradients for empty phrases even if its output is clamped afterwards.
    normalized_overlap = overlap / (trace1 * trace2).clamp_min(1e-12).sqrt()
    return torch.stack([
        incl12, incl21, normalized_overlap,
        trace1 / ambient, trace2 / ambient,
        (trace1 - trace2) / ambient,
        (p1 - p2).square().mean((-2, -1)),
        overlap / ambient,
    ], -1)


def vector_features(basis1, basis2):
    """Eight ordinary vector statistics: no learned subspace inclusion."""
    a, b = basis1.flatten(1).float(), basis2.flatten(1).float()
    norm1, norm2 = a.norm(dim=-1), b.norm(dim=-1)
    return torch.stack([
        F.cosine_similarity(a, b, dim=-1, eps=1e-6),
        (a * b).mean(-1),
        norm1 / math.sqrt(a.shape[-1]),
        norm2 / math.sqrt(b.shape[-1]),
        (norm1 - norm2) / math.sqrt(a.shape[-1]),
        (a - b).abs().mean(-1),
        (a - b).square().mean(-1),
        (a + b).mean(-1),
    ], -1)


class GeometryHead(nn.Module):
    """Ordered phrase geometry -> additive four-label residual logits.

    Args:
        hidden_size: backbone hidden dimension.
        cfg: optional geometry_dim, geometry_rank, geometry_ridge,
             geometry_width keys (defaults 16, 4, 0.1, 32).
        arm: 'geometry' or the exactly capacity-matched 'vector' control.

    The common linear projection is shared across the two phrase roles. Inputs
    are pooled then normalized per phrase; no absolute token positions or pair
    metadata enter the head. The causal backbone may still contextualize the
    second phrase by the first, so this pilot does not guarantee a globally
    context-independent partial order or exact reversal equivariance.
    """
    def __init__(self, hidden_size, cfg, arm="geometry"):
        super().__init__()
        if arm not in {"geometry", "vector"}:
            raise ValueError(f"unknown geometry arm: {arm}")
        self.arm = arm
        self.ambient = int(cfg.get("geometry_dim", 16))
        self.rank = int(cfg.get("geometry_rank", 4))
        self.ridge = float(cfg.get("geometry_ridge", 0.1))
        width = int(cfg.get("geometry_width", 32))
        if not 0 < self.rank <= self.ambient or self.ridge <= 0 or width < 1:
            raise ValueError("require 0 < rank <= ambient, ridge > 0, width > 0")
        self.projection = nn.Linear(hidden_size, self.ambient * self.rank, bias=False)
        self.readout = nn.Linear(8, width)
        self.output = nn.Linear(width, 4)
        nn.init.normal_(self.output.weight, std=1e-3)
        nn.init.zeros_(self.output.bias)

    def _project(self, pooled):
        # Parameter casts also keep this path valid if an enclosing model is
        # explicitly converted to half precision; casts preserve gradient flow.
        pooled = F.layer_norm(pooled.float(), (pooled.shape[-1],))
        basis = F.linear(pooled, self.projection.weight.float())
        return basis.reshape(-1, self.ambient, self.rank)

    def features(self, hidden, mask1, mask2):
        context = (torch.autocast(device_type=hidden.device.type, enabled=False)
                   if hidden.device.type in {"cpu", "cuda"} else nullcontext())
        with context:
            basis1 = self._project(masked_pool(hidden, mask1))
            basis2 = self._project(masked_pool(hidden, mask2))
            if self.arm == "geometry":
                return subspace_features(basis1, basis2, self.ridge)
            return vector_features(basis1, basis2)

    def forward(self, hidden, mask1, mask2):
        context = (torch.autocast(device_type=hidden.device.type, enabled=False)
                   if hidden.device.type in {"cpu", "cuda"} else nullcontext())
        with context:
            features = self.features(hidden, mask1, mask2)
            middle = F.gelu(F.linear(features, self.readout.weight.float(),
                                     self.readout.bias.float()))
            return F.linear(middle, self.output.weight.float(), self.output.bias.float())
