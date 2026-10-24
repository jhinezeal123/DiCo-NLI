"""Differentiable latent alignment residual for the DiCo-NLI exploration pilot.

The four output coordinates follow the runner's alphabetical label order:
BACKWARD_ENTAILMENT, EQUIVALENCE, FORWARD_ENTAILMENT, NEGATIVE_OTHER.
Local relation states are *latent*: no teacher labels, trace generation, or
logical interpretation is assumed. All four final logits, including NEG, are
learned directly from the final supervised objective.
"""
from contextlib import nullcontext
import math

import torch
from torch import nn
from torch.nn import functional as F


def _masked_mean(values, mask):
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(1) / weights.sum(1).clamp_min(1.0)


def _safe_attention(scores, query_mask, key_mask):
    """Masked attention; an absent query/key phrase yields exactly zero rows."""
    valid = query_mask.unsqueeze(-1) & key_mask.unsqueeze(1)
    weights = scores.masked_fill(~valid, -10000.0).softmax(-1)
    weights = weights * valid.to(weights.dtype)
    return weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)


class AlignmentHead(nn.Module):
    """Soft cross alignment plus latent relation aggregation and free residual.

    Both arms use the same projection and relation/readout parameters. The
    ``uniform_alignment`` arm changes only attention weights; matching vectors
    remain active in its local features, avoiding unused control parameters.
    The module should remain fp32 when the backbone uses fp16 AMP.
    """

    def __init__(self, hidden_size: int, cfg: dict, arm: str = "alignment"):
        super().__init__()
        if arm not in ("alignment", "uniform_alignment"):
            raise ValueError(f"Unknown alignment arm: {arm}")
        self.arm = arm
        self.dim = int(cfg.get("alignment_dim", 64))
        self.states = int(cfg.get("alignment_states", 7))
        self.temperature = float(cfg.get("alignment_temperature", 1.0))
        if self.dim < 1 or self.states < 2 or self.temperature <= 0:
            raise ValueError("Positive dimension/temperature and >=2 states required")
        self.normalizer = nn.LayerNorm(hidden_size)
        self.project = nn.Linear(hidden_size, self.dim)
        self.match = nn.Linear(self.dim, self.dim, bias=False)
        self.relation = nn.Sequential(
            nn.Linear(4 * self.dim, self.dim), nn.GELU(),
            nn.Linear(self.dim, self.states),
        )
        # Mean + smooth max per relation state in each direction, and four
        # coverage/entropy diagnostics. Different directions retain orientation.
        self.readout = nn.Sequential(
            nn.Linear(4 * self.states + 4, self.dim), nn.GELU(),
            nn.Linear(self.dim, 4),
        )
        nn.init.normal_(self.readout[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.readout[-1].bias)

    def _direction(self, vectors, aligned, mask, attention, key_mask):
        features = torch.cat(
            (vectors, aligned, vectors - aligned, vectors * aligned), dim=-1
        )
        states = self.relation(features).softmax(-1)
        mean_states = _masked_mean(states, mask)
        # Smooth max at fixed temperature; normalize by phrase length so
        # padding and repeated masked slots never alter the statistics.
        count = mask.sum(1, keepdim=True).clamp_min(1).to(states.dtype)
        masked = (states * 5.0).masked_fill(~mask.unsqueeze(-1), -10000.0)
        smooth_max = (masked.logsumexp(1) - count.log()) / 5.0
        smooth_max = torch.where(mask.any(1, keepdim=True), smooth_max,
                                 torch.zeros_like(smooth_max))
        # Entropy is normalized by valid key length and zero for <=1 keys.
        entropy = -(attention * attention.clamp_min(1e-8).log()).sum(-1)
        denom = key_mask.sum(1).clamp_min(2).to(entropy.dtype).log()
        entropy = entropy / denom.unsqueeze(-1)
        entropy = _masked_mean(entropy.unsqueeze(-1), mask)
        entropy = entropy * (key_mask.sum(1, keepdim=True) > 1)
        coverage = _masked_mean(attention.sum(-1, keepdim=True), mask)
        return torch.cat((mean_states, smooth_max, entropy, coverage), dim=-1)

    def alignment_weights(self, hidden, mask1, mask2):
        """Return both attention matrices for diagnostics; no labels required."""
        mask1, mask2 = mask1.bool(), mask2.bool()
        # Disable outer AMP: softmax/logsumexp and relation aggregation remain
        # stable even with fp16 backbone states and very short/empty phrases.
        context = (torch.autocast(device_type=hidden.device.type, enabled=False)
                   if hidden.device.type in ("cuda", "cpu") else nullcontext())
        with context:
            projected = F.gelu(self.project(self.normalizer(hidden.float())))
            vectors = self.match(projected)
            scores = torch.matmul(vectors, vectors.transpose(1, 2))
            scores = scores / (math.sqrt(self.dim) * self.temperature)
            if self.arm == "uniform_alignment":
                scores = torch.zeros_like(scores)
            a12 = _safe_attention(scores, mask1, mask2)
            a21 = _safe_attention(scores.transpose(1, 2), mask2, mask1)
            return vectors, a12, a21

    def forward(self, hidden, mask1, mask2):
        if hidden.ndim != 3 or mask1.shape != hidden.shape[:2] or mask2.shape != mask1.shape:
            raise ValueError("Expected hidden [B,L,H] and phrase masks [B,L]")
        context = (torch.autocast(device_type=hidden.device.type, enabled=False)
                   if hidden.device.type in ("cuda", "cpu") else nullcontext())
        with context:
            mask1, mask2 = mask1.bool(), mask2.bool()
            vectors, a12, a21 = self.alignment_weights(hidden, mask1, mask2)
            aligned12 = torch.matmul(a12, vectors)
            aligned21 = torch.matmul(a21, vectors)
            f12 = self._direction(vectors, aligned12, mask1, a12, mask2)
            f21 = self._direction(vectors, aligned21, mask2, a21, mask1)
            return self.readout(torch.cat((f12, f21), dim=-1))
