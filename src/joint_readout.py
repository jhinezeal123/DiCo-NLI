"""Matched independent versus interacting reversal-equivariant readouts."""
import torch
from torch import nn

REV = [2, 1, 0, 3]


class Readout(nn.Module):
    def __init__(self, dim, arm):
        super().__init__()
        assert arm in ("independent", "joint")
        self.arm = arm
        self.norm = nn.LayerNorm(dim)
        width = 128 if arm == "independent" else 64
        self.net = nn.Sequential(nn.Linear(dim if arm == "independent" else 2*dim, width),
                                 nn.GELU(), nn.Linear(width, 4))

    def forward(self, a, b):
        a, b = self.norm(a), self.norm(b)
        if self.arm == "independent":
            p, q = self.net(a), self.net(b)
        else:
            p, q = self.net(torch.cat((a, b), -1)), self.net(torch.cat((b, a), -1))
        return .5 * (p + q[:, REV])


def check_readouts():
    """Small synthetic structural/gradient checks, no fitting or model download."""
    with torch.random.fork_rng():
        torch.manual_seed(2087)
        a, b = torch.randn(5, 12, dtype=torch.float64), torch.randn(5, 12, dtype=torch.float64)
        for arm in ("independent", "joint"):
            head = Readout(12, arm).double()
            z = head(a, b)
            torch.testing.assert_close(z, head(b, a)[:, REV], atol=1e-12, rtol=1e-12)
            if arm == "independent":
                expected = .5*(head.net(head.norm(a)) + head.net(head.norm(b))[:, REV])
                torch.testing.assert_close(z, expected, atol=1e-12, rtol=1e-12)
            loss = torch.nn.functional.cross_entropy(z, torch.tensor([0, 1, 2, 3, 3]), label_smoothing=.1)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
        counts = {arm: sum(p.numel() for p in Readout(1536, arm).parameters())
                  for arm in ("independent", "joint")}
        assert counts == {"independent": 200324, "joint": 200004}, counts
    return {"reversal_all_classes": True, "finite_gradients": True,
            "independent_decomposition": True, "head_parameters_dim1536": counts}
