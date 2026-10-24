"""Meaningful mathematical and gradient checks; requires PyTorch only."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from explore_geometry import GeometryHead, masked_pool, soft_projection, subspace_features


class GeometryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_inclusion_direction_and_reversal(self):
        # e1 is contained in span(e1,e2); the converse is false.
        narrow = torch.tensor([[[1., 0.], [0., 0.], [0., 0.]]])
        broad = torch.tensor([[[1., 0.], [0., 1.], [0., 0.]]])
        f = subspace_features(narrow, broad, ridge=1e-3)
        swapped = subspace_features(broad, narrow, ridge=1e-3)
        self.assertGreater(f[0, 0].item(), 0.99)
        self.assertAlmostEqual(f[0, 1].item(), 0.4995, places=3)
        self.assertTrue(torch.allclose(f[:, 0], swapped[:, 1]))
        self.assertTrue(torch.allclose(f[:, 1], swapped[:, 0]))
        self.assertTrue(torch.allclose(f[:, 5], -swapped[:, 5]))

    def test_projection_positive_and_bounded_with_rank_deficiency(self):
        basis = torch.tensor([[[1., 1.], [0., 0.], [0., 0.]],
                              [[0., 0.], [0., 0.], [0., 0.]]])
        p = soft_projection(basis)
        self.assertTrue(torch.isfinite(p).all())
        self.assertTrue(torch.allclose(p, p.transpose(-1, -2), atol=1e-6))
        eig = torch.linalg.eigvalsh(p)
        self.assertGreaterEqual(eig.min().item(), -1e-6)
        self.assertLessEqual(eig.max().item(), 1.0 + 1e-6)
        with self.assertRaises(ValueError):
            soft_projection(basis, ridge=0)

    def test_gradient_through_both_phrases_and_projection(self):
        hidden = torch.randn(3, 7, 24, requires_grad=True)
        first = torch.tensor([[1,1,0,0,0,0,0]] * 3)
        second = torch.tensor([[0,0,0,1,1,1,0]] * 3)
        head = GeometryHead(24, {"geometry_dim": 8, "geometry_rank": 3})
        logits = head(hidden, first, second)
        self.assertEqual(tuple(logits.shape), (3, 4))
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 2, 3])).backward()
        self.assertGreater(hidden.grad[:, :2].abs().sum().item(), 0)
        self.assertGreater(hidden.grad[:, 3:6].abs().sum().item(), 0)
        self.assertGreater(head.projection.weight.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(hidden.grad).all())

    def test_empty_masks_and_fp16_inputs(self):
        hidden = torch.randn(2, 4, 16, dtype=torch.float16, requires_grad=True)
        empty = torch.zeros(2, 4)
        valid = torch.tensor([[1,1,0,0], [0,0,1,1]])
        self.assertTrue(torch.equal(masked_pool(hidden, empty), torch.zeros(2, 16)))
        for arm in ["geometry", "vector"]:
            head = GeometryHead(16, {}, arm=arm).half()
            logits = head(hidden, empty, valid)
            self.assertEqual(logits.dtype, torch.float32)
            self.assertTrue(torch.isfinite(logits).all())
            logits.square().sum().backward(retain_graph=True)
        self.assertTrue(torch.isfinite(hidden.grad).all())

    def test_control_capacity_exactly_matches_and_uses_other_features(self):
        geometry = GeometryHead(24, {})
        control = GeometryHead(24, {}, arm="vector")
        control.load_state_dict(geometry.state_dict())
        self.assertEqual(sum(p.numel() for p in geometry.parameters()),
                         sum(p.numel() for p in control.parameters()))
        hidden = torch.randn(2, 4, 24)
        first, second = torch.tensor([[1,1,0,0]] * 2), torch.tensor([[0,0,1,1]] * 2)
        self.assertFalse(torch.allclose(geometry.features(hidden, first, second),
                                       control.features(hidden, first, second)))


if __name__ == "__main__":
    unittest.main()
