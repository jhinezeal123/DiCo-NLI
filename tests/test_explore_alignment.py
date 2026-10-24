"""Small CPU checks for the learned alignment mechanism, without model downloads."""
import pathlib
import sys
import unittest

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from explore_alignment import AlignmentHead


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.hidden = torch.randn(3, 7, 12, requires_grad=True)
        self.m1 = torch.tensor([[1, 1, 0, 0, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0]], dtype=torch.bool)
        self.m2 = torch.tensor([[0, 0, 1, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 1, 1, 0, 0]], dtype=torch.bool)

    def make(self, arm="alignment"):
        return AlignmentHead(12, {"alignment_dim": 8, "alignment_states": 5}, arm)

    def test_finite_gradients_and_empty_masks(self):
        for arm in ("alignment", "uniform_alignment"):
            head = self.make(arm)
            delta = head(self.hidden, self.m1, self.m2)
            self.assertEqual(tuple(delta.shape), (3, 4))
            self.assertTrue(torch.isfinite(delta).all())
            torch.nn.functional.cross_entropy(delta, torch.tensor([0, 1, 3])).backward()
            self.assertTrue(torch.isfinite(self.hidden.grad).all())
            for name, parameter in head.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(head.match.weight.grad.abs().sum().item(), 0)

    def test_padding_noise_and_extension_do_not_change_outputs(self):
        head = self.make().eval()
        original = head(self.hidden, self.m1, self.m2)
        noise = self.hidden.detach().clone()
        noise[~(self.m1 | self.m2)] = torch.randn_like(noise[~(self.m1 | self.m2)]) * 1000
        torch.testing.assert_close(original, head(noise, self.m1, self.m2))
        extended = torch.cat((noise, torch.randn(3, 4, 12)), dim=1)
        zeros = torch.zeros(3, 4, dtype=torch.bool)
        torch.testing.assert_close(original, head(extended, torch.cat((self.m1, zeros), 1),
                                                  torch.cat((self.m2, zeros), 1)))

    def test_learned_matching_differs_from_uniform_with_identical_parameters(self):
        learned = self.make()
        uniform = self.make("uniform_alignment")
        uniform.load_state_dict(learned.state_dict())
        self.assertEqual(sum(p.numel() for p in learned.parameters()),
                         sum(p.numel() for p in uniform.parameters()))
        _, weights, _ = learned.alignment_weights(self.hidden, self.m1, self.m2)
        _, baseline, _ = uniform.alignment_weights(self.hidden, self.m1, self.m2)
        self.assertGreater((weights - baseline).abs().max().item(), 1e-5)
        self.assertTrue((weights[2] == 0).all())
        self.assertTrue((weights[1] == 0).all())
        torch.testing.assert_close(baseline[0, 0, 2:4], torch.tensor([0.5, 0.5]))

    def test_outer_autocast_still_outputs_fp32(self):
        head = self.make()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            delta = head(self.hidden.to(torch.bfloat16), self.m1, self.m2)
        self.assertEqual(delta.dtype, torch.float32)
        self.assertTrue(torch.isfinite(delta).all())

    def test_invalid_configuration_rejected(self):
        with self.assertRaises(ValueError):
            AlignmentHead(12, {"alignment_temperature": 0})
        with self.assertRaises(ValueError):
            self.make("unknown")


if __name__ == "__main__":
    unittest.main()
