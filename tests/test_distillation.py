import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distill_core import REV, clean_train_groups, softmax, targets


class DistillationTests(unittest.TestCase):
    def test_projection_equivariance_and_negative_exclusion(self):
        p = np.array([[5, 1, 0, -1], [1, 4, 0, -1], [-1, 0, 1, 5]])
        q = np.array([[0, 1, 3, -1], [2, 0, 1, -1], [3, 1, 0, -1]])
        a, b = targets(p, q, 2, [True, True, False], True)
        np.testing.assert_allclose(a[:2], b[:2, REV], atol=1e-7)
        np.testing.assert_allclose(a[2], softmax(p, 2)[2], atol=1e-7)
        np.testing.assert_allclose(b[2], softmax(q, 2)[2], atol=1e-7)
        np.testing.assert_allclose(a.sum(1), 1, atol=1e-7)

    def test_unprojected_preserves_teacher_and_label_mapping(self):
        p = np.arange(12).reshape(3, 4)
        a, b = targets(p, -p, 2, [True, False, True], False)
        np.testing.assert_allclose(a, softmax(p, 2), atol=1e-7)
        np.testing.assert_allclose(b, softmax(-p, 2), atol=1e-7)
        np.testing.assert_array_equal(REV[REV], np.arange(4))
        self.assertEqual(REV.tolist(), [2, 1, 0, 3])

    def test_remove_whole_pair_group_using_only_dev_inputs(self):
        tr = [dict(pair_id="1", text1=" A  B ", text2="C"),
              dict(pair_id="1", text1="different", text2="related"),
              dict(pair_id="2", text1="safe", text2="row")]
        dv = [dict(pair_id="9", text1="c", text2="a b")]
        kept, removed = clean_train_groups(tr, dv)
        self.assertEqual(removed, ["1"])
        self.assertEqual([r["pair_id"] for r in kept], ["2"])

    def test_invalid_temperature_and_row_alignment(self):
        for t in [0, -1, float("nan")]:
            with self.assertRaises(ValueError):
                softmax(np.zeros((2, 4)), t)
        with self.assertRaises(ValueError):
            targets(np.zeros((2, 4)), np.zeros((1, 4)), 2, [True, True], True)

    def test_torch_loss_orientation_and_temperature_scaling(self):
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            self.skipTest("torch checked on the experiment host before training")
        from distill_core import per_row_loss
        z = torch.tensor([[2., 1., -1., 0.]], requires_grad=True)
        y = torch.tensor([0])
        target = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        actual = per_row_loss(z, y, target, 0.5, 2.)
        logp = F.log_softmax(z / 2, -1)
        manual = 0.5 * F.cross_entropy(z, y) + 0.5 * 4 * (target * (target.log() - logp)).sum()
        torch.testing.assert_close(actual[0], manual)
        actual.mean().backward()
        self.assertTrue(torch.isfinite(z.grad).all())


if __name__ == "__main__":
    unittest.main()
