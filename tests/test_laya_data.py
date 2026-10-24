import importlib.util
import itertools
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import laya_data as data


class SchemaTests(unittest.TestCase):
    def test_each_official_label_has_correct_target(self):
        for label, key in zip(data.LABELS, data.OPTION_KEYS):
            record = data.training_record(dict(text1="a dog", text2="an animal", label=label))
            self.assertEqual(record["expected"], {"relation": key})
            self.assertEqual(list(record["questions"]["relation"]["criteria"]), data.OPTION_KEYS)

    def test_inference_has_no_gold_and_swap_only_changes_text(self):
        row = dict(text1="a red car", text2="a car", label="FORWARD_ENTAILMENT")
        record = data.inference_record(row, swap=True)
        self.assertEqual(record["state"], "Text 1: a car\nText 2: a red car")
        self.assertNotIn("expected", record)
        self.assertNotIn("gold", record)
        self.assertEqual(record["questions"], data.inference_record(row)["questions"])

    def test_remove_entire_group_and_reversed_text_duplicates(self):
        train = [dict(pair_id="a", text1=" The  dog ", text2="AN ANIMAL"),
                 dict(pair_id="a", text1="group member", text2="other order"),
                 dict(pair_id="b", text1="safe", text2="pair")]
        dev = [dict(pair_id="z", text1="an animal", text2="the dog")]
        kept, excluded = data.clean_train_groups(train, dev)
        self.assertEqual(excluded, ["a"])
        self.assertEqual([r["pair_id"] for r in kept], ["b"])

    def test_reversal_permutation(self):
        self.assertEqual([data.REV[i] for i in data.REV], list(range(4)))
        self.assertEqual(data.LABELS[data.REV[0]], "FORWARD_ENTAILMENT")

    @unittest.skipUnless(importlib.util.find_spec("laya"), "SDK integration checked on compute host")
    def test_sdk_targets_and_every_option_permutation(self):
        from laya.train import to_internal, target_from_expected
        from laya.common import unpermute_probs
        import numpy as np
        q = to_internal("relation", data.question())
        target = target_from_expected(q, "C")
        self.assertEqual(target, [0., 0., 1., 0.])
        for order in itertools.permutations(range(4)):
            permuted = np.asarray([target[i] for i in order])
            np.testing.assert_array_equal(unpermute_probs(permuted, list(order)), target)


if __name__ == "__main__":
    unittest.main()
