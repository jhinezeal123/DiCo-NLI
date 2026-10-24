import copy
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from explore_counterfactual import LABELS, prepare, unordered_pair


def row(i, a, b, label, group=None):
    return {"instance_id": str(i), "pair_id": str(group or i),
            "text1": a, "text2": b, "label": label,
            "text1_lang": "en", "text2_lang": "en"}


class CounterfactualDataTests(unittest.TestCase):
    def setUp(self):
        self.train = [
            row("s", "A red car", "a vehicle", "FORWARD_ENTAILMENT", "cars"),
            row("b", "a vehicle", "A red car", "BACKWARD_ENTAILMENT", "cars"),
            row("e", "a couch", "a sofa", "EQUIVALENCE", "seats"),
            row("n", "a bus", "a photograph", "NEGATIVE_OTHER", "negative"),
        ]
        self.dev = [row("d", "an elephant", "an animal", "FORWARD_ENTAILMENT", "dev")]

    def test_relation_directions_and_nontrivial_equivalence(self):
        out, _, audit = prepare(self.train, self.dev, "counterfactual", 42, {})
        examples = {(r["text1"], r["text2"]): r["label"] for r in out[4:]}
        self.assertEqual(examples[("a car", "a vehicle")], "FORWARD_ENTAILMENT")
        self.assertEqual(examples[("a vehicle", "a car")], "BACKWARD_ENTAILMENT")
        self.assertEqual(examples[("a car", "an automobile")], "EQUIVALENCE")
        self.assertEqual(examples[("a car", "a bicycle")], "NEGATIVE_OTHER")
        self.assertEqual(examples[("a sofa", "some furniture")], "FORWARD_ENTAILMENT")
        self.assertNotIn("a furniture", {r["text1"] for r in out} | {r["text2"] for r in out})
        self.assertTrue(all(r["text1"] != r["text2"] for r in out[4:] if r["label"] == "EQUIVALENCE"))
        self.assertFalse(audit["eval_labels_used"])

    def test_matched_counts_steps_and_original_labels(self):
        cf, _, ca = prepare(self.train, self.dev, "counterfactual", 13, {})
        rs, _, ra = prepare(self.train, self.dev, "resample", 13, {})
        self.assertEqual(len(cf), len(rs))
        self.assertEqual(Counter(r["label"] for r in cf), Counter(r["label"] for r in rs))
        self.assertEqual(ca["appended_label_counts"], ra["appended_label_counts"])
        allowed = {(r["text1"], r["text2"], r["label"], r["pair_id"]) for r in self.train}
        self.assertTrue(all((r["text1"], r["text2"], r["label"], r["pair_id"]) in allowed for r in rs[4:]))

    def test_no_eval_label_dependency_and_no_mutation(self):
        train_original, eval_original = copy.deepcopy(self.train), copy.deepcopy(self.dev)
        unlabeled = [{k: v for k, v in r.items() if k != "label"} for r in self.dev]
        poisoned = [dict(r, label="UNKNOWN_DO_NOT_READ") for r in self.dev]
        for arm in ("counterfactual", "resample"):
            a, _, audit_a = prepare(self.train, self.dev, arm, 7, {})
            b, _, audit_b = prepare(self.train, unlabeled, arm, 7, {})
            c, _, audit_c = prepare(self.train, poisoned, arm, 7, {})
            self.assertEqual(a, b)
            self.assertEqual(a, c)
            self.assertEqual(audit_a, audit_b)
            self.assertEqual(audit_a, audit_c)
        self.assertEqual(train_original, self.train)
        self.assertEqual(eval_original, self.dev)

    def test_whole_family_rejected_on_synthetic_overlap(self):
        dev = [row("d", "  A CAR ", "a   bicycle", "EQUIVALENCE", "dev")]
        out, _, audit = prepare(self.train, dev, "counterfactual", 42, {})
        generated = out[4:]
        # Only the indefinite surface family is rejected; its reverse and EQ
        # derivatives cannot remain under a different ID.
        self.assertNotIn(("a car", "a vehicle"), {(r["text1"], r["text2"]) for r in generated})
        self.assertGreater(audit["rejected"]["family_eval_text_overlap"], 0)
        self.assertTrue(all(unordered_pair(r) != unordered_pair(dev[0]) for r in out))

    def test_original_overlap_removes_entire_group_and_derivatives_keep_groups(self):
        bad = [row("leak", "an elephant", "an animal", "FORWARD_ENTAILMENT", "badgroup"),
               row("sibling", "an animal", "an elephant", "BACKWARD_ENTAILMENT", "badgroup")]
        out, _, audit = prepare(self.train + bad, self.dev, "counterfactual", 42, {})
        self.assertEqual(audit["removed_train_rows"], 2)
        self.assertEqual(audit["excluded_source_pair_ids"], ["badgroup"])
        self.assertNotIn("badgroup", {r["pair_id"] for r in out})
        original_groups = {r["pair_id"] for r in self.train}
        self.assertTrue(all(r["pair_id"] in original_groups for r in out))
        self.assertEqual(len(out), len({r["instance_id"] for r in out}))

    def test_invalid_arm_and_empty_generation_fail_explicitly(self):
        with self.assertRaises(ValueError):
            prepare(self.train, self.dev, "invalid", 42, {})
        with self.assertRaises(ValueError):
            prepare(self.train, self.dev, "counterfactual", 42, {"counterfactual_max_families": 0})
        missing = [row("a", "a hedgehog", "an animal", LABELS[0])]
        with self.assertRaises(ValueError):
            prepare(missing, self.dev, "counterfactual", 42, {})


if __name__ == "__main__":
    unittest.main()
