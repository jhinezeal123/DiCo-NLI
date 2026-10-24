import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from explore_retrieval import differing_spans, edit_representation, prepare, unordered_pair


def row(i, a, b, label, group=None):
    return {"instance_id": str(i), "pair_id": str(group or i),
            "text1": a, "text2": b, "label": label}


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.train = [
            row("f", "a dog", "an animal", "FORWARD_ENTAILMENT", "dog"),
            row("b", "an animal", "a dog", "BACKWARD_ENTAILMENT", "dog"),
            row("duplicate", " A  DOG ", "AN ANIMAL", "FORWARD_ENTAILMENT", "other_group"),
            row("cf", "a cat", "an animal", "FORWARD_ENTAILMENT", "cat"),
            row("cb", "an animal", "a cat", "BACKWARD_ENTAILMENT", "cat"),
            row("eq", "a car", "an automobile", "EQUIVALENCE", "car"),
            row("neg", "a red car", "a blue car", "NEGATIVE_OTHER", "color"),
        ]
        self.dev = [{"instance_id": "dev", "pair_id": "new", "text1": "a horse", "text2": "an animal"}]

    def test_ordered_edit_representation(self):
        self.assertEqual(differing_spans("a red car", "a blue car"), (["red"], ["blue"], ["a", "car"]))
        self.assertIn("leftchange red rightchange blue", edit_representation("a red car", "a blue car"))
        self.assertIn("leftchange blue rightchange red", edit_representation("a blue car", "a red car"))
        self.assertNotEqual(edit_representation("dog", "animal"), edit_representation("animal", "dog"))

    def test_source_group_and_all_unordered_text_duplicates_excluded(self):
        for arm in ("topic_retrieval", "edit_retrieval"):
            _, _, audit = prepare(self.train, self.dev, arm, 42, {})
            query = next(p for p in audit["neighbor_provenance"]
                         if p["query_instance_id"] == "f" and p["orientation"] == "original")
            ids = {r["instance_id"] for r in query["neighbors"]}
            self.assertFalse(ids & {"f", "b", "duplicate"})
            self.assertEqual(query["excluded_source_group_rows"], 2)
            self.assertEqual(query["excluded_unordered_text_rows"], 3)

    def test_eval_labels_never_affect_training_context_or_provenance(self):
        for arm in ("topic_retrieval", "edit_retrieval"):
            original = prepare(self.train, self.dev, arm, 42, {})
            poisoned = prepare(self.train, [dict(r, label="SECRET_GOLD") for r in self.dev], arm, 42, {})
            self.assertEqual(original[0], poisoned[0])
            self.assertEqual(original[2], poisoned[2])
            self.assertEqual(original[1][0]["context"], poisoned[1][0]["context"])
            self.assertFalse(original[2]["eval_labels_used"])

    def test_swap_context_retrieved_as_its_own_query(self):
        _, eval_out, audit = prepare(self.train, self.dev, "edit_retrieval", 13, {"retrieval_k": 1})
        orig = next(p for p in audit["neighbor_provenance"] if p["split"] == "eval" and p["orientation"] == "original")
        swap = next(p for p in audit["neighbor_provenance"] if p["split"] == "eval" and p["orientation"] == "swap")
        self.assertEqual(orig["query_representation"], edit_representation("a horse", "an animal"))
        self.assertEqual(swap["query_representation"], edit_representation("an animal", "a horse"))
        self.assertNotEqual(orig["neighbors"], swap["neighbors"])
        self.assertIn("FORWARD_ENTAILMENT", eval_out[0]["context"])
        self.assertIn("BACKWARD_ENTAILMENT", eval_out[0]["swap_context"])

    def test_full_contaminated_group_removed_and_input_not_mutated(self):
        original = copy.deepcopy(self.train)
        dev = [dict(self.dev[0], text1="an automobile", text2="a car")]
        train_out, _, audit = prepare(self.train, dev, "topic_retrieval", 7, {})
        self.assertNotIn("car", {r["pair_id"] for r in train_out})
        self.assertEqual(audit["excluded_source_pair_ids"], ["car"])
        self.assertEqual(original, self.train)
        self.assertTrue(all(unordered_pair(r) != unordered_pair(dev[0]) for r in train_out))

    def test_deterministic_and_no_metadata_in_context(self):
        a = prepare(self.train, self.dev, "edit_retrieval", 7, {})
        b = prepare(self.train, self.dev, "edit_retrieval", 7, {})
        self.assertEqual(a, b)
        context = a[1][0]["context"]
        self.assertNotIn("pair_id", context)
        self.assertNotIn("instance_id", context)
        self.assertNotIn("other_group", context)

    def test_empty_query_falls_back_without_using_labels(self):
        blank = [dict(self.dev[0], text1="", text2=" ")]
        _, out, audit = prepare(self.train, blank, "edit_retrieval", 42, {})
        self.assertEqual(out[0]["context"], "")
        self.assertEqual(out[0]["swap_context"], "")
        self.assertEqual(audit["counts"]["fallback_empty_lexical_query_or_vocabulary"], 2)


if __name__ == "__main__":
    unittest.main()
