"""Pure evidence semantics checks; no model downloads or inference."""
import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from explore_witness import (attach_features, build_prompt, canonical_pair, features,
                             parse_generation, reverse_features, verification_requests)


class WitnessTests(unittest.TestCase):
    def setUp(self):
        self.parsed = parse_generation(json.dumps({
            "a": "There is an animal.", "b": "There is a dog.",
            "r": "A dog is an animal.", "f": "There is only a cat.", "g": "unknown",
        }))
        self.scores = {"fa": [0.01, 0.97, 0.02], "fb": [0.92, 0.02, 0.06],
                       "ra": [0.03, 0.91, 0.06], "rb": [0.07, 0.83, 0.10]}

    def test_no_witness_and_parse_failure_are_unknown(self):
        feature = features(self.parsed, self.scores)["extra_counterexample"]
        self.assertAlmostEqual(feature[0], 0.97 * 0.92)
        self.assertEqual(feature[1], 0.0)
        self.assertEqual(feature[7], 0.0)
        invalid = features(parse_generation("not JSON"), self.scores)
        self.assertEqual(invalid["extra_counterexample"], [0.0] * 8)
        self.assertEqual(invalid["extra_rationale"], [0.0] * 8)

    def test_verbal_abstention_is_unknown_and_truncated_json_is_not_repaired(self):
        for abstention in ("No counterexample exists.", "No valid witness can be constructed.",
                           "I cannot find a counterexample.", "unable to construct one"):
            parsed = parse_generation(json.dumps({"a": "A dog exists.", "b": "An animal exists.",
                                                  "r": "Dogs are animals.", "f": abstention, "g": ""}))
            self.assertEqual(parsed["f"], "")
            self.assertNotIn("fa", verification_requests(parsed))
            self.assertEqual(features(parsed, self.scores)["extra_counterexample"], [0.0] * 8)
        truncated = '{"a":"A dog exists.","b":"An animal exists.","r":"Dogs are animals.","f":"", "g":"A cat'
        self.assertFalse(parse_generation(truncated)["parsed"])
        self.assertEqual(features(parse_generation(truncated), {})["extra_counterexample"], [0.0] * 8)

    def test_verification_requires_positive_claim_and_negative_other_claim(self):
        requests = verification_requests(self.parsed)
        self.assertEqual(requests["fa"], ("There is only a cat.", "There is an animal."))
        self.assertEqual(requests["fb"], ("There is only a cat.", "There is a dog."))
        self.assertNotIn("ga", requests)
        missing = dict(self.scores)
        del missing["fb"]
        self.assertEqual(features(self.parsed, missing)["extra_counterexample"][0], 0.0)

    def test_reverse_orientation_and_involution(self):
        for key in ("extra_counterexample", "extra_rationale"):
            vector = features(self.parsed, self.scores)[key]
            reversed_ = reverse_features(vector, key)
            self.assertEqual(reverse_features(reversed_, key), vector)
        base = features(self.parsed, self.scores)["extra_counterexample"]
        self.assertEqual(reverse_features(base)[1], base[0])
        self.assertEqual(reverse_features(base)[7], base[6])
        self.assertEqual(reverse_features(base, "counterexample"), reverse_features(base))
        rationale = features(self.parsed, self.scores)["extra_rationale"]
        self.assertEqual(reverse_features(rationale, "rationale"),
                         reverse_features(rationale, "extra_rationale"))

    def test_attachment_is_gold_invariant_and_swap_aware(self):
        rows = [{"text1": "animal", "text2": "dog", "label": "FORWARD_ENTAILMENT"},
                {"text1": "dog", "text2": "animal", "label": "BACKWARD_ENTAILMENT"}]
        cache = {("animal", "dog"): {"parsed": self.parsed, "scores": self.scores}}
        result = attach_features(rows, cache)
        relabeled = attach_features([{**row, "label": "NEGATIVE_OTHER"} for row in rows], cache)
        for original, changed in zip(result, relabeled):
            self.assertEqual(original["extra_counterexample"], changed["extra_counterexample"])
            self.assertEqual(original["extra_rationale"], changed["extra_rationale"])
        self.assertEqual(result[1]["extra_counterexample"],
                         reverse_features(result[0]["extra_counterexample"]))

    def test_prompt_and_canonicalization_do_not_use_metadata(self):
        self.assertEqual(canonical_pair(" dog ", "animal"), ("animal", "dog", True))
        prompt = build_prompt("animal", "dog")
        self.assertNotIn("ENTAILMENT", prompt)
        self.assertNotIn("label", prompt.lower())
        fenced = "output:\n```json\n" + json.dumps({k: self.parsed[k] for k in "abrfg"}) + "\n```"
        self.assertTrue(parse_generation(fenced)["parsed"])
        self.assertFalse(parse_generation('{"a":"a","b":"b"}')["parsed"])


if __name__ == "__main__":
    unittest.main()
