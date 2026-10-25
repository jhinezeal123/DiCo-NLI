import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from causal_public_grounding import public_graph, build_public_grounded_requests, development_label, assemble_public
from causal_public_protocol import decode_native, native_messages


class PublicGroundingTests(unittest.TestCase):
    def fixture(self):
        text = ("Imagine a hypothetical world: Alice has a direct effect on dan. Bob has a direct effect on dan. "
                "Cara has a direct effect on dan.\n\nDan is on when alice is on or bob is on and cara is off. "
                "We observe bob on and cara on.\n\nWould dan be on if alice were on instead of off?")
        meaning = {alias: {"name": name, "one": name+" on", "zero": name+" off"}
                   for alias, name in zip("ABCD", ["alice", "bob", "cara", "dan"])}
        table = [[[int(a or (b and not c)) for b in (0, 1)] for a in (0, 1)] for c in (0, 1)]
        program = {"meaning": meaning, "params": {"p(A)": .5, "p(B)": .5, "p(C)": .5, "p(D | C,A,B)": table},
                   "observed": {"B": 1, "C": 1}, "intervene": {"A": 1}, "outcome": "D", "polarity": 1}
        demo = {"text": text, "program": program, "answer": "yes"}
        return demo, {"text": text, "question_id": 123, "answer": "yes"}

    def test_oracle_choices_preserve_three_parent_order_and_point_counterfactual(self):
        demo, row = self.fixture()
        requests, plans = build_public_grounded_requests([row], [demo])
        self.assertEqual(public_graph(row["text"])["parents"]["dan"], ["alice", "bob", "cara"])
        decisions = {spec["request_id"]: {"choice_index": development_label(spec, demo["program"])}
                     for spec in plans[123]["specs"]}
        program = assemble_public(plans[123], decisions)
        self.assertEqual(program["observed"], {"bob": 1, "cara": 1})
        self.assertEqual(program["intervene"], {"alice": 1})
        self.assertEqual(decode_native(json.dumps(program), "native_parser_solver")["prediction"], "yes")
        expected = [[[int(a or (b and not c)) for c in (0, 1)] for b in (0, 1)] for a in (0, 1)]
        self.assertEqual(program["params"]["p(dan | alice,bob,cara)"], expected)

    def test_evaluation_requests_ignore_hidden_cpts_mapping_query_and_label(self):
        demo, row = self.fixture()
        expected = build_public_grounded_requests([row], [demo])
        poisoned = copy.deepcopy(row)
        poisoned.update({"answer": "SECRET", "program": "SECRET", "meta": "SECRET", "params": "SECRET"})
        self.assertEqual(expected, build_public_grounded_requests([poisoned], [demo]))
        self.assertNotIn("SECRET", json.dumps(expected))

    def test_unobserved_declaration_does_not_become_factual_evidence(self):
        demo, row = self.fixture()
        row["text"] = row["text"].replace("Cara has a direct effect on dan.",
                                         "Cara has a direct effect on dan. Bob is unobserved.")
        _, plans = build_public_grounded_requests([row], [demo])
        self.assertFalse(any(s["stage"] == "evidence" and s["name"] == "bob" for s in plans[123]["specs"]))
        with self.assertRaises(ValueError):
            public_graph("Unsupported free text")

    def test_stronger_control_has_executable_development_reasoning(self):
        demo, row = self.fixture()
        messages = native_messages(row, "structured_program_demos", [demo])
        self.assertIn("Abduction:", messages[2]["content"])
        self.assertIn("Intervention:", messages[2]["content"])
        self.assertIn("Prediction:", messages[2]["content"])
        self.assertTrue(messages[2]["content"].endswith("<answer>yes</answer>"))
