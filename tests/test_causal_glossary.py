import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from causal_state_glossary import parse_glossary, glossary_messages, glossary_context
from causal_public_grounding import build_public_grounded_requests


class StateGlossaryTests(unittest.TestCase):
    def fixture(self):
        text = ("Imagine a hypothetical world: Alice has a direct effect on cholesterol level.\n\n"
                "Alice being active causes low cholesterol.\n\nWould cholesterol be low if alice were active?")
        meaning = {"A": {"name": "alice", "one": "active alice", "zero": "inactive alice"},
                   "B": {"name": "cholesterol level", "one": "low cholesterol", "zero": "high cholesterol"}}
        demo = {"text": text, "answer": "yes", "program": {"meaning": meaning,
                "params": {"p(A)": .5, "p(B | A)": [0, 1]}, "observed": {},
                "intervene": {"A": 1}, "outcome": "B", "polarity": 1}}
        glossary = {v["name"]: {"one": v["one"], "zero": v["zero"]} for v in meaning.values()}
        return demo, {"text": text, "question_id": 12}, glossary

    def test_state_meaning_can_be_low_for_one_and_is_shared_by_all_stages(self):
        demo, row, glossary = self.fixture()
        requests, _ = build_public_grounded_requests([row], [demo], {12: glossary})
        context = glossary_context(glossary)
        self.assertEqual({r["stage"] for r in requests}, {"cell", "evidence", "treatment", "value", "outcome", "polarity"})
        for request in requests:
            self.assertIn(context, request["messages"][-1]["content"])
            self.assertIn("overrides ALL default", request["messages"][0]["content"])
        self.assertEqual(parse_glossary(json.dumps(glossary), glossary), glossary)

    def test_invalid_missing_or_duplicate_glossary_cannot_be_silently_repaired(self):
        _, _, glossary = self.fixture()
        with self.assertRaises(ValueError):
            parse_glossary('{"alice":{"one":"on","zero":"off"}}', glossary)
        with self.assertRaises(ValueError):
            parse_glossary('{"alice":{"one":"on","zero":"off"},"alice":{"one":"on","zero":"off"}}', ["alice"])
        with self.assertRaises(ValueError):
            parse_glossary('{"alice":{"one":"on","zero":"ON"}}', ["alice"])

    def test_predicted_glossary_and_choice_prompts_ignore_hidden_evaluation_annotations(self):
        demo, row, glossary = self.fixture()
        poisoned = copy.deepcopy(row)
        poisoned.update({"program": "SECRET_PROGRAM", "meta": "SECRET_META", "answer": "SECRET_LABEL"})
        self.assertEqual(glossary_messages(row, [demo]), glossary_messages(poisoned, [demo]))
        self.assertEqual(build_public_grounded_requests([row], [demo], {12: glossary}),
                         build_public_grounded_requests([poisoned], [demo], {12: glossary}))
