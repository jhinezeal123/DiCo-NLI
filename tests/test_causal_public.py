import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from causal_public_protocol import build_public_split, public_text, native_messages, decode_native, demonstration_program


class PublicProtocolTests(unittest.TestCase):
    def fixture(self):
        questions, models = [], []
        for graph, model_ids in (("chain", [1, 2, 3]), ("mediation", [3220, 4, 5])):
            for mid in model_ids:
                models.append({"model_id": mid, "background": "A affects B.",
                    "params": {"p(X)": .5, "p(Y | X)": [1, 0]},
                    "variable_mapping": {"Xname": "A", "X1": "A on", "X0": "A off",
                                         "Yname": "B", "Y1": "B on", "Y0": "B off"}})
                for i in range(2):
                    questions.append({"question_id": mid*100+i, "given_info": "A turns B off.",
                        "question": "Would B be on if A were off?", "answer": "yes",
                        "meta": {"query_type": "det-counterfactual", "graph_id": graph, "model_id": mid,
                                 "given_info": {}, "treatment": "X", "action": 0, "outcome": "Y", "polarity": 1}})
        return questions, models

    def test_model_group_split_excludes_whole_development_world(self):
        questions, models = self.fixture()
        split = build_public_split(questions, models, models_per_graph=2)
        ids = [r["model_id"] for r in split["rows"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertNotIn(3220, ids)
        self.assertFalse(set(ids) & {d["model_id"] for d in split["demonstrations"]})
        self.assertEqual(split, build_public_split(list(reversed(questions)), list(reversed(models)), 2))

    def test_evaluation_prompt_allowlist_ignores_all_hidden_fields(self):
        questions, models = self.fixture()
        q, m = questions[0], models[0]
        original = public_text(q, m)
        q = copy.deepcopy(q); m = copy.deepcopy(m)
        q.update({"answer": "SECRET_ANSWER", "meta": {"SECRET_META": True}, "reasoning": "SECRET_REASONING"})
        m.update({"params": "SECRET_CPT", "variable_mapping": "SECRET_MAP", "structure": "SECRET_GRAPH"})
        self.assertEqual(original, public_text(q, m))
        row = {"text": original, "answer": "SECRET_TEST_LABEL", "program": "SECRET_TEST_PROGRAM"}
        # CausalCoT aliases also come exclusively from the public background.
        row["text"] = "Imagine a world: A has a direct effect on B.\n\n" + original
        for method in ("direct", "structured", "native_parser_solver", "native_causal_cot"):
            self.assertNotIn("SECRET", json.dumps(native_messages(row, method, [])))

    def test_train_program_preserves_native_semantics_and_no_xor_noise(self):
        qs, ms = self.fixture()
        program = demonstration_program(qs[0], ms[0])
        result = decode_native(json.dumps(program), "native_parser_solver")
        self.assertTrue(result["valid"])
        self.assertEqual(result["prediction"], "yes")
        self.assertEqual(result["value"], 1)
        program["params"]["p(B | A)"] = [0, 1]
        self.assertEqual(decode_native(json.dumps(program), "native_parser_solver")["prediction"], "no")

    def test_invalid_or_unidentified_predictions_do_not_receive_answers(self):
        self.assertFalse(decode_native('{"params": {}}', "native_parser_solver")["valid"])
        self.assertFalse(decode_native('<answer>yes</answer><answer>no</answer>', "direct")["valid"])
        program = {"params": {"p(A)": .5, "p(B)": .5}, "observed": {},
                   "intervene": {"A": 1}, "outcome": "B", "polarity": 1}
        result = decode_native(json.dumps(program), "native_parser_solver")
        self.assertEqual(result["prediction"], "unknown")


if __name__ == "__main__":
    unittest.main()
