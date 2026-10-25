"""Decoder semantics, unchanged public inputs and frozen learned-output guards."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from causal_action_algebra import TRANSITIONS, intervention_mask
from causal_dataset_v5 import build_dataset_v5
from causal_dataset_v3 import solve_conditioned
from causal_operator_replay import load_operator_replay
from causal_semantic_strategies import build_semantic_requests, slot_transition, project_transition, infer_semantic_state, parse_cot_state
from causal_sequence_compiler import scm_for_mask
from causal_v3_pipeline import load_control_replay
from causal_semantic_methods import run_semantic_methods

SLOTS = [(i, 0) for i in range(9)] + [(0, 2), (1, 1), (2, 2), (3, 1), (6, 2), (7, 1), (9, 0)]


class SemanticDecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = build_dataset_v5(20261030, 6)
        cls.cfg = json.loads(Path("configs/causal_v3.json").read_text())

    def test_slot_composition_and_denotational_projection_recover_all_public_functions(self):
        for action, (effect, guard) in enumerate(SLOTS):
            self.assertEqual(slot_transition(effect, guard), TRANSITIONS[action])
            probabilities = [[.97 if j == TRANSITIONS[action][i] else .01 for j in range(4)] for i in range(4)]
            table, chosen, scores = project_transition(probabilities)
            self.assertEqual(table, TRANSITIONS[action])
            self.assertEqual(chosen, action)
        # A model may predict a well-defined effect/guard combination outside the
        # original16 classes. Execute it, without silently repairing it to gold.
        self.assertEqual(slot_transition(5, 1), (0, 0, 2, 0))
        self.assertNotIn(slot_transition(5, 1), TRANSITIONS)
        with self.assertRaises(ValueError):
            project_transition([[.25]*4])

    def test_all_three_strategies_with_oracle_unit_responses(self):
        for mode in ("slots", "probes", "state_cot"):
            operators, requests, plans = build_semantic_requests(self.data["rows"], self.data["demonstrations"], mode)
            decisions, outputs = {}, {}
            for row in self.data["rows"]:
                plan = plans[row["case_id"]]
                truth = {n["name"]: n for n in row["scm"]["variables"]}
                for node in plan["nodes"]:
                    if "request_id" in node:
                        ops = ["and", "or", "xor"] if node["stage"] == "binary" else ["majority3", "parity3"]
                        decisions[node["request_id"]] = {"choice_index": ops.index(truth[node["name"]]["op"])}
                if mode == "state_cot":
                    mask = intervention_mask(row["scm"]["intervene"], plan["proposed"])
                    outputs[plan["cot_request"]] = {"raw_output": f"Unit trace. <state>{'ABCD'[mask]}</state>"}
                else:
                    for old, action in zip(plan["action_requests"], row["program_actions"]):
                        keys = plan["semantic_requests"][old]
                        if mode == "slots":
                            outputs[keys[0]] = {"choice_index": SLOTS[action][0]}
                            outputs[keys[1]] = {"choice_index": SLOTS[action][1]}
                        else:
                            for i, key in enumerate(keys):
                                outputs[key] = {"conditional_choice_probs": [.97 if j == TRANSITIONS[action][i] else .01 for j in range(4)]}
                mask, info = infer_semantic_state(plan, outputs, mode)
                compiled = scm_for_mask(plan, decisions, mask)
                self.assertEqual(solve_conditioned(compiled)["answer"], row["answer"])

    def test_new_requests_ignore_test_labels_and_training_intermediate_traces(self):
        rows, demos = copy.deepcopy(self.data["rows"]), copy.deepcopy(self.data["demonstrations"])
        for row in rows:
            row.update({"scm": "SECRET", "answer": "SECRET", "view": "SECRET", "program_actions": "SECRET", "program_true_states": "SECRET"})
        for demo in demos:
            demo.update({"program_actions": "SECRET", "program_true_states": "SECRET"})
        for mode in ("slots", "probes", "state_cot"):
            expected = build_semantic_requests(self.data["rows"], self.data["demonstrations"], mode)
            found = build_semantic_requests(rows, demos, mode)
            self.assertEqual(expected, found)
            self.assertNotIn("SECRET", json.dumps(found))

    def test_cot_requires_an_explicit_generated_final_marker(self):
        self.assertEqual(parse_cot_state("First A, later <state>D</state>"), 3)
        self.assertEqual(parse_cot_state("<state>A</state> then <state>b</state>"), 1)
        for text in ("The state is D", "<state>5</state>", ""):
            with self.assertRaises(ValueError):
                parse_cot_state(text)

    def test_real_control_and_operator_replays_are_complete_and_preserve_wrong_predictions(self):
        operators, _, _ = build_semantic_requests(self.data["rows"], self.data["demonstrations"], "slots")
        learned = load_operator_replay(self.cfg, self.data, operators)
        self.assertEqual(len(learned), 82)
        controls = load_control_replay(self.cfg, self.data)
        self.assertEqual(len(controls), 8)
        self.assertEqual(sum(len(rows) for rows in controls.values()), 768)
        for change in ({"pilot_only": False}, {"model_revision": "wrong"}):
            with self.assertRaises(ValueError):
                load_operator_replay({**self.cfg, **change}, self.data, operators)
        with tempfile.TemporaryDirectory() as directory:
            spec = self.cfg["operator_replay"]
            rows = [json.loads(line) for line in Path(spec["path"]).read_text().splitlines()]
            expected = {r["request_id"]: r for r in operators}
            first = rows[0]
            altered = (first["choice_index"] + 1) % expected[first["request_id"]]["choice_count"]
            first["choice_index"] = altered
            first["raw_output"] = "ABC"[altered]
            raw = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
            path = Path(directory) / "predictions.jsonl"
            path.write_bytes(raw)
            cfg = {**self.cfg, "operator_replay": {**spec, "path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}}
            loaded = load_operator_replay(cfg, self.data, operators)
            self.assertEqual(loaded[first["request_id"]]["choice_index"], altered)
            cfg["operator_replay"]["sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                load_operator_replay(cfg, self.data, operators)

    def test_runner_routes_each_decoder_and_accounts_new_work_separately(self):
        for mode in ("slots", "probes", "state_cot"):
            operators, requests, plans = build_semantic_requests(self.data["rows"], self.data["demonstrations"], mode)
            op_choices, answers = {}, {}
            for row in self.data["rows"]:
                plan = plans[row["case_id"]]
                truth = {n["name"]: n for n in row["scm"]["variables"]}
                for node in plan["nodes"]:
                    if "request_id" in node:
                        choices = ["and", "or", "xor"] if node["stage"] == "binary" else ["majority3", "parity3"]
                        op_choices[node["request_id"]] = {"choice_index": choices.index(truth[node["name"]]["op"]), "stage": node["stage"]}
                if mode == "state_cot":
                    mask = intervention_mask(row["scm"]["intervene"], plan["proposed"])
                    answers[plan["cot_request"]] = {"raw_output": f"<state>{'ABCD'[mask]}</state>"}
                else:
                    for old, action in zip(plan["action_requests"], row["program_actions"]):
                        keys = plan["semantic_requests"][old]
                        if mode == "slots":
                            answers[keys[0]] = {"choice_index": SLOTS[action][0]}
                            answers[keys[1]] = {"choice_index": SLOTS[action][1]}
                        else:
                            for state, key in enumerate(keys):
                                answers[key] = {"conditional_choice_probs": [.97 if j == TRANSITIONS[action][state] else .01 for j in range(4)]}
            outputs = {r["request_id"]: {"input_tokens": 100, "output_tokens": 5 if mode == "state_cot" else 1,
                                         **answers[r["request_id"]]} for r in requests}
            captured = []
            def score(method, raw, costs, usage, **kwargs):
                self.assertAlmostEqual(sum(v["input_tokens"] for v in costs.values()), usage["input_tokens"])
                self.assertAlmostEqual(sum(v["output_tokens"] for v in costs.values()), usage["output_tokens"])
                if method == "grounded_solver":
                    for row in self.data["rows"]:
                        self.assertEqual(solve_conditioned(json.loads(raw[row["case_id"]]))["answer"], row["answer"])
                captured.append(method)
            with patch("causal_semantic_methods.load_operator_replay", return_value=op_choices), \
                 patch("causal_semantic_methods.decide", return_value=outputs):
                run_semantic_methods(None, None, self.data, {**self.cfg, "sequence_strategy": mode},
                                     lambda *args: None, score, lambda *args: outputs)
            self.assertEqual(captured, ["grounded_solver", "grounded_operator_blind", "grounded_action_blind"])


if __name__ == "__main__":
    unittest.main()
