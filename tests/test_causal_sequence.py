"""Independent state semantics, matched permutations, leakage and run accounting."""
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from causal_action_algebra import ACTION_CODES, fold_actions, intervention_mask
from causal_choice import LETTERS
from causal_dataset_v3 import solve_conditioned
from causal_dataset_v5 import build_dataset_v5
from causal_scm import solve
from causal_sequence_compiler import program_document, build_sequence_requests, assemble_sequence, used_requests
from causal_sequence_methods import run_sequence_methods
from causal_v3_protocol import decode_v3


def independent_set_execution(actions, initial=()):
    """Set semantics, independent of the implementation's bitmask truth tables."""
    active = set(initial)
    for action in actions:
        code = ACTION_CODES[action]
        operation, _, condition = code.partition("_if_")
        if condition and condition not in active:
            continue
        if operation == "no_change":
            continue
        if operation == "swap":
            active = {"second" if item == "first" else "first" for item in active}
            continue
        verb, _, subject = operation.partition("_")
        subjects = {"first", "second"} if subject == "both" else {subject}
        if verb == "enable":
            active.update(subjects)
        elif verb == "restore":
            active.difference_update(subjects)
        elif verb == "toggle":
            active.symmetric_difference_update(subjects)
        else:
            raise AssertionError(code)
    return int("first" in active) + 2 * int("second" in active)


class ConditionalProgramTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = build_dataset_v5(20261030, 6)
        cls.requests, cls.plans = build_sequence_requests(cls.data["rows"], cls.data["demonstrations"])

    def oracle_decisions(self):
        # Unit-test responses only. Production builders never read these fields.
        choices = {}
        for row in self.data["rows"]:
            plan = self.plans[row["case_id"]]
            truth = {n["name"]: n for n in row["scm"]["variables"]}
            for node in plan["nodes"]:
                if "request_id" in node:
                    ops = ["and", "or", "xor"] if node["stage"] == "binary" else ["majority3", "parity3"]
                    choices[node["request_id"]] = ops.index(truth[node["name"]]["op"])
            for key, action in zip(plan["action_requests"], row["program_actions"]):
                self.assertEqual(choices.setdefault(key, action), action)
            choices[plan["state_request"]] = intervention_mask(row["scm"]["intervene"], plan["proposed"])
        self.assertEqual(set(choices), {r["request_id"] for r in self.requests})
        return {r["request_id"]: {"request_id": r["request_id"], "stage": r["stage"],
                                  "choice_index": choices[r["request_id"]], "input_tokens": 100,
                                  "output_tokens": 1, "seconds": .01,
                                  "raw_output": LETTERS[choices[r["request_id"]]]} for r in self.requests}

    def test_all_transition_functions_against_independent_set_interpreter(self):
        for action in range(16):
            for mask in range(4):
                initial = [name for i, name in enumerate(("first", "second")) if mask & (1 << i)]
                self.assertEqual(fold_actions([action], mask)[-1], independent_set_execution([action], initial))
        rng = random.Random(1407)
        for _ in range(100):
            actions = [rng.randrange(16) for _ in range(8)]
            self.assertEqual(fold_actions(actions)[-1], independent_set_execution(actions))

    def test_equal_length_equal_words_different_orders_and_endpoints(self):
        families = defaultdict(list)
        for row in self.data["rows"]:
            families[row["family_id"]].append(row)
            actions = row["program_actions"]
            self.assertEqual(len(actions), 8)
            self.assertEqual(sum(9 <= a < 15 for a in actions), 4)
            self.assertGreaterEqual(len(row["program_relevant_steps"]), 4)
            self.assertGreaterEqual(sum(i < 4 for i in row["program_relevant_steps"]), 2)
            self.assertNotEqual(fold_actions(actions)[-1], fold_actions(list(reversed(actions)))[-1])
            self.assertEqual(fold_actions(actions)[-1], independent_set_execution(actions))
        for rows in families.values():
            self.assertEqual(len({r["text"] for r in rows}), 4)
            self.assertTrue(all(Counter(r["text"].split()) == Counter(rows[0]["text"].split()) for r in rows))
            masks = {r["view"]: fold_actions(r["program_actions"])[-1] for r in rows}
            self.assertEqual((masks["factual"], masks["intervention"], masks["restored"]), (0, 3, 0))
            self.assertIn(masks["partial_release"], (1, 2))

    def test_old_worlds_and_new_seeds_are_excluded_and_state_demos_cover_four_options(self):
        masks = Counter(intervention_mask(row["scm"]["intervene"], program_document(row["text"])["proposed"])
                        for row in self.data["demonstrations"])
        self.assertEqual(masks, {0: 2, 1: 1, 2: 1, 3: 2})
        seen = set()
        first = build_dataset_v5(2001, 6, seen)
        second = build_dataset_v5(2002, 6, seen)
        ids = lambda data: {row["family_id"] for row in data["rows"] + data["demonstrations"]}
        self.assertFalse(ids(first) & ids(second))
        self.assertEqual(first["audit"]["prior_excluded_families"], 30)
        self.assertEqual(second["audit"]["prior_excluded_families"], 60)
        self.assertEqual(len(seen), 90)

    def test_program_endpoints_match_independent_causal_oracle(self):
        selected = {next(r["family_id"] for r in self.data["rows"] if r["split"] == split)
                    for split in self.data["audit"]["splits"]}
        for row in self.data["rows"]:
            doc = program_document(row["text"])
            mask = independent_set_execution(row["program_actions"])
            self.assertEqual(mask, intervention_mask(row["scm"]["intervene"], doc["proposed"]))
            if row["family_id"] in selected:
                self.assertEqual(solve_conditioned(row["scm"]), solve(row["scm"]))

    def test_oracle_semantics_compile_all_queries_and_strong_state_control_is_independent(self):
        decisions = self.oracle_decisions()
        for row in self.data["rows"]:
            plan = self.plans[row["case_id"]]
            for variant in ("sequence", "state_choice"):
                required = {key: decisions[key] for key in used_requests(plan, variant)}
                scm, diagnostics = assemble_sequence(plan, required, variant)
                self.assertEqual(solve_conditioned(scm)["answer"], row["answer"])
                if variant == "state_choice":
                    self.assertEqual(diagnostics["predicted_action_choices"], [])
                    self.assertFalse(set(plan["action_requests"]) & set(required))

    def test_compiler_ignores_all_hidden_program_labels_and_world_metadata(self):
        poisoned = copy.deepcopy(self.data["rows"])
        for row in poisoned:
            row.update({"scm": "SECRET", "answer": "SECRET", "view": "SECRET",
                        "program_actions": "SECRET", "program_true_states": "SECRET",
                        "program_relevant_steps": "SECRET", "changed": "SECRET"})
        self.assertEqual((self.requests, self.plans), build_sequence_requests(poisoned, self.data["demonstrations"]))
        self.assertNotIn("SECRET", json.dumps(build_sequence_requests(poisoned, self.data["demonstrations"])))
        actions = [r for r in self.requests if r["stage"] == "action"]
        self.assertTrue(actions)
        self.assertTrue(all(len(r["messages"]) == 2 and r["choice_count"] == 16 for r in actions))
        self.assertTrue(all("current state is NOT supplied" in r["messages"][0]["content"] for r in actions))
        self.assertTrue(all("Begin with neither" not in r["messages"][0]["content"] for r in actions))

    def test_rejects_ambiguous_target_or_malformed_instruction_order(self):
        row = self.data["rows"][0]
        doc = program_document(row["text"])
        other = next(name for name in doc["names"] if name != doc["target"])
        with self.assertRaises(ValueError):
            program_document(row["text"].replace("Readout: What", f"Readout: For {other}, what"))
        with self.assertRaises(ValueError):
            program_document(row["text"].replace("\n2. ", "\n9. "))

    def test_tracked_method_routing_and_request_costs_with_unit_oracle(self):
        decisions = self.oracle_decisions()
        captured = {}
        def scorer(method, raw, costs, usage, derived=None, diagnostics_by_case=None):
            self.assertEqual(len(raw), 96)
            self.assertAlmostEqual(sum(r["input_tokens"] for r in costs.values()), usage["input_tokens"])
            self.assertAlmostEqual(sum(r["output_tokens"] for r in costs.values()), usage["output_tokens"])
            if method in ("grounded_solver", "grounded_state_choice"):
                for row in self.data["rows"]:
                    self.assertEqual(decode_v3(raw[row["case_id"]], "parser_solver")["prediction"], row["answer"])
            captured[method] = derived
        with patch("causal_sequence_methods.decide", return_value=decisions):
            run_sequence_methods(None, None, self.data, {"batch_size": 4}, lambda *args: None, scorer)
        self.assertEqual(set(captured), {"grounded_solver", "grounded_state_choice", "grounded_operator_blind", "grounded_action_blind"})
        self.assertIsNone(captured["grounded_state_choice"])


if __name__ == "__main__":
    unittest.main()
