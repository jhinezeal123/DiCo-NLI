"""Supervision independence, executable labels, completion boundaries and scoring."""
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from causal_action_algebra import intervention_mask
from causal_choice import LETTERS
from causal_dataset_v5 import build_dataset_v5
from causal_intervention_supervision import (build_intervention_training, build_supervised_requests,
    training_feature, render_training_action, atomic_request)
from causal_sequence_compiler import build_sequence_requests
from train_intervention_semantics import score_semantics
from test_causal_sequence import independent_set_execution


class CharacterTokenizer:
    """A small deterministic tokenizer to exercise label boundaries without ML."""
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        result = "".join(row["role"] + ":" + row["content"] + "<end>" for row in messages)
        return result + ("assistant:" if add_generation_prompt else "")

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(char) for char in text]}


class InterventionSupervisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training, cls.audit = build_intervention_training()
        cls.data = build_dataset_v5(20261030, 6)
        cls.operators, cls.queries, cls.plans = build_supervised_requests(cls.data["rows"], cls.data["demonstrations"])

    def test_independent_short_training_is_reproducible_balanced_and_label_correct(self):
        second, audit = build_intervention_training()
        self.assertEqual(self.training, second)
        self.assertEqual(self.audit, audit)
        self.assertEqual(self.audit["tasks"], {"action": 256, "state_probe": 1024, "program_state": 256, "program_cot": 256})
        self.assertEqual(self.audit["program_final_classes"], {letter: 64 for letter in "ABCD"})
        self.assertFalse(self.audit["heldout_language_renderer_used"])
        self.assertEqual(set(self.audit["names"]) & {name for plan in self.plans.values() for name in plan["names"]}, set())
        for row in self.training:
            self.assertLessEqual(len(row["actions"]), 4)
            if row["stage"] == "action":
                self.assertEqual(row["completion"], LETTERS[row["actions"][0]])
                continue
            initial = [name for index, name in enumerate(("first", "second")) if row["initial_state"] & (1 << index)]
            expected = independent_set_execution(row["actions"], initial)
            if row["stage"] == "program_cot":
                states = [independent_set_execution(row["actions"][:index]) for index in range(1, len(row["actions"]) + 1)]
                text = " ".join(f"{index}:{LETTERS[state]}" for index, state in enumerate(states, 1)) + f"\n<state>{LETTERS[expected]}</state>"
                self.assertEqual(row["completion"], text)
            else:
                self.assertEqual(row["completion"], LETTERS[expected])
        with self.assertRaises(ValueError):
            render_training_action(0, "first", "second", 3)

    def test_test_requests_are_public_only_and_atomic_prompts_unchanged(self):
        old, old_plans = build_sequence_requests(self.data["rows"], self.data["demonstrations"])
        old_actions = {row["request_id"]: row for row in old if row["stage"] == "action"}
        new_actions = {row["request_id"]: row for row in self.queries if row["stage"] == "action"}
        self.assertEqual(new_actions, old_actions)
        self.assertEqual(len(new_actions), 162)
        self.assertEqual(Counter(row["stage"] for row in self.queries), {"action": 162, "program_state": 96, "program_cot": 96})
        rows = copy.deepcopy(self.data["rows"])
        for row in rows:
            row.update(scm="PRIVATE", answer="PRIVATE", view="PRIVATE", program_actions="PRIVATE", program_true_states="PRIVATE")
        observed = build_supervised_requests(rows, self.data["demonstrations"])
        self.assertEqual(observed, (self.operators, self.queries, self.plans))
        self.assertNotIn("PRIVATE", json.dumps(observed))
        self.assertTrue(all(len(row["messages"]) == 2 for row in self.queries))

    def test_training_feature_masks_only_prompt_and_rejects_truncation_or_bad_boundary(self):
        row = self.training[0]
        tokenizer = CharacterTokenizer()
        feature = training_feature(tokenizer, row, 10000)
        supervised = [label for label in feature["labels"] if label != -100]
        self.assertEqual(supervised, [ord(char) for char in row["completion"] + "<end>"])
        self.assertEqual(sum(label == -100 for label in feature["labels"]), len(feature["input_ids"]) - len(supervised))
        with self.assertRaises(ValueError):
            training_feature(tokenizer, row, 10)
        class BrokenBoundary(CharacterTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                return ("wrong-prefix" if kwargs.get("add_generation_prompt") else "") + super().apply_chat_template(messages, **kwargs)
        with self.assertRaises(ValueError):
            training_feature(BrokenBoundary(), row, 10000)

    def oracle_outputs(self):
        operators, outputs = {}, {}
        for row in self.data["rows"]:
            plan = self.plans[row["case_id"]]
            truth = {node["name"]: node for node in row["scm"]["variables"]}
            for node in plan["nodes"]:
                if "request_id" in node:
                    options = ["and", "or", "xor"] if node["stage"] == "binary" else ["majority3", "parity3"]
                    operators[node["request_id"]] = {"choice_index": options.index(truth[node["name"]]["op"])}
            for key, action in zip(plan["action_requests"], row["program_actions"]):
                outputs[key] = {"choice_index": action}
            mask = intervention_mask(row["scm"]["intervene"], plan["proposed"])
            outputs[plan["program_state_request"]] = {"choice_index": mask}
            outputs[plan["program_cot_request"]] = {"raw_output": f"Test trace <state>{LETTERS[mask]}</state>"}
        for row in outputs.values():
            row.update(input_tokens=100, output_tokens=1, seconds=.01)
        return operators, outputs

    def test_each_decoder_and_ablations_use_same_kernel_and_account_shared_requests(self):
        operators, outputs = self.oracle_outputs()
        for mode in ("action", "program_state", "program_cot"):
            captured = []
            method, records = score_semantics(self.data, self.plans, operators, outputs, "unit", mode,
                                              lambda tag, value: captured.append((tag, value)))
            self.assertEqual(method, "unit:" + mode)
            self.assertTrue(all(row["correct"] for row in records))
            summary = captured[-1][1]
            self.assertAlmostEqual(sum(row["input_tokens"] for row in records), summary["input_tokens"])
            self.assertAlmostEqual(sum(row["output_tokens"] for row in records), summary["output_tokens"])
        _, blind = score_semantics(self.data, self.plans, operators, outputs, "unit", "action", lambda *args: None, "action_blind")
        self.assertTrue(all(row["parsed_scm"]["intervene"] == {} and row["input_tokens"] == 0 for row in blind))
        _, blind = score_semantics(self.data, self.plans, operators, outputs, "unit", "action", lambda *args: None, "operator_blind")
        for row in blind:
            self.assertTrue(all(node["op"] == "and" for node in row["parsed_scm"]["variables"] if len(node["parents"]) == 2))

    def test_malformed_trace_stays_invalid_and_wrong_output_is_not_gold_repaired(self):
        operators, outputs = self.oracle_outputs()
        first = next(iter(self.plans.values()))
        outputs[first["program_cot_request"]]["raw_output"] = "The state is D; no explicit marker."
        _, rows = score_semantics(self.data, self.plans, operators, outputs, "unit", "program_cot", lambda *args: None)
        observed = next(row for row in rows if row["case_id"] == next(iter(self.plans)))
        self.assertFalse(observed["valid"])
        self.assertIsNone(observed["final_intervention_mask"])
        key = first["program_state_request"]
        wrong = (outputs[key]["choice_index"] + 1) % 4
        outputs[key]["choice_index"] = wrong
        _, rows = score_semantics(self.data, self.plans, operators, outputs, "unit", "program_state", lambda *args: None)
        self.assertEqual(rows[0]["final_intervention_mask"], wrong)


if __name__ == "__main__":
    unittest.main()
