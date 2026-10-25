"""Stress-suite semantics, heldouts, leakage guards, and eight-node compatibility."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from causal_dataset_v3 import build_dataset_v3, solve_conditioned
from causal_dataset_v4 import build_dataset_v4
from causal_scm import solve, normalize_scm
from causal_v3_compiler import build_grounded_v3, assemble_grounded_v3, build_local_v3


class TwelveVariableStressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = build_dataset_v4(20261029, 6)

    def test_exact_stress_dimensions_and_label_balance(self):
        self.assertEqual(len(self.data["rows"]), 96)
        for info in self.data["audit"]["splits"].values():
            self.assertEqual(info["intervention_labels"], {"0": 2, "1": 2, "unknown": 2})
        for row in self.data["rows"] + self.data["demonstrations"]:
            nodes = row["scm"]["variables"]
            self.assertEqual(len(nodes), 12)
            self.assertTrue(6 <= len(row["scm"]["observed"]) <= 8)
            self.assertIn(row["compatible_worlds"], (16, 32, 64))
            binary = sum(len(n["parents"]) == 2 for n in nodes)
            ternary = sum(len(n["parents"]) == 3 for n in nodes)
            if row["split"] == "ood_mechanism":
                self.assertEqual((binary, ternary), (0, 3))
            elif row["split"] == "ood_topology":
                self.assertTrue(6 <= binary <= 8)
                self.assertEqual(ternary, 0)
            else:
                self.assertEqual((binary, ternary), (2, 0))

    def test_missing_endogenous_enumeration_matches_independent_noise_oracle(self):
        # One whole family per split, all four views, exhaustive 2**12 noise worlds.
        selected = {next(row["family_id"] for row in self.data["rows"] if row["split"] == split)
                    for split in self.data["audit"]["splits"]}
        for row in self.data["rows"]:
            if row["family_id"] in selected:
                self.assertEqual(solve_conditioned(row["scm"]), solve(row["scm"]))

    def test_twelve_choice_compiler_with_oracle_semantic_decisions(self):
        requests, plans = build_grounded_v3(self.data["rows"], self.data["demonstrations"])
        by_key = {r["request_id"]: r for r in requests}
        for row in self.data["rows"]:
            plan = plans[row["case_id"]]
            self.assertEqual(by_key[plan["target_request"]]["choice_count"], 12)
            self.assertIn("L. ", by_key[plan["target_request"]]["messages"][-1]["content"])
            gold = {n["name"]: n for n in row["scm"]["variables"]}
            decisions = {}
            for node in plan["nodes"]:
                if node["op"] is None:
                    ops = ["and", "or", "xor"] if node["stage"] == "binary" else ["majority3", "parity3"]
                    decisions[node["request_id"]] = {"choice_index": ops.index(gold[node["name"]]["op"])}
            decisions[plan["target_request"]] = {"choice_index": plan["names"].index(row["scm"]["target"])}
            for name, key in plan["active_requests"].items():
                decisions[key] = {"choice_index": int(name in row["scm"]["intervene"])}
            self.assertEqual(solve_conditioned(assemble_grounded_v3(plan, decisions))["answer"], row["answer"])

    def test_both_compilers_still_ignore_hidden_test_fields(self):
        poisoned = copy.deepcopy(self.data["rows"])
        for row in poisoned:
            row.update({"scm": {"hidden": "SECRET"}, "answer": "SECRET", "view": "SECRET", "compatible_worlds": -1})
        for builder in (build_grounded_v3, build_local_v3):
            expected = builder(self.data["rows"], self.data["demonstrations"])
            self.assertEqual(expected, builder(poisoned, self.data["demonstrations"]))
            self.assertNotIn("SECRET", json.dumps(expected))

    def test_cross_seed_world_and_demo_exclusion(self):
        seen = set()
        first = build_dataset_v4(1001, 6, seen)
        second = build_dataset_v4(1002, 6, seen)
        ids = lambda data: {row["family_id"] for row in data["rows"] + data["demonstrations"]}
        self.assertFalse(ids(first) & ids(second))
        self.assertEqual(len(seen), 60)
        self.assertEqual(second["audit"]["prior_excluded_families"], 30)

    def test_original_eight_node_dataset_unchanged_and_size_limit_remains_bounded(self):
        old = build_dataset_v3(20261027, 12)
        self.assertEqual(old["audit"]["dataset_sha256"], "9db48daf42be9109ef4049ee9f6875ce2c76e246519e79710d03727e14bdd717")
        scm = copy.deepcopy(self.data["rows"][0]["scm"])
        scm["variables"].append({"name": "extra", "op": "root", "parents": []})
        with self.assertRaises(ValueError):
            normalize_scm(scm)


if __name__ == "__main__":
    unittest.main()
