import copy
import itertools
import json
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from causal_scm import solve, mechanism
from causal_dataset_v3 import build_dataset_v3, solve_conditioned, VIEWS
from causal_v3_protocol import metrics_v3, paired_v3
from causal_v3_compiler import build_grounded_v3, assemble_grounded_v3, build_local_v3


class HarderCausalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = build_dataset_v3(20261027, 6)

    def test_ternary_gate_truth_tables(self):
        for bits in itertools.product((0,1), repeat=3):
            self.assertEqual(mechanism("majority3",bits), int(sum(bits)>=2))
            self.assertEqual(mechanism("parity3",bits), sum(bits)%2)

    def test_conditioned_world_enumeration_matches_independent_noise_enumeration(self):
        for row in self.data["rows"]:
            self.assertEqual(solve_conditioned(row["scm"]), solve(row["scm"]))

    def test_four_views_partial_observations_and_balanced_unknown(self):
        data = self.data
        self.assertEqual(len(data["rows"]), 96)
        self.assertEqual(sorted(r["answer"] for r in data["demonstrations"]), ["0","0","1","1","unknown","unknown"])
        for audit in data["audit"]["splits"].values():
            self.assertEqual(audit["intervention_labels"], {"0":2,"1":2,"unknown":2})
        for row in data["rows"]:
            nodes = row["scm"]["variables"]
            self.assertEqual(len(nodes),8)
            self.assertTrue(4 <= len(row["scm"]["observed"]) <= 6)
            self.assertEqual(sum(len(n["parents"]) == 3 for n in nodes), int(row["split"] == "ood_mechanism"))
            self.assertEqual(len(row["scm"]["intervene"]), {"factual":0,"intervention":2,"partial_release":1,"restored":0}[row["view"]])

    def test_partial_noise_output_negation_quotient_is_exact(self):
        for a in (0,1):
            for observed in ({}, {"A":a}, {"B":a}):
                base = {"variables":[{"name":"A","op":"root","parents":[]},
                                     {"name":"B","op":"copy","parents":["A"]}],
                        "observed":observed,"intervene":{"A":1-a},"target":"B"}
                variant = copy.deepcopy(base); variant["variables"][1]["op"] = "not"
                self.assertEqual(solve(base),solve(variant))

    def test_oracle_semantic_choices_compile_correct_four_view_queries(self):
        requests, plans = build_grounded_v3(self.data["rows"], self.data["demonstrations"])
        for row in self.data["rows"]:
            plan=plans[row["case_id"]]; decisions={}; gold={n["name"]:n for n in row["scm"]["variables"]}
            for node in plan["nodes"]:
                if node["op"] is None:
                    ops=["and","or","xor"] if node["stage"] == "binary" else ["majority3","parity3"]
                    decisions[node["request_id"]]={"choice_index":ops.index(gold[node["name"]]["op"])}
            decisions[plan["target_request"]]={"choice_index":plan["names"].index(row["scm"]["target"])}
            for name,key in plan["active_requests"].items():
                decisions[key]={"choice_index":int(name in row["scm"]["intervene"])}
            self.assertEqual(solve_conditioned(assemble_grounded_v3(plan,decisions))["answer"], row["answer"])

    def test_both_compilers_ignore_hidden_evaluation_metadata(self):
        rows=copy.deepcopy(self.data["rows"])
        for builder in (build_grounded_v3,build_local_v3):
            expected=builder(rows,self.data["demonstrations"])
            poisoned=copy.deepcopy(rows)
            for row in poisoned:
                row.update({"scm":{"hidden":"SECRET"},"answer":"SECRET","view":"SECRET","compatible_worlds":-1})
            self.assertEqual(expected,builder(poisoned,self.data["demonstrations"]))
            self.assertNotIn("SECRET",json.dumps(expected))

    def test_four_view_gate_rejects_missing_view_and_constant_unknown(self):
        records=[{**r,"prediction":"unknown","valid":True,"correct":r["answer"] == "unknown"} for r in self.data["rows"]]
        self.assertTrue(all(v["joint_accuracy"] == 0 for v in metrics_v3(records).values()))
        with self.assertRaises(ValueError):
            metrics_v3(records[1:])
        self.assertTrue(all(v["joint_delta"] == 0 for v in paired_v3(records,records).values()))

    def test_cross_seed_world_exclusion(self):
        seen=set(); first=build_dataset_v3(101,6,seen); second=build_dataset_v3(102,6,seen)
        self.assertEqual(len(seen),60)
        self.assertFalse({r["family_id"] for r in first["rows"]}&{r["family_id"] for r in second["rows"]})
