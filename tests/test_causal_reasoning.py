"""Analytic regression cases for same-world semantics and evaluation isolation."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from causal_scm import canonical, normalize_scm, parse_scm, solve
from causal_dataset import build_dataset, TOPOLOGIES
from causal_protocol import covered_demonstrations, decode_output, messages, metrics, paired_bootstrap
from causal_decomposition import build_requests, demonstrations as local_demonstrations, split_document
from causal_dataset_v2 import build_dataset_v2, family_signature


def two_node(observed=None, intervene=None):
    return {"variables": [{"name": "x", "op": "root", "parents": []},
                           {"name": "y", "op": "copy", "parents": ["x"]}],
            "observed": {"x": 1, "y": 0} if observed is None else observed,
            "intervene": {} if intervene is None else intervene, "target": "y"}


class CausalSemanticsTest(unittest.TestCase):
    def test_singleton_boolean_reductions_preserve_counterfactuals(self):
        for op, reduced in [("and", "copy"), ("or", "copy"), ("xor", "copy"),
                            ("nand", "not"), ("nor", "not")]:
            for x in (0, 1):
                for y in (0, 1):
                    scm = two_node(observed={"x": x, "y": y}, intervene={"x": 1-x})
                    scm["variables"][1]["op"] = op
                    reference = copy.deepcopy(scm)
                    reference["variables"][1]["op"] = reduced
                    normalized = normalize_scm(scm, reduce_unary=True)
                    self.assertEqual(solve(normalized), solve(reference))
                    with self.assertRaises(ValueError):
                        normalize_scm(scm)

    def test_normalization_keeps_binary_xor_and_rejects_missing_names(self):
        scm = {"variables": [{"name": "a", "op": "root", "parents": []},
                              {"name": "b", "op": "root", "parents": []},
                              {"name": "c", "op": "xor", "parents": ["a", "b"]}],
               "observed": {"a": 0, "b": 1, "c": 1}, "intervene": {}, "target": "c"}
        self.assertEqual(normalize_scm(scm, reduce_unary=True)["variables"][2]["op"], "xor")
        scm["variables"][2]["parents"] = ["missing"]
        with self.assertRaises(ValueError):
            normalize_scm(scm, reduce_unary=True)

    def test_abduction_preserves_noise_in_counterfactual(self):
        # X=1,Y=0 implies Ux=1,Uy=1; setting X=0 makes Y=1.
        result = solve(two_node(intervene={"x": 0}))
        self.assertEqual(result, {"answer": "1", "compatible_worlds": 1, "outcomes": [1]})

    def test_observed_intervention_target_not_conditioned_in_changed_world(self):
        # The old observation Y=0 must not override a surgical do(Y=1).
        self.assertEqual(solve(two_node(intervene={"y": 1}))["answer"], "1")

    def test_partial_observation_does_not_invent_a_unique_counterfactual(self):
        result = solve(two_node(observed={"y": 1}, intervene={"x": 0}))
        self.assertEqual(result["answer"], "unknown")
        self.assertEqual(result["compatible_worlds"], 2)

    def test_restoring_equation_restores_factual_value(self):
        scm = two_node(intervene={"x": 0})
        self.assertEqual(solve(scm)["answer"], "1")
        scm["intervene"] = {}
        self.assertEqual(solve(scm)["answer"], "0")

    def test_nand_and_post_mechanism_noise(self):
        scm = {"variables": [{"name": "a", "op": "root", "parents": []},
                              {"name": "b", "op": "root", "parents": []},
                              {"name": "c", "op": "nand", "parents": ["a", "b"]}],
               "observed": {"a": 1, "b": 1, "c": 1}, "intervene": {"a": 0}, "target": "c"}
        # NAND(1,1)=0 and C=1 imply Uc=1; NAND(0,1) XOR 1 = 0.
        self.assertEqual(solve(scm)["answer"], "0")

    def test_cycle_unknown_parent_and_duplicate_names_are_rejected(self):
        for nodes in [
            [{"name": "x", "op": "copy", "parents": ["x"]}],
            [{"name": "x", "op": "copy", "parents": ["missing"]}],
            [{"name": "x", "op": "root", "parents": []}] * 2,
        ]:
            with self.assertRaises(ValueError):
                normalize_scm({"variables": nodes, "observed": {}, "intervene": {}, "target": "x"})

    def test_safe_parser_accepts_code_fence_and_no_code_execution(self):
        scm = two_node()
        self.assertEqual(canonical(parse_scm("```json\n" + json.dumps(scm) + "\n```")), canonical(scm))
        with self.assertRaises(ValueError):
            parse_scm("__import__('os').system('echo unsafe')")

    def test_gold_schema_only_affects_diagnostics_not_prediction(self):
        scm = two_node(intervene={"x": 0})
        other_gold = two_node()
        first = decode_output(json.dumps(scm), "parser_solver", scm)
        second = decode_output(json.dumps(scm), "parser_solver", other_gold)
        self.assertEqual(first["prediction"], second["prediction"])
        self.assertEqual(first["prediction"], "1")
        self.assertTrue(first["schema_exact"])
        self.assertFalse(second["schema_exact"])


class DatasetProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = build_dataset(20261025, 4)

    def test_grouping_strata_and_holdouts(self):
        rows = self.dataset["rows"]
        families = {r["family_id"] for r in rows}
        demos = {d["family_id"] for d in self.dataset["demonstrations"]}
        self.assertFalse(families & demos)
        self.assertEqual(len(rows), 48)
        for split, audit in self.dataset["audit"]["splits"].items():
            self.assertEqual(audit["changed_families"], 2)
            self.assertEqual(audit["labels"], {"0": 6, "1": 6})
            group = [r for r in rows if r["split"] == split]
            if split == "ood_topology":
                self.assertTrue(all(r["topology"] in ("collider", "skip") for r in group))
            else:
                self.assertTrue(all(r["topology"] in ("chain", "fork", "diamond") for r in group))
            for row in group:
                ops = {v["op"] for v in row["scm"]["variables"]}
                self.assertEqual(bool(ops & {"nand", "nor"}), split == "ood_mechanism")

    def test_prompt_final_user_input_has_no_gold_injected(self):
        row = self.dataset["rows"][0]
        altered = copy.deepcopy(row)
        altered["answer"] = "hidden-gold-sentinel"
        altered["scm"]["target"] = "hidden-scm-sentinel"
        for method in ("direct", "structured", "parser_solver"):
            prompt = messages(altered, method, self.dataset["demonstrations"])
            self.assertEqual(prompt[-1]["content"], row["text"])
            self.assertNotIn("sentinel", json.dumps(prompt))

    def test_four_demonstrations_balance_answers_and_change(self):
        demos = self.dataset["demonstrations"]
        self.assertEqual(sorted(d["answer"] for d in demos), ["0", "0", "1", "1"])
        self.assertEqual(sum(d["changed"] for d in demos), 2)
        for method in ("direct", "structured", "parser_solver"):
            prompt = messages(self.dataset["rows"][0], method, demos, shots=4)
            self.assertEqual(sum(m["role"] == "assistant" for m in prompt), 4)

    def test_constant_answer_can_be_consistent_and_still_fail_joint(self):
        records = [{**r, "prediction": "0", "valid": True, "correct": r["answer"] == "0"}
                   for r in self.dataset["rows"]]
        for result in metrics(records).values():
            self.assertEqual(result["dcc"], 1.0)
            self.assertEqual(result["joint_accuracy"], .25)
        differences = paired_bootstrap(records, records, repetitions=100)
        self.assertTrue(all(v["ci95"] == [0.0, 0.0] for v in differences.values()))

    def test_invalid_output_is_never_silently_replaced_with_gold(self):
        row = self.dataset["rows"][0]
        decoded = decode_output("I cannot parse the model", "parser_solver", row["scm"])
        self.assertEqual(decoded["prediction"], "invalid")
        self.assertFalse(decoded["valid"])

    def test_local_parser_reuses_only_identical_public_rules(self):
        rows = self.dataset["rows"][:3]
        requests, cases = build_requests(rows, self.dataset["demonstrations"])
        self.assertEqual(sum(r["stage"] == "rule" for r in requests), 5)
        self.assertEqual(sum(r["stage"] == "query" for r in requests), 3)
        self.assertEqual(cases[rows[0]["case_id"]]["rules"], cases[rows[1]["case_id"]]["rules"])
        self.assertNotEqual(cases[rows[0]["case_id"]]["query"], cases[rows[1]["case_id"]]["query"])

    def test_local_parser_requests_cannot_access_evaluation_gold(self):
        rows = copy.deepcopy(self.dataset["rows"][:3])
        expected = build_requests(rows, self.dataset["demonstrations"])
        for row in rows:
            row["answer"] = "hidden-answer-sentinel"
            row["scm"] = {"hidden": "gold-schema-sentinel"}
            row["view"] = "hidden-view-sentinel"
        actual = build_requests(rows, self.dataset["demonstrations"])
        self.assertEqual(actual, expected)
        self.assertNotIn("sentinel", json.dumps(actual))


class SixVariableValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = build_dataset_v2(20261026, 4)

    def test_true_operator_and_topology_holdouts(self):
        data = build_dataset_v2(20261026, 4)
        for row in data["rows"]:
            nodes = row["scm"]["variables"]
            self.assertEqual(len(nodes), 6)
            self.assertEqual(sum(len(n["parents"]) == 2 for n in nodes) >= 2,
                             row["split"] == "ood_topology")
            self.assertEqual(any(n["op"] == "xor" for n in nodes), row["split"] == "ood_mechanism")
        demo_ops = {n["op"] for r in data["demonstrations"] for n in r["scm"]["variables"]}
        self.assertTrue({"and", "or"} <= demo_ops)
        self.assertNotIn("xor", demo_ops)
        for audit in data["audit"]["splits"].values():
            self.assertEqual(audit["labels"], {"0": 6, "1": 6})
            self.assertEqual(audit["changed_families"], 2)

    def test_family_signatures_group_noise_equivalent_mechanisms(self):
        parents = [[], [0], [1]]
        first = family_signature(parents, ["root", "copy", "not"], [1, 0, 1], 0, 0)
        second = family_signature(parents, ["root", "not", "copy"], [1, 0, 1], 0, 0)
        self.assertEqual(first, second)

    def test_shared_exclusion_set_prevents_cross_seed_overlap(self):
        seen = set()
        first = build_dataset_v2(20261026, 4, seen)
        second = build_dataset_v2(271828, 4, seen)
        self.assertEqual(len(seen), 40)
        self.assertFalse({r["family_id"] for r in first["rows"]} & {r["family_id"] for r in second["rows"]})

    def test_control_and_local_parser_share_query_demonstrations(self):
        views = ["factual", "intervention", "restored", "intervention"]
        examples = covered_demonstrations(self.dataset["demonstrations"], views)
        self.assertEqual(sorted(r["answer"] for r in examples), ["0", "0", "1", "1"])
        _, local_queries = local_demonstrations(self.dataset["demonstrations"])
        for row, (query, answer) in zip(examples, local_queries):
            self.assertEqual(split_document(row["text"])[1], query)
            self.assertEqual({k: row["scm"][k] for k in ("observed", "intervene", "target")}, answer)


if __name__ == "__main__":
    unittest.main()
