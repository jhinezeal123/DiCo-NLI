import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from causal_reference import load_reference
from causal_grounding import final_interrogative, build_grounded_requests
from causal_dataset_v2 import build_dataset_v2


class ReferenceAndFocusTests(unittest.TestCase):
    def test_target_focus_removes_earlier_intervention_without_extracting_answer(self):
        question = "Force cedar to 0. Then remove it. After restoration, what is the value of lime?"
        self.assertEqual(final_interrogative(question), "After restoration, what is the value of lime?")
        with self.assertRaises(ValueError):
            final_interrogative("No explicit question here.")
        data = build_dataset_v2(20261026, 4)
        before, _ = build_grounded_requests(data["rows"], data["demonstrations"])
        after, _ = build_grounded_requests(data["rows"], data["demonstrations"], True)
        for stage in ("op", "active"):
            self.assertEqual([r for r in before if r["stage"] == stage], [r for r in after if r["stage"] == stage])
        mutated = copy.deepcopy(data["rows"])
        for row in mutated:
            row.update({"scm": {"hidden": True}, "answer": "hidden", "view": "hidden"})
        self.assertEqual(build_grounded_requests(data["rows"], data["demonstrations"], True),
                         build_grounded_requests(mutated, data["demonstrations"], True))

    def test_reuse_rejects_wrong_hash_cases_and_duplicate_predictions(self):
        row = {"case_id": "one", "family_id": "f", "split": "id", "view": "factual",
               "answer": "1", "changed": True, "method": "direct", "prediction": "1"}
        dataset = {"audit": {"dataset_sha256": "dataset"}, "rows": [row]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.jsonl"
            path.write_text(json.dumps(row) + "\n")
            cfg = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                   "dataset_sha256": "dataset", "run_id": "prior", "commit": "abc"}
            self.assertEqual(load_reference(cfg, dataset)["direct"][0]["generation_reused_from_run"], "prior")
            with self.assertRaises(ValueError):
                load_reference({**cfg, "dataset_sha256": "different"}, dataset)
            with self.assertRaises(ValueError):
                load_reference({**cfg, "sha256": "wrong"}, dataset)
            path.write_text((json.dumps(row) + "\n") * 2)
            cfg["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):
                load_reference(cfg, dataset)
