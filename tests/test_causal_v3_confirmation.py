"""Leakage, lossless partitioning, and frozen-protocol checks for fresh v3."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from causal_dataset import digest
from causal_v3_confirmation import confirmation_datasets, prepare_dataset, shard_dataset, FRESH_SEEDS


class FreshConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = json.loads(Path("configs/causal_v3.json").read_text())
        cls.excluded, cls.datasets, cls.excluded_count, cls.total_count = confirmation_datasets()

    def test_all_consumed_worlds_and_demonstrations_are_disjoint_from_fresh_worlds(self):
        seen = set()
        for dataset in self.excluded + list(self.datasets.values()):
            worlds = {r["family_id"] for r in dataset["rows"] + dataset["demonstrations"]}
            self.assertFalse(seen & worlds)
            seen.update(worlds)
        self.assertEqual(len(seen), 1242)
        self.assertEqual(self.excluded_count, 648)
        self.assertEqual(self.total_count, 1242)

    def test_four_shards_reassemble_every_case_and_all_views_without_overlap(self):
        for dataset in self.datasets.values():
            merged, cases = [], set()
            for index in range(4):
                part = shard_dataset(dataset, index)
                ids = {r["case_id"] for r in part["rows"]}
                self.assertFalse(cases & ids)
                cases.update(ids)
                merged.extend(part["rows"])
                self.assertEqual(part["demonstrations"], dataset["demonstrations"])
                self.assertEqual(len(ids), 192)
                families = {r["family_id"] for r in part["rows"]}
                self.assertEqual(len(families), 48)
                for family in families:
                    self.assertEqual({r["view"] for r in part["rows"] if r["family_id"] == family},
                                     {"factual", "intervention", "partial_release", "restored"})
            self.assertEqual(sorted(merged, key=lambda r: r["case_id"]),
                             sorted(dataset["rows"], key=lambda r: r["case_id"]))
            self.assertEqual(len(cases), 768)

    def test_real_lock_checks_every_fresh_dataset_and_shard(self):
        for seed in FRESH_SEEDS:
            for index in range(4):
                cfg = {**self.cfg, "dataset_seed": seed, "shard_index": index}
                part = prepare_dataset(cfg)
                self.assertEqual(part["audit"]["dataset_sha256"], digest(part["rows"]))
                self.assertEqual(part["audit"]["full_dataset_sha256"], self.datasets[seed]["audit"]["dataset_sha256"])

    def test_rejects_replay_opened_data_config_drift_and_poisoned_lock(self):
        for change in ({"control_replay": {"bad": True}}, {"development_input": {"bad": True}},
                       {"dataset_seed": 390001}, {"pilot_only": True}, {"shots": 5},
                       {"batch_size": 8}, {"model_revision": "unfrozen"}, {"shard_index": 4}):
            with self.subTest(change=change), self.assertRaises((ValueError, KeyError)):
                prepare_dataset({**self.cfg, **change})
        with self.assertRaisesRegex(ValueError, "checksum"):
            prepare_dataset({**self.cfg, "confirmation_plan_sha256": "0" * 64})
        plan = json.loads(Path(self.cfg["confirmation_plan"]).read_text())
        plan = copy.deepcopy(plan)
        plan["frozen_inference_file_sha256"]["src/causal_v3_compiler.py"] = "0" * 64
        raw = json.dumps(plan).encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poisoned.json"
            path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "frozen inference changed"):
                prepare_dataset({**self.cfg, "confirmation_plan": str(path),
                                 "confirmation_plan_sha256": hashlib.sha256(raw).hexdigest()})


if __name__ == "__main__":
    unittest.main()
