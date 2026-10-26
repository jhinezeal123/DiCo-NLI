"""Transfer audit, public prompt boundaries, pairing and adapter validation."""
import copy
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from causal_dico_transfer import LABELS, audit_and_demonstrations, requests_for, paired_official_bootstrap
from evaluate_causal_dico_transfer import extract_adapter


def fixture():
    train = [{"instance_id": f"t{i}", "pair_id": f"train{i}", "text1": f"training left {i}",
              "text2": f"training right {i}", "label": LABELS[i % 4]} for i in range(16)]
    dev = [{"instance_id": "d0", "pair_id": "forward", "text1": "all birds", "text2": "some birds", "label": LABELS[1]},
           {"instance_id": "d1", "pair_id": "forward", "text1": "some birds", "text2": "all birds", "label": LABELS[2]},
           {"instance_id": "d2", "pair_id": "equivalent", "text1": "large car", "text2": "big car", "label": LABELS[0]},
           {"instance_id": "d3", "pair_id": "equivalent", "text1": "big car", "text2": "large car", "label": LABELS[0]},
           {"instance_id": "d4", "pair_id": "negative", "text1": "stone", "text2": "cloud", "label": LABELS[3]}]
    references = [{**row, "reverse_pair_id": f"d{index ^ 1}" if index < 4 else ""} for index, row in enumerate(dev)]
    return train, dev, references


class DicoTransferTests(unittest.TestCase):
    def test_overlap_filter_removes_whole_source_pairs_and_demos_stay_disjoint(self):
        train, dev, ref = fixture()
        train[0].update(text1=dev[0]["text1"], text2=dev[0]["text2"])
        audit, demos, clean = audit_and_demonstrations(train, dev, ref)
        self.assertEqual(audit["train_dev_unordered_text_overlap_rows"], 2)
        self.assertEqual(clean, {"d2", "d3", "d4"})
        self.assertEqual(len(demos), 8)
        self.assertEqual(audit["demonstration_labels"], {label: 2 for label in LABELS})
        self.assertNotIn(train[0], demos)
        self.assertEqual(audit_and_demonstrations(train, dev, ref), (audit, demos, clean))

    def test_prompt_never_reads_dev_labels_or_reference_metadata(self):
        train, dev, ref = fixture()
        _, demos, _ = audit_and_demonstrations(train, dev, ref)
        original = requests_for(dev, demos)
        poisoned = copy.deepcopy(dev)
        for row in poisoned:
            row.update(label="PRIVATE_GOLD", pair_id="PRIVATE_PAIR", reverse_pair_id="PRIVATE_REVERSE")
        self.assertEqual(original, requests_for(poisoned, demos))
        self.assertNotIn("PRIVATE", json.dumps(original))

    def test_identical_texts_share_one_inference_but_cover_all_instance_ids(self):
        train, dev, ref = fixture()
        _, demos, _ = audit_and_demonstrations(train, dev, ref)
        dev.append({**dev[-1], "instance_id": "d5", "pair_id": "another_negative"})
        requests, by_id = requests_for(dev, demos)
        self.assertEqual(len(requests), 5)
        self.assertEqual(len(by_id), 6)
        self.assertEqual(by_id["d4"], by_id["d5"])

    def test_id_leakage_or_mismatched_gold_is_rejected(self):
        for mode in ("id", "pair", "label"):
            train, dev, ref = fixture()
            if mode == "id":
                train[0]["instance_id"] = dev[0]["instance_id"]
            elif mode == "pair":
                train[0]["pair_id"] = dev[0]["pair_id"]
            else:
                ref[0]["label"] = LABELS[0]
            with self.assertRaises(ValueError):
                audit_and_demonstrations(train, dev, ref)

    def test_bootstrap_keeps_reciprocal_pairs_and_equal_predictions_have_zero_delta(self):
        @dataclass(frozen=True)
        class Gold:
            instance_id: str
            pair_id: str
            reverse_pair_id: str | None
            label: str
        gold = {"a": Gold("a", "p", "b", LABELS[1]), "b": Gold("b", "p", "a", LABELS[2]),
                "c": Gold("c", "n", None, LABELS[3])}
        prediction = {key: value.label for key, value in gold.items()}
        calls = []
        class Report:
            def to_dict(self):
                return {metric: 1.0 for metric in ("weighted_f1", "soft_cons", "hard_cons")}
        def official_callback(rows, labels):
            self.assertEqual(set(rows), set(labels))
            for row in rows.values():
                if row.reverse_pair_id:
                    self.assertEqual(rows[row.reverse_pair_id].reverse_pair_id, row.instance_id)
                    self.assertEqual(rows[row.reverse_pair_id].pair_id, row.pair_id)
            calls.append(1)
            return Report()
        result = paired_official_bootstrap(gold, prediction, prediction, official_callback, samples=10)
        self.assertEqual(len(calls), 22)
        self.assertTrue(all(value["delta"] == 0 and value["ci95"] == [0, 0] for value in result.values()))
        with self.assertRaises(ValueError):
            paired_official_bootstrap(gold, {}, prediction, official_callback, samples=10)

    def test_adapter_checksum_and_member_allowlist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for malicious in (False, True):
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, "w") as archive:
                    archive.writestr("adapter_config.json", json.dumps({"base_model_name_or_path": "base-model"}))
                    archive.writestr("adapter_model.safetensors", "unit bytes")
                    archive.writestr("README.md", "unit adapter")
                    if malicious:
                        archive.writestr("../outside.txt", "bad")
                raw = buffer.getvalue()
                path = root / "adapter.zip"
                path.write_bytes(raw)
                spec = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "base_model_id": "base-model"}
                if malicious:
                    with self.assertRaises(ValueError):
                        extract_adapter(spec, root / "unpacked")
                else:
                    extract_adapter(spec, root / "unpacked")
                    self.assertTrue((root / "unpacked/adapter_config.json").exists())
                with self.assertRaises(ValueError):
                    extract_adapter({**spec, "sha256": "0" * 64}, root / "unpacked")
                self.assertFalse((root / "outside.txt").exists())


if __name__ == "__main__":
    unittest.main()
