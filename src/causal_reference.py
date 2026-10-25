"""Immutable matched prediction reuse, exclusively for scoring/comparison."""
import hashlib
import json
from pathlib import Path


def load_reference(config, dataset):
    if config["dataset_sha256"] != dataset["audit"]["dataset_sha256"]:
        raise ValueError("reference dataset fingerprint differs")
    content = Path(config["path"]).read_bytes()
    if hashlib.sha256(content).hexdigest() != config["sha256"]:
        raise ValueError("reference predictions checksum differs")
    expected = {r["case_id"]: r for r in dataset["rows"]}
    records = {}
    for line in content.decode().splitlines():
        row = json.loads(line)
        gold = expected[row["case_id"]]
        if any(row[k] != gold[k] for k in ("family_id", "split", "view", "answer", "changed")):
            raise ValueError("reference case does not match current dataset")
        row = {**row, "generation_reused_from_run": config["run_id"],
               "generation_reused_from_commit": config["commit"], "new_lm_calls": 0}
        records.setdefault(row["method"], []).append(row)
    for method, rows in records.items():
        ids = [r["case_id"] for r in rows]
        if len(ids) != len(set(ids)) or set(ids) != set(expected):
            raise ValueError("reference method must cover each evaluation case exactly once: " + method)
    return records
