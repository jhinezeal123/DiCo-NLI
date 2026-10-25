"""Validate frozen learned operator decisions; never replace them with labels."""
import hashlib
import json
from pathlib import Path


def load_operator_replay(cfg, dataset, requests):
    spec = cfg.get("operator_replay")
    if not spec:
        return {}
    if cfg.get("pilot_only") is not True:
        raise ValueError("operator replay is restricted to opened development")
    for key in ("model_id", "model_revision", "shots", "batch_size"):
        if spec[key] != cfg[key]:
            raise ValueError(f"operator replay config mismatch: {key}")
    if spec["dataset_sha256"] != dataset["audit"]["dataset_sha256"]:
        raise ValueError("operator replay dataset mismatch")
    raw = Path(spec["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec["sha256"]:
        raise ValueError("operator replay checksum mismatch")
    expected = {r["request_id"]: r for r in requests}
    found = {}
    for line in raw.decode().splitlines():
        row = json.loads(line)
        key = row["request_id"]
        if key not in expected or key in found:
            raise ValueError("unexpected or duplicate operator decision")
        request = expected[key]
        if row["stage"] != request["stage"] or not 0 <= row["choice_index"] < request["choice_count"]:
            raise ValueError("invalid operator decision")
        found[key] = {**row, "source_run": spec["source_run"], "new_lm_calls": 0}
    if set(found) != set(expected):
        raise ValueError("incomplete operator replay")
    return found
