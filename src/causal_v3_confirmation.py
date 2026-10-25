"""Fresh confirmation inputs, frozen inference guards, and lossless sharding."""
import copy
from functools import lru_cache
import hashlib
import json
from pathlib import Path

from causal_dataset import digest
from causal_dataset_v3 import build_dataset_v3

FRESH_SEEDS = (490001, 490002, 490003)
EXCLUDED_RECIPES = ((20261027, 12), (390001, 48), (390002, 48), (390003, 48))
SPLITS = ("id", "ood_topology", "ood_mechanism", "ood_language")


@lru_cache(maxsize=1)
def confirmation_datasets():
    """Regenerate every consumed world, including demos and cancelled shards."""
    seen = set()
    excluded = [build_dataset_v3(seed, count, seen) for seed, count in EXCLUDED_RECIPES]
    excluded_count = len(seen)
    fresh = {seed: build_dataset_v3(seed, 48, seen) for seed in FRESH_SEEDS}
    return excluded, fresh, excluded_count, len(seen)


def shard_dataset(dataset, index):
    if type(index) is not int or index not in range(4):
        raise ValueError("v3 requires shard index 0..3")
    families = {split: [] for split in SPLITS}
    for row in dataset["rows"]:
        group = families[row["split"]]
        if row["family_id"] not in group:
            group.append(row["family_id"])
    if any(len(group) != 48 for group in families.values()):
        raise ValueError("confirmation requires 48 families per split")
    selected = {family for group in families.values() for family in group[index*12:(index+1)*12]}
    result = copy.deepcopy(dataset)
    result["rows"] = [row for row in result["rows"] if row["family_id"] in selected]
    if len(result["rows"]) != 192:
        raise ValueError("confirmation shard must contain all four views of 48 families")
    full_audit = result["audit"]
    result["audit"] = {
        "full_dataset_audit": full_audit,
        "full_dataset_sha256": full_audit["dataset_sha256"],
        "dataset_sha256": digest(result["rows"]),
        "shard_index": index, "shard_count": 4,
        "cases": len(result["rows"]), "families_per_split": 12,
        "views": full_audit["views"], "fresh_confirmation_seeds": list(FRESH_SEEDS),
        "scope": "computational shard; per-seed gate requires all four shards",
    }
    return result


def prepare_dataset(cfg):
    if "confirmation_plan" not in cfg:
        return build_dataset_v3(cfg["dataset_seed"], cfg["families_per_split"])
    if (cfg.get("control_replay") or cfg.get("development_input") or cfg.get("pilot_only") is not False
            or cfg["families_per_split"] != 48 or cfg["shots"] != 6
            or cfg["dataset_seed"] not in FRESH_SEEDS):
        raise ValueError("invalid fresh confirmation recipe")
    raw = Path(cfg["confirmation_plan"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != cfg["confirmation_plan_sha256"]:
        raise ValueError("confirmation plan checksum mismatch")
    plan = json.loads(raw)
    if plan["seeds"] != list(FRESH_SEEDS):
        raise ValueError("fresh confirmation seed plan changed")
    for path, expected in plan["frozen_inference_file_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"frozen inference changed: {path}")
    for key, value in plan["frozen_config"].items():
        if cfg[key] != value:
            raise ValueError(f"frozen confirmation config changed: {key}")
    excluded, datasets, excluded_count, total_count = confirmation_datasets()
    if excluded_count != plan["excluded_worlds_including_demonstrations"] or total_count != plan["total_worlds_after_future_sampling"]:
        raise ValueError("world exclusion count mismatch")
    for dataset, expected in zip(excluded, plan["excluded_inputs"]):
        if dataset["seed"] != expected["seed"] or dataset["audit"]["dataset_sha256"] != expected["dataset_sha256"]:
            raise ValueError("consumed input exclusion mismatch")
    for seed, dataset in datasets.items():
        expected = plan["datasets"][str(seed)]
        if (dataset["audit"]["dataset_sha256"] != expected["dataset_sha256"]
                or digest(dataset["demonstrations"]) != expected["demonstrations_sha256"]):
            raise ValueError("fresh confirmation dataset mismatch")
    shard = shard_dataset(datasets[cfg["dataset_seed"]], cfg["shard_index"])
    expected = plan["datasets"][str(cfg["dataset_seed"])]["shards"][str(cfg["shard_index"])]["dataset_sha256"]
    if shard["audit"]["dataset_sha256"] != expected:
        raise ValueError("confirmation shard mismatch")
    return shard
