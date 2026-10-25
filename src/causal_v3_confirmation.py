"""Shared world exclusions and lossless computational sharding for v3."""
import copy
import hashlib
import json
from pathlib import Path

from causal_dataset import digest
from causal_dataset_v3 import build_dataset_v3, RESERVED_SEEDS


def confirmation_datasets():
    seen=set()
    development=build_dataset_v3(20261027,12,seen)
    datasets={seed:build_dataset_v3(seed,48,seen) for seed in RESERVED_SEEDS}
    return development,datasets


def shard_dataset(dataset,index):
    if index not in range(4): raise ValueError("v3 requires shard index0..3")
    families={}
    for row in dataset["rows"]:
        group=families.setdefault(row["split"],[])
        if row["family_id"] not in group: group.append(row["family_id"])
    if any(len(group)!=48 for group in families.values()):
        raise ValueError("v3 confirmation requires48 families per split")
    selected={family for group in families.values() for family in group[index*12:(index+1)*12]}
    result=copy.deepcopy(dataset)
    result["rows"]=[row for row in result["rows"] if row["family_id"] in selected]
    full_audit=result["audit"]
    result["audit"]={"full_dataset_audit":full_audit,"full_dataset_sha256":full_audit["dataset_sha256"],
        "dataset_sha256":digest(result["rows"]),"shard_index":index,"shard_count":4,
        "cases":len(result["rows"]),"families_per_split":12,"views":full_audit["views"],
        "scope":"computational shard; per-seed gate requires all four shards"}
    return result


def prepare_dataset(cfg):
    if "confirmation_plan" not in cfg:
        return build_dataset_v3(cfg["dataset_seed"],cfg["families_per_split"])
    if cfg.get("control_replay") or cfg["families_per_split"]!=48 or cfg["shots"]!=6:
        raise ValueError("invalid confirmation recipe")
    raw=Path(cfg["confirmation_plan"]).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=cfg["confirmation_plan_sha256"]:
        raise ValueError("v3 confirmation plan checksum mismatch")
    plan=json.loads(raw)
    for path,expected in plan["frozen_inference_file_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
            raise ValueError(f"frozen inference changed: {path}")
    for key in ("model_id","model_revision","batch_size","max_new_tokens","shots","methods","primary_method"):
        if cfg[key]!=plan["frozen_config"][key]:
            raise ValueError(f"frozen confirmation config changed: {key}")
    development,datasets=confirmation_datasets()
    if development["audit"]["dataset_sha256"]!=plan["development_dataset_sha256"]:
        raise ValueError("development exclusion mismatch")
    for seed,dataset in datasets.items():
        if dataset["audit"]["dataset_sha256"]!=plan["datasets"][str(seed)]["full_dataset_sha256"]:
            raise ValueError("confirmation dataset mismatch")
    shard=shard_dataset(datasets[cfg["dataset_seed"]],cfg["shard_index"])
    expected=plan["datasets"][str(cfg["dataset_seed"])]["shards"][str(cfg["shard_index"])]["dataset_sha256"]
    if shard["audit"]["dataset_sha256"]!=expected: raise ValueError("shard mismatch")
    return shard
