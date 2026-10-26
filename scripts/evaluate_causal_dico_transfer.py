"""Official DiCo-NLI scoring of frozen base and causal-SFT checkpoints."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from causal_choice import decide
from causal_dico_transfer import LABELS, read_csv, audit_and_demonstrations, requests_for, paired_official_bootstrap


def emit(tag, value):
    print(tag + " " + json.dumps(value, sort_keys=True), flush=True)


def load_source(cfg):
    root = Path("external/dico_transfer_source")
    for spec in cfg["source_files"]:
        relative = Path(spec["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe source manifest path")
        path = root / relative
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == spec["sha256"]:
            continue
        url = f"https://raw.githubusercontent.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI/{cfg['data_revision']}/{spec['path']}"
        raw = urllib.request.urlopen(url, timeout=60).read()
        if hashlib.sha256(raw).hexdigest() != spec["sha256"]:
            raise ValueError("organizer source checksum mismatch: " + spec["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return root


def extract_adapter(spec, destination):
    path = Path(spec["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
        raise ValueError("adapter checksum mismatch")
    with zipfile.ZipFile(path) as archive:
        if set(archive.namelist()) != {"adapter_config.json", "adapter_model.safetensors", "README.md"}:
            raise ValueError("adapter-only archive has unexpected members")
        config = json.loads(archive.read("adapter_config.json"))
        if config["base_model_name_or_path"] != spec["base_model_id"]:
            raise ValueError("adapter base model mismatch")
        destination.mkdir(parents=True, exist_ok=True)
        archive.extractall(destination)
    return destination


def main():
    cfg = json.loads(Path("configs/causal_dico_transfer.json").read_text())
    emit("DICO_TRANSFER_CONFIG", {**cfg, "run_id": os.environ.get("ORX_RUN_ID"), "commit": os.environ.get("ORX_COMMIT")})
    source = load_source(cfg)
    train = read_csv(source / cfg["data_paths"]["train"])
    dev = read_csv(source / cfg["data_paths"]["dev"])
    reference = read_csv(source / cfg["data_paths"]["reference"])
    audit, demonstrations, clean_ids = audit_and_demonstrations(train, dev, reference, cfg["seed"])
    if audit != cfg["expected_data_audit"]:
        raise ValueError("DiCo-NLI input or demonstration audit mismatch")
    requests, by_id = requests_for(dev, demonstrations)
    fingerprint = hashlib.sha256(json.dumps(requests, sort_keys=True).encode()).hexdigest()
    if fingerprint != cfg["requests_sha256"]:
        raise ValueError("fixed NLI prompt changed")
    emit("DICO_TRANSFER_DATA_AUDIT", {**audit, "unique_model_requests": len(requests), "prompt_sha256": fingerprint,
         "parameter_training_on_dico": False, "scm_solver_used_for_nli": False})
    output = Path("outputs") / cfg["name"]
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(source.resolve()))
    from evaluation_functions.scorer import score_files
    from evaluation_functions.io import read_reference_file
    from evaluation_functions.validation import validate_reference_records
    from evaluation_functions.metrics import compute_evaluation
    import csv
    import torch
    import transformers
    import peft
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    if not torch.cuda.is_available():
        raise RuntimeError("NLI checkpoint inference requires configured remote GPU")
    torch.manual_seed(cfg["seed"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["model_revision"], padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], revision=cfg["model_revision"],
        torch_dtype=torch.float16, attn_implementation="sdpa", low_cpu_mem_usage=True).to("cuda:0")
    emit("DICO_TRANSFER_ENVIRONMENT", {"python": sys.version.split()[0], "torch": torch.__version__,
         "transformers": transformers.__version__, "peft": peft.__version__, "gpu": torch.cuda.get_device_name(0)})
    gold = validate_reference_records(read_reference_file(source / cfg["data_paths"]["reference"]))
    subsets = {"official_dev": set(gold), "clean_dev": clean_ids}
    references = {}
    for name, ids in subsets.items():
        target = output / f"{name}-reference.csv"
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(reference[0]))
            writer.writeheader()
            writer.writerows(row for row in reference if row["instance_id"] in ids)
        references[name] = target
    all_predictions, scores, costs = {}, {}, {}
    for name in cfg["checkpoints"]:
        if name != "base3b":
            spec = cfg["adapters"][name]
            adapter_path = extract_adapter({**spec, "base_model_id": cfg["model_id"]}, output / name / "adapter")
            if isinstance(model, PeftModel):
                model.load_adapter(adapter_path, adapter_name=name)
                model.set_adapter(name)
            else:
                model = PeftModel.from_pretrained(model, adapter_path, adapter_name=name, is_trainable=False)
        model.eval()
        started = time.monotonic()
        decisions = decide(model, tokenizer, requests, cfg["batch_size"],
            lambda tag, row: emit(tag.replace("PUBLIC_", "DICO_TRANSFER_"), {**row, "checkpoint": name}))
        elapsed = time.monotonic() - started
        predictions = {row["instance_id"]: LABELS[decisions[by_id[row["instance_id"]]]["choice_index"]] for row in dev}
        if set(predictions) != set(gold):
            raise ValueError("prediction coverage mismatch")
        all_predictions[name] = predictions
        for row in dev:
            emit("DICO_TRANSFER_CASE", {"checkpoint": name, "instance_id": row["instance_id"], "pair_id": row["pair_id"],
                 "answer": row["label"], "prediction": predictions[row["instance_id"]],
                 "request_id": by_id[row["instance_id"]], "correct": predictions[row["instance_id"]] == row["label"]})
        scores[name] = {}
        for subset, ids in subsets.items():
            pred_path = output / f"{name}-{subset}-predictions.csv"
            with pred_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["instance_id", "label"])
                writer.writeheader()
                writer.writerows({"instance_id": row["instance_id"], "label": predictions[row["instance_id"]]}
                                 for row in dev if row["instance_id"] in ids)
            scores[name][subset] = score_files(references[subset], pred_path, output_dir=output / name / subset).to_dict()
        costs[name] = {"elapsed_seconds": elapsed, "unique_query_calls": len(requests),
                       "input_tokens": sum(row["input_tokens"] for row in decisions.values()), "output_tokens": len(requests)}
        emit("DICO_TRANSFER_METHOD_SUMMARY", {"checkpoint": name, "scores": scores[name], "cost": costs[name]})
    comparisons = {}
    for name in cfg["checkpoints"][1:]:
        comparisons[name + "_vs_base3b"] = {}
        for subset, ids in subsets.items():
            comparisons[name + "_vs_base3b"][subset] = paired_official_bootstrap(
                {key: gold[key] for key in ids}, {key: all_predictions[name][key] for key in ids},
                {key: all_predictions["base3b"][key] for key in ids}, compute_evaluation, cfg["bootstrap_samples"])
    final = {"config": cfg, "data_audit": audit, "scores": scores, "comparisons": comparisons, "costs": costs,
             "scope": "fixed eight-shot direct transfer on public DiCo-NLI dev; no NLI parameter training, no hidden-test or leaderboard claim",
             "main_result": "official_dev metrics; clean_dev is a secondary source-overlap sensitivity analysis"}
    (output / "summary.json").write_text(json.dumps(final, indent=2))
    emit("DICO_TRANSFER_FINAL_SUMMARY", final)


if __name__ == "__main__":
    main()
