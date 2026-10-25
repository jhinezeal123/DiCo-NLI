"""Tracked causal-reasoning evaluation; called exclusively by the fixed run.sh."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from causal_dataset import build_dataset
from causal_protocol import decode_output, messages, metrics, paired_bootstrap
from causal_scm import solve


def emit(kind, value):
    print(kind + " " + json.dumps(value, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    output = Path("outputs") / cfg["name"]
    output.mkdir(parents=True, exist_ok=True)
    emit("CAUSAL_CONFIG", {**cfg, "commit": os.environ.get("ORX_COMMIT"), "run_id": os.environ.get("ORX_RUN_ID")})
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    random.seed(cfg["dataset_seed"])
    torch.manual_seed(cfg["dataset_seed"])
    if not torch.cuda.is_available():
        raise RuntimeError("GPU evaluation requested but CUDA is unavailable")
    emit("ENVIRONMENT", {"python": sys.version.split()[0], "torch": torch.__version__,
                          "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(0),
                          "gpu_count": torch.cuda.device_count(),
                          "packages": {p: importlib.metadata.version(p) for p in ("huggingface-hub", "tokenizers")}})
    if cfg.get("generator_version") == "v2":
        from causal_dataset_v2 import build_dataset_v2
        dataset = build_dataset_v2(cfg["dataset_seed"], cfg["families_per_split"])
    else:
        dataset = build_dataset(cfg["dataset_seed"], cfg["families_per_split"])
    emit("DATASET_AUDIT", dataset["audit"])
    (output / "dataset.json").write_text(json.dumps(dataset, indent=2))
    oracle = [solve(r["scm"])["answer"] == r["answer"] for r in dataset["rows"]]
    assert all(oracle)
    emit("ORACLE_CONTROL", {"accuracy": 1.0, "cases": len(oracle),
                            "note": "By-construction oracle agreement, not LLM performance; analytic tests check semantics."})
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["model_revision"], padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], revision=cfg["model_revision"],
                                                torch_dtype=torch.float16, attn_implementation="sdpa").to("cuda:0").eval()
    emit("MODEL_READY", {"model_id": cfg["model_id"], "revision": cfg["model_revision"],
                         "parameters": model.num_parameters(), "dtype": str(model.dtype)})
    all_records = {}
    for method in cfg["methods"]:
        if method == "decomposed_solver":
            continue
        records, started = [], time.monotonic()
        for offset in range(0, len(dataset["rows"]), cfg["batch_size"]):
            batch = dataset["rows"][offset:offset + cfg["batch_size"]]
            prompts = [tokenizer.apply_chat_template(messages(r, method, dataset["demonstrations"], cfg["shots"],
                                                           cfg.get("demonstration_query_views")),
                       tokenize=False, add_generation_prompt=True) for r in batch]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False).to(model.device)
            torch.cuda.synchronize()
            batch_started = time.monotonic()
            with torch.inference_mode():
                generated = model.generate(**encoded, do_sample=False, max_new_tokens=cfg["max_new_tokens"],
                                           pad_token_id=tokenizer.pad_token_id,
                                           eos_token_id=tokenizer.eos_token_id)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - batch_started
            continuation = generated[:, encoded["input_ids"].shape[1]:]
            texts = tokenizer.batch_decode(continuation, skip_special_tokens=True)
            for i, (row, text) in enumerate(zip(batch, texts)):
                decoded = decode_output(text, method, row["scm"] if method == "parser_solver" else None)
                record = {k: row[k] for k in ("case_id", "family_id", "split", "view", "answer", "changed")}
                record.update({**decoded, "method": method, "correct": decoded["prediction"] == row["answer"],
                               "raw_output": text, "input_tokens": int(encoded["attention_mask"][i].sum()),
                               "output_tokens": int((continuation[i] != tokenizer.pad_token_id).sum()),
                               "batch_seconds_per_case": elapsed / len(batch),
                               "prompt_sha256": hashlib.sha256(prompts[i].encode()).hexdigest()})
                records.append(record)
                emit("CAUSAL_CASE", record)
            emit("CAUSAL_PROGRESS", {"method": method, "done": len(records), "total": len(dataset["rows"]),
                                     "seconds": round(time.monotonic() - started, 2)})
        all_records[method] = records
        summary = {"method": method, "metrics": metrics(records),
                   "output_tokens": sum(r["output_tokens"] for r in records),
                   "input_tokens": sum(r["input_tokens"] for r in records),
                   "elapsed_seconds": time.monotonic() - started}
        emit("CAUSAL_METHOD_SUMMARY", summary)
        (output / f"{method}-predictions.json").write_text(json.dumps(records, indent=2))
    if "decomposed_solver" in cfg["methods"]:
        from causal_decomposition import run_decomposed
        records, usage = run_decomposed(model, tokenizer, dataset, cfg, emit)
        all_records["decomposed_solver"] = records
        emit("CAUSAL_METHOD_SUMMARY", {"method": "decomposed_solver", "metrics": metrics(records), **usage})
        (output / "decomposed_solver-predictions.json").write_text(json.dumps(records, indent=2))
    normalized_methods = []
    if cfg.get("unary_normalization"):
        case_lookup = {row["case_id"]: row for row in dataset["rows"]}
        for original_method in ("parser_solver", "decomposed_solver"):
            if original_method not in all_records:
                continue
            new_method = original_method + "_normalized"
            normalized = []
            excluded = {"prediction", "valid", "error", "schema_exact", "fields_wrong",
                        "parsed_scm", "compatible_worlds", "correct"}
            for original in all_records[original_method]:
                case = case_lookup[original["case_id"]]
                decoded = decode_output(original["raw_output"], "parser_solver", case["scm"], reduce_unary=True)
                record = {**{k: v for k, v in original.items() if k not in excluded}, **decoded,
                          "method": new_method, "correct": decoded["prediction"] == case["answer"],
                          "generation_reused_from": original_method}
                normalized.append(record)
                emit("CAUSAL_CASE", record)
            all_records[new_method] = normalized
            normalized_methods.append(new_method)
            emit("CAUSAL_METHOD_SUMMARY", {"method": new_method, "metrics": metrics(normalized),
                 "new_lm_calls": 0, "generation_reused_from": original_method,
                 "source_generation_input_tokens": sum(r["input_tokens"] for r in normalized),
                 "source_generation_output_tokens": sum(r["output_tokens"] for r in normalized)})
    comparisons = {method: paired_bootstrap(all_records["parser_solver"], all_records[method])
                   for method in cfg["methods"] if method != "parser_solver"}
    final = {"config": cfg, "audit": dataset["audit"], "metrics": {m: metrics(v) for m, v in all_records.items()},
             "parser_solver_vs": comparisons, "oracle_agreement": 1.0,
             "pilot_only": cfg["pilot_only"], "gate": {"joint_minimum": .9, "delta_minimum": .05,
                 "requires_positive_ci_lower": True, "confirm_dataset_seeds": 3,
                 "status": "not_evaluated_on_small_pilot" if cfg["pilot_only"] else "requires_three_seed_confirmation"}}
    if "decomposed_solver" in all_records:
        final["decomposed_solver_vs"] = {m: paired_bootstrap(all_records["decomposed_solver"], rows)
                                         for m, rows in all_records.items() if m != "decomposed_solver"}
    if normalized_methods:
        final["primary_method"] = cfg["primary_method"]
        final["normalization_vs"] = {candidate: {m: paired_bootstrap(all_records[candidate], rows)
                                                for m, rows in all_records.items() if m not in normalized_methods}
                                     for candidate in normalized_methods}
    (output / "summary.json").write_text(json.dumps(final, indent=2))
    emit("CAUSAL_FINAL_SUMMARY", final)


if __name__ == "__main__":
    main()
