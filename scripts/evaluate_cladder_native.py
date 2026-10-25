"""Tracked public-text CLadder pilot. Hidden test annotations never enter prompts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from audit_cladder_native import load_source, emit
from causal_public_protocol import build_public_split, native_messages, decode_native, public_metrics, public_bootstrap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg = json.loads(Path(parser.parse_args().config).read_text())
    emit("PUBLIC_CONFIG", {**cfg, "commit": os.environ.get("ORX_COMMIT"), "run_id": os.environ.get("ORX_RUN_ID")})
    questions, models = [load_source(f["name"], cfg["revision"], f["sha256"]) for f in cfg["files"]]
    dataset = build_public_split(questions, models, cfg["models_per_graph"])
    emit("PUBLIC_DATASET_AUDIT", dataset["audit"])
    output = Path("outputs") / cfg["name"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset.json").write_text(json.dumps(dataset, indent=2))
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(1729)
    if not torch.cuda.is_available():
        raise RuntimeError("native LM evaluation requires GPU")
    emit("PUBLIC_ENVIRONMENT", {"torch": torch.__version__, "transformers": transformers.__version__,
                                "python": sys.version.split()[0], "gpu_count": torch.cuda.device_count(),
                                "gpu": torch.cuda.get_device_name(0)})
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["model_revision"], padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], revision=cfg["model_revision"],
               torch_dtype=torch.float16, attn_implementation="sdpa", device_map="balanced", low_cpu_mem_usage=True,
               max_memory={i: "13GiB" for i in range(torch.cuda.device_count())})
    model.eval()
    all_records = {}
    for method in cfg["methods"]:
        records, started = [], time.monotonic()
        for offset in range(0, len(dataset["rows"]), cfg["batch_size"]):
            batch = dataset["rows"][offset:offset + cfg["batch_size"]]
            prompts = [tokenizer.apply_chat_template(native_messages(r, method, dataset["demonstrations"]),
                        tokenize=False, add_generation_prompt=True) for r in batch]
            encoded = tokenizer(prompts, padding=True, truncation=False, return_tensors="pt").to(model.device)
            for device in range(torch.cuda.device_count()):
                torch.cuda.synchronize(device)
            batch_started = time.monotonic()
            with torch.inference_mode():
                generated = model.generate(**encoded, do_sample=False, max_new_tokens=cfg["max_new_tokens"],
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            for device in range(torch.cuda.device_count()):
                torch.cuda.synchronize(device)
            elapsed = time.monotonic() - batch_started
            continuation = generated[:, encoded["input_ids"].shape[1]:]
            texts = tokenizer.batch_decode(continuation, skip_special_tokens=True)
            for i, (row, text) in enumerate(zip(batch, texts)):
                decoded = decode_native(text, method)
                record = {k: row[k] for k in ("question_id", "model_id", "graph", "answer")}
                record.update({**decoded, "method": method, "correct": decoded["prediction"] == row["answer"],
                    "raw_output": text, "input_tokens": int(encoded["attention_mask"][i].sum()),
                    "output_tokens": int((continuation[i] != tokenizer.pad_token_id).sum()),
                    "batch_seconds_per_case": elapsed / len(batch),
                    "prompt_sha256": hashlib.sha256(prompts[i].encode()).hexdigest()})
                records.append(record)
                emit("PUBLIC_CASE", record)
            emit("PUBLIC_PROGRESS", {"method": method, "done": len(records), "total": len(dataset["rows"]),
                                     "seconds": time.monotonic() - started})
        all_records[method] = records
        emit("PUBLIC_METHOD_SUMMARY", {"method": method, "metrics": public_metrics(records),
             "elapsed_seconds": time.monotonic() - started, "input_tokens": sum(r["input_tokens"] for r in records),
             "output_tokens": sum(r["output_tokens"] for r in records)})
    if cfg.get("native_grounded_solver"):
        from causal_public_grounding import run_public_grounded
        records, usage = run_public_grounded(model, tokenizer, dataset, cfg, emit)
        all_records["native_grounded_solver"] = records
        emit("PUBLIC_METHOD_SUMMARY", {"method": "native_grounded_solver", "metrics": public_metrics(records), **usage})
    primary = cfg.get("primary_method", "native_parser_solver")
    final = {"config": cfg, "audit": dataset["audit"], "metrics": {m: public_metrics(rs) for m, rs in all_records.items()},
             "primary_method": primary,
             "parser_vs": {m: public_bootstrap(all_records[primary], rs)
                           for m, rs in all_records.items() if m != primary},
             "pilot_only": True, "scope": "native public text, deterministic-counterfactual subset only; no fine-tuning"}
    (output / "results.json").write_text(json.dumps({"summary": final, "records": all_records}, indent=2))
    emit("PUBLIC_FINAL_SUMMARY", final)


if __name__ == "__main__":
    main()
