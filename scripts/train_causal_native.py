"""Small tracked LoRA pilot; all ML training/evaluation runs on configured GPU."""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import random
import sys
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from audit_cladder_native import load_source, emit
from causal_public_protocol import build_public_split, native_messages, decode_native, public_metrics, public_bootstrap
from causal_supervision import build_supervision, training_pair


def persist_adapter(adapter_dir, emit):
    if sum(path.stat().st_size for path in adapter_dir.iterdir() if path.is_file()) > 32*1024*1024:
        raise ValueError("adapter export exceeds 32MiB cap; refusing to log full base weights")
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(adapter_dir.iterdir()):
            if path.is_file():
                zipped.write(path, path.name)
    content = archive.getvalue()
    encoded = base64.b64encode(content).decode("ascii")
    chunks = [encoded[i:i+32768] for i in range(0, len(encoded), 32768)]
    manifest = {"name": "adapter.zip", "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                "chunks": len(chunks), "transport": "persisted run stdout; adapter only, no base weights"}
    emit("SFT_ADAPTER_MANIFEST", manifest)
    for index, chunk in enumerate(chunks):
        emit("SFT_ADAPTER_CHUNK", {"index": index, "base64": chunk})
    emit("SFT_ADAPTER_COMPLETE", {"sha256": manifest["sha256"], "chunks": len(chunks)})
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg = json.loads(Path(parser.parse_args().config).read_text())
    emit("SFT_CONFIG", {**cfg, "commit": os.environ.get("ORX_COMMIT"), "run_id": os.environ.get("ORX_RUN_ID")})
    questions, models = [load_source(f["name"], cfg["revision"], f["sha256"]) for f in cfg["files"]]
    plan_path = Path(cfg["split_plan_path"])
    if hashlib.sha256(plan_path.read_bytes()).hexdigest() != cfg["split_plan_sha256"]:
        raise ValueError("split plan checksum mismatch")
    plan = json.loads(plan_path.read_bytes())
    training, audit = build_supervision(questions, models, plan, cfg["training_questions"], cfg["seed"])
    dataset = build_public_split(questions, models, 4)
    if {r["model_id"] for r in dataset["rows"]} != set(plan["pilot_model_ids"]):
        raise ValueError("pilot differs from fixed split plan")
    emit("SFT_DATA_AUDIT", audit)
    output = Path("outputs") / cfg["name"]
    output.mkdir(parents=True, exist_ok=True)
    (output/"training.json").write_text(json.dumps(training, indent=2))
    import torch
    import transformers
    import peft
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, TaskType, get_peft_model
    if not torch.cuda.is_available():
        raise RuntimeError("causal SFT requires GPU")
    random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    emit("SFT_ENVIRONMENT", {"python": sys.version.split()[0], "torch": torch.__version__,
         "transformers": transformers.__version__, "peft": peft.__version__, "gpu": torch.cuda.get_device_name(0),
         "gpu_count": torch.cuda.device_count(), "training_device": "cuda:0"})
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["model_revision"], padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], revision=cfg["model_revision"],
                torch_dtype=torch.float16, attn_implementation="sdpa", low_cpu_mem_usage=True).to("cuda:0")
    all_records = {}

    def evaluate(phase):
        model.eval()
        model.config.use_cache = True
        for method in ("direct", "native_parser_solver"):
            records, started = [], time.monotonic()
            for offset in range(0, len(dataset["rows"]), cfg["eval_batch_size"]):
                batch = dataset["rows"][offset:offset+cfg["eval_batch_size"]]
                prompts = [tokenizer.apply_chat_template(native_messages(row, method, dataset["demonstrations"]),
                           tokenize=False, add_generation_prompt=True) for row in batch]
                encoded = tokenizer(prompts, padding=True, truncation=False, return_tensors="pt").to("cuda:0")
                torch.cuda.synchronize(); batch_started = time.monotonic()
                with torch.inference_mode():
                    generated = model.generate(**encoded, do_sample=False, max_new_tokens=cfg["max_new_tokens"],
                        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                torch.cuda.synchronize(); elapsed = time.monotonic()-batch_started
                continuation = generated[:, encoded["input_ids"].shape[1]:]
                texts = tokenizer.batch_decode(continuation, skip_special_tokens=True)
                for i, (row, text) in enumerate(zip(batch, texts)):
                    decoded = decode_native(text, method)
                    record = {k: row[k] for k in ("question_id", "model_id", "graph", "answer")}
                    record.update({**decoded, "phase": phase, "method": method,
                        "correct": decoded["prediction"] == row["answer"], "raw_output": text,
                        "input_tokens": int(encoded["attention_mask"][i].sum()),
                        "output_tokens": int((continuation[i] != tokenizer.pad_token_id).sum()),
                        "batch_seconds_per_case": elapsed/len(batch),
                        "prompt_sha256": hashlib.sha256(prompts[i].encode()).hexdigest()})
                    records.append(record); emit("SFT_PUBLIC_CASE", record)
                emit("SFT_PROGRESS", {"phase": phase, "method": method, "done": len(records), "total": len(dataset["rows"])})
            all_records[phase+":"+method] = records
            emit("SFT_METHOD_SUMMARY", {"phase": phase, "method": method, "metrics": public_metrics(records),
                 "seconds": time.monotonic()-started, "input_tokens": sum(r["input_tokens"] for r in records),
                 "output_tokens": sum(r["output_tokens"] for r in records)})

    evaluate("base")
    torch.manual_seed(cfg["seed"])
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=cfg["lora_rank"],
             lora_alpha=cfg["lora_alpha"], lora_dropout=.05, target_modules=["q_proj", "v_proj"]))
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    features = [training_pair(tokenizer, row, cfg["max_length"]) for row in training]
    dropped = sum(pair is None for pair in features)
    features = [pair for pair in features if pair is not None]
    if not features:
        raise ValueError("all supervision pairs exceed sequence budget")
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    emit("SFT_MODEL", {"trainable_parameters": sum(p.numel() for p in params), "total_parameters": model.num_parameters(),
         "trainable_dtypes": sorted({str(p.dtype) for p in params}), "question_pairs": len(features),
         "dropped_whole_pairs_for_length": dropped, "max_length": cfg["max_length"]})
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()
    optimizer = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=0, eps=1e-6)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, max(1, cfg["max_steps"]//10), cfg["max_steps"])
    scaler = torch.amp.GradScaler("cuda")
    rng = random.Random(cfg["seed"])
    order = list(range(len(features))); rng.shuffle(order)
    torch.cuda.reset_peak_memory_stats()
    steps, attempts, input_count, target_count, started = 0, 0, 0, 0, time.monotonic()
    while steps < cfg["max_steps"]:
        if attempts >= cfg["max_steps"]*2:
            raise FloatingPointError("too many mixed-precision skipped updates")
        index = order[attempts % len(order)]
        optimizer.zero_grad(set_to_none=True)
        losses = {}
        for feature in features[index]:
            ids = torch.tensor([feature["input_ids"]], dtype=torch.long, device="cuda:0")
            labels = torch.tensor([feature["labels"]], dtype=torch.long, device="cuda:0")
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels).loss
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite native supervision loss")
            scaler.scale(loss/2).backward()
            losses[feature["method"]] = float(loss.detach())
            input_count += ids.numel(); target_count += int((labels != -100).sum())
        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        old_scale = scaler.get_scale()
        scaler.step(optimizer); scaler.update()
        skipped = scaler.get_scale() < old_scale
        attempts += 1
        if not skipped:
            scheduler.step(); steps += 1
        if attempts == 1 or steps % 10 == 0 or skipped:
            emit("SFT_TRAIN", {"updates": steps, "attempts": attempts, "losses": losses, "grad_norm": grad_norm,
                 "skipped": skipped, "learning_rate": optimizer.param_groups[0]["lr"],
                 "input_tokens": input_count, "supervised_target_tokens": target_count,
                 "seconds": time.monotonic()-started, "peak_vram_bytes": torch.cuda.max_memory_allocated()})
    train_summary = {"updates": steps, "attempts": attempts, "input_tokens": input_count,
                     "supervised_target_tokens": target_count, "seconds": time.monotonic()-started}
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    evaluate("adapter")
    adapter_dir = output/"adapter"; adapter_dir.mkdir(exist_ok=True)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    adapter_manifest = persist_adapter(adapter_dir, emit)
    final = {"config": cfg, "data_audit": audit, "training": train_summary,
        "metrics": {name: public_metrics(rows) for name, rows in all_records.items()},
        "adapter_parser_vs": {name: public_bootstrap(all_records["adapter:native_parser_solver"], rows)
                              for name, rows in all_records.items() if name != "adapter:native_parser_solver"},
        "adapter_direct_vs_base_direct": public_bootstrap(all_records["adapter:direct"], all_records["base:direct"]),
        "adapter_artifact": adapter_manifest, "pilot_only": True,
        "scope": f"{audit['questions_used']}-question model-disjoint supervision pilot; shared direct/program adapter, not an ablation of supervision style"}
    (output/"summary.json").write_text(json.dumps(final, indent=2))
    emit("SFT_FINAL_SUMMARY", final)


if __name__ == "__main__":
    main()
