"""Tracked query-semantics SFT, with frozen learned 7B operators and exact SCMs."""
from collections import Counter
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
from causal_choice import decide
from causal_dataset_v5 import build_dataset_v5, RESERVED_SEEDS
from causal_intervention_supervision import build_intervention_training, build_supervised_requests, infer_supervised_mask, training_feature
from causal_operator_replay import load_operator_replay
from causal_sequence_compiler import scm_for_mask
from causal_v3_pipeline import load_control_replay
from causal_v3_protocol import decode_v3, metrics_v3, paired_v3


def emit(tag, value):
    print(tag + " " + json.dumps(value, sort_keys=True), flush=True)


def persist_adapter(directory):
    files = sorted(path for path in directory.iterdir() if path.is_file())
    if sum(path.stat().st_size for path in files) > 32 * 1024 * 1024:
        raise ValueError("adapter-only export exceeds 32MiB")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.name)
    raw = buffer.getvalue()
    encoded = base64.b64encode(raw).decode("ascii")
    chunks = [encoded[index:index+32768] for index in range(0, len(encoded), 32768)]
    manifest = {"name": "adapter.zip", "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                "chunks": len(chunks), "transport": "persisted run stdout; adapter only, no base weights"}
    emit("SFT_ADAPTER_MANIFEST", manifest)
    for index, chunk in enumerate(chunks):
        emit("SFT_ADAPTER_CHUNK", {"index": index, "base64": chunk})
    emit("SFT_ADAPTER_COMPLETE", {"sha256": manifest["sha256"], "chunks": len(chunks)})
    return manifest


def generate_traces(model, tokenizer, requests, batch_size, phase):
    import torch
    outputs = {}
    for offset in range(0, len(requests), batch_size):
        batch = requests[offset:offset+batch_size]
        prompts = [tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True) for row in batch]
        encoded = tokenizer(prompts, padding=True, truncation=False, return_tensors="pt").to(model.device)
        torch.cuda.synchronize()
        started = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(**encoded, do_sample=False, max_new_tokens=512,
                                       pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        continuation = generated[:, encoded["input_ids"].shape[1]:]
        for index, (row, text) in enumerate(zip(batch, tokenizer.batch_decode(continuation, skip_special_tokens=True))):
            value = {"request_id": row["request_id"], "phase": phase, "stage": row["stage"], "raw_output": text,
                     "input_tokens": int(encoded["attention_mask"][index].sum()),
                     "output_tokens": int((continuation[index] != tokenizer.pad_token_id).sum()),
                     "seconds": elapsed / len(batch), "max_new_tokens": 512,
                     "prompt_sha256": hashlib.sha256(prompts[index].encode()).hexdigest()}
            outputs[row["request_id"]] = value
            emit("CAUSAL_SUBCALL", value)
        emit("CAUSAL_PROGRESS", {"phase": phase, "method": "program_cot", "done": len(outputs), "total": len(requests)})
    return outputs


def score_semantics(dataset, plans, operators, outputs, phase, mode, emit_fn=emit, blind=None):
    """Compile using public plans and model outputs; labels enter only scoring."""
    method = phase + ":" + (blind if blind else mode)
    keys_by_case = {case: ([] if blind == "action_blind" else plan["action_requests"] if mode == "action"
                          else [plan[mode + "_request"]]) for case, plan in plans.items()}
    uses = Counter(key for keys in keys_by_case.values() for key in keys)
    records = []
    for row in dataset["rows"]:
        case = row["case_id"]
        plan = plans[case]
        try:
            mask, details = (0, {}) if blind == "action_blind" else infer_supervised_mask(plan, outputs, mode)
            scm = scm_for_mask(plan, operators, mask, blind == "operator_blind")
            raw = json.dumps(scm, separators=(",", ":"))
        except ValueError as exc:
            mask, raw = None, "QUERY_COMPILATION_ERROR: " + str(exc)
            key = plan.get(mode + "_request")
            details = {"query_compilation_error": str(exc), "query_model_output": outputs.get(key, {}).get("raw_output", "")}
        decoded = decode_v3(raw, "parser_solver", row["scm"])
        keys = keys_by_case[case]
        record = {key: row[key] for key in ("case_id", "family_id", "split", "view", "answer", "changed")}
        record.update({**decoded, **details, "method": method, "phase": phase, "raw_output": raw,
                       "correct": decoded["prediction"] == row["answer"], "final_intervention_mask": mask,
                       "subcall_ids": keys, "input_tokens": sum(outputs[key]["input_tokens"] / uses[key] for key in keys),
                       "output_tokens": sum(outputs[key]["output_tokens"] / uses[key] for key in keys),
                       "operator_source": "frozen 7B learned decisions, not gold operators"})
        if blind:
            record.update(generation_reused_from=phase + ":action", new_lm_calls=0)
        records.append(record)
        emit_fn("CAUSAL_CASE", record)
    summary = {"method": method, "metrics": metrics_v3(records), "unique_query_subcalls": len(uses),
               "input_tokens": sum(outputs[key]["input_tokens"] for key in uses),
               "output_tokens": sum(outputs[key]["output_tokens"] for key in uses),
               "model_forward_seconds": sum(outputs[key]["seconds"] for key in uses),
               "operator_cost_scope": "frozen actual 7B predictions; query model is 3B; costs exclude original 7B inference",
               "world_demonstrations_for_query_model": 0}
    if blind:
        summary.update(generation_reused_from=phase + ":action", new_lm_calls=0)
    emit_fn("CAUSAL_METHOD_SUMMARY", summary)
    return method, records


def train(model, tokenizer, features, cfg):
    import torch
    import transformers
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    emit("SFT_MODEL", {"trainable_parameters": sum(parameter.numel() for parameter in params),
         "total_parameters": model.num_parameters(), "trainable_dtypes": sorted({str(parameter.dtype) for parameter in params}),
         "examples": len(features), "maximum_feature_length": max(len(row["input_ids"]) for row in features),
         "dropped_for_length": 0, "completion_only_loss": True})
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()
    optimizer = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=0, eps=1e-6)
    total_updates = len(features) * cfg["epochs"]
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, max(1, total_updates // 10), total_updates)
    scaler = torch.amp.GradScaler("cuda")
    rng = random.Random(cfg["seed"])
    steps = attempts = input_count = target_count = 0
    stage_exposures = Counter()
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(cfg["epochs"]):
        order = list(range(len(features)))
        rng.shuffle(order)
        for index in order:
            feature = features[index]
            # Retry an overflow on the SAME example; do not silently drop supervision.
            for retry in range(10):
                optimizer.zero_grad(set_to_none=True)
                ids = torch.tensor([feature["input_ids"]], dtype=torch.long, device=model.device)
                labels = torch.tensor([feature["labels"]], dtype=torch.long, device=model.device)
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite intervention supervision loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped = scaler.get_scale() < old_scale
                attempts += 1
                input_count += ids.numel()
                target_count += int((labels != -100).sum())
                if not skipped:
                    scheduler.step()
                    steps += 1
                    stage_exposures[feature["stage"]] += 1
                if attempts == 1 or steps % 25 == 0 or skipped:
                    emit("SFT_TRAIN", {"updates": steps, "attempts": attempts, "epoch": epoch + 1,
                         "stage": feature["stage"], "loss": float(loss.detach()), "grad_norm": grad_norm,
                         "skipped": skipped, "learning_rate": optimizer.param_groups[0]["lr"],
                         "input_tokens": input_count, "supervised_target_tokens": target_count,
                         "seconds": time.monotonic() - started, "peak_vram_bytes": torch.cuda.max_memory_allocated()})
                if not skipped:
                    break
            else:
                raise FloatingPointError("ten repeated overflows on the same example")
    assert steps == total_updates
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    return {"updates": steps, "attempts": attempts, "epochs": cfg["epochs"], "stage_exposures": dict(stage_exposures),
            "seconds": time.monotonic() - started, "input_tokens": input_count, "supervised_target_tokens": target_count}


def main():
    cfg = json.loads(Path("configs/causal_semantic_sft.json").read_text())
    if cfg["pilot_only"] is not True or cfg["dataset_seed"] in RESERVED_SEEDS:
        raise ValueError("this runner is restricted to opened development")
    emit("CAUSAL_CONFIG", {**cfg, "commit": os.environ.get("ORX_COMMIT"), "run_id": os.environ.get("ORX_RUN_ID")})
    ref_raw = Path(cfg["reference_config"]["path"]).read_bytes()
    if hashlib.sha256(ref_raw).hexdigest() != cfg["reference_config"]["sha256"]:
        raise ValueError("frozen reference configuration mismatch")
    reference_cfg = json.loads(ref_raw)
    dataset = build_dataset_v5(cfg["dataset_seed"], cfg["families_per_split"])
    if dataset["audit"]["dataset_sha256"] != cfg["dataset_sha256"]:
        raise ValueError("development dataset mismatch")
    operators, requests, plans = build_supervised_requests(dataset["rows"], dataset["demonstrations"])
    operator_outputs = load_operator_replay(reference_cfg, dataset, operators)
    all_records = load_control_replay(reference_cfg, dataset)
    if set(all_records) != set(cfg["reference_methods"]):
        raise ValueError("frozen reference methods mismatch")
    training, audit = build_intervention_training(**cfg["training_data"])
    if audit["training_sha256"] != cfg["training_sha256"]:
        raise ValueError("independent training data mismatch")
    if len(training) * cfg["epochs"] != cfg["expected_updates"] or cfg["max_new_tokens"] != 512:
        raise ValueError("fixed training or inference budget mismatch")
    evaluation_names = {name for plan in plans.values() for name in plan["names"]}
    if evaluation_names & set(audit["names"]):
        raise ValueError("training/evaluation variable-name overlap")
    emit("DATASET_AUDIT", dataset["audit"])
    emit("SFT_DATA_AUDIT", {**{key: value for key, value in audit.items() if key != "names"},
         "training_name_count": len(audit["names"]), "training_evaluation_name_overlap": 0,
         "scope": "short independent programs; no private SCM answers or eight-step evaluation traces"})
    output = Path("outputs") / cfg["name"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "training.json").write_text(json.dumps(training, indent=2))
    for value in operator_outputs.values():
        emit("CAUSAL_SUBCALL", {**value, "phase": "frozen7b"})
    for method, records in all_records.items():
        for row in records:
            emit("CAUSAL_CASE", {**row, "new_lm_calls": 0})
        emit("CAUSAL_METHOD_SUMMARY", {"method": method, "metrics": metrics_v3(records), "new_lm_calls": 0,
             "source_run": reference_cfg["control_replay"]["source_run"]})
    import torch
    import transformers
    import peft
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import get_peft_model, LoraConfig, TaskType
    if not torch.cuda.is_available():
        raise RuntimeError("semantic SFT requires configured remote GPU")
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["model_revision"], padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    features = [training_feature(tokenizer, row, cfg["max_length"]) for row in training]
    emit("SFT_TOKEN_AUDIT", {"examples": len(features), "maximum_length": max(len(row["input_ids"]) for row in features),
         "dropped_for_length": 0, "completion_only_loss": True})
    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], revision=cfg["model_revision"],
            torch_dtype=torch.float16, attn_implementation="sdpa", low_cpu_mem_usage=True).to("cuda:0")
    emit("ENVIRONMENT", {"python": sys.version.split()[0], "torch": torch.__version__,
         "transformers": transformers.__version__, "peft": peft.__version__,
         "gpu": torch.cuda.get_device_name(0), "gpu_count": torch.cuda.device_count(), "query_model_device": "cuda:0",
         "operator_model": reference_cfg["model_id"], "operator_inference": "checksum-verified frozen decisions"})

    def evaluate(phase):
        model.eval()
        model.config.use_cache = True
        decisions = decide(model, tokenizer, [row for row in requests if "choice_count" in row], cfg["batch_size"],
                           lambda tag, value: emit(tag.replace("PUBLIC_", "CAUSAL_"), {**value, "phase": phase}))
        traces = generate_traces(model, tokenizer, [row for row in requests if row["stage"] == "program_cot"], cfg["batch_size"], phase)
        decisions.update(traces)
        for mode in ("action", "program_state", "program_cot"):
            method, records = score_semantics(dataset, plans, operator_outputs, decisions, phase, mode)
            all_records[method] = records
        if phase == "adapter3b":
            for blind in ("operator_blind", "action_blind"):
                method, records = score_semantics(dataset, plans, operator_outputs, decisions, phase, "action", blind=blind)
                all_records[method] = records

    evaluate("base3b")
    torch.manual_seed(cfg["seed"])
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=cfg["lora_rank"],
             lora_alpha=cfg["lora_alpha"], lora_dropout=.05, target_modules=["q_proj", "v_proj"]))
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    train_summary = train(model, tokenizer, features, cfg)
    emit("SFT_TRAINING_COMPLETE", train_summary)
    adapter_dir = output / "adapter"
    adapter_dir.mkdir(exist_ok=True)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    adapter_manifest = persist_adapter(adapter_dir)
    evaluate("adapter3b")
    if set(all_records) != set(cfg["methods"]):
        raise ValueError("incomplete registered evaluation")
    final = {"config": cfg, "audit": dataset["audit"], "training_audit": audit,
             "training": train_summary, "adapter_artifact": adapter_manifest,
             "metrics": {method: metrics_v3(rows) for method, rows in all_records.items()},
             "primary_method": cfg["primary_method"], "primary_vs": {method: paired_v3(all_records[cfg["primary_method"]], rows)
                          for method, rows in all_records.items() if method != cfg["primary_method"]},
             "pilot_only": True, "parser_solver_vs": {}, "gate": {"status": "development_only", "joint_minimum": .9,
                 "delta_minimum": .05, "requires_positive_ci_lower": True,
                 "confirm_dataset_seeds": RESERVED_SEEDS, "required_views": 4, "controls": cfg["gate_controls"]},
             "scope": "3B query semantics plus frozen learned 7B operators and exact solver; not a standalone 3B causal model"}
    (output / "summary.json").write_text(json.dumps(final, indent=2))
    emit("CAUSAL_FINAL_SUMMARY", final)


if __name__ == "__main__":
    main()
