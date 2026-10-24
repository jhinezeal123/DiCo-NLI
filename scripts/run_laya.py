"""Laya fine-tuning through the pinned SDK and fixed bash run.sh contract."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import laya_data as data


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields or list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inference_items(tok, rows, max_len, head_len, swap=False):
    from laya.train import encode_state, make_item, to_internal
    items = []
    for row in rows:
        rendered = data.inference_record(row, swap)
        q = to_internal(data.QUESTION_ID, rendered["questions"][data.QUESTION_ID])
        ids = encode_state(tok, rendered["state"], max_len)
        # A uniform placeholder satisfies the collator; inference never uses targets.
        item, reason = make_item(tok, q, [0.25] * 4, ids, head_len, max_len)
        if reason:
            raise ValueError(f"invalid inference item {row['instance_id']}: {reason}")
        items.append(item)
    return items


def predict(model, tok, items, cfg, device):
    import numpy as np
    from laya.train import calibration_records
    # This function only reads logits; no calibration is fitted.
    records = calibration_records(model, tok, items, device, cfg["max_length"],
                                  cfg["head_max_length"], cfg["eval_batch_size"])
    logits = np.stack([r[1] for r in records]).astype(np.float64)
    if logits.shape != (len(items), 4) or not np.isfinite(logits).all():
        raise ValueError("expected finite N x 4 logits")
    logits = logits / cfg["temperature"]
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def worker(job):
    import numpy as np
    import torch
    from laya.common import build_head
    from laya.train import TrainConfig, items_from_rows, load_checkpoint, save_checkpoint, train_model, to_internal
    cfg, seed = job["config"], job["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "this experiment expects the configured GPU host"
    print("WORKER_CONFIG " + json.dumps(dict(config=cfg, seed=seed, device=str(device),
          gpu=torch.cuda.get_device_name(0), label_order=data.LABELS)), flush=True)
    model, tok, sdk_cfg = load_checkpoint(job["checkpoint"])
    model.to(device)
    sdk_cfg = dict(sdk_cfg, max_len=cfg["max_length"], head_max_len=cfg["head_max_length"], temperature=[1., 1., 1.])
    sdk_cfg.pop("temperature_by_options", None)
    with torch.no_grad():
        model.temperature.fill_(1.)
    train_rows, dev_inputs = read_csv(job["train"]), read_csv(job["dev_inputs"])
    assert all("label" not in r for r in dev_inputs), "workers must not receive dev gold"
    train_items, skipped = items_from_rows(tok, [data.training_record(r) for r in train_rows],
                                           cfg["max_length"], cfg["head_max_length"])
    assert not skipped and len(train_items) == len(train_rows), skipped
    dev_items = inference_items(tok, dev_inputs, cfg["max_length"], cfg["head_max_length"])
    swap_items = inference_items(tok, dev_inputs, cfg["max_length"], cfg["head_max_length"], swap=True)
    q = to_internal(data.QUESTION_ID, data.question())
    head_ids, markers, head_stats = build_head(tok, q, cfg["head_max_length"])
    assert len(markers) == head_stats["options_distinct"] == 4
    all_items = train_items + dev_items + swap_items
    truncated = sum(len(head_ids) + len(it["state_ids"]) + 1 > cfg["max_length"] for it in all_items)
    print("TOKEN_AUDIT " + json.dumps(dict(head_tokens=len(head_ids), head_stats=head_stats,
          train_items=len(train_items), dev_items=len(dev_items), truncated_states=truncated,
          max_state_tokens=max(len(it["state_ids"]) for it in all_items))), flush=True)
    assert not truncated, "increase committed max_length; do not silently truncate phrase inputs"
    parameters = dict(total=sum(p.numel() for p in model.parameters()),
                      encoder=sum(p.numel() for p in model.encoder.parameters()),
                      trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("MODEL_PARAMETERS " + json.dumps(parameters), flush=True)
    out = Path(job["out"])
    result = {}
    if job["measure_pretrained"]:
        result["pretrained_orig"] = predict(model, tok, dev_items, cfg, device)
        result["pretrained_swap"] = predict(model, tok, swap_items, cfg, device)
        print("PRETRAINED_EVALUATION_COMPLETE", flush=True)
    settings = TrainConfig(epochs=cfg["epochs"], micro_batch=cfg["micro_batch"], grad_accum=cfg["grad_accum"],
        encoder_lr=cfg["encoder_lr"], head_lr=cfg["head_lr"], weight_decay=cfg["weight_decay"],
        loss=cfg["loss"], shuffle_options=("choice",) if cfg["shuffle_options"] else (),
        seed=seed, amp=cfg["amp"], gradient_checkpointing=cfg["gradient_checkpointing"],
        max_len=cfg["max_length"], head_max_len=cfg["head_max_length"],
        rl_samples=cfg["rl_samples"], sigma_start=cfg["sigma_start"], sigma_end=cfg["sigma_end"],
        w_sph=cfg["w_sph"], w_rps=cfg["w_rps"], calib_frac=0., calib_max=0, log_every=100)
    settings.validate()
    t0 = time.perf_counter()
    def epoch_end(epoch, loss):
        if not math.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss at epoch {epoch + 1}")
        print("TRAIN_EPOCH " + json.dumps(dict(seed=seed, epoch=epoch + 1, loss=loss,
              wall_seconds=time.perf_counter() - t0)), flush=True)
    trajectory = train_model(model, tok, train_items, settings, device, cfg["max_length"],
                             cfg["head_max_length"], on_epoch_end=epoch_end)
    train_seconds = time.perf_counter() - t0
    result["orig"] = predict(model, tok, dev_items, cfg, device)
    result["swap"] = predict(model, tok, swap_items, cfg, device)
    np.savez(out / "probabilities.npz", **result)
    sdk_cfg["training"] = dict(project="DiCo-NLI", seed=seed, objective=cfg["loss"],
                               epochs=cfg["epochs"], dev_selection=False,
                               source_model=cfg["model_id"], source_revision=cfg["model_revision"])
    save_checkpoint(model, tok, sdk_cfg, str(out / "model"))
    meta = dict(seed=seed, objective=cfg["loss"], training_seconds=train_seconds,
                parameters=parameters, epoch_losses=trajectory,
                gpu=torch.cuda.get_device_name(0), max_memory_allocated=torch.cuda.max_memory_allocated(),
                sdk_version=importlib.metadata.version("laya"), checkpoint=str(out / "model"))
    (out / "worker-summary.json").write_text(json.dumps(meta, indent=2))
    print("WORKER_SUMMARY " + json.dumps(meta), flush=True)


def run_jobs(jobs, n_gpu):
    pending, running, free = list(jobs), {}, list(range(n_gpu))
    while pending or running:
        while pending and free:
            job, gpu = pending.pop(0), free.pop(0)
            path = Path(job["out"])
            path.mkdir()
            spec = path / "job.json"
            spec.write_text(json.dumps(job, indent=2))
            log = open(path / "train.log", "w", encoding="utf-8")
            proc = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--worker", str(spec)],
                                    stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
            running[job["seed"]] = (proc, gpu, log, path, time.time(), 0)
            print(f"WORKER_START seed={job['seed']} gpu={gpu}", flush=True)
        time.sleep(2)
        for seed in list(running):
            proc, gpu, log, path, started, offset = running[seed]
            with open(path / "train.log", "r", encoding="utf-8", errors="replace") as stream:
                stream.seek(offset)
                chunk = stream.read()
                next_offset = stream.tell()
            if chunk:
                print(f"<<<WORKER_LOG seed={seed}\n{chunk}", end="" if chunk.endswith("\n") else "\n", flush=True)
                print("WORKER_LOG>>>", flush=True)
            running[seed] = (proc, gpu, log, path, started, next_offset)
            if proc.poll() is None:
                continue
            log.close()
            del running[seed]
            free.append(gpu)
            print(f"WORKER_DONE seed={seed} rc={proc.returncode} wall_seconds={time.time() - started:.1f}", flush=True)
            if proc.returncode:
                for other, _, other_log, *_ in running.values():
                    other.terminate()
                    other.wait()
                    other_log.close()
                raise RuntimeError(f"seed {seed} failed; see streamed worker log")


def run_experiment(cfg, upstream):
    import numpy as np
    from huggingface_hub import snapshot_download
    from run_pipeline import audit, environment, gpu_count, official_score, validate
    assert importlib.metadata.version("laya") == cfg["laya_version"]
    print("LAYA_CONFIG " + json.dumps(cfg), flush=True)
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(ROOT / "tests"), "-v"], check=True)
    out = ROOT / "outputs" / cfg["name"]
    out.mkdir(parents=True)
    track = f"track{cfg['track']}"
    paths = {"train": upstream / "final_data/train" / f"dico_nli_train_{track}_participant_labeled.csv",
             "dev": upstream / "final_data/dev" / f"dico_nli_dev_{track}_participant_labeled.csv",
             "dev_reference": upstream / "final_data/dev" / f"dico_nli_dev_{track}_reference.csv"}
    train_rows, dev_rows = read_csv(paths["train"]), read_csv(paths["dev"])
    audit(train_rows, dev_rows, paths)
    dev_inputs = [{k: v for k, v in r.items() if k != "label"} for r in dev_rows]
    cleaned, excluded = data.clean_train_groups(train_rows, dev_inputs)
    audit_info = dict(original_train_rows=len(train_rows), cleaned_train_rows=len(cleaned),
                      removed_rows=len(train_rows) - len(cleaned), excluded_pair_ids=excluded,
                      dev_rows=len(dev_rows), remaining_unordered_text_overlap=0,
                      hashes={key: sha256(path) for key, path in paths.items()})
    print("CLEAN_DATA_AUDIT " + json.dumps(audit_info), flush=True)
    (out / "data-audit.json").write_text(json.dumps(audit_info, indent=2))
    train_path, dev_path = out / "train-clean.csv", out / "dev-inputs-unlabeled.csv"
    write_csv(train_path, cleaned)
    write_csv(dev_path, dev_inputs)
    environment(cfg["model_id"])
    checkpoint = snapshot_download(cfg["model_id"], revision=cfg["model_revision"],
                                   allow_patterns=["rl_agent_config.json", "model.safetensors", "encoder/*", "tokenizer/*"])
    print("PINNED_CHECKPOINT " + json.dumps(dict(model=cfg["model_id"], revision=cfg["model_revision"],
          config_sha256=sha256(Path(checkpoint) / "rl_agent_config.json"))), flush=True)
    jobs = [dict(config=cfg, seed=seed, train=str(train_path.resolve()), dev_inputs=str(dev_path.resolve()),
                 checkpoint=checkpoint, out=str((out / f"seed{seed}").resolve()),
                 measure_pretrained=seed == cfg["seeds"][0]) for seed in cfg["seeds"]]
    run_jobs(jobs, gpu_count())
    results = []
    ensemble = {mode: [] for mode in ("standard", "bidir")}

    def evaluate(name, seed, p, q, dest):
        assert p.shape == q.shape == (len(dev_rows), 4)
        np.testing.assert_allclose(p.sum(1), 1., atol=1e-6)
        np.testing.assert_allclose(q.sum(1), 1., atol=1e-6)
        probabilities = {"standard": p, "bidir": 0.5 * (p + q[:, data.REV])}
        for decoding, prob in probabilities.items():
            pred_path = (dest / f"{name}-{decoding}-predictions.csv").resolve()
            write_csv(pred_path, [dict(instance_id=r["instance_id"], label=data.LABELS[int(v.argmax())])
                                  for r, v in zip(dev_rows, prob)], ["instance_id", "label"])
            validate(pred_path, dev_rows)
            score = official_score(upstream, paths["dev_reference"], pred_path,
                                   (dest / f"scores-{name}-{decoding}").resolve())
            result = dict(variant=name, seed=seed, objective=cfg["loss"], decoding=decoding,
                          **{k: score[k] for k in ("weighted_f1", "soft_cons", "hard_cons")})
            results.append(result)
            print("LAYA_METRICS " + json.dumps(result), flush=True)
            print(f"<<<PREDICTIONS_CSV variant={name} seed={seed} decoding={decoding}")
            print(pred_path.read_text(), end="")
            print("PREDICTIONS_CSV>>>", flush=True)
            if name == "finetuned":
                ensemble[decoding].append(prob)
        print(f"<<<DEV_PROBS variant={name} seed={seed}")
        writer = csv.writer(sys.stdout, lineterminator="\n")
        writer.writerow(["instance_id"] + [f"p_{l}" for l in data.LABELS] + [f"q_{l}" for l in data.LABELS])
        for row, a, b in zip(dev_rows, p, q):
            writer.writerow([row["instance_id"]] + a.tolist() + b.tolist())
        print("DEV_PROBS>>>", flush=True)

    workers = []
    for seed in cfg["seeds"]:
        dest = out / f"seed{seed}"
        payload = np.load(dest / "probabilities.npz", allow_pickle=False)
        if "pretrained_orig" in payload:
            evaluate("pretrained", seed, payload["pretrained_orig"], payload["pretrained_swap"], dest)
        evaluate("finetuned", seed, payload["orig"], payload["swap"], dest)
        workers.append(json.loads((dest / "worker-summary.json").read_text()))
    ensemble_results = []
    for mode, values in ensemble.items():
        prob = np.mean(values, axis=0)
        pred_path = (out / f"ensemble-{mode}-predictions.csv").resolve()
        write_csv(pred_path, [dict(instance_id=r["instance_id"], label=data.LABELS[int(v.argmax())])
                              for r, v in zip(dev_rows, prob)], ["instance_id", "label"])
        validate(pred_path, dev_rows)
        score = official_score(upstream, paths["dev_reference"], pred_path,
                               (out / f"scores-ensemble-{mode}").resolve())
        record = dict(objective=cfg["loss"], decoding=mode, n_seeds=len(values),
                      **{k: score[k] for k in ("weighted_f1", "soft_cons", "hard_cons")})
        ensemble_results.append(record)
        print("ENSEMBLE_METRICS " + json.dumps(record), flush=True)
        print(f"<<<PREDICTIONS_CSV variant=ensemble seed=all decoding={mode}")
        print(pred_path.read_text(), end="")
        print("PREDICTIONS_CSV>>>", flush=True)
    for mode in ("standard", "bidir"):
        records = [r for r in results if r["variant"] == "finetuned" and r["decoding"] == mode]
        aggregate = {key: dict(mean=statistics.mean(r[key] for r in records),
                               stdev=statistics.stdev(r[key] for r in records),
                               minimum=min(r[key] for r in records))
                     for key in ("weighted_f1", "soft_cons", "hard_cons")}
        print("FINAL_LAYA_METRICS " + json.dumps(dict(objective=cfg["loss"], decoding=mode,
              seeds=cfg["seeds"], metrics=aggregate)), flush=True)
    summary = dict(config=cfg, data_audit=audit_info, results=results, workers=workers,
                   ensemble_results=ensemble_results, dev_checkpoint_selection=False,
                   calibration="fixed T=1; no fitting on dev")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("LAYA_SUMMARY_JSON " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=Path, required=True)
    args = ap.parse_args()
    worker(json.loads(args.worker.read_text()))
