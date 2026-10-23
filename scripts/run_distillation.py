"""A controlled pilot with one refit teacher and four student treatments.

The fixed `bash run.sh` contract enters here via run_pipeline.py. Worker processes
are local to the allocated experiment host and each owns one visible GPU.
"""
import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def write_rows(path, rows, fields=None):
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields or list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with open(path, encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def resolve_revision(cfg):
    from huggingface_hub import model_info
    cfg = dict(cfg)
    cfg["revision"] = model_info(cfg["model_id"], revision=cfg.get("revision")).sha
    print(f"MODEL_REVISION model_id={cfg['model_id']} revision={cfg['revision']}", flush=True)
    return cfg


def run_workers(jobs, out_root, n_gpu):
    pending, running, free = list(jobs), {}, list(range(n_gpu))
    while pending or running:
        while pending and free:
            job, gpu = pending.pop(0), free.pop(0)
            name = job["job_name"]
            dest = out_root / name
            dest.mkdir()
            spec = dest / "job.json"
            spec.write_text(json.dumps(job, indent=2))
            log = open(dest / "train.log", "w", encoding="utf-8")
            proc = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                                     "--worker", str(spec)], stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "TOKENIZERS_PARALLELISM": "false"})
            running[name] = proc, gpu, log
            print(f"WORKER_START name={name} gpu={gpu}", flush=True)
        time.sleep(2)
        for name in list(running):
            proc, gpu, log = running[name]
            if proc.poll() is None:
                continue
            log.close()
            del running[name]
            free.append(gpu)
            print(f"<<<WORKER_LOG name={name}", flush=True)
            print((out_root / name / "train.log").read_text(errors="replace"), flush=True)
            print("WORKER_LOG>>>", flush=True)
            if proc.returncode:
                for other_proc, _, other_log in running.values():
                    other_proc.terminate()
                    other_proc.wait()
                    other_log.close()
                raise RuntimeError(f"worker {name} failed rc={proc.returncode}")


def run_pilot(cfg, upstream):
    import numpy as np
    from run_pipeline import audit, environment, gpu_count, official_score, validate
    from distill_core import LABELS, REV, clean_train_groups, softmax
    section = cfg["distillation"]
    print("PILOT_CONFIG " + json.dumps(cfg), flush=True)
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(ROOT / "tests"), "-v"], check=True)
    out = (ROOT / "outputs" / cfg["name"]).resolve()
    out.mkdir(parents=True)  # Fresh immutable source directory on every run.
    track = f"track{cfg['track']}"
    data = upstream / "final_data"
    paths = dict(train=data / "train" / f"dico_nli_train_{track}_participant_labeled.csv",
                 dev=data / "dev" / f"dico_nli_dev_{track}_participant_labeled.csv",
                 dev_reference=data / "dev" / f"dico_nli_dev_{track}_reference.csv")
    train, dev = read_rows(paths["train"]), read_rows(paths["dev"])
    audit(train, dev, paths)
    dev_inputs = [{k: v for k, v in row.items() if k != "label"} for row in dev]
    cleaned, excluded = clean_train_groups(train, dev_inputs)
    audit_path = out / "data_audit.json"
    audit_info = dict(original_train_rows=len(train), cleaned_train_rows=len(cleaned),
                      removed_train_rows=len(train) - len(cleaned), removed_pair_ids=excluded,
                      dev_rows=len(dev), remaining_unordered_text_overlaps=0)
    audit_path.write_text(json.dumps(audit_info, indent=2))
    print("CLEAN_DATA_AUDIT " + json.dumps(audit_info), flush=True)
    train_path, dev_path = out / "train_clean.csv", out / "dev_inputs_unlabeled.csv"
    write_rows(train_path, cleaned)
    write_rows(dev_path, dev_inputs)
    environment(cfg["model_id"])
    teacher_cfg = resolve_revision({k: v for k, v in cfg.items() if k not in ("distillation", "seeds")})
    student_cfg = resolve_revision(section["student"])
    n_gpu = gpu_count()
    print(f"PILOT_RUNTIME gpus={n_gpu} seeds={cfg['seeds']} dev_checkpoint_selection=false", flush=True)
    all_results = []

    def evaluate(name, path, seed):
        payload = np.load(path, allow_pickle=False)
        p, q = softmax(payload["dev_orig_logits"], 1), softmax(payload["dev_swap_logits"], 1)
        # Diagnostic inference projection; NEG identity here is a heuristic, not official mapping.
        probs = dict(standard=p, bidir=0.5 * (p + q[:, REV]))
        meta = json.loads(path.with_suffix(".json").read_text())
        for decoding, probability in probs.items():
            pred = path.parent / f"dev_predictions_{decoding}.csv"
            write_rows(pred, [dict(instance_id=r["instance_id"], label=LABELS[int(v.argmax())])
                              for r, v in zip(dev, probability)], ["instance_id", "label"])
            validate(pred, dev)
            scores = official_score(upstream, paths["dev_reference"], pred, path.parent / f"scores_{decoding}")
            record = dict(name=name, seed=seed, decoding=decoding,
                          **{k: scores[k] for k in ("weighted_f1", "soft_cons", "hard_cons")}, **meta)
            all_results.append(record)
            print("PILOT_METRICS " + json.dumps(record), flush=True)
            print(f"<<<PREDICTIONS_CSV name={name} seed={seed} decoding={decoding}")
            print(pred.read_text(), end="")
            print("PREDICTIONS_CSV>>>", flush=True)
        print(f"<<<DEV_PROBS name={name} seed={seed}")
        writer = csv.writer(sys.stdout, lineterminator="\n")
        writer.writerow(["instance_id"] + [f"p_{l}" for l in LABELS] + [f"q_{l}" for l in LABELS])
        for row, pp, qq in zip(dev, p, q):
            writer.writerow([row["instance_id"]] + pp.tolist() + qq.tolist())
        print("DEV_PROBS>>>", flush=True)

    for seed in cfg["seeds"]:
        teacher_name = f"teacher_seed{seed}"
        teacher_path = out / teacher_name / "probs.npz"
        common = dict(train_path=str(train_path), dev_path=str(dev_path), seed=seed)
        run_workers([dict(job_name=teacher_name, cfg=teacher_cfg, **common)], out, 1)
        evaluate("teacher", teacher_path, seed)
        jobs = [dict(job_name=f"{arm['name']}_seed{seed}", cfg=student_cfg, arm=arm,
                     teacher_cache=str(teacher_path), kd_cfg=section, **common) for arm in section["arms"]]
        run_workers(jobs, out, n_gpu)
        for job in jobs:
            evaluate(job["arm"]["name"], out / job["job_name"] / "probs.npz", seed)
    summary = dict(config=cfg, teacher_revision=teacher_cfg["revision"],
                   student_revision=student_cfg["revision"], data_audit=audit_info, results=all_results)
    (out / "pilot_summary.json").write_text(json.dumps(summary, indent=2))
    for name in ["teacher"] + [arm["name"] for arm in section["arms"]]:
        for decoding in ("standard", "bidir"):
            records = [r for r in all_results if r["name"] == name and r["decoding"] == decoding]
            final = dict(name=name, decoding=decoding, seeds=cfg["seeds"],
                         **{k: statistics.mean(r[k] for r in records)
                            for k in ("weighted_f1", "soft_cons", "hard_cons")})
            print("FINAL_PILOT_METRICS " + json.dumps(final), flush=True)
    print("PILOT_SUMMARY_JSON " + json.dumps(summary), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=Path, required=True)
    args = ap.parse_args()
    job = json.loads(args.worker.read_text())
    from distill_trainer import fit
    fit(job["cfg"], job["seed"], job["train_path"], job["dev_path"],
        args.worker.parent / "probs.npz", arm=job.get("arm"), cache=job.get("teacher_cache"),
        kd_cfg=job.get("kd_cfg"))


if __name__ == "__main__":
    main()
