"""VUAR pipeline modes, dispatched from run_pipeline.py by cfg["mode"].

vuar_teacher: label-conditioned rationalization of train originals + zero-shot diagnostics.
"""
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import vuar  # noqa: E402

LABELS = ["BACKWARD_ENTAILMENT", "EQUIVALENCE", "FORWARD_ENTAILMENT", "NEGATIVE_OTHER"]


def clean(s):
    return re.sub(r"\s+", " ", s).strip()


def run_workers(tasks, n_gpu):
    """tasks: [(name, argv, log_path)]; at most one live process per GPU (CUDA_VISIBLE_DEVICES)."""
    pending, running, free = list(tasks), {}, list(range(n_gpu))
    while pending or running:
        while pending and free:
            name, argv, log_path = pending.pop(0)
            gpu = free.pop(0)
            log = open(log_path, "w")
            proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
            running[name] = (proc, gpu, log, log_path, time.time())
            print(f"launched {name} on gpu={gpu}", flush=True)
        time.sleep(5)
        for name in list(running):
            proc, gpu, log, log_path, t0 = running[name]
            if proc.poll() is None:
                continue
            log.close()
            del running[name]
            free.append(gpu)
            print(f"{name} finished rc={proc.returncode} gpu={gpu} wall={time.time() - t0:.0f}s", flush=True)
            if proc.returncode != 0:
                print(Path(log_path).read_text(errors="replace")[-3500:])
                for p2, *_ in running.values():
                    p2.kill()
                sys.exit(f"worker {name} failed")


def teacher(cfg, config_path, upstream, paths, out_root, rp):
    from sklearn.metrics import f1_score
    train_rows, dev_rows = rp.read_csv(paths["train"]), rp.read_csv(paths["dev"])
    originals = [r for r in train_rows if r["instance_id"].endswith("__original")]
    mk = lambda r, gid, group, gold: dict(id=gid, group=group, t1=clean(r["text1"]), t2=clean(r["text2"]),
                                          gold=gold, ref=r["label"])
    jobs = [mk(r, r["instance_id"], "train_cond", r["label"]) for r in originals]
    sub = random.Random(0).sample(originals, min(int(cfg.get("teacher_zero_shot_train", 300)), len(originals)))
    jobs += [mk(r, r["instance_id"] + "#zs", "train_zero", None) for r in sub]
    jobs += [mk(r, r["instance_id"], "dev_zero", None) for r in dev_rows]
    print(f"jobs: {dict(Counter(j['group'] for j in jobs))}", flush=True)
    jobs_path = out_root / "jobs.json"
    jobs_path.write_text(json.dumps(jobs, ensure_ascii=False), encoding="utf-8")

    section = rp.section
    section("TEACHER (shards in parallel, one per GPU)")
    n = rp.gpu_count()
    t0 = time.time()
    worker = Path(__file__).resolve().parent / "teacher_worker.py"
    run_workers([(f"shard{i}", [sys.executable, "-u", str(worker), "--config", str(config_path),
                                "--jobs", str(jobs_path), "--shard", str(i), "--nshards", str(n),
                                "--out", str(out_root / f"shard{i}.jsonl")],
                  out_root / f"shard{i}.log") for i in range(n)], n)
    print(f"teacher_wall_seconds={time.time() - t0:.0f}")
    for i in range(n):
        print(f"--- shard{i} log tail ---")
        print("\n".join((out_root / f"shard{i}.log").read_text(errors="replace").splitlines()[-12:]))
    res = [json.loads(line) for i in range(n)
           for line in (out_root / f"shard{i}.jsonl").read_text(encoding="utf-8").splitlines()]
    by_id = {j["id"]: j for j in jobs}

    section("DSL FIT: label-conditioned rationalization of train originals")
    cond = [r for r in res if r["group"] == "train_cond"]
    first = sum(1 for r in cond if r["accepted"] and r["attempts"] == 1)
    acc = sum(1 for r in cond if r["accepted"])
    print(f"train originals={len(cond)} accepted_greedy={first} ({first / len(cond):.1%}) "
          f"accepted_after_retries={acc} ({acc / len(cond):.1%})")
    for lab in LABELS:
        rows = [r for r in cond if r["gold"] == lab]
        if not rows:
            continue
        rels = Counter(rel for r in rows if r["accepted"] for _, _, rel in vuar.parse(r["units"]))
        loose_conf = sum(1 for r in rows if r["accepted"]
                         and any(rel in ("LOOSE", "CONF") for _, _, rel in vuar.parse(r["units"])))
        print(f"  {lab:20s} n={len(rows):4d} accepted={sum(r['accepted'] for r in rows) / len(rows):.1%} "
              f"greedy={sum(1 for r in rows if r['accepted'] and r['attempts'] == 1) / len(rows):.1%} "
              f"rels={dict(rels.most_common())} traces_with_LOOSE_or_CONF={loose_conf}")

    section("ZERO-SHOT (label-free) teacher traces: accuracy of verifier-composed label")
    zero = {}
    for group in ("train_zero", "dev_zero"):
        rows = [r for r in res if r["group"] == group]
        pred = [r["label"] or "NEGATIVE_OTHER" for r in rows]
        ref = [r["ref"] for r in rows]
        wf1 = f1_score(ref, pred, average="weighted")
        accu = sum(p == g for p, g in zip(pred, ref)) / len(rows)
        zero[group] = (accu, wf1)
        print(f"{group}: n={len(rows)} unparsable={sum(1 for r in rows if not r['label'])} "
              f"ungrounded={sum(1 for r in rows if not r['grounded'])} accuracy={accu:.4f} weighted_f1={wf1:.4f} "
              f"pred_dist={dict(Counter(pred))}")
    dev = {r["id"]: r for r in res if r["group"] == "dev_zero"}
    pred_path = out_root / "dev_zero_shot_predictions.csv"
    with open(pred_path, "w", encoding="utf-8", newline="") as f:
        f.write("instance_id,label\n")
        for r in dev_rows:
            f.write(f"{r['instance_id']},{dev[r['instance_id']]['label'] or 'NEGATIVE_OTHER'}\n")
    rp.validate(pred_path, dev_rows)
    scores = rp.official_score(upstream, paths["dev_reference"], pred_path.resolve(),
                               (out_root / "scores_dev_zero").resolve())
    pairs = {}
    for r in dev_rows:
        pairs.setdefault(r["pair_id"], []).append(r["instance_id"])
    both = [v for v in pairs.values() if len(v) == 2]
    mirror_ok = sum(1 for a, b in both
                    if dev[b]["label"] == vuar.LABEL_REV.get(dev[a]["label"] or "", None))
    print(f"zero-shot mirror consistency on dev reversible pairs: {mirror_ok}/{len(both)} = {mirror_ok / len(both):.1%}")

    section("ARTIFACTS (copied from log)")
    print("<<<TRACES_JSONL")
    for r in cond:
        if r["accepted"]:
            j = by_id[r["id"]]
            print(json.dumps({"id": r["id"], "t1": j["t1"], "t2": j["t2"], "label": r["gold"],
                              "units": r["units"]}, ensure_ascii=False))
    print("TRACES_JSONL>>>")
    print("<<<ZERO_SHOT_JSONL")
    for r in res:
        if r["group"] != "train_cond":
            print(json.dumps({k: r[k] for k in ("id", "group", "ref", "label", "grounded", "units")},
                             ensure_ascii=False))
    print("ZERO_SHOT_JSONL>>>")
    section("SUMMARY")
    print(f"FINAL_METRICS mode=vuar_teacher train_cond_accept={acc / len(cond):.4f} "
          f"train_cond_accept_greedy={first / len(cond):.4f} "
          f"dev_zero_shot_weighted_f1={scores['weighted_f1']:.4f} soft_cons={scores['soft_cons']:.4f} "
          f"hard_cons={scores['hard_cons']:.4f} train_zero_shot_weighted_f1={zero['train_zero'][1]:.4f}")


def run(mode, cfg, config_path, upstream, paths, out_root, rp):
    if mode == "vuar_teacher":
        return teacher(cfg, config_path, upstream, paths, out_root, rp)
    raise SystemExit(f"unknown mode {mode}")
