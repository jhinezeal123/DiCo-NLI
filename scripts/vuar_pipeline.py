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


def spans(cfg, config_path, upstream, paths, out_root, rp):
    """Stage 1b: deterministic alignment; the teacher only scores the relation of replaced spans."""
    import math
    from sklearn.metrics import f1_score
    train_rows, dev_rows = rp.read_csv(paths["train"]), rp.read_csv(paths["dev"])
    originals = [r for r in train_rows if r["instance_id"].endswith("__original")]
    items, jobs = [], []

    def add(group, row):
        t1, t2 = clean(row["text1"]), clean(row["text2"])
        units = vuar.diff_units(t1, t2)
        items.append((group, row["instance_id"], row["label"], t1, t2, units))
        for n, i in enumerate(vuar.rep_indices(units)):
            a, b, _ = units[i]
            key = f"{group}|{row['instance_id']}|{n}"
            jobs.append(dict(key=key, t1=t1, t2=t2, a=a, b=b))
            jobs.append(dict(key=key + "|sw", t1=t2, t2=t1, a=b, b=a))

    for r in originals:
        add("train", r)
    for r in dev_rows:
        add("dev", r)
    print(f"items: train={sum(1 for x in items if x[0] == 'train')} dev={sum(1 for x in items if x[0] == 'dev')} "
          f"span_queries={len(jobs)}", flush=True)
    jobs_path = out_root / "jobs.json"
    jobs_path.write_text(json.dumps(jobs, ensure_ascii=False), encoding="utf-8")

    section = rp.section
    section("TEACHER SPAN SCORING (shards in parallel, one per GPU)")
    n = rp.gpu_count()
    t0 = time.time()
    worker = Path(__file__).resolve().parent / "span_worker.py"
    run_workers([(f"shard{i}", [sys.executable, "-u", str(worker), "--config", str(config_path),
                                "--jobs", str(jobs_path), "--shard", str(i), "--nshards", str(n),
                                "--out", str(out_root / f"shard{i}.json")],
                  out_root / f"shard{i}.log") for i in range(n)], n)
    print(f"span_scoring_wall_seconds={time.time() - t0:.0f}")
    raw = {}
    for i in range(n):
        raw.update(json.loads((out_root / f"shard{i}.json").read_text(encoding="utf-8")))

    def rel_scores(group, rid, k):
        """Log-probs over relations, averaging the query with its mirrored (swapped) counterpart."""
        f, w = raw[f"{group}|{rid}|{k}"], raw[f"{group}|{rid}|{k}|sw"]
        out = {}
        for rel in vuar.SPAN_RELS:
            p = 0.5 * (math.exp(f[vuar.REL_LETTER[rel]]) + math.exp(w[vuar.REL_LETTER[vuar.MIRROR[rel]]]))
            out[rel] = math.log(max(p, 1e-9))
        return out

    def scores_for(item):
        group, rid, _, _, _, units = item
        return [rel_scores(group, rid, k) for k in range(len(vuar.rep_indices(units)))]

    uniform = lambda units: [{r: 0.0 for r in vuar.SPAN_RELS} for _ in vuar.rep_indices(units)]

    section("DSL FIT (deterministic alignment): label-conditioned relation assignment on train originals")
    tr_items = [x for x in items if x[0] == "train"]
    accepted, per_label = [], {}
    zero_pred, zero_ref = [], []
    for it in tr_items:
        group, rid, gold, t1, t2, units = it
        sc = scores_for(it)
        trace = vuar.best_assignment(units, sc, gold)
        per_label.setdefault(gold, Counter())["n"] += 1
        per_label[gold]["structural_feasible"] += int(vuar.best_assignment(units, uniform(units), gold) is not None)
        per_label[gold]["has_rep"] += int(bool(vuar.rep_indices(units)))
        if trace is not None:
            accepted.append((it, trace))
            per_label[gold]["accepted"] += 1
            for _, _, rel in trace:
                per_label[gold]["rel_" + rel] += 1
        z = vuar.compose(vuar.best_assignment(units, sc, None))
        per_label[gold]["zero_shot_correct"] += int(z == gold)
        zero_pred.append(z)
        zero_ref.append(gold)
    tot = len(tr_items)
    print(f"train originals={tot} accepted={len(accepted)} ({len(accepted) / tot:.1%})")
    for lab in LABELS:
        c = per_label.get(lab)
        if not c:
            continue
        rels = {k[4:]: v for k, v in c.items() if k.startswith("rel_")}
        print(f"  {lab:20s} n={c['n']:4d} feasible={c['structural_feasible'] / c['n']:.1%} "
              f"has_replaced_span={c['has_rep'] / c['n']:.1%} zero_shot_acc={c['zero_shot_correct'] / c['n']:.1%} "
              f"relations={dict(sorted(rels.items(), key=lambda kv: -kv[1]))}")
    tr_wf1 = f1_score(zero_ref, zero_pred, average="weighted")
    print(f"train zero-shot (teacher argmax, no labels) accuracy={sum(a == b for a, b in zip(zero_pred, zero_ref)) / tot:.4f} "
          f"weighted_f1={tr_wf1:.4f}")

    section("ZERO-SHOT on dev: deterministic alignment + teacher span relations, label composed by rule")
    dv_items = {x[1]: x for x in items if x[0] == "dev"}
    preds, feasible = {}, 0
    for rid, it in dv_items.items():
        z = vuar.compose(vuar.best_assignment(it[5], scores_for(it), None))
        preds[rid] = z
        feasible += int(vuar.best_assignment(it[5], uniform(it[5]), it[2]) is not None)
    print(f"dev rows whose gold label is reachable by the composition rule (structural ceiling): "
          f"{feasible}/{len(dv_items)} = {feasible / len(dv_items):.1%}")
    pred_path = out_root / "dev_zero_shot_predictions.csv"
    with open(pred_path, "w", encoding="utf-8", newline="") as f:
        f.write("instance_id,label\n")
        for r in dev_rows:
            f.write(f"{r['instance_id']},{preds[r['instance_id']]}\n")
    rp.validate(pred_path, dev_rows)
    scores = rp.official_score(upstream, paths["dev_reference"], pred_path.resolve(),
                               (out_root / "scores_dev_zero").resolve())

    section("ARTIFACTS (copied from log)")
    print("<<<TRACES_JSONL")
    for (group, rid, gold, t1, t2, units), trace in accepted:
        print(json.dumps({"id": rid, "t1": t1, "t2": t2, "label": gold, "units": vuar.format_units(trace)},
                         ensure_ascii=False))
    print("TRACES_JSONL>>>")
    print("<<<RAW_SCORES_JSON")
    print(json.dumps(raw))
    print("RAW_SCORES_JSON>>>")
    section("SUMMARY")
    print(f"FINAL_METRICS mode=vuar_spans train_accept={len(accepted) / tot:.4f} "
          f"dev_ceiling={feasible / len(dv_items):.4f} dev_zero_shot_weighted_f1={scores['weighted_f1']:.4f} "
          f"soft_cons={scores['soft_cons']:.4f} hard_cons={scores['hard_cons']:.4f} "
          f"train_zero_shot_weighted_f1={tr_wf1:.4f}")


def reconcile(orig, swap):
    """Final label from the trace of (t1,t2) and the trace of (t2,t1) (mirrored back).

    Returns (standard_label, bidir_label, mirror_consistent, orig_label, swap_label_mirrored).
    """
    l1 = vuar.compose(vuar.parse(orig[0]))
    l2 = vuar.compose(vuar.parse(swap[0]))
    l2m = vuar.LABEL_REV[l2] if l2 else None
    if l1 and l2m:
        bidir = l1 if (l1 == l2m or orig[1] >= swap[1]) else l2m
    else:
        bidir = l1 or l2m or "NEGATIVE_OTHER"
    return l1 or "NEGATIVE_OTHER", bidir, (l1 is not None and l1 == l2m), l1, l2m


def student(cfg, config_path, upstream, paths, out_root, rp):
    import statistics
    repo = Path(__file__).resolve().parents[1]
    traces_path = repo / cfg["traces_file"]
    dev_rows = rp.read_csv(paths["dev"])
    ref = {r["instance_id"]: r["label"] for r in dev_rows}
    section = rp.section
    section("STUDENT TRAIN + DECODE (seeds in parallel, one per GPU)")
    t0 = time.time()
    worker = Path(__file__).resolve().parent / "student_worker.py"
    seeds = cfg["seeds"]
    run_workers([(f"seed{sd}", [sys.executable, "-u", str(worker), "--config", str(config_path), "--seed", str(sd),
                                "--traces", str(traces_path), "--train", str(paths["train"]),
                                "--dev", str(paths["dev"]), "--out", str(out_root / f"seed{sd}.json")],
                  out_root / f"seed{sd}.log") for sd in seeds], rp.gpu_count())
    print(f"student_wall_seconds_all_seeds={time.time() - t0:.0f}")

    headline = cfg.get("decode", "bidir")
    modes = ["standard", "bidir"]
    results = {m: {} for m in modes}
    for sd in seeds:
        section(f"SEED {sd}")
        print((out_root / f"seed{sd}.log").read_text(errors="replace"), end="")
        gen = json.loads((out_root / f"seed{sd}.json").read_text(encoding="utf-8"))
        rec = [reconcile(o, w) for o, w in zip(gen["orig"], gen["swap"])]
        ids = [r["instance_id"] for r in dev_rows]
        n = len(ids)
        unparsable = sum(1 for r in rec if r[3] is None)
        agree = [i for i, r in enumerate(rec) if r[2]]
        dis = [i for i in range(n) if not rec[i][2]]
        acc = lambda sel, k: sum(rec[i][k] == ref[ids[i]] for i in sel) / max(1, len(sel))
        print(f"trace stats seed {sd}: unparsable_orig={unparsable}/{n} mirror_consistent={len(agree)}/{n} "
              f"acc_standard_on_consistent={acc(agree, 0):.3f} acc_bidir_on_inconsistent={acc(dis, 1):.3f} "
              f"acc_standard_on_inconsistent={acc(dis, 0):.3f}")
        print(f"<<<DEV_TRACES seed={sd}")
        print("\t".join(["instance_id", "label_orig", "label_swap_mirrored", "lp_orig", "lp_swap",
                         "trace_orig", "trace_swap"]))
        for i in range(n):
            print("\t".join([ids[i], str(rec[i][3]), str(rec[i][4]), f"{gen['orig'][i][1]:.4f}",
                             f"{gen['swap'][i][1]:.4f}", gen["orig"][i][0], gen["swap"][i][0]]))
        print("DEV_TRACES>>>")
        for mode, k in (("standard", 0), ("bidir", 1)):
            pred_path = out_root / f"seed{sd}_{mode}_predictions.csv"
            with open(pred_path, "w", encoding="utf-8", newline="") as f:
                f.write("instance_id,label\n")
                for i in range(n):
                    f.write(f"{ids[i]},{rec[i][k]}\n")
            rp.validate(pred_path, dev_rows)
            sc = rp.official_score(upstream, paths["dev_reference"], pred_path.resolve(),
                                   (out_root / f"seed{sd}_scores_{mode}").resolve())
            print(f"<<<PREDICTIONS_CSV seed={sd} decoding={mode}")
            print(pred_path.read_text(encoding="utf-8"), end="")
            print("PREDICTIONS_CSV>>>")
            results[mode][sd] = sc
            print(f"SEED_METRICS decoding={mode} seed={sd} weighted_f1={sc['weighted_f1']:.4f} "
                  f"soft_cons={sc['soft_cons']:.4f} hard_cons={sc['hard_cons']:.4f}", flush=True)
    section("SUMMARY")
    for mode in modes:
        tag = "FINAL_METRICS" if mode == headline else "DIAG_METRICS"
        agg = {}
        for m in ("weighted_f1", "soft_cons", "hard_cons"):
            vals = [r[m] for r in results[mode].values()]
            agg[m] = (statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0, min(vals))
        print(f"{tag} decoding={mode} split=dev track={cfg['track']} seeds={list(results[mode])} "
              + " ".join(f"{m}={a[0]:.4f}+-{a[1]:.4f}(min {a[2]:.4f})" for m, a in agg.items()))


def run(mode, cfg, config_path, upstream, paths, out_root, rp):
    if mode == "vuar_spans":
        return spans(cfg, config_path, upstream, paths, out_root, rp)
    if mode == "vuar_student":
        return student(cfg, config_path, upstream, paths, out_root, rp)
    if mode == "vuar_teacher":
        return teacher(cfg, config_path, upstream, paths, out_root, rp)
    raise SystemExit(f"unknown mode {mode}")
