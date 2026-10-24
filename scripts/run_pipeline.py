"""Track baseline pipeline: audit data -> official starter kit train (train only)
-> predict dev -> validate submission -> official scorer -> compact summary.

The dev split is only used for prediction/scoring with a fixed epoch budget;
no checkpoint or hyperparameter is selected on it here.
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

LABELS = ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"]
REVERSE = {
    "EQUIVALENCE": "EQUIVALENCE",
    "FORWARD_ENTAILMENT": "BACKWARD_ENTAILMENT",
    "BACKWARD_ENTAILMENT": "FORWARD_ENTAILMENT",
}


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def section(title):
    print(f"\n===== {title} =====", flush=True)


def audit(train_rows, dev_rows, paths):
    section("DATA AUDIT")
    for name, p in paths.items():
        print(f"{name}: {p} sha256={sha256(p)}")
    for split, rows in (("train", train_rows), ("dev", dev_rows)):
        ids = [r["instance_id"] for r in rows]
        dist = Counter(r["label"] for r in rows)
        bad = set(dist) - set(LABELS)
        print(f"{split}: rows={len(rows)} unique_ids={len(set(ids))} pairs={len({r['pair_id'] for r in rows})} "
              f"labels={dict(sorted(dist.items()))}")
        assert len(ids) == len(set(ids)), f"duplicate instance_id in {split}"
        assert not bad, f"unknown labels in {split}: {bad}"
    train_pairs = {r["pair_id"] for r in train_rows}
    dev_pairs = {r["pair_id"] for r in dev_rows}
    norm = lambda r: frozenset((r["text1"].strip().lower(), r["text2"].strip().lower()))
    text_overlap = {norm(r) for r in train_rows} & {norm(r) for r in dev_rows}
    print(f"pair_id overlap train/dev: {len(train_pairs & dev_pairs)}")
    print(f"unordered normalized text-pair overlap train/dev: {len(text_overlap)}")
    assert not (train_pairs & dev_pairs), "pair_id leakage between train and dev"


def environment(model_id, revision=None):
    section("ENVIRONMENT")
    import numpy
    import pandas
    import sklearn
    import torch
    import transformers
    print(f"python={sys.version.split()[0]} torch={torch.__version__} transformers={transformers.__version__} "
          f"numpy={numpy.__version__} pandas={pandas.__version__} sklearn={sklearn.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}"
          + (f" gpu={torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
    try:
        from huggingface_hub import model_info
        print(f"model_id={model_id} model_revision={model_info(model_id, revision=revision).sha}")
    except Exception as exc:  # network/auth hiccups must not block the run
        print(f"model_id={model_id} model_revision=unavailable ({exc.__class__.__name__})")


def gpu_count():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
        return max(1, sum(1 for line in out.splitlines() if line.startswith("GPU ")))
    except (OSError, subprocess.CalledProcessError):
        return 1


def train_seeds_parallel(config_path, seeds, paths, out_root):
    """Train each seed in its own process pinned to one GPU; at most one live job per GPU."""
    import numpy as np
    n_gpu = gpu_count()
    print(f"gpus_available={n_gpu} seeds={list(seeds)}", flush=True)
    worker = Path(__file__).resolve().parent / "train_worker.py"
    pending, running, trained = list(seeds), {}, {}
    progress_lines = {}
    free_gpus = list(range(n_gpu))
    while pending or running:
        while pending and free_gpus:
            seed, gpu = pending.pop(0), free_gpus.pop(0)
            out = out_root / f"seed{seed}"
            out.mkdir()
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
            log = open(out / "train.log", "w")
            proc = subprocess.Popen(
                [sys.executable, "-u", str(worker), "--config", str(config_path), "--seed", str(seed),
                 "--train", str(paths["train"]), "--dev", str(paths["dev"]), "--out", str(out / "probs.npz")],
                stdout=log, stderr=subprocess.STDOUT, env=env)
            running[seed] = (proc, gpu, log, time.time())
            print(f"launched seed={seed} on gpu={gpu}", flush=True)
        time.sleep(5)
        for seed in list(running):
            proc, gpu, log, t0 = running[seed]
            lines = (out_root / f"seed{seed}" / "train.log").read_text(errors="replace").splitlines()
            for line in lines[progress_lines.get(seed, 0):]:
                if line.startswith(("TRAIN_SETUP", "TRAIN_PROGRESS", "Epoch ", "PARTIAL_CANDIDATES")):
                    print(f"WORKER_PROGRESS seed={seed} {line}", flush=True)
            progress_lines[seed] = len(lines)
            if proc.poll() is None:
                continue
            log.close()
            del running[seed]
            free_gpus.append(gpu)
            print(f"seed={seed} finished rc={proc.returncode} gpu={gpu} wall={time.time() - t0:.0f}s", flush=True)
            if proc.returncode != 0:
                print((out_root / f"seed{seed}" / "train.log").read_text(errors="replace")[-3000:])
                for p2, _, _, _ in running.values():
                    p2.kill()
                sys.exit(f"worker for seed {seed} failed")
            trained[seed] = dict(np.load(out_root / f"seed{seed}" / "probs.npz"))
    return trained


def validate(pred_path, dev_rows):
    section("SUBMISSION VALIDATION")
    with open(pred_path, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    rows = read_csv(pred_path)
    ids = [r["instance_id"] for r in rows]
    dev_ids = {r["instance_id"] for r in dev_rows}
    assert header == ["instance_id", "label"], header
    assert len(ids) == len(set(ids)), "duplicate ids"
    assert set(ids) == dev_ids, "id set mismatch"
    assert all(r["label"] in LABELS for r in rows), "invalid label"
    print(f"OK header={header} rows={len(rows)} pred_dist={dict(sorted(Counter(r['label'] for r in rows).items()))}")
    return {r["instance_id"]: r["label"] for r in rows}


def official_score(upstream, reference_path, pred_path, out_dir):
    section("OFFICIAL SCORER")
    subprocess.run([sys.executable, "-m", "evaluation_functions", "--gold", str(reference_path),
                    "--predictions", str(pred_path), "--output-dir", str(out_dir)],
                   cwd=upstream, check=True, stdout=subprocess.DEVNULL)
    print((out_dir / "scores.txt").read_text())
    scores = json.loads((out_dir / "scores.json").read_text())
    print("classification:", json.dumps(scores["classification"]))
    print("consistency_counts:", json.dumps({k: v for k, v in scores["consistency"].items()
                                              if not isinstance(v, (list, dict))}))
    return scores


def error_analysis(pred, ref_rows):
    section("ERROR ANALYSIS")
    gold = {r["instance_id"]: r["label"] for r in ref_rows}
    conf = Counter((gold[i], pred[i]) for i in gold)
    print("confusion (gold -> pred):")
    for g in LABELS:
        print(f"  {g:20s} " + " ".join(f"{p[:5]}={conf[(g, p)]:3d}" for p in LABELS))
    seen, buckets = set(), Counter()
    for r in ref_rows:
        a, b = r["instance_id"], r["reverse_pair_id"]
        if not b or (b, a) in seen:
            continue
        seen.add((a, b))
        g = gold[a]
        soft = pred[a] in REVERSE and REVERSE[pred[a]] == pred[b]
        hard = pred[a] == g and pred[b] == gold[b]
        kind = "EQUIV" if g == "EQUIVALENCE" else "ENTAIL"
        buckets[(kind, "pairs")] += 1
        buckets[(kind, "soft_ok")] += soft
        buckets[(kind, "hard_ok")] += hard
        buckets[(kind, "consistent_but_wrong")] += soft and not hard
    for kind in ("EQUIV", "ENTAIL"):
        n = buckets[(kind, "pairs")]
        print(f"  {kind}: pairs={n} soft_ok={buckets[(kind, 'soft_ok')]} hard_ok={buckets[(kind, 'hard_ok')]} "
              f"consistent_but_wrong={buckets[(kind, 'consistent_but_wrong')]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--upstream", required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    upstream = Path(args.upstream).resolve()
    t = f"track{cfg['track']}"
    data = upstream / "final_data"
    paths = {
        "train": data / "train" / f"dico_nli_train_{t}_participant_labeled.csv",
        "dev": data / "dev" / f"dico_nli_dev_{t}_participant_labeled.csv",
        "dev_reference": data / "dev" / f"dico_nli_dev_{t}_reference.csv",
    }
    out_root = Path("outputs") / cfg["name"]
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    section("CONFIG")
    print(json.dumps(cfg, indent=1))
    train_rows, dev_rows = read_csv(paths["train"]), read_csv(paths["dev"])
    ref_rows = read_csv(paths["dev_reference"])
    audit(train_rows, dev_rows, paths)
    environment(cfg["model_id"], cfg.get("model_revision"))

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import dico_trainer as dt
    import numpy as np
    if cfg.get("bidirectional_objective"):
        from bidirectional_objectives import validate_objectives
        validate_objectives()

    section("TRAIN (seeds in parallel, one per GPU)")
    t0 = time.time()
    trained = train_seeds_parallel(args.config, cfg["seeds"], paths, out_root)
    print(f"train_wall_seconds_all_seeds={time.time() - t0:.1f}")

    headline = cfg.get("decode", "standard")
    modes = ["standard", "bidir"]
    results = {m: {} for m in modes}
    ens_sum = {}
    for seed in cfg["seeds"]:
        section(f"SEED {seed}")
        out = out_root / f"seed{seed}"
        print((out / "train.log").read_text(errors="replace"), end="")
        p, q = trained[seed]["orig"], trained[seed]["swap"]
        # q is the prediction for the swapped input; map it back through the reversal.
        final = {"standard": p, "bidir": 0.5 * (p + q[:, dt.REV])}
        for mode in final:
            ens_sum[mode] = ens_sum.get(mode, 0) + final[mode]
        print(f"<<<DEV_PROBS seed={seed}")
        print("instance_id," + ",".join(f"p_{l}" for l in dt.LABELS) + "," + ",".join(f"q_{l}" for l in dt.LABELS))
        for r, pi, qi in zip(dev_rows, p, q):
            print(r["instance_id"] + "," + ",".join(f"{v:.6f}" for v in list(pi) + list(qi)))
        print("DEV_PROBS>>>")
        for mode in modes:
            pred_path = out / f"{t}_dev_predictions_{mode}.csv"
            with open(pred_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f, lineterminator="\n")
                w.writerow(["instance_id", "label"])
                for r, pr in zip(dev_rows, final[mode]):
                    w.writerow([r["instance_id"], dt.LABELS[int(np.argmax(pr))]])
            section(f"seed {seed} decoding={mode}")
            pred = validate(pred_path, dev_rows)
            scores = official_score(upstream, paths["dev_reference"], pred_path.resolve(),
                                    (out / f"scores_{mode}").resolve())
            if mode == headline:
                error_analysis(pred, ref_rows)
            print(f"<<<PREDICTIONS_CSV seed={seed} decoding={mode}")
            print(pred_path.read_text(), end="")
            print("PREDICTIONS_CSV>>>")
            results[mode][seed] = scores
            print(f"SEED_METRICS decoding={mode} seed={seed} weighted_f1={scores['weighted_f1']:.4f} "
                  f"soft_cons={scores['soft_cons']:.4f} hard_cons={scores['hard_cons']:.4f}", flush=True)

    section("SEED ENSEMBLE (diagnostic: mean probabilities over seeds)")
    ens_scores = {}
    for mode in modes:
        pred_path = out_root / f"{t}_dev_predictions_ensemble_{mode}.csv"
        with open(pred_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["instance_id", "label"])
            for r, pr in zip(dev_rows, ens_sum[mode]):
                w.writerow([r["instance_id"], dt.LABELS[int(np.argmax(pr))]])
        validate(pred_path, dev_rows)
        ens_scores[mode] = official_score(upstream, paths["dev_reference"], pred_path.resolve(),
                                          (out_root / f"scores_ensemble_{mode}").resolve())

    section("SUMMARY")
    for mode in modes:
        e = ens_scores[mode]
        print(f"ENSEMBLE_METRICS decoding={mode} n_seeds={len(cfg['seeds'])} weighted_f1={e['weighted_f1']:.4f} "
              f"soft_cons={e['soft_cons']:.4f} hard_cons={e['hard_cons']:.4f}")
    for mode in modes:
        tag = "FINAL_METRICS" if mode == headline else "DIAG_METRICS"
        agg = {}
        for m in ("weighted_f1", "soft_cons", "hard_cons"):
            vals = [r[m] for r in results[mode].values()]
            agg[m] = (statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0, min(vals))
        print(f"{tag} decoding={mode} split=dev track={cfg['track']} seeds={list(results[mode])} "
              + " ".join(f"{m}={a[0]:.4f}+-{a[1]:.4f}(min {a[2]:.4f})" for m, a in agg.items()))


if __name__ == "__main__":
    main()
