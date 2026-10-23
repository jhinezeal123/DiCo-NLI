"""Track baseline pipeline: audit data -> official starter kit train (train only)
-> predict dev -> validate submission -> official scorer -> compact summary.

The dev split is only used for prediction/scoring with a fixed epoch budget;
no checkpoint or hyperparameter is selected on it here.
"""
import argparse
import csv
import hashlib
import json
import shutil
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


def environment(model_id):
    section("ENVIRONMENT")
    import torch
    import transformers
    print(f"python={sys.version.split()[0]} torch={torch.__version__} transformers={transformers.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}"
          + (f" gpu={torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
    try:
        from huggingface_hub import model_info
        print(f"model_id={model_id} model_revision={model_info(model_id).sha}")
    except Exception as exc:  # network/auth hiccups must not block the run
        print(f"model_id={model_id} model_revision=unavailable ({exc.__class__.__name__})")


def train_and_predict(cfg, upstream, dev_path, train_path, reference_path, results_dir):
    section("TRAIN (starter kit, train split only)")
    cmd = [
        sys.executable, "-u", "main.py",
        "--model_path", cfg["model_id"], "--tokenizer_path", cfg["model_id"],
        "--do_train", "--train_path", str(train_path),
        "--do_dev", "--dev_path", str(dev_path),
        "--no-do_test",
        "--seed", str(cfg["seed"]), "--dropout", str(cfg["dropout"]),
        "--warmup_pcrt", str(cfg["warmup_pcrt"]), "--lr", str(cfg["lr"]),
        "--batch_size", str(cfg["batch_size"]), "--wd", str(cfg["wd"]),
        "--num_epochs", str(cfg["num_epochs"]), "--no-is_optuna_trial",
        "--max_length", str(cfg["max_length"]),
        "--results_dir", str(results_dir), "--reference_path", str(reference_path),
    ]
    print("cmd:", " ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=upstream / "starter_kit", stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    per_example = 0
    for line in proc.stdout:
        # The kit prints one line per dev example each epoch; keep the log compact.
        if line.startswith("ID: "):
            per_example += 1
            continue
        print(line, end="", flush=True)
    if proc.wait() != 0:
        sys.exit(f"starter kit failed with exit code {proc.returncode}")
    print(f"train_wall_seconds={time.time() - t0:.1f} suppressed_per_example_lines={per_example}")
    preds = sorted(results_dir.glob("predictions_dev_*.csv"))
    assert len(preds) == cfg["num_epochs"], f"expected one dev prediction per epoch, got {len(preds)}"
    return preds[-1]  # final epoch: fixed budget, no selection on dev


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
    out = Path("outputs") / cfg["name"]
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    section("CONFIG")
    print(json.dumps(cfg, indent=1))
    train_rows, dev_rows = read_csv(paths["train"]), read_csv(paths["dev"])
    ref_rows = read_csv(paths["dev_reference"])
    audit(train_rows, dev_rows, paths)
    environment(cfg["model_id"])

    final_pred = train_and_predict(cfg, upstream, paths["dev"], paths["train"], paths["dev_reference"],
                                   (out / "kit_results").resolve())
    pred_path = out / f"{t}_dev_predictions.csv"
    shutil.copy(final_pred, pred_path)
    pred = validate(pred_path, dev_rows)
    scores = official_score(upstream, paths["dev_reference"], pred_path.resolve(), (out / "scores").resolve())
    error_analysis(pred, ref_rows)

    section("SUMMARY")
    print(f"FINAL_METRICS split=dev track={cfg['track']} weighted_f1={scores['weighted_f1']:.4f} "
          f"soft_cons={scores['soft_cons']:.4f} hard_cons={scores['hard_cons']:.4f}")


if __name__ == "__main__":
    main()
