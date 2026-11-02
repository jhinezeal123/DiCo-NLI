"""Transductive self-training on the evaluation inputs.

Stage 1: train the frozen recipe on train only and predict the eval split.
Pseudo-label: for each eval row, take the seed-ensemble bidirectional probability vector,
apply the NEG decision weight (chosen on train OOF), and keep rows whose normalised max
probability reaches `threshold` with the argmax label. No gold labels are used.
Stage 2: retrain the same seeds on train + pseudo rows and predict the eval split again.
The official scorer is the only consumer of gold labels.
"""
import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_pipeline as rp  # noqa: E402

LABELS = ["BACKWARD_ENTAILMENT", "EQUIVALENCE", "FORWARD_ENTAILMENT", "NEGATIVE_OTHER"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
REV = [LABEL2ID["FORWARD_ENTAILMENT"], LABEL2ID["EQUIVALENCE"], LABEL2ID["BACKWARD_ENTAILMENT"],
       LABEL2ID["NEGATIVE_OTHER"]]


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def ensemble_probs(out_root, seeds):
    probs = []
    for seed in seeds:
        z = np.load(Path(out_root) / f"seed{seed}" / "probs.npz")
        probs.append(0.5 * (z["orig"] + z["swap"][:, REV]))
    return np.mean(probs, axis=0)


def unit(probs, neg_weight):
    w = np.ones(4)
    w[LABEL2ID["NEGATIVE_OTHER"]] = float(neg_weight)
    p = probs * w
    return p / p.sum(1, keepdims=True)


def pseudo_rows(dev_rows, probs, neg_weight, threshold):
    p = unit(probs, neg_weight)
    kept = []
    for row, pr in zip(dev_rows, p):
        if float(np.max(pr)) >= threshold:
            kept.append({**row, "label": LABELS[int(np.argmax(pr))]})
    return kept, p


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

    rp.section("CONFIG")
    print(json.dumps(cfg, indent=1))
    train_rows, dev_rows = rp.read_csv(paths["train"]), rp.read_csv(paths["dev"])
    ref_rows = rp.read_csv(paths["dev_reference"])
    rp.audit(train_rows, dev_rows, paths)
    rp.environment(cfg["model_id"])

    st = cfg.get("self_train", {})
    threshold = float(st.get("threshold", 0.9))
    neg_weight = float(cfg.get("neg_weight", 1.0))

    rp.section("STAGE 1: train on train only, predict eval inputs")
    (out_root / "stage1").mkdir()
    rp.train_seeds_parallel(args.config, cfg["seeds"], paths, out_root / "stage1")
    probs1 = ensemble_probs(out_root / "stage1", cfg["seeds"])

    rp.section("PSEUDO-LABEL")
    pseudo, p1 = pseudo_rows(dev_rows, probs1, neg_weight, threshold)
    dist = Counter(r["label"] for r in pseudo)
    print(f"threshold={threshold} kept={len(pseudo)}/{len(dev_rows)} label_dist={dict(sorted(dist.items()))}")
    gold = {r["instance_id"]: r["label"] for r in ref_rows}
    agree = sum(r["label"] == gold[r["instance_id"]] for r in pseudo)
    print(f"PSEUDO_DIAG threshold={threshold} n={len(pseudo)} agreement_with_gold={agree / max(1, len(pseudo)):.4f}")

    fields = list(train_rows[0].keys())
    aug_rows = list(train_rows) + [{k: r.get(k, "") for k in fields} for r in pseudo]
    aug_path = out_root / "train_plus_pseudo.csv"
    write_csv(aug_path, aug_rows, fields)
    print(f"augmented_train_rows={len(aug_rows)} -> {aug_path}")

    rp.section("STAGE 2: retrain on train + pseudo rows")
    cfg2 = {**cfg, "name": cfg["name"] + "_stage2"}
    cfg2_path = out_root / "config_stage2.json"
    cfg2_path.write_text(json.dumps(cfg2, indent=2) + "\n")
    paths2 = {"train": aug_path, "dev": paths["dev"], "dev_reference": paths["dev_reference"]}
    (out_root / "stage2").mkdir()
    rp.train_seeds_parallel(cfg2_path, cfg["seeds"], paths2, out_root / "stage2")
    probs2 = ensemble_probs(out_root / "stage2", cfg["seeds"])

    rp.section("FINAL PREDICTIONS AND SCORES")
    headline = cfg.get("decode", "bidir_negw")
    results = {}
    for stage, probs in (("stage1", probs1), ("stage2", probs2)):
        for mode, negw in (("bidir", 1.0), ("bidir_negw", neg_weight)):
            p = unit(probs, negw)
            pred_path = out_root / f"{t}_dev_predictions_{stage}_{mode}.csv"
            with open(pred_path, "w", newline="", encoding="utf-8") as f:
                wr = csv.writer(f, lineterminator="\n")
                wr.writerow(["instance_id", "label"])
                for r, pr in zip(dev_rows, p):
                    wr.writerow([r["instance_id"], LABELS[int(np.argmax(pr))]])
            rp.validate(pred_path, dev_rows)
            sc = rp.official_score(upstream, paths["dev_reference"], pred_path.resolve(),
                                   (out_root / f"scores_{stage}_{mode}").resolve())
            results[(stage, mode)] = sc
            tag = "SELFTRAIN_METRICS" if (stage == "stage2" and mode == headline) else "DIAG"
            print(f"{tag} stage={stage} decoding={mode} n_pseudo={len(pseudo)} "
                  f"weighted_f1={sc['weighted_f1']:.4f} soft_cons={sc['soft_cons']:.4f} "
                  f"hard_cons={sc['hard_cons']:.4f}", flush=True)

    p2 = unit(probs2, neg_weight if headline == "bidir_negw" else 1.0)
    pred2 = {r["instance_id"]: LABELS[int(np.argmax(pr))] for r, pr in zip(dev_rows, p2)}
    rp.error_analysis(pred2, ref_rows)


if __name__ == "__main__":
    main()
