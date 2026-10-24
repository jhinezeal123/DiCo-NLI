"""Fuse two independent view runs (seq-cls + LM-token readout) and score the result.

Each view directory holds `seed<seed>/probs.npz` with `orig` and `swap` dev probabilities.
Per view: average the seeds' bidirectional probability vectors. Across views: arithmetic
mean of those vectors. The NEG decision weight (chosen on train OOF, never on dev) is then
applied before argmax, exactly as in the single-view pipeline.
"""
import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# Keep the fusion free of torch/transformers imports: only label bookkeeping is needed.
LABELS = ["BACKWARD_ENTAILMENT", "EQUIVALENCE", "FORWARD_ENTAILMENT", "NEGATIVE_OTHER"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
REV = [LABEL2ID["FORWARD_ENTAILMENT"], LABEL2ID["EQUIVALENCE"], LABEL2ID["BACKWARD_ENTAILMENT"],
       LABEL2ID["NEGATIVE_OTHER"]]


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def section(title):
    print(f"\n===== {title} =====", flush=True)


def view_probs(view_dir, seeds):
    """Per-view seed-mean bidirectional probabilities, plus the per-seed vectors."""
    per_seed = []
    ids = None
    for seed in seeds:
        z = np.load(Path(view_dir) / f"seed{seed}" / "probs.npz")
        p, q = z["orig"], z["swap"]
        v = 0.5 * (p + q[:, REV])
        per_seed.append(v)
        if ids is None:
            ids = list(range(len(v)))
    return np.mean(per_seed, axis=0), per_seed


def write_pred(path, dev_rows, probs, neg_weight=None):
    w = np.ones(len(LABELS))
    if neg_weight is not None:
        w[LABEL2ID["NEGATIVE_OTHER"]] = float(neg_weight)
    probs = probs * w
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(["instance_id", "label"])
        for r, pr in zip(dev_rows, probs):
            wr.writerow([r["instance_id"], LABELS[int(np.argmax(pr))]])
    return path


def validate(pred_path, dev_rows):
    section("SUBMISSION VALIDATION")
    with open(pred_path, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    rows = read_csv(pred_path)
    ids = [r["instance_id"] for r in rows]
    assert header == ["instance_id", "label"], header
    assert len(ids) == len(set(ids)), "duplicate ids"
    assert set(ids) == {r["instance_id"] for r in dev_rows}, "id set mismatch"
    assert all(r["label"] in LABELS for r in rows), "invalid label"
    print(f"OK rows={len(rows)} pred_dist={dict(sorted(Counter(r['label'] for r in rows).items()))}")


def official_score(upstream, reference_path, pred_path, out_dir):
    section("OFFICIAL SCORER")
    subprocess.run([sys.executable, "-m", "evaluation_functions", "--gold", str(reference_path),
                    "--predictions", str(pred_path), "--output-dir", str(out_dir)],
                   cwd=upstream, check=True, stdout=subprocess.DEVNULL)
    print((Path(out_dir) / "scores.txt").read_text())
    return json.loads((Path(out_dir) / "scores.json").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="primary config (track/seed list/neg weight)")
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", nargs="+", required=True, help="view output directories")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    upstream = Path(args.upstream).resolve()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t = f"track{cfg['track']}"
    data = upstream / "final_data"
    dev_path = data / "dev" / f"dico_nli_dev_{t}_participant_labeled.csv"
    ref_path = data / "dev" / f"dico_nli_dev_{t}_reference.csv"
    dev_rows = read_csv(dev_path)

    section("FUSED CONFIG")
    print(json.dumps({"views": args.views, "seeds": cfg["seeds"], "neg_weight": cfg.get("neg_weight"),
                      "fusion": "arithmetic mean of per-view seed-mean bidirectional probs"}, indent=1))

    member_probs, singles = [], {}
    for vdir in args.views:
        vp, per_seed = view_probs(vdir, cfg["seeds"])
        member_probs.append(vp)
        singles[vdir] = per_seed
        pred = write_pred(out / f"single_{Path(vdir).name}.csv", dev_rows, vp, cfg.get("neg_weight"))
        validate(pred, dev_rows)
        s = official_score(upstream, ref_path, pred.resolve(), (out / f"scores_single_{Path(vdir).name}").resolve())
        print(f"VIEW_METRICS view={Path(vdir).name} weighted_f1={s['weighted_f1']:.4f} "
              f"soft_cons={s['soft_cons']:.4f} hard_cons={s['hard_cons']:.4f}")

    fused = np.mean(member_probs, axis=0)
    section("FUSED DECODE")
    results = {}
    for mode, w in (("bidir", None), ("bidir_negw", cfg.get("neg_weight"))):
        pred = write_pred(out / f"{t}_dev_predictions_{mode}.csv", dev_rows, fused, w)
        validate(pred, dev_rows)
        s = official_score(upstream, ref_path, pred.resolve(), (out / f"scores_{mode}").resolve())
        results[mode] = s
        tag = "FUSED_METRICS" if mode == cfg.get("decode", "bidir_negw") else "FUSED_DIAG"
        print(f"{tag} decoding={mode} n_views={len(args.views)} n_models={len(args.views) * len(cfg['seeds'])} "
              f"weighted_f1={s['weighted_f1']:.4f} soft_cons={s['soft_cons']:.4f} hard_cons={s['hard_cons']:.4f}",
              flush=True)

    # diagnostic: mean of per-seed vectors across both views (12 models) vs seed-mean-then-fuse
    section("DECOMPOSITION")
    print("Each view's seed-mean is fused. View X seed counts:")
    for vdir, per_seed in singles.items():
        print(f"  {Path(vdir).name}: {len(per_seed)} seeds")
    headline = results[cfg.get("decode", "bidir_negw")]
    print(f"FINAL_FUSED weighted_f1={headline['weighted_f1']:.4f} soft_cons={headline['soft_cons']:.4f} "
          f"hard_cons={headline['hard_cons']:.4f}")


if __name__ == "__main__":
    main()
