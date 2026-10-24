"""Tracked evaluation of fixed OOF-trained boundary calibration variants."""
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from boundary_calibration import BoundaryCalibrator, LABELS, REV, normalize


def read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def key(row):
    return tuple(sorted((" ".join(row["text1"].lower().split()),
                         " ".join(row["text2"].lower().split()))))


def probabilities(path, ids):
    rows = {r["instance_id"]: r for r in read(path)}
    assert set(rows) == set(ids), f"Prediction ID mismatch: {path}"
    p = np.array([[float(rows[i][f"p_{l}"]) for l in LABELS] for i in ids])
    q = np.array([[float(rows[i][f"q_{l}"]) for l in LABELS] for i in ids])
    return normalize(0.5 * (p + q[:, REV]))


def run(cfg, upstream):
    sys.path.insert(0, str(upstream))
    from evaluation_functions import score_files
    from evaluation_functions.schemas import GoldInstance
    from evaluation_functions import compute_evaluation
    root = Path("configs/calibration")
    provenance = json.loads((root / "provenance.json").read_text())
    for filename, meta in provenance["files"].items():
        assert hashlib.sha256((root / filename).read_bytes()).hexdigest() == meta["sha256"]
    print("CONFIG", json.dumps(cfg, sort_keys=True), flush=True)
    print("PROVENANCE", json.dumps(provenance, sort_keys=True), flush=True)
    data = upstream / "final_data"
    train = read(data / "train/dico_nli_train_track1_participant_labeled.csv")
    # Only ID/text columns survive; dev labels cannot enter calibration fitting.
    dev = [{k: r[k] for k in ("instance_id", "pair_id", "text1", "text2")}
           for r in read(data / "dev/dico_nli_dev_track1_participant_labeled.csv")]
    assert not ({r["pair_id"] for r in train} & {r["pair_id"] for r in dev})
    oof = {r["instance_id"]: r for r in read(root / "train_oof.csv")}
    assert set(oof) == {r["instance_id"] for r in train}
    assert all(oof[r["instance_id"]]["label"] == r["label"] for r in train)
    dev_text_keys = {key(r) for r in dev}
    retained = [r for r in train if key(r) not in dev_text_keys]
    print(f"FIT_AUDIT all_train={len(train)} retained={len(retained)} "
          f"excluded_text_overlap={len(train)-len(retained)} dev={len(dev)} "
          "metadata_features=[] dev_labels_used_for_fit=False", flush=True)
    train_ids = [r["instance_id"] for r in train]
    all_p = probabilities(root / "train_oof.csv", train_ids)
    index = {i: j for j, i in enumerate(train_ids)}
    p = all_p[[index[r["instance_id"]] for r in retained]]
    texts = [(r["text1"], r["text2"]) for r in retained]
    labels = np.array([LABELS.index(r["label"]) for r in retained])
    cal = BoundaryCalibrator(cfg["calibration_mode"], cfg["calibration_l2"]).fit(texts, p, labels)
    print("FROZEN_CALIBRATOR", json.dumps(cal.diagnostics, sort_keys=True), flush=True)
    out = Path("outputs") / cfg["name"]
    out.mkdir(parents=True, exist_ok=False)
    np.savez(out / "calibrator.npz", weights=cal.weights, mean=cal.mean, scale=cal.scale)
    # Scoring reference is first opened after all fitting has finished.
    reference = data / "dev/dico_nli_dev_track1_reference.csv"
    dev_ids = [r["instance_id"] for r in dev]
    dev_texts = [(r["text1"], r["text2"]) for r in dev]
    ref_rows = read(reference)
    gold = {r["instance_id"]: GoldInstance(instance_id=r["instance_id"], pair_id=r["pair_id"],
             label=r["label"], reverse_pair_id=r["reverse_pair_id"] or None) for r in ref_rows}
    truth = {r["instance_id"]: r["label"] for r in ref_rows}
    clean_ids = {r["instance_id"] for r in dev if key(r) not in {key(t) for t in train}}
    result = {}
    for name in [f"seed{s}" for s in cfg["seeds"]] + ["fold_ensemble"]:
        if name == "fold_ensemble":
            base = np.mean([probabilities(root / f"dev_fold{j}.csv", dev_ids) for j in range(5)], axis=0)
        else:
            base = probabilities(root / f"dev_{name}.csv", dev_ids)
        calibrated = cal.predict_proba(dev_texts, base)
        backwards = cal.predict_proba([(b, a) for a, b in dev_texts], base[:, REV])[:, REV]
        assert np.allclose(calibrated, backwards, atol=1e-12), "Broken reversal equivariance"
        result[name] = {}
        preds = {}
        for mode, pr in (("parent", base), ("calibrated", calibrated)):
            path = out / f"{name}_{mode}.csv"
            pred = dict(zip(dev_ids, [LABELS[i] for i in pr.argmax(axis=1)]))
            preds[mode] = pred
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(["instance_id", "label"])
                writer.writerows(pred.items())
            scores = score_files(reference, path, output_dir=out / f"{name}_{mode}_scores").to_dict()
            result[name][mode] = scores
            print("SCORES", json.dumps({"sample": name, "mode": mode, **scores}, sort_keys=True), flush=True)
            clean_scores = compute_evaluation({i: gold[i] for i in clean_ids},
                                              {i: pred[i] for i in clean_ids}).to_dict()
            print("CLEAN_TEXT_OVERLAP_SCORES", json.dumps({"sample": name, "mode": mode,
                   "n": len(clean_ids), **{k: clean_scores[k] for k in ("weighted_f1", "soft_cons", "hard_cons")}}), flush=True)
            print(f"<<<CALIBRATION_PREDICTIONS sample={name} mode={mode}")
            print(path.read_text(), end="")
            print("CALIBRATION_PREDICTIONS>>>", flush=True)
        fixes = sum(preds["parent"][i] != truth[i] and preds["calibrated"][i] == truth[i] for i in truth)
        breaks = sum(preds["parent"][i] == truth[i] and preds["calibrated"][i] != truth[i] for i in truth)
        print("CHANGES", json.dumps({"sample": name, "fixed": fixes, "broken": breaks}), flush=True)
    means = {}
    for mode in ("parent", "calibrated"):
        means[mode] = {metric: statistics.mean(result[f"seed{s}"][mode][metric] for s in cfg["seeds"])
                       for metric in ("weighted_f1", "soft_cons", "hard_cons")}
        for label in ("NEGATIVE_OTHER", "EQUIVALENCE"):
            means[mode][label + "_f1"] = statistics.mean(result[f"seed{s}"][mode]["classification"]["per_label"][label]["f1"] for s in cfg["seeds"])
        print("FINAL_METRICS", json.dumps({"mode": mode, "n_seeds": len(cfg["seeds"]), **means[mode]}, sort_keys=True), flush=True)
    d = {k: means["calibrated"][k] - means["parent"][k] for k in means["parent"]}
    passed = (d["NEGATIVE_OTHER_f1"] >= 0.05 and d["EQUIVALENCE_f1"] >= -0.01
              and d["weighted_f1"] >= 0 and d["hard_cons"] >= 0)
    print("PREREGISTERED_GATE", json.dumps({"passed": passed, "deltas": d}, sort_keys=True), flush=True)
    (out / "summary.json").write_text(json.dumps({"results": result, "means": means, "deltas": d,
                                                  "gate_passed": passed}, indent=2) + "\n")
