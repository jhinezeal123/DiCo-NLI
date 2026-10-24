"""Freeze candidate labels from persisted train OOF predictions; never read dev gold."""
import argparse
import csv
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

LABELS = ["BACKWARD_ENTAILMENT", "EQUIVALENCE", "FORWARD_ENTAILMENT", "NEGATIVE_OTHER"]
REV = {"BACKWARD_ENTAILMENT": "FORWARD_ENTAILMENT", "FORWARD_ENTAILMENT": "BACKWARD_ENTAILMENT",
       "EQUIVALENCE": "EQUIVALENCE", "NEGATIVE_OTHER": "NEGATIVE_OTHER"}
OOF_RUN = "81fc6753-4916-40ae-950d-195e6ba11157"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof-log", type=Path, required=True)
    ap.add_argument("--train", type=Path, required=True)
    args = ap.parse_args()
    raw = args.oof_log.read_bytes()
    block = re.search(r"<<<OOF_PROBS\r?\n(.*?)\r?\nOOF_PROBS>>>", raw.decode("utf-8", errors="replace"), re.S).group(1)
    oof = {r["instance_id"]: r for r in csv.DictReader(io.StringIO(block))}
    with args.train.open(encoding="utf-8-sig", newline="") as f:
        train = list(csv.DictReader(f))
    assert set(oof) == {r["instance_id"] for r in train}
    groups = defaultdict(list)
    for row in train:
        pred = oof[row["instance_id"]]
        assert pred["label"] == row["label"]
        p = {l: 0.5 * (float(pred["p_" + l]) + float(pred["q_" + REV[l]])) for l in LABELS}
        groups[row["pair_id"]].append((row, p))
    categories = defaultdict(list)
    for pair, rows in groups.items():
        labels = {r["label"] for r, _ in rows}
        kind = "EQ" if labels == {"EQUIVALENCE"} else "NO" if labels == {"NEGATIVE_OTHER"} else "DIR"
        assert len(rows) == (1 if kind == "NO" else 2)
        confidence = sum(p[r["label"]] for r, p in rows) / len(rows)
        categories[kind].append((confidence, pair, rows))
    counts = {"EQ": 20, "DIR": 44, "NO": 24}
    selected = []
    for kind, limit in counts.items():
        for _, pair, rows in sorted(categories[kind], key=lambda x: (x[0], x[1]))[:limit]:
            for row, p in rows:
                alternative = max(p, key=p.get)
                assert alternative != row["label"], "Candidate selection must use disagreeing rows"
                selected.append({"instance_id": row["instance_id"], "pair_id": pair,
                                 "gold_label": row["label"], "alternative_label": alternative,
                                 "p_gold": p[row["label"]], "p_alternative": p[alternative],
                                 "oof_source_run": OOF_RUN})
    assert len(selected) == len({r["instance_id"] for r in selected}) == 152
    target_groups = defaultdict(list)
    for row in selected:
        target_groups[row["pair_id"]].append({row["gold_label"], row["alternative_label"]})
    for masks in target_groups.values():
        if len(masks) == 2:
            assert {REV[l] for l in masks[0]} == masks[1]
    dest = Path("configs/partial_candidates.csv")
    with dest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(selected[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(selected, key=lambda r: r["instance_id"]))
    manifest = {"source_oof_run": OOF_RUN, "source_oof_commit": "807f7fd",
                "source_log_sha256": hashlib.sha256(raw).hexdigest(),
                "train_lf_sha256": hashlib.sha256(args.train.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
                "candidates_sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
                "rows": len(selected), "source_pairs": len(target_groups),
                "gold_counts": dict(Counter(r["gold_label"] for r in selected)),
                "mean_p_gold": sum(r["p_gold"] for r in selected) / len(selected),
                "all_gold_retained": True, "train_reversal_closed": True,
                "dev_gold_read": False, "human_adjudicated": False}
    Path("configs/partial_candidates_provenance.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
