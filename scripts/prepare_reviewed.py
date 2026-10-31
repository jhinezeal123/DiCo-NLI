"""Build AI-reviewed paraphrase or matched source-repetition training CSV.

The review is a single, non-blinded AI judgment, not independent human gold.
The selected source IDs and replacements are fixed in the committed manifest.
Only participant dev texts are read for exact-pair leakage checks.
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def read_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def norm(text):
    return " ".join(str(text).casefold().split())


def pair_key(row):
    return tuple(sorted((norm(row["text1"]), norm(row["text2"]))))


def prepare(cfg, train_path, dev_path, output_dir):
    train, dev = read_rows(train_path), read_rows(dev_path)
    selected = read_rows(cfg["reviewed_manifest"])
    if len(selected) != cfg["expected_augmented_rows"]:
        raise ValueError(f"Unexpected reviewed row count {len(selected)}")
    lookup = {r["instance_id"]: r for r in train}
    if len(lookup) != len(train):
        raise ValueError("Original train instance IDs are not unique")
    if len({m["source_id"] for m in selected}) != len(selected):
        raise ValueError("Reviewed manifest contains duplicate source IDs")
    dev_pairs = {pair_key(r) for r in dev}
    augmented = []
    for i, m in enumerate(selected):
        original = lookup[m["source_id"]]
        if (original["pair_id"] != m["source_pair"] or original["label"] != m["label"]
                or m["side"] not in ("text1", "text2")
                or norm(original[m["side"]]) != norm(m["original"])):
            raise ValueError(f"Reviewed source mismatch for {m['source_id']}")
        revised = dict(original)
        revised[m["side"]] = m["paraphrase"]
        if pair_key(original) in dev_pairs or pair_key(revised) in dev_pairs:
            raise ValueError(f"Reviewed source or paraphrase overlaps dev: {m['source_id']}")
        if cfg["augmentation_mode"] == "paraphrase":
            extra = revised
        elif cfg["augmentation_mode"] == "duplicate":
            extra = dict(original)
        else:
            raise ValueError("Invalid augmentation_mode")
        extra["instance_id"] = f"ai_reviewed_aug_{i:05d}"
        extra["pair_id"] = f"ai_reviewed_pair_{i:05d}"
        augmented.append(extra)
    digest = hashlib.sha256("\n".join(m["source_id"] for m in selected).encode()).hexdigest()
    labels = dict(sorted(Counter(r["label"] for r in augmented).items()))
    print(f"AI_REVIEW_AUGMENT_SUMMARY mode={cfg['augmentation_mode']} rows={len(selected)} classes={labels} source_hash={digest}", flush=True)
    print("AI_REVIEW_SELECTED_IDS_JSON " + json.dumps([m["source_id"] for m in selected]), flush=True)
    output = Path(output_dir) / "train_augmented.csv"
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(train[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(train + augmented)
    return output
