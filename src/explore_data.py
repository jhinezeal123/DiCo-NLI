"""Input-only leakage protection and serialization for the exploration pilots."""
import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

LABELS = ["BACKWARD_ENTAILMENT", "EQUIVALENCE", "FORWARD_ENTAILMENT", "NEGATIVE_OTHER"]
REV = [2, 1, 0, 3]


def normalized_pair(row):
    clean = lambda s: re.sub(r"\s+", " ", str(s)).strip().casefold()
    return tuple(sorted((clean(row["text1"]), clean(row["text2"]))))


def without_labels(rows):
    # Deliberately construct an allowlist: no reference/reverse-pair metadata reaches a model.
    return [{k: r[k] for k in ("instance_id", "pair_id", "text1", "text2")} for r in rows]


def clean_train(train, evaluation):
    eval_text = {normalized_pair(r) for r in evaluation}
    eval_groups = {r["pair_id"] for r in evaluation}
    removed = {r["pair_id"] for r in train
               if normalized_pair(r) in eval_text or r["pair_id"] in eval_groups}
    kept = [copy.deepcopy(r) for r in train if r["pair_id"] not in removed]
    assert not ({normalized_pair(r) for r in kept} & eval_text)
    return kept, {"original_train_rows": len(train), "cleaned_train_rows": len(kept),
                  "removed_rows": len(train) - len(kept), "removed_groups": sorted(removed),
                  "remaining_unordered_text_overlaps": 0,
                  "cleaned_label_counts": dict(Counter(r["label"] for r in kept))}


def dump_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def validate_prepared(train, evaluation):
    assert train and evaluation
    assert all(r.get("label") in LABELS for r in train)
    assert all("label" not in r for r in evaluation), "evaluation gold reached worker"
    assert len({r["instance_id"] for r in train}) == len(train), "duplicate training ids"
    assert len({r["instance_id"] for r in evaluation}) == len(evaluation)
    assert not ({normalized_pair(r) for r in train} & {normalized_pair(r) for r in evaluation})
