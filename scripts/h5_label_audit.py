"""Read-only label audit using structural checks and a frozen NLI teacher.

The teacher is independent of the fine-tuned classifiers but is not a human
annotator. Teacher disagreement never establishes an incorrect gold label.
"""
import collections
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REV = {"EQUIVALENCE": "EQUIVALENCE", "FORWARD_ENTAILMENT": "BACKWARD_ENTAILMENT",
       "BACKWARD_ENTAILMENT": "FORWARD_ENTAILMENT"}


def norm(text):
    return " ".join(str(text).casefold().split())


def structural_audit(train_rows, dev_rows, reference_path):
    ordered, canonical, by_id = collections.defaultdict(list), collections.defaultdict(list), {}
    for split, rows in (("train", train_rows), ("dev", dev_rows)):
        for r in rows:
            a, b = norm(r["text1"]), norm(r["text2"])
            item = {"split": split, "id": r["instance_id"], "label": r["label"]}
            ordered[(a, b)].append(item)
            if r["label"] in REV:
                canonical[tuple(sorted((a, b)))].append(
                    dict(item, canonical_label=r["label"] if a <= b else REV[r["label"]]))
            if split == "dev":
                by_id[r["instance_id"]] = r
    exact_conflicts = [v for v in ordered.values() if len({x["label"] for x in v}) > 1]
    reverse_conflicts = [v for v in canonical.values() if len({x["canonical_label"] for x in v}) > 1]
    with open(reference_path, encoding="utf-8-sig", newline="") as handle:
        refs = {r["instance_id"]: r for r in csv.DictReader(handle)}
    errors, linked = [], set()
    for iid, ref in refs.items():
        rev_id = ref.get("reverse_pair_id", "").strip()
        if not rev_id:
            continue
        key = tuple(sorted((iid, rev_id)))
        if key in linked:
            continue
        linked.add(key)
        if rev_id not in refs or iid not in by_id or rev_id not in by_id:
            errors.append({"ids": key, "reason": "missing reverse id"})
            continue
        a, b = by_id[iid], by_id[rev_id]
        if (norm(a["text1"]) != norm(b["text2"]) or norm(a["text2"]) != norm(b["text1"])
                or REV.get(ref["label"]) != refs[rev_id]["label"]):
            errors.append({"ids": key, "reason": "text or reversible label mismatch"})
    result = {"train_rows": len(train_rows), "dev_rows": len(dev_rows),
              "distinct_ordered_text_pairs": len(ordered),
              "exact_ordered_conflict_groups": len(exact_conflicts),
              "canonical_reversible_conflict_groups": len(reverse_conflicts),
              "official_dev_reverse_links": len(linked), "official_dev_reverse_link_violations": len(errors),
              "exact_conflict_examples": exact_conflicts[:10],
              "canonical_conflict_examples": reverse_conflicts[:10], "link_violations": errors[:10],
              "caveat": "Exact phrase pairs can carry different unstated contexts; conflicts are audit candidates, not proven annotation mistakes. NEG has no official reversal."}
    print("LABEL_STRUCTURAL_AUDIT " + json.dumps(result, sort_keys=True), flush=True)
    return result


def fisher_two_sided(a, b, c, d):
    row1, row2, col1 = a + b, c + d, a + c
    n = row1 + row2
    denom = math.comb(n, col1)
    if denom == 0:
        return None
    def prob(k):
        return math.comb(row1, k) * math.comb(row2, col1 - k) / denom
    observed = prob(a)
    lo, hi = max(0, col1 - row2), min(row1, col1)
    return min(1.0, sum(prob(k) for k in range(lo, hi + 1) if prob(k) <= observed + 1e-12))


@torch.no_grad()
def entailment_probs(model, tokenizer, pairs, device):
    names = {int(k): str(v).casefold() for k, v in model.config.id2label.items()}
    idx = [k for k, name in names.items() if name == "entailment"]
    if len(idx) != 1:
        raise ValueError(f"Expected exactly one entailment label in model config, got {names}")
    values = []
    for start in range(0, len(pairs), 32):
        part = pairs[start:start + 32]
        batch = tokenizer([a for a, _ in part], [b for _, b in part],
                          padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(**batch).logits
        values.extend(torch.softmax(logits.float(), dim=-1)[:, idx[0]].cpu().tolist())
    return np.array(values, dtype=np.float64)


def from_entailment(ab, ba, threshold):
    f, b = ab >= threshold, ba >= threshold
    if f and b:
        return "EQUIVALENCE"
    if f:
        return "FORWARD_ENTAILMENT"
    if b:
        return "BACKWARD_ENTAILMENT"
    return "NEGATIVE_OTHER"


def run(cfg, paths, train_rows, dev_rows):
    source = Path(__file__).resolve().parents[1] / "data" / "shared_errors_with_qwen.csv"
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reviewed = list(csv.DictReader(handle))
    dev = {r["instance_id"]: r for r in dev_rows}
    for r in reviewed:
        actual = dev.get(r["instance_id"])
        if not actual or any(actual[k] != r[k if k != "label" else "gold"]
                             for k in ("text1", "text2", "label")):
            raise ValueError(f"reviewed CSV does not exactly match official dev: {r['instance_id']}")
    print("LABEL_REVIEW_INPUT " + json.dumps({"rows": len(reviewed),
          "pairs": len({r["pair_id"] for r in reviewed}),
          "verdicts": dict(collections.Counter(r["verdict"] for r in reviewed)),
          "source": str(source), "role": "single-reader assessment, evaluation only"}, sort_keys=True), flush=True)
    structural_audit(train_rows, dev_rows, paths["dev_reference"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("LABEL_TEACHER_CONFIG " + json.dumps({"model_id": cfg["model_id"],
          "revision": cfg["model_revision"], "device": str(device),
          "entailment_threshold": 0.5, "data_used_for_training": False}, sort_keys=True), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["model_revision"])
    model = AutoModelForSequenceClassification.from_pretrained(cfg["model_id"], revision=cfg["model_revision"])
    model = model.to(device).eval()
    pairs = [(r["text1"], r["text2"]) for r in dev_rows]
    forward = entailment_probs(model, tokenizer, pairs, device)
    backward = entailment_probs(model, tokenizer, [(b, a) for a, b in pairs], device)
    predicted = [from_entailment(a, b, 0.5) for a, b in zip(forward, backward)]
    teacher = {r["instance_id"]: {"prediction": p, "entail_ab": float(a), "entail_ba": float(b)}
               for r, p, a, b in zip(dev_rows, predicted, forward, backward)}

    print("<<<LABEL_AUDIT_CSV")
    fields = ["instance_id", "pair_id", "text1", "text2", "gold", "verdict", "reason",
              "teacher_prediction", "teacher_entail_ab", "teacher_entail_ba"]
    writer = csv.DictWriter(sys.stdout, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in reviewed:
        score = teacher[row["instance_id"]]
        writer.writerow({k: row[k] for k in fields[:7]} | {"teacher_prediction": score["prediction"],
                        "teacher_entail_ab": round(score["entail_ab"], 6),
                        "teacher_entail_ba": round(score["entail_ba"], 6)})
    print("LABEL_AUDIT_CSV>>>")

    groups = collections.defaultdict(list)
    for row in reviewed:
        groups[row["pair_id"]].append(row)
    by_verdict = collections.defaultdict(lambda: {"pairs": 0, "teacher_disagrees_with_gold": 0})
    examples = []
    for pid, values in sorted(groups.items()):
        assert len({r["verdict"] for r in values}) == 1
        row = next((r for r in values if r["instance_id"].endswith("__original")), values[0])
        info = teacher[row["instance_id"]]
        disagree = info["prediction"] != row["gold"]
        bucket = by_verdict[row["verdict"]]
        bucket["pairs"] += 1
        bucket["teacher_disagrees_with_gold"] += int(disagree)
        if row["verdict"] == "GOLD_DOUBTFUL" and disagree:
            examples.append({"pair_id": pid, "gold": row["gold"], "teacher": info["prediction"],
                             "text1": row["text1"], "text2": row["text2"]})
    doubtful, clear = by_verdict["GOLD_DOUBTFUL"], by_verdict["GOLD_CLEAR"]
    a, c = doubtful["teacher_disagrees_with_gold"], clear["teacher_disagrees_with_gold"]
    b, d = doubtful["pairs"] - a, clear["pairs"] - c
    summary = {"reviewed_rows": len(reviewed), "reviewed_pairs": len(groups),
               "full_dev_rows": len(dev_rows),
               "teacher_gold_agreement_all_dev": sum(p == r["label"] for p, r in zip(predicted, dev_rows)) / len(dev_rows),
               "by_single_reader_verdict_pair_level": dict(by_verdict),
               "doubtful_vs_clear_teacher_disagreement_fisher_two_sided": fisher_two_sided(a, b, c, d),
               "flagged_doubtful_examples": examples,
               "limitations": ["Teacher disagreement is a screening signal, not ground truth or proof of label noise.",
                               "Reviewer verdicts are single-reader judgments on cases selected because three other models erred.",
                               "The two-direction 0.5 threshold is an uncalibrated NLI-to-four-label mapping.",
                               "Examples lack original context; human adjudication is required."]}
    print("LABEL_AUDIT_FINAL_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
