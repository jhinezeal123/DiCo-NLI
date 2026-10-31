"""Prepare identical-source paraphrase and duplicate arms without dev labels.

Candidates are existing train-anchor LLM paraphrases, screened in both directions
by a pinned, frozen MNLI model. This is a model-based filter, not human review.
The printed manifest and hashes permit verifying that both arms select the same
source rows. The participant dev texts are used solely for overlap exclusion.
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def norm(text):
    return " ".join(str(text).casefold().split())


def pair_key(a, b):
    return tuple(sorted((norm(a), norm(b))))


def read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def prepare(cfg, train_path, dev_path, output_dir):
    candidates_path = Path(cfg["paraphrase_candidates"])
    candidates = [json.loads(line) for line in candidates_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    train, dev = read_rows(train_path), read_rows(dev_path)
    train_phrases = {norm(r[k]) for r in train for k in ("text1", "text2")}
    dev_phrases = {norm(r[k]) for r in dev for k in ("text1", "text2")}
    dev_pairs = {pair_key(r["text1"], r["text2"]) for r in dev}
    available = {}
    for row in candidates:
        source, target = row["x"].strip(), row["para"].strip()
        key = norm(source)
        if key in train_phrases and key not in dev_phrases and norm(target) != key and target:
            # Duplicate generation for one anchor would be an ambiguous recipe.
            if key in available and available[key][1] != target:
                raise ValueError(f"conflicting paraphrases for train anchor {source!r}")
            available[key] = (source, target)
    ordered = [available[k] for k in sorted(available)]
    if not ordered:
        raise RuntimeError("No train-only candidate anchors found")

    model_id, revision = cfg["nli_model_id"], cfg["nli_revision"]
    print(f"NLI_SCREEN_START model={model_id} revision={revision} anchors={len(ordered)} threshold={cfg['nli_threshold']}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision)
    label_ids = [int(k) for k, v in model.config.id2label.items() if str(v).casefold() == "entailment"]
    if len(label_ids) != 1:
        raise ValueError(f"Cannot identify entailment label: {model.config.id2label}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    pairs = [(a, b) for a, b in ordered] + [(b, a) for a, b in ordered]
    scores = []
    with torch.inference_mode():
        for i in range(0, len(pairs), 16):
            chunk = pairs[i:i + 16]
            inputs = tokenizer([x[0] for x in chunk], [x[1] for x in chunk],
                               truncation=True, max_length=128, padding=True, return_tensors="pt").to(device)
            scores.extend(torch.softmax(model(**inputs).logits.float(), dim=-1)[:, label_ids[0]].cpu().tolist())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    n, threshold = len(ordered), float(cfg["nli_threshold"])
    approved = {norm(a): (b, scores[i], scores[i+n]) for i, (a, b) in enumerate(ordered)
                if scores[i] >= threshold and scores[i+n] >= threshold}
    print(f"NLI_SCREEN_RESULT accepted_anchors={len(approved)} rejected_anchors={n-len(approved)} device={device}", flush=True)

    # Stable per-class ranking; choose one edit side deterministically and preserve
    # the original row's label. Both arms use the identical selected source IDs.
    potential = []
    for row in train:
        if pair_key(row["text1"], row["text2"]) in dev_pairs:
            continue
        for side in ("text1", "text2"):
            if norm(row[side]) not in approved:
                continue
            replacement, forward, backward = approved[norm(row[side])]
            revised = dict(row)
            revised[side] = replacement
            if pair_key(revised["text1"], revised["text2"]) in dev_pairs:
                continue
            potential.append((row, revised, side, forward, backward))
            break
    potential.sort(key=lambda x: hashlib.sha256(("pilot-v1:" + x[0]["instance_id"]).encode()).hexdigest())
    selected, class_counts = [], Counter()
    for item in potential:
        label = item[0]["label"]
        if class_counts[label] < int(cfg["max_per_class"]):
            selected.append(item)
            class_counts[label] += 1
    if len(selected) < int(cfg["min_augmented_rows"]):
        raise RuntimeError(f"Only {len(selected)} eligible rows after filtering; require {cfg['min_augmented_rows']}")
    mode = cfg["augmentation_mode"]
    if mode not in {"paraphrase", "duplicate"}:
        raise ValueError(f"Unexpected augmentation mode {mode}")
    augmented = []
    manifest = []
    for i, (original, revised, side, forward, backward) in enumerate(selected):
        extra = dict(revised if mode == "paraphrase" else original)
        extra["instance_id"] = f"pilot_aug_{i:05d}"
        extra["pair_id"] = f"pilot_aug_pair_{i:05d}"
        augmented.append(extra)
        entry = {"source_id": original["instance_id"], "source_pair": original["pair_id"],
                 "label": original["label"], "side": side, "original": original[side],
                 "paraphrase": revised[side], "nli_fwd": round(forward, 6), "nli_rev": round(backward, 6)}
        manifest.append(entry)
    source_hash = hashlib.sha256("\n".join(x["source_id"] for x in manifest).encode()).hexdigest()
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "train_augmented.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(train[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(train + augmented)
    (out / "augmentation_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"AUGMENT_SUMMARY mode={mode} rows={len(selected)} classes={dict(sorted(class_counts.items()))} source_hash={source_hash} manifest_hash={manifest_hash}", flush=True)
    print("AUGMENT_MANIFEST_JSON " + json.dumps(manifest, ensure_ascii=False, separators=(",", ":")), flush=True)
    return out / "train_augmented.csv"
