"""Screen proposed H2 development examples; never read the heldout subset."""
import collections
import csv
import gc
import hashlib
import io
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

TEACHERS = [
    ("deberta", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
     "6f5cf0a2b59cabb106aca4c287eed12e357e90eb"),
    ("bart", "facebook/bart-large-mnli", "d7645e127eaf1aefc7862fd59a17a5aa8558b8ce"),
]


@torch.no_grad()
def predict(model, tok, examples, device):
    mapping = {int(k): str(v).casefold() for k, v in model.config.id2label.items()}
    matches = [k for k, name in mapping.items() if name == "entailment"]
    if len(matches) != 1:
        raise ValueError(f"invalid entailment label map: {mapping}")
    result = []
    for start in range(0, len(examples), 16):
        block = examples[start:start + 16]
        inputs = tok([x[0] for x in block], [x[1] for x in block],
                     padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(**inputs).logits.float()
        result.extend(torch.softmax(logits, -1)[:, matches[0]].cpu().tolist())
    return result


def map_label(a, b):
    a, b = a >= 0.5, b >= 0.5
    if a and b:
        return "EQUIVALENCE"
    if a:
        return "FORWARD_ENTAILMENT"
    if b:
        return "BACKWARD_ENTAILMENT"
    return "NEGATIVE_OTHER"


def run(cfg):
    path = Path("data/h2_screen_development.csv")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 48 and {r["split"] for r in rows} == {"development"}
    assert len({r["contrast_group_id"] for r in rows}) == 12
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {"rows": len(rows), "groups": 12, "split": "development",
              "holdout_loaded": False, "entailment_threshold": 0.5,
              "teachers": TEACHERS, "device": str(device),
              "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    print("H2_SCREEN_CONFIG " + json.dumps(config, sort_keys=True), flush=True)
    sentences = [(r["text1"], r["text2"]) for r in rows]
    predictions = {}
    for name, model_id, revision in TEACHERS:
        print(f"H2_SCREEN_LOADING teacher={name} model_id={model_id} revision={revision}", flush=True)
        tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision).to(device).eval()
        forward = predict(model, tok, sentences, device)
        backward = predict(model, tok, [(b, a) for a, b in sentences], device)
        predictions[name] = [{"label": map_label(a, b), "entail_ab": a, "entail_ba": b}
                             for a, b in zip(forward, backward)]
        print(f"H2_SCREEN_TEACHER_DONE teacher={name} rows={len(rows)}", flush=True)
        del model, tok
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    fields = ["instance_id", "pair_id", "contrast_group_id", "review_id", "stratum", "condition",
              "text1", "text2", "proposed_label", "deberta_label", "deberta_entail_ab",
              "deberta_entail_ba", "bart_label", "bart_entail_ab", "bart_entail_ba",
              "both_agree_proposal", "both_disagree_proposal"]
    print("<<<H2_SCREEN_CSV", flush=True)
    import sys
    writer = csv.DictWriter(sys.stdout, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    agree_counts = collections.Counter()
    by_category = collections.defaultdict(collections.Counter)
    for i, row in enumerate(rows):
        first, second = predictions["deberta"][i], predictions["bart"][i]
        proposal = row["proposed_label"]
        agree = int(first["label"] == proposal) + int(second["label"] == proposal)
        agree_counts[str(agree)] += 1
        by_category[row["condition"]][str(agree)] += 1
        writer.writerow({k: row[k] for k in fields[:9]} |
                        {"deberta_label": first["label"], "deberta_entail_ab": f"{first['entail_ab']:.7f}",
                         "deberta_entail_ba": f"{first['entail_ba']:.7f}",
                         "bart_label": second["label"], "bart_entail_ab": f"{second['entail_ab']:.7f}",
                         "bart_entail_ba": f"{second['entail_ba']:.7f}",
                         "both_agree_proposal": str(agree == 2).lower(),
                         "both_disagree_proposal": str(agree == 0).lower()})
    print("H2_SCREEN_CSV>>>", flush=True)
    summary = {"rows": len(rows), "groups": 12, "split": "development",
               "holdout_loaded": False, "number_teachers_agreeing": dict(agree_counts),
               "agreement_by_condition": {k: dict(v) for k, v in by_category.items()},
               "limitations": [
                   "Agreement is an uncalibrated model-based screening signal, not human validation.",
                   "Teachers share training data and may make correlated mistakes.",
                   "Models are trained on sentence NLI, while the inputs are short DiCo phrase fragments.",
                   "No labels were modified based on this screen.",
               ]}
    print("H2_SCREEN_FINAL_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
