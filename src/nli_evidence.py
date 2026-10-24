"""Generate independent, frozen NLI scores on the experiment compute host."""
import json

import numpy as np


def score_text_pairs(cfg, texts):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model_id, revision = cfg["evidence_model_id"], cfg["evidence_revision"]
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.requires_grad_(False).to(device).eval()
    label_ids = {str(v).lower(): int(k) for k, v in model.config.id2label.items()}
    order = [label_ids[l] for l in ("entailment", "neutral", "contradiction")]
    print("EVIDENCE_MODEL", json.dumps({"model_id": model_id, "revision": revision,
          "source_id2label": model.config.id2label, "device": str(device),
          "trainable": False, "label_order": ["entailment", "neutral", "contradiction"]}), flush=True)
    bs = int(cfg["evidence_batch_size"])
    both = list(texts) + [(b, a) for a, b in texts]
    outputs = []
    truncated = 0
    with torch.inference_mode():
        for start in range(0, len(both), bs):
            pairs = both[start:start + bs]
            encoded = tok([a for a, _ in pairs], [b for _, b in pairs], padding=True,
                          truncation=True, max_length=cfg["max_length"], return_tensors="pt")
            truncated += int((encoded["attention_mask"].sum(1) >= cfg["max_length"]).sum())
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(**encoded.to(device)).logits
            outputs.append(torch.softmax(logits.float(), dim=1)[:, order].cpu().numpy())
            if start % (20 * bs) == 0:
                print(f"EVIDENCE_PROGRESS scored={min(start+bs, len(both))}/{len(both)}", flush=True)
    scores = np.concatenate(outputs).astype(np.float64)
    result = np.c_[scores[:len(texts)], scores[len(texts):]]
    assert result.shape == (len(texts), 6)
    assert np.allclose(result[:, :3].sum(1), 1, atol=1e-6)
    print(f"EVIDENCE_COMPLETE unordered_rows={len(texts)} ordered_scores={len(both)} at_length_limit={truncated}", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result
