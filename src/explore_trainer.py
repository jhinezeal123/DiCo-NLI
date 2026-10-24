"""Matched Qwen LoRA trainer with optional trainable semantic residual heads.

All model selection is fixed in committed configuration; evaluation inputs carry no gold.
The underlying Qwen model returns its final hidden state directly, avoiding a copy
of every intermediate layer when token-level heads are active.
"""
import gc
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import transformers
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

from explore_data import LABELS, read_rows


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    transformers.set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    return torch.Generator().manual_seed(seed)


def encode_row(tokenizer, row, cfg, swapped=False):
    a, b = (row["text2"], row["text1"]) if swapped else (row["text1"], row["text2"])
    template = cfg["pair_template"]
    text = template.format(t1=a, t2=b)
    a0 = len(template.split("{t1}")[0])
    b0 = len(template.split("{t2}")[0].format(t1=a))
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                        truncation=True, max_length=int(cfg["max_length"]))
    offsets = encoded.pop("offset_mapping")
    masks = [[int(end > start and end > lo and start < hi) for start, end in offsets]
             for lo, hi in ((a0, a0 + len(a)), (b0, b0 + len(b)))]
    context = row.get("swap_context" if swapped else "context", "")
    context_ids = tokenizer(context, add_special_tokens=False)["input_ids"] if context else []
    room = max(0, int(cfg["max_length"]) - len(encoded["input_ids"]))
    context_ids = context_ids[:min(room, int(cfg.get("context_tokens", 96)))]
    ids = context_ids + encoded["input_ids"]
    # Unknown evidence has explicit flags, rather than becoming a positive entailment label.
    arm = cfg["arm"]
    extra_key = "extra_" + arm
    extra = list(row.get(extra_key, [0.] * 8))
    if swapped and arm in ("counterexample", "rationale"):
        from explore_witness import reverse_features
        extra = reverse_features(extra, extra_key)
    assert len(extra) == 8 and all(math.isfinite(float(v)) for v in extra)
    return {"input_ids": ids, "attention_mask": [1] * len(ids),
            "mask1": [0] * len(context_ids) + masks[0],
            "mask2": [0] * len(context_ids) + masks[1], "extra": extra}


def collate(features, indices, pad_id, device):
    selected = [features[i] for i in indices]
    width = max(len(f["input_ids"]) for f in selected)
    batch = {}
    for key in ("input_ids", "attention_mask", "mask1", "mask2"):
        value = pad_id if key == "input_ids" else 0
        batch[key] = torch.tensor([f[key] + [value] * (width - len(f[key])) for f in selected],
                                  dtype=torch.long, device=device)
    batch["extra"] = torch.tensor([f["extra"] for f in selected], dtype=torch.float32, device=device)
    return batch


class EvidenceHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(8, 32), nn.GELU(), nn.Linear(32, 4))
        nn.init.normal_(self.network[-1].weight, std=1e-3)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, evidence):
        with torch.autocast(device_type=evidence.device.type, enabled=False):
            return self.network(evidence.float())


class SemanticClassifier(nn.Module):
    def __init__(self, backbone, cfg):
        super().__init__()
        self.backbone = backbone
        self.arm = cfg["arm"]
        hidden = backbone.config.hidden_size
        self.head = None
        if self.arm in ("geometry", "vector"):
            from explore_geometry import GeometryHead
            self.head = GeometryHead(hidden, cfg, self.arm)
        elif self.arm in ("alignment", "uniform_alignment"):
            from explore_alignment import AlignmentHead
            self.head = AlignmentHead(hidden, cfg, self.arm)
        elif self.arm in ("rationale", "counterexample"):
            self.head = EvidenceHead()

    def forward(self, input_ids, attention_mask, mask1, mask2, extra):
        base = self.backbone.get_base_model()
        hidden = base.model(input_ids=input_ids, attention_mask=attention_mask,
                            use_cache=False, return_dict=True).last_hidden_state
        last = attention_mask.sum(dim=1).long() - 1
        pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last]
        logits = base.score(pooled).float()
        if self.head is not None:
            delta = self.head(extra) if self.arm in ("rationale", "counterexample") else self.head(hidden, mask1, mask2)
            logits = logits + delta
        return logits


def build(cfg, device):
    revision = cfg["model_revision"]
    config = AutoConfig.from_pretrained(cfg["model_id"], revision=revision)
    config.num_labels = 4
    config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], revision=revision)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    config.pad_token_id = tokenizer.pad_token_id
    backbone = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_id"], revision=revision, config=config,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa")
    lc = cfg["lora"]
    backbone = get_peft_model(backbone, LoraConfig(
        task_type=TaskType.SEQ_CLS, r=lc["r"], lora_alpha=lc["alpha"],
        lora_dropout=lc["dropout"], target_modules=lc["target_modules"]))
    # FP16 base stays frozen, while trainable adapters/readout must have FP32 grads for GradScaler.
    for parameter in backbone.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    model = SemanticClassifier(backbone, cfg).to(device)
    print("MODEL_PARAMETERS " + json.dumps({"total": sum(p.numel() for p in model.parameters()),
          "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
          "head": sum(p.numel() for p in model.head.parameters()) if model.head is not None else 0}), flush=True)
    return tokenizer, model


@torch.no_grad()
def predict(model, tokenizer, features, cfg, device):
    model.eval()
    output = []
    for start in range(0, len(features), int(cfg.get("eval_batch_size", 24))):
        indices = range(start, min(len(features), start + int(cfg.get("eval_batch_size", 24))))
        batch = collate(features, indices, tokenizer.pad_token_id, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            output.append(model(**batch).softmax(-1).cpu().numpy())
    result = np.concatenate(output).astype(np.float64)
    assert np.isfinite(result).all() and np.allclose(result.sum(1), 1., atol=1e-5)
    return result


def run_worker(cfg, seed, train_path, evaluation_path, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = seed_all(seed)
    train, evaluation = read_rows(train_path), read_rows(evaluation_path)
    assert all("label" not in r for r in evaluation)
    print("EFFECTIVE_CONFIG " + json.dumps({**cfg, "seed": seed, "train_rows": len(train),
                                           "eval_rows": len(evaluation), "device": str(device)}), flush=True)
    tokenizer, model = build(cfg, device)
    features = [encode_row(tokenizer, row, cfg) for row in train]
    eval_orig = [encode_row(tokenizer, row, cfg) for row in evaluation]
    eval_swap = [encode_row(tokenizer, row, cfg, swapped=True) for row in evaluation]
    for split, values in (("train", features), ("eval", eval_orig), ("eval_swap", eval_swap)):
        if any(not any(f["mask1"]) or not any(f["mask2"]) for f in values):
            raise ValueError(f"a phrase was empty/truncated in {split}; increase the committed target length")
    print("TOKEN_AUDIT " + json.dumps({"train_mean": float(np.mean([len(f["input_ids"]) for f in features])),
           "eval_mean": float(np.mean([len(f["input_ids"]) for f in eval_orig])),
           "missing_phrase_masks": sum(not any(f["mask1"]) or not any(f["mask2"]) for f in features)}), flush=True)
    target = torch.tensor([LABELS.index(r["label"]) for r in train], device=device)
    all_params = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": cfg["lr"]}]
    if model.head is not None:
        groups.append({"params": list(model.head.parameters()), "lr": cfg.get("head_lr", 1e-3)})
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg["wd"], betas=(0.9, 0.999), eps=1e-6)
    batch_size, accumulation = int(cfg["batch_size"]), int(cfg.get("grad_accum", 1))
    batches = math.ceil(len(features) / batch_size)
    steps = math.ceil(batches / accumulation) * int(cfg["num_epochs"])
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, int(steps * cfg["warmup_pcrt"]), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    skipped, successful = 0, 0
    for epoch in range(int(cfg["num_epochs"])):
        model.train()
        order = torch.randperm(len(features), generator=generator).tolist()
        loss_sum = 0.
        optimizer.zero_grad(set_to_none=True)
        for bi, start in enumerate(range(0, len(order), batch_size)):
            indices = order[start:start + batch_size]
            batch = collate(features, indices, tokenizer.pad_token_id, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(**batch)
                loss = F.cross_entropy(logits.float(), target[indices], label_smoothing=cfg["label_smoothing"])
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss epoch={epoch} batch={bi}")
            # Last accumulation window may be shorter than the normal window.
            window = min(accumulation, batches - (bi // accumulation) * accumulation)
            scaler.scale(loss / window).backward()
            loss_sum += float(loss.detach())
            if (bi + 1) % accumulation == 0 or bi + 1 == batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(all_params, 1.)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < old_scale:
                    skipped += 1
                else:
                    successful += 1
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if (bi + 1) % 100 == 0:
                print(f"TRAIN_STEP epoch={epoch+1} batch={bi+1}/{batches} loss={float(loss.detach()):.6f}", flush=True)
        print(f"TRAIN_EPOCH epoch={epoch+1} loss={loss_sum/batches:.6f} successful_steps={successful} skipped_steps={skipped}", flush=True)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    p, q = predict(model, tokenizer, eval_orig, cfg, device), predict(model, tokenizer, eval_swap, cfg, device)
    np.savez(out / "probs.npz", orig=p, swap=q)
    model.backbone.save_pretrained(out / "adapter")
    tokenizer.save_pretrained(out / "adapter")
    if model.head is not None:
        torch.save(model.head.state_dict(), out / "semantic_head.pt")
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"WORKER_DONE arm={cfg['arm']} seed={seed} checkpoint={out / 'adapter'}", flush=True)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
