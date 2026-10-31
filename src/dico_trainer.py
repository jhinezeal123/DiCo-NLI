"""In-repo trainer mirroring the official starter-kit recipe, but exposing logits.

Kit parity: AdamW(betas=(0.9, 0.999), eps=1e-6), linear warmup/decay, grad clip 1.0,
classifier_dropout from config, CE loss, deterministic seeding, truncation="only_second".
Difference: per-batch dynamic padding instead of padding to max_length (identical
under the attention mask, much faster).
"""
import os
from pathlib import Path
import random
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from sklearn.metrics import f1_score
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

# Same order as the kit's sklearn LabelEncoder (alphabetical).
LABELS = ["BACKWARD_ENTAILMENT", "EQUIVALENCE", "FORWARD_ENTAILMENT", "NEGATIVE_OTHER"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
# Label of the reversed pair; NEGATIVE_OTHER has no official reversal, kept as identity.
REV = [LABEL2ID["FORWARD_ENTAILMENT"], LABEL2ID["EQUIVALENCE"], LABEL2ID["BACKWARD_ENTAILMENT"],
       LABEL2ID["NEGATIVE_OTHER"]]


def set_seed(seed):
    transformers.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    # warn_only: some LLM ops have no deterministic kernel; deterministic ops are unchanged.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def load_split(path):
    df = pd.read_csv(path, sep=",", header=0)
    clean = lambda s: re.sub(r"\s+", " ", str(s)).strip()
    ids = df["instance_id"].astype(str).tolist()
    t1 = [clean(s) for s in df["text1"]]
    t2 = [clean(s) for s in df["text2"]]
    y = np.array([LABEL2ID[l] for l in df["label"]]) if "label" in df.columns else None
    return ids, t1, t2, y


HINT_WORDS = {"EQV": "equivalent", "FWD": "text 1 entails text 2", "REV": "text 2 entails text 1",
              "EXC": "contradictory", "IND": "neutral"}
SWAP_Z = {"EQV": "EQV", "FWD": "REV", "REV": "FWD", "EXC": "EXC", "IND": "IND"}


def encode(tok, t1, t2, max_length, template=None, hints=None):
    if template:
        # Decoder tokenizers add no separator between a text pair; format explicitly.
        hints = hints or [None] * len(t1)
        enc = tok([template.format(t1=a, t2=b, z=HINT_WORDS.get(h, "")) for a, b, h in zip(t1, t2, hints)],
                  truncation=True, max_length=max_length)
    else:
        enc = tok(t1, t2, truncation="only_second", max_length=max_length)
    keys = [k for k in ("input_ids", "attention_mask", "token_type_ids") if k in enc]
    return [{k: enc[k][i] for k in keys} for i in range(len(t1))]


def batch(tok, feats, idx, device):
    return tok.pad([feats[i] for i in idx], return_tensors="pt").to(device)


@torch.no_grad()
def predict_probs(model, tok, feats, bs, device, amp=False):
    bs = max(bs, 64)  # eval-only: larger batches, no effect on training
    model.eval()
    out = []
    for s in range(0, len(feats), bs):
        b = batch(tok, feats, range(s, min(s + bs, len(feats))), device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits = model(**b).logits
        out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def build_model(cfg, num_labels, device):
    config = AutoConfig.from_pretrained(cfg["model_id"])
    # Dropout handling mirrors the starter kit's Model class.
    if hasattr(config, "classifier_dropout"):
        config.classifier_dropout = float(cfg["dropout"])
    elif hasattr(config, "pooler_dropout") and not hasattr(config, "cls_dropout"):
        config.cls_dropout = float(cfg["dropout"])
    config.num_labels = num_labels
    tok = AutoTokenizer.from_pretrained(cfg["model_id"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    config.pad_token_id = tok.pad_token_id
    model = AutoModelForSequenceClassification.from_pretrained(cfg["model_id"], config=config)
    if cfg.get("lora"):
        from peft import LoraConfig, TaskType, get_peft_model
        lc = cfg["lora"]
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.SEQ_CLS, r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
            target_modules=lc["target_modules"]))
        model.print_trainable_parameters()
    return tok, model.to(device)


def factorized_probs(e_fwd, e_bwd):
    """4-label distribution from P(t1 entails t2) and P(t2 entails t1), in LABELS order."""
    return np.stack([(1 - e_fwd) * e_bwd, e_fwd * e_bwd, e_fwd * (1 - e_bwd),
                     (1 - e_fwd) * (1 - e_bwd)], axis=1)


def run(cfg, seed, train_path, dev_path, device):
    """Train on train only; return dev probabilities for the original and swapped inputs."""
    g = set_seed(seed)
    tr_ids, tr1, tr2, ytr = load_split(train_path)
    dv_ids, dv1, dv2, ydv = load_split(dev_path)
    h_tr = h_dv = h_dv_sw = None
    if cfg.get("logic_hint"):
        hint = dict(pd.read_csv(Path("outputs") / cfg["name"] / "logic_hints.csv").values)
        h_tr, h_dv = [hint[k] for k in tr_ids], [hint[k] for k in dv_ids]
        h_dv_sw = [SWAP_Z[h] for h in h_dv]
    factorized = bool(cfg.get("factorized", False))
    amp = bool(cfg.get("amp", False)) and device.type == "cuda"
    if factorized:
        # Binary target: does text1 entail text2?
        ytr = np.isin(ytr, [LABEL2ID["FORWARD_ENTAILMENT"], LABEL2ID["EQUIVALENCE"]]).astype(int)

    tok, model = build_model(cfg, 2 if factorized else len(LABELS), device)

    L = cfg["max_length"]
    tpl = cfg.get("pair_template")
    f_train = encode(tok, tr1, tr2, L, tpl, h_tr)
    f_dev, f_dev_swap = encode(tok, dv1, dv2, L, tpl, h_dv), encode(tok, dv2, dv1, L, tpl, h_dv_sw)

    bs, epochs = int(cfg["batch_size"]), int(cfg["num_epochs"])
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=float(cfg["wd"]),
                            betas=(0.9, 0.999), eps=1e-6)
    steps_per_epoch = (len(f_train) + bs - 1) // bs
    total = steps_per_epoch * epochs
    sched = transformers.get_linear_schedule_with_warmup(opt, int(total * float(cfg["warmup_pcrt"])), total)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ytr_t = torch.tensor(ytr, device=device)

    def dev_probs():
        p = predict_probs(model, tok, f_dev, bs, device, amp)
        q = predict_probs(model, tok, f_dev_swap, bs, device, amp)
        if factorized:
            p, q = factorized_probs(p[:, 1], q[:, 1]), factorized_probs(q[:, 1], p[:, 1])
        return p, q

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(f_train), generator=g).tolist()
        total_loss = 0.0
        for s in range(0, len(perm), bs):
            idx = perm[s:s + bs]
            b = batch(tok, f_train, idx, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = model(**b).logits
            loss = F.cross_entropy(logits.float(), ytr_t[idx], label_smoothing=float(cfg.get("label_smoothing", 0.0)))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total_loss += loss.item()
        p, _ = dev_probs()
        acc = float((p.argmax(1) == ydv).mean())
        wf1 = f1_score(ydv, p.argmax(1), average="weighted")
        # Dev numbers per epoch are diagnostics only; the final epoch is always used.
        print(f"Epoch {epoch + 1} average loss: {total_loss / steps_per_epoch:.4f} "
              f"| dev acc {acc:.4f} weighted_f1 {wf1:.4f}", flush=True)

    p, q = dev_probs()
    if h_dv is not None:
        hint_interventions(model, tok, dv1, dv2, ydv, h_dv, L, tpl, bs, device, amp)
    return {"orig": p, "swap": q}


def hint_interventions(model, tok, dv1, dv2, ydv, h_dv, L, tpl, bs, device, amp):
    """Counterfactual logic hints on dev: how far does the predicted label follow the stated logic?"""
    ident = {"EQV": "EQUIVALENCE", "FWD": "FORWARD_ENTAILMENT", "REV": "BACKWARD_ENTAILMENT",
             "EXC": "NEGATIVE_OTHER", "IND": "NEGATIVE_OTHER"}
    base = predict_probs(model, tok, encode(tok, dv1, dv2, L, tpl, h_dv), bs, device, amp).argmax(1)
    for z in HINT_WORDS:
        pred = predict_probs(model, tok, encode(tok, dv1, dv2, L, tpl, [z] * len(dv1)), bs, device, amp).argmax(1)
        dist = np.bincount(pred, minlength=len(LABELS)) / len(pred)
        print(f"HINT_INTERVENTION z={z} follow_rate={np.mean(pred == LABEL2ID[ident[z]]):.4f} "
              f"changed_vs_actual_hint={np.mean(pred != base):.4f} acc={np.mean(pred == ydv):.4f} "
              + " ".join(f"P_{l[:2]}={v:.3f}" for l, v in zip(LABELS, dist)), flush=True)
