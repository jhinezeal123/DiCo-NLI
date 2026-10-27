"""In-repo trainer mirroring the official starter-kit recipe, but exposing logits.

Kit parity: AdamW(betas=(0.9, 0.999), eps=1e-6), linear warmup/decay, grad clip 1.0,
classifier_dropout from config, CE loss, deterministic seeding, truncation="only_second".
Difference: per-batch dynamic padding instead of padding to max_length (identical
under the attention mask, much faster).
"""
import os
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


# PhrasIS source subtypes of NEGATIVE_OTHER, recovered from the organizers' pair_id numbering
# (class-ordered inventory). Used only to build TRAIN targets, never as a model input.
NEG_SUBTYPES = ["OPPO", "REL", "SIMI"]
SUBTYPE_RANGES = {"OPPO": (1947, 1965), "REL": (1966, 2376), "SIMI": (2377, 3321)}


def neg_subtype(pair_id):
    n = int(str(pair_id).split("_")[-1])
    for i, name in enumerate(NEG_SUBTYPES):
        lo, hi = SUBTYPE_RANGES[name]
        if lo <= n <= hi:
            return i
    return -1


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
    sub = np.array([neg_subtype(p) for p in df["pair_id"]]) if "pair_id" in df.columns else None
    return ids, t1, t2, y, sub


def encode(tok, t1, t2, max_length, template=None):
    if template:
        # Decoder tokenizers add no separator between a text pair; format explicitly.
        enc = tok([template.format(t1=a, t2=b) for a, b in zip(t1, t2)], truncation=True,
                  max_length=max_length)
    else:
        enc = tok(t1, t2, truncation="only_second", max_length=max_length)
    keys = [k for k in ("input_ids", "attention_mask", "token_type_ids") if k in enc]
    return [{k: enc[k][i] for k in keys} for i in range(len(t1))]


def batch(tok, feats, idx, device):
    return tok.pad([feats[i] for i in idx], return_tensors="pt").to(device)


@torch.no_grad()
def predict_probs(model, tok, feats, bs, device, amp=False, fine_neg=None):
    bs = max(bs, 64)  # eval-only: larger batches, no effect on training
    model.eval()
    out = []
    for s in range(0, len(feats), bs):
        b = batch(tok, feats, range(s, min(s + bs, len(feats))), device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits = model(**b).logits
        out.append(to_probs4(logits.float(), fine_neg).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def to_probs4(logits, fine_neg):
    """Map head logits to the 4 official labels (LABELS order)."""
    if fine_neg == "flat6":
        p = torch.softmax(logits, dim=-1)
        return torch.cat([p[:, :3], p[:, 3:].sum(1, keepdim=True)], dim=1)
    if fine_neg == "aux":
        return torch.softmax(logits[:, :4], dim=-1)
    return torch.softmax(logits, dim=-1)


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
    _, tr1, tr2, ytr, sub_tr = load_split(train_path)
    _, dv1, dv2, ydv, _ = load_split(dev_path)
    factorized = bool(cfg.get("factorized", False))
    fine_neg = cfg.get("fine_neg")
    aux_w = float(cfg.get("aux_weight", 0.5))
    if fine_neg:
        is_neg = ytr == LABEL2ID["NEGATIVE_OTHER"]
        assert (sub_tr[is_neg] >= 0).all() and (sub_tr[~is_neg] == -1).all(), "subtype/label mismatch"
        print("train NEG subtypes:", {n: int((sub_tr == i).sum()) for i, n in enumerate(NEG_SUBTYPES)}, flush=True)
    if fine_neg == "flat6":
        # 6-way target: BE, EQ, FE, NEG/OPPO, NEG/REL, NEG/SIMI
        ytr = np.where(ytr == LABEL2ID["NEGATIVE_OTHER"], 3 + sub_tr, ytr)
    n_out = 2 if factorized else {"flat6": 6, "aux": 7}.get(fine_neg, len(LABELS))
    amp = bool(cfg.get("amp", False)) and device.type == "cuda"
    if factorized:
        # Binary target: does text1 entail text2?
        ytr = np.isin(ytr, [LABEL2ID["FORWARD_ENTAILMENT"], LABEL2ID["EQUIVALENCE"]]).astype(int)

    tok, model = build_model(cfg, n_out, device)

    L = cfg["max_length"]
    tpl = cfg.get("pair_template")
    f_train = encode(tok, tr1, tr2, L, tpl)
    f_dev, f_dev_swap = encode(tok, dv1, dv2, L, tpl), encode(tok, dv2, dv1, L, tpl)

    bs, epochs = int(cfg["batch_size"]), int(cfg["num_epochs"])
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=float(cfg["wd"]),
                            betas=(0.9, 0.999), eps=1e-6)
    steps_per_epoch = (len(f_train) + bs - 1) // bs
    total = steps_per_epoch * epochs
    sched = transformers.get_linear_schedule_with_warmup(opt, int(total * float(cfg["warmup_pcrt"])), total)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ytr_t = torch.tensor(ytr, device=device)
    sub_t = torch.tensor(sub_tr, device=device)

    def dev_probs():
        p = predict_probs(model, tok, f_dev, bs, device, amp, fine_neg)
        q = predict_probs(model, tok, f_dev_swap, bs, device, amp, fine_neg)
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
            ls = float(cfg.get("label_smoothing", 0.0))
            logits = logits.float()
            if fine_neg == "aux":
                loss = F.cross_entropy(logits[:, :4], ytr_t[idx], label_smoothing=ls)
                st = sub_t[idx]
                m = st >= 0
                if m.any():  # subtype CE on NEG rows, averaged over the batch
                    aux = F.cross_entropy(logits[m][:, 4:], st[m], label_smoothing=ls, reduction="sum")
                    loss = loss + aux_w * aux / len(idx)
            else:
                loss = F.cross_entropy(logits, ytr_t[idx], label_smoothing=ls)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total_loss += loss.item()
        p, _ = dev_probs()
        acc = float((p.argmax(1) == ydv).mean())  # p is always 4-way
        wf1 = f1_score(ydv, p.argmax(1), average="weighted")
        # Dev numbers per epoch are diagnostics only; the final epoch is always used.
        print(f"Epoch {epoch + 1} average loss: {total_loss / steps_per_epoch:.4f} "
              f"| dev acc {acc:.4f} weighted_f1 {wf1:.4f}", flush=True)

    p, q = dev_probs()
    return {"orig": p, "swap": q}
