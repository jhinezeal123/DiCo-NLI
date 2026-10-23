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


def set_seed(seed):
    transformers.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)
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


def encode(tok, t1, t2, max_length):
    enc = tok(t1, t2, truncation="only_second", max_length=max_length)
    keys = [k for k in ("input_ids", "attention_mask", "token_type_ids") if k in enc]
    return [{k: enc[k][i] for k in keys} for i in range(len(t1))]


def batch(tok, feats, idx, device):
    return tok.pad([feats[i] for i in idx], return_tensors="pt").to(device)


@torch.no_grad()
def predict_probs(model, tok, feats, bs, device):
    bs = max(bs, 64)  # eval-only: larger batches, no effect on training
    model.eval()
    out = []
    for s in range(0, len(feats), bs):
        b = batch(tok, feats, range(s, min(s + bs, len(feats))), device)
        out.append(torch.softmax(model(**b).logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def run(cfg, seed, train_path, dev_path, device):
    """Train on train only; return dev probabilities for the original and swapped inputs."""
    g = set_seed(seed)
    _, tr1, tr2, ytr = load_split(train_path)
    _, dv1, dv2, ydv = load_split(dev_path)

    config = AutoConfig.from_pretrained(cfg["model_id"])
    config.classifier_dropout = float(cfg["dropout"])
    config.num_labels = len(LABELS)
    tok = AutoTokenizer.from_pretrained(cfg["model_id"])
    model = AutoModelForSequenceClassification.from_pretrained(cfg["model_id"], config=config).to(device)

    L = cfg["max_length"]
    f_train = encode(tok, tr1, tr2, L)
    f_dev, f_dev_swap = encode(tok, dv1, dv2, L), encode(tok, dv2, dv1, L)

    bs, epochs = int(cfg["batch_size"]), int(cfg["num_epochs"])
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["wd"]),
                            betas=(0.9, 0.999), eps=1e-6)
    steps_per_epoch = (len(f_train) + bs - 1) // bs
    total = steps_per_epoch * epochs
    sched = transformers.get_linear_schedule_with_warmup(opt, int(total * float(cfg["warmup_pcrt"])), total)
    ytr_t = torch.tensor(ytr, device=device)
    lam = float(cfg.get("lambda_cons", 0.0))
    paired = bool(cfg.get("paired_ce", False))
    rev_t = torch.tensor(REV, device=device)
    f_train_swap = encode(tok, tr2, tr1, L) if (lam > 0 or paired) else None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(f_train), generator=g).tolist()
        total_loss = 0.0
        for s in range(0, len(perm), bs):
            idx = perm[s:s + bs]
            b = batch(tok, f_train, idx, device)
            y = ytr_t[idx]
            logits = model(**b).logits
            loss = F.cross_entropy(logits, y)
            if lam > 0 or paired:
                logits_q = model(**batch(tok, f_train_swap, idx, device)).logits
                if paired:
                    loss = 0.5 * (loss + F.cross_entropy(logits_q, rev_t[y]))
                if lam > 0:
                    # Reversal consistency only where the gold label is reversible.
                    m = y != LABEL2ID["NEGATIVE_OTHER"]
                    if m.any():
                        logp = F.log_softmax(logits[m], dim=-1)
                        logq = F.log_softmax(logits_q[m], dim=-1)[:, rev_t]
                        kl = 0.5 * (F.kl_div(logq, logp, log_target=True, reduction="batchmean")
                                    + F.kl_div(logp, logq, log_target=True, reduction="batchmean"))
                        loss = loss + lam * kl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total_loss += loss.item()
        p = predict_probs(model, tok, f_dev, bs, device)
        acc = float((p.argmax(1) == ydv).mean())
        wf1 = f1_score(ydv, p.argmax(1), average="weighted")
        # Dev numbers per epoch are diagnostics only; the final epoch is always used.
        print(f"Epoch {epoch + 1} average loss: {total_loss / steps_per_epoch:.4f} "
              f"| dev acc {acc:.4f} weighted_f1 {wf1:.4f}", flush=True)

    return {"orig": predict_probs(model, tok, f_dev, bs, device),
            "swap": predict_probs(model, tok, f_dev_swap, bs, device)}
