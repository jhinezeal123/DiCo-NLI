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
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

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


def lm_readout_setup(model, tok, label_words):
    """Token readout: logits of the pretrained LM head at the last prompt position, restricted
    to the first sub-token of each label word (in LABELS order)."""
    ids = [tok(label_words[l], add_special_tokens=False)["input_ids"][0] for l in LABELS]
    assert len(set(ids)) == len(ids), f"label words share a first token: {ids}"
    print("label_word_tokens: " + ", ".join(f"{l}={label_words[l]!r}->{i}:{tok.decode([i])!r}"
                                             for l, i in zip(LABELS, ids)), flush=True)
    model.dico_label_ids = torch.tensor(ids)
    import inspect
    fwd = inspect.signature(type(model.get_base_model() if hasattr(model, "get_base_model") else model).forward)
    model.dico_keep_kw = "logits_to_keep" if "logits_to_keep" in fwd.parameters else "num_logits_to_keep"


def class_logits(model, b):
    """[batch, 4] class logits for either head type."""
    if not hasattr(model, "dico_label_ids"):
        return model(**b).logits
    # Left padding: the last position is the final prompt token for every row.
    mask = b["attention_mask"]
    pos = (mask.long().cumsum(-1) - 1).clamp(min=0)
    out = model(input_ids=b["input_ids"], attention_mask=mask, position_ids=pos, **{model.dico_keep_kw: 1})
    return out.logits[:, -1, :][:, model.dico_label_ids.to(out.logits.device)]


@torch.no_grad()
def predict_probs(model, tok, feats, bs, device, amp=False):
    bs = max(bs, 64)  # eval-only: larger batches, no effect on training
    model.eval()
    out = []
    for s in range(0, len(feats), bs):
        b = batch(tok, feats, range(s, min(s + bs, len(feats))), device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits = class_logits(model, b)
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
    load_kwargs = {}
    quant = cfg.get("quant_4bit", False)
    if quant:
        from transformers import BitsAndBytesConfig
        load_kwargs.update(device_map={"": device.index or 0}, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16))
    elif cfg.get("base_dtype") == "float16":
        load_kwargs["torch_dtype"] = torch.float16
    lm_readout = cfg.get("readout") == "lm_tokens"
    if lm_readout:
        tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], **load_kwargs)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(cfg["model_id"], config=config, **load_kwargs)
    if cfg.get("lora"):
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        if quant:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        lc = cfg["lora"]
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM if lm_readout else TaskType.SEQ_CLS, r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
            target_modules=lc["target_modules"]))
        # Low-precision bases: keep trainable (LoRA + head) weights in fp32 for stable AdamW/GradScaler.
        for prm in model.parameters():
            if prm.requires_grad:
                prm.data = prm.data.float()
        model.print_trainable_parameters()
    if lm_readout:
        assert num_labels == len(LABELS), "lm_tokens readout supports the 4-label head only"
        lm_readout_setup(model, tok, cfg["label_words"])
    return tok, model if quant else model.to(device)


def factorized_probs(e_fwd, e_bwd):
    """4-label distribution from P(t1 entails t2) and P(t2 entails t1), in LABELS order."""
    return np.stack([(1 - e_fwd) * e_bwd, e_fwd * e_bwd, e_fwd * (1 - e_bwd),
                     (1 - e_fwd) * (1 - e_bwd)], axis=1)


def run(cfg, seed, train_path, dev_path, device):
    """Train on train only; return dev probabilities for the original and swapped inputs."""
    g = set_seed(seed)
    _, tr1, tr2, ytr = load_split(train_path)
    if cfg.get("aug_neg_swap"):
        # NEGATIVE_OTHER rows exist in one order only; add the swapped copy with the same label.
        neg = [i for i, y in enumerate(ytr) if y == LABEL2ID["NEGATIVE_OTHER"]]
        tr1, tr2 = tr1 + [tr2[i] for i in neg], tr2 + [tr1[i] for i in neg]
        ytr = np.concatenate([ytr, ytr[neg]])
        print(f"aug_neg_swap: +{len(neg)} rows -> {len(ytr)} train rows", flush=True)
    _, dv1, dv2, ydv = load_split(dev_path)
    factorized = bool(cfg.get("factorized", False))
    amp = bool(cfg.get("amp", False)) and device.type == "cuda"
    if factorized:
        # Binary target: does text1 entail text2?
        ytr = np.isin(ytr, [LABEL2ID["FORWARD_ENTAILMENT"], LABEL2ID["EQUIVALENCE"]]).astype(int)

    tok, model = build_model(cfg, 2 if factorized else len(LABELS), device)

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
                logits = class_logits(model, b)
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
    return {"orig": p, "swap": q}
