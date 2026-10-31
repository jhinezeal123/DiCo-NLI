"""Noisy-channel classifier: fine-tune a causal LM to generate text2 given (text1, relation).

Score of relation r for a pair (t1, t2): sum_t log p(t2_t | prompt(t1, r), t2_<t) + log prior(r).
The target string is identical across relations, so no length normalisation is needed.
Probabilities = softmax over the four relation scores. Returned in dico_trainer.LABELS order so the
pipeline's bidirectional decoding, scoring and fusion apply unchanged.
"""
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from sklearn.metrics import f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer

import dico_trainer as dt

# FORWARD_ENTAILMENT: text1 entails text2, i.e. text2 is the more general phrase.
RELATION_WORDS = {
    "BACKWARD_ENTAILMENT": "text 2 is more specific",
    "EQUIVALENCE": "same meaning",
    "FORWARD_ENTAILMENT": "text 2 is more general",
    "NEGATIVE_OTHER": "neither entails the other",
}


def prompt(t1, label):
    return f"Text 1: {t1}\nRelation: {RELATION_WORDS[label]}\nText 2:"


def build(tok, t1, t2, labels, max_length):
    feats = []
    for a, b, l in zip(t1, t2, labels):
        p_ids = tok(prompt(a, l), add_special_tokens=False)["input_ids"]
        t_ids = tok(" " + b, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        ids = (p_ids + t_ids)[:max_length]
        lab = ([-100] * len(p_ids) + t_ids)[:max_length]
        feats.append((ids, lab))
    return feats


def collate(feats, idx, pad_id, device):
    n = max(len(feats[i][0]) for i in idx)
    ids = torch.full((len(idx), n), pad_id, dtype=torch.long)
    lab = torch.full((len(idx), n), -100, dtype=torch.long)
    att = torch.zeros((len(idx), n), dtype=torch.long)
    for k, i in enumerate(idx):
        a, b = feats[i]
        ids[k, :len(a)], lab[k, :len(b)], att[k, :len(a)] = torch.tensor(a), torch.tensor(b), 1
    return ids.to(device), att.to(device), lab.to(device)


def seq_logprob(model, ids, att, lab, amp):
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        logits = model(input_ids=ids, attention_mask=att).logits
    logits, tgt = logits[:, :-1].float(), lab[:, 1:]
    nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100,
                          reduction="none").view(tgt.shape)
    return -(nll * (tgt != -100)).sum(1), (tgt != -100).sum(1)


@torch.no_grad()
def relation_probs(model, tok, t1, t2, log_prior, max_length, device, amp, bs=32):
    model.eval()
    n = len(t1)
    scores = np.zeros((n, len(dt.LABELS)))
    for j, l in enumerate(dt.LABELS):
        feats = build(tok, t1, t2, [l] * n, max_length)
        for s in range(0, n, bs):
            idx = list(range(s, min(s + bs, n)))
            lp, _ = seq_logprob(model, *collate(feats, idx, tok.pad_token_id, device), amp)
            scores[s:s + len(idx), j] = lp.cpu().numpy()
    scores = scores + log_prior[None, :]
    scores -= scores.max(1, keepdims=True)
    p = np.exp(scores)
    return p / p.sum(1, keepdims=True)


def run(cfg, seed, train_path, dev_path, device):
    g = dt.set_seed(seed)
    _, tr1, tr2, ytr = dt.load_split(train_path)
    _, dv1, dv2, ydv = dt.load_split(dev_path)
    amp = bool(cfg.get("amp", False)) and device.type == "cuda"
    tok = AutoTokenizer.from_pretrained(cfg["model_id"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"])
    from peft import LoraConfig, TaskType, get_peft_model
    lc = cfg["lora"]
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=lc["r"], lora_alpha=lc["alpha"],
                                             lora_dropout=lc["dropout"], target_modules=lc["target_modules"]))
    model.print_trainable_parameters()
    model.to(device)

    counts = np.bincount(ytr, minlength=len(dt.LABELS)).astype(float)
    log_prior = np.log(counts / counts.sum())
    print("relation_words:", RELATION_WORDS, "train_prior:", dict(zip(dt.LABELS, np.round(counts / counts.sum(), 4))))
    L = int(cfg["max_length"])
    f_train = build(tok, tr1, tr2, [dt.LABELS[y] for y in ytr], L)
    contrast_weight = float(cfg.get("channel_contrast_weight", 0.0))
    assert contrast_weight >= 0
    print(f"channel_contrast_weight={contrast_weight} negative_sampling=uniform_wrong_label "
          f"negative_seed={seed + 104729}", flush=True)
    # Keep negative sampling separate from the training permutation generator.
    neg_gen = torch.Generator().manual_seed(seed + 104729)
    f_by_label = ([build(tok, tr1, tr2, [label] * len(tr1), L) for label in dt.LABELS]
                  if contrast_weight else None)

    bs, epochs = int(cfg["batch_size"]), int(cfg["num_epochs"])
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=float(cfg["wd"]), betas=(0.9, 0.999), eps=1e-6)
    steps_per_epoch = (len(f_train) + bs - 1) // bs
    total = steps_per_epoch * epochs
    sched = transformers.get_linear_schedule_with_warmup(opt, int(total * float(cfg["warmup_pcrt"])), total)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(f_train), generator=g).tolist()
        total_loss = total_lm = total_rank = 0.0
        for s in range(0, len(perm), bs):
            batch_idx = perm[s:s + bs]
            ids, att, lab = collate(f_train, batch_idx, tok.pad_token_id, device)
            lp, ntok = seq_logprob(model, ids, att, lab, amp)
            assert bool((ntok > 0).all()), "target truncated entirely"
            lm_loss = -(lp.sum() / ntok.sum())  # token-averaged LM loss on text2 only
            rank_loss = lp.new_zeros(())
            if contrast_weight:
                offsets = torch.randint(1, len(dt.LABELS), (len(batch_idx),), generator=neg_gen).tolist()
                neg_labels = [(int(ytr[i]) + offset) % len(dt.LABELS)
                              for i, offset in zip(batch_idx, offsets)]
                neg_feats = [f_by_label[label][i] for i, label in zip(batch_idx, neg_labels)]
                neg_ids, neg_att, neg_lab = collate(neg_feats, range(len(neg_feats)), tok.pad_token_id, device)
                lp_neg, neg_ntok = seq_logprob(model, neg_ids, neg_att, neg_lab, amp)
                assert bool((ntok == neg_ntok).all()), "relation prompt changed target truncation"
                rank_loss = F.softplus(lp_neg - lp).mean()
            loss = lm_loss + contrast_weight * rank_loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total_loss += loss.item()
            total_lm += lm_loss.item()
            total_rank += rank_loss.item()
        p = relation_probs(model, tok, dv1, dv2, log_prior, L, device, amp)
        print(f"Epoch {epoch + 1} average loss: {total_loss / steps_per_epoch:.4f} "
              f"lm_loss={total_lm / steps_per_epoch:.4f} rank_loss={total_rank / steps_per_epoch:.4f} | dev acc "
              f"{float((p.argmax(1) == ydv).mean()):.4f} weighted_f1 {f1_score(ydv, p.argmax(1), average='weighted'):.4f}",
              flush=True)
    p = relation_probs(model, tok, dv1, dv2, log_prior, L, device, amp)
    q = relation_probs(model, tok, dv2, dv1, log_prior, L, device, amp)
    return {"orig": p, "swap": q}
