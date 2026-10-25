"""VUAR-H student: LoRA CausalLM that writes unit traces AND carries a 4-way class head.

Two objectives: cross-entropy on the trace tokens (as stage 2) plus an auxiliary
cross-entropy on a class head read from the last prompt token.  The head lets the same
network act as a plain classifier, so the trace becomes auxiliary supervision instead of
a bottleneck.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import dico_trainer as dt  # noqa: E402
import vuar  # noqa: E402

PROMPT = "Text 1: {t1}\nText 2: {t2}\nUnits:"
LABELS = ["BACKWARD_ENTAILMENT", "EQUIVALENCE", "FORWARD_ENTAILMENT", "NEGATIVE_OTHER"]


def build_examples(traces):
    ex = []
    for tr in traces:
        units = vuar.parse(tr["units"])
        y = LABELS.index(tr["label"])
        ex.append((tr["t1"], tr["t2"], vuar.format_units(units), y))
        ex.append((tr["t2"], tr["t1"], vuar.format_units(vuar.mirror(units)),
                   LABELS.index(vuar.LABEL_REV[tr["label"]])))
    return ex


def collate(tok, items, device):
    n = max(len(p) + len(t) for p, t, _, _ in items)
    ids = torch.full((len(items), n), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(items), n), dtype=torch.long)
    labels = torch.full((len(items), n), -100, dtype=torch.long)
    plen, ys = [], []
    for i, (p, t, _, y) in enumerate(items):
        seq = p + t
        ids[i, :len(seq)] = torch.tensor(seq)
        mask[i, :len(seq)] = 1
        labels[i, len(p):len(seq)] = torch.tensor(t)
        plen.append(len(p))
        ys.append(y)
    return (ids.to(device), mask.to(device), labels.to(device),
            torch.tensor(plen, device=device), torch.tensor(ys, device=device))


def head_logits(model, head, ids, mask, plen):
    out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    h = out.hidden_states[-1][torch.arange(ids.size(0), device=ids.device), plen - 1]
    return head(h.float())


@torch.no_grad()
def decode_and_classify(model, head, tok, t1s, t2s, bs, max_new, device, amp):
    """Greedy trace decoding plus class-head probabilities, in input order."""
    model.eval()
    tok.padding_side = "left"
    order = sorted(range(len(t1s)), key=lambda i: len(t1s[i]) + len(t2s[i]))
    texts = [None] * len(t1s)
    lps = [None] * len(t1s)
    probs = [None] * len(t1s)
    eos = tok.eos_token_id
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tok([PROMPT.format(t1=t1s[i], t2=t2s[i]) for i in idx], return_tensors="pt", padding=True).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            res = model.generate(**enc, max_new_tokens=max_new, do_sample=False, output_scores=True,
                                 return_dict_in_generate=True, pad_token_id=tok.pad_token_id, eos_token_id=eos)
        gen = res.sequences[:, enc["input_ids"].shape[1]:]
        lp = torch.stack([F.log_softmax(sc.float(), -1).gather(1, gen[:, t:t + 1]).squeeze(1)
                          for t, sc in enumerate(res.scores)], 1)
        is_eos = (gen == eos).long()
        keep = ((is_eos.cumsum(1) - is_eos) == 0).float()
        mean_lp = ((lp * keep).sum(1) / keep.sum(1).clamp(min=1)).cpu().tolist()
        # left padding: the last position (index -1) is the end of the prompt for every row
        plen = torch.full((enc["input_ids"].size(0),), enc["input_ids"].size(1),
                          dtype=torch.long, device=enc["input_ids"].device)
        p = torch.softmax(head_logits(model, head, enc["input_ids"], enc["attention_mask"], plen).float(), -1)
        for k, i in enumerate(idx):
            texts[i] = tok.decode(gen[k], skip_special_tokens=True).strip().split("\n")[0]
            lps[i] = mean_lp[k]
            probs[i] = p[k].cpu().tolist()
    return texts, lps, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-train", default=None,
                    help="CSV whose rows are also decoded, storing traces as train_traces in --out")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(cfg.get("amp", False)) and device.type == "cuda"
    g = dt.set_seed(args.seed)

    traces = [json.loads(line) for line in Path(args.traces).read_text(encoding="utf-8").splitlines() if line.strip()]
    examples = build_examples(traces)
    aux_w = float(cfg.get("aux_ce_weight", 1.0))
    print(f"seed={args.seed} traces={len(traces)} examples={len(examples)} aux_ce_weight={aux_w} "
          f"gpu={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}", flush=True)

    name = cfg["model_id"]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if cfg.get("load_in_4bit"):
        from transformers import BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(name, device_map={"": 0}, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16))
    else:
        model = AutoModelForCausalLM.from_pretrained(name)
    from peft import LoraConfig, TaskType, get_peft_model
    lc = cfg["lora"]
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=lc["r"], lora_alpha=lc["alpha"],
                                             lora_dropout=lc["dropout"], target_modules=lc["target_modules"]))
    model.print_trainable_parameters()
    if not cfg.get("load_in_4bit"):
        model.to(device)
    head = torch.nn.Linear(model.config.hidden_size, len(LABELS)).to(device)

    items = []
    for t1, t2, units, y in examples:
        p = tok(PROMPT.format(t1=t1, t2=t2)).input_ids
        t = tok(" " + units).input_ids + [tok.eos_token_id]
        items.append((p, t, units, y))
    print(f"max sequence length={max(len(p) + len(t) for p, t, _, _ in items)}", flush=True)

    bs, epochs = int(cfg["batch_size"]), int(cfg["num_epochs"])
    params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    for p in params:
        p.data = p.data.float()
    opt = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=float(cfg["wd"]), betas=(0.9, 0.999), eps=1e-6)
    steps = (len(items) + bs - 1) // bs
    sched = transformers.get_linear_schedule_with_warmup(opt, int(steps * epochs * float(cfg["warmup_pcrt"])),
                                                         steps * epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        head.train()
        perm = torch.randperm(len(items), generator=g).tolist()
        tot_t = tot_c = 0.0
        for s in range(0, len(perm), bs):
            ids, mask, labels, plen, ys = collate(tok, [items[i] for i in perm[s:s + bs]], device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                # one forward pass: HF shifts labels internally, -100 masks the prompt
                out = model(input_ids=ids, attention_mask=mask, labels=labels,
                            output_hidden_states=True)
                loss_t = out.loss
                h = out.hidden_states[-1][torch.arange(ids.size(0), device=ids.device), plen - 1]
                loss_c = F.cross_entropy(head(h.float()), ys)
                loss = loss_t + aux_w * loss_c
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot_t += loss_t.item()
            tot_c += loss_c.item()
        print(f"Epoch {epoch + 1} trace_loss={tot_t / steps:.4f} aux_ce={tot_c / steps:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    _, d1, d2, _ = dt.load_split(args.dev)
    dec_bs = int(cfg.get("decode_batch_size", 48))
    max_new = int(cfg.get("max_new_tokens", 110))
    t0 = time.time()
    t1o, l1o, p1o = decode_and_classify(model, head, tok, d1, d2, dec_bs, max_new, device, amp)
    t2o, l2o, p2o = decode_and_classify(model, head, tok, d2, d1, dec_bs, max_new, device, amp)
    print(f"decode_wall_seconds={time.time() - t0:.0f}", flush=True)
    out = {
        "orig": [[a, b] for a, b in zip(t1o, l1o)],
        "swap": [[a, b] for a, b in zip(t2o, l2o)],
        "head_orig": p1o,
        "head_swap": p2o,
    }
    if args.dump_train:
        ids, r1, r2, _ = dt.load_split(args.dump_train)
        tr1, _, _ = decode_and_classify(model, head, tok, r1, r2, dec_bs, max_new, device, amp)
        out["train_traces"] = dict(zip(ids, tr1))
        print(f"train_traces={len(ids)}", flush=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
