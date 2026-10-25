"""VUAR student: LoRA-finetune a causal LM to write unit-alignment traces, then decode dev traces.

Runs on the single GPU made visible to this process. Training uses the verifier-accepted
teacher traces (original direction) and their exact mirrors (swapped direction); the loss is
on the trace tokens only. Decoding is greedy, for the original and the swapped dev input.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import dico_trainer as dt  # noqa: E402
import vuar  # noqa: E402

PROMPT = "Text 1: {t1}\nText 2: {t2}\nUnits:"


def build_examples(traces):
    ex = []
    for tr in traces:
        units = vuar.parse(tr["units"])
        ex.append((tr["t1"], tr["t2"], vuar.format_units(units)))
        ex.append((tr["t2"], tr["t1"], vuar.format_units(vuar.mirror(units))))
    return ex


def collate(tok, items, device):
    n = max(len(p) + len(t) for p, t in items)
    ids = torch.full((len(items), n), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(items), n), dtype=torch.long)
    labels = torch.full((len(items), n), -100, dtype=torch.long)
    for i, (p, t) in enumerate(items):
        seq = p + t
        ids[i, :len(seq)] = torch.tensor(seq)
        mask[i, :len(seq)] = 1
        labels[i, len(p):len(seq)] = torch.tensor(t)
    return ids.to(device), mask.to(device), labels.to(device)


@torch.no_grad()
def decode(model, tok, t1s, t2s, bs, max_new, device, amp):
    """Greedy trace decoding; returns [(text, mean token log-prob)] in input order."""
    model.eval()
    tok.padding_side = "left"
    order = sorted(range(len(t1s)), key=lambda i: len(t1s[i]) + len(t2s[i]))
    out = [None] * len(t1s)
    eos = tok.eos_token_id
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tok([PROMPT.format(t1=t1s[i], t2=t2s[i]) for i in idx], return_tensors="pt", padding=True).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            res = model.generate(**enc, max_new_tokens=max_new, do_sample=False, output_scores=True,
                                 return_dict_in_generate=True, pad_token_id=tok.pad_token_id, eos_token_id=eos)
        gen = res.sequences[:, enc["input_ids"].shape[1]:]
        lps = torch.stack([F.log_softmax(sc.float(), -1).gather(1, gen[:, t:t + 1]).squeeze(1)
                           for t, sc in enumerate(res.scores)], 1)
        is_eos = (gen == eos).long()
        keep = ((is_eos.cumsum(1) - is_eos) == 0).float()  # tokens up to and including the first eos
        mean_lp = ((lps * keep).sum(1) / keep.sum(1).clamp(min=1)).cpu().tolist()
        for i, text, lp in zip(idx, tok.batch_decode(gen, skip_special_tokens=True), mean_lp):
            out[i] = (text.strip().split("\n")[0], lp)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(cfg.get("amp", False)) and device.type == "cuda"
    g = dt.set_seed(args.seed)

    traces = [json.loads(line) for line in Path(args.traces).read_text(encoding="utf-8").splitlines() if line.strip()]
    examples = build_examples(traces)
    print(f"seed={args.seed} accepted_original_traces={len(traces)} training_examples={len(examples)} "
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

    items = []
    for t1, t2, units in examples:
        p = tok(PROMPT.format(t1=t1, t2=t2)).input_ids
        t = tok(" " + units).input_ids + [tok.eos_token_id]
        items.append((p, t))
    print(f"max sequence length={max(len(p) + len(t) for p, t in items)}", flush=True)

    bs, epochs = int(cfg["batch_size"]), int(cfg["num_epochs"])
    params = [p for p in model.parameters() if p.requires_grad]
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
        perm = torch.randperm(len(items), generator=g).tolist()
        total = 0.0
        for s in range(0, len(perm), bs):
            ids, mask, labels = collate(tok, [items[i] for i in perm[s:s + bs]], device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = model(input_ids=ids, attention_mask=mask).logits
            loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                                   labels[:, 1:].reshape(-1), ignore_index=-100)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        print(f"Epoch {epoch + 1} average trace loss: {total / steps:.4f} ({time.time() - t0:.0f}s)", flush=True)

    _, d1, d2, _ = dt.load_split(args.dev)
    dec_bs = int(cfg.get("decode_batch_size", 48))
    max_new = int(cfg.get("max_new_tokens", 110))
    t0 = time.time()
    orig = decode(model, tok, d1, d2, dec_bs, max_new, device, amp)
    swap = decode(model, tok, d2, d1, dec_bs, max_new, device, amp)
    print(f"decode_wall_seconds={time.time() - t0:.0f}", flush=True)
    Path(args.out).write_text(json.dumps({"orig": orig, "swap": swap}, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
