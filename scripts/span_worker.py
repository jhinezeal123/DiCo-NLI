"""VUAR stage 1b teacher: score the relation of replaced spans by next-token logits (no generation).

Each job is {key, t1, t2, a, b}. The output is, per job, the log-probabilities of the answer letters
A-E (EQ, SPEC, GEN, CONF, LOOSE), normalised over those five letters.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import vuar  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    jobs = json.loads(Path(args.jobs).read_text())[args.shard::args.nshards]
    bs = int(cfg.get("span_batch_size", 16))

    name = cfg["teacher_model"]
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        name, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_use_double_quant=True,
                                               bnb_4bit_compute_dtype=torch.float16))
    model.eval()
    letters = list(vuar.LETTER)
    letter_ids = [tok.encode(l, add_special_tokens=False) for l in letters]
    assert all(len(i) == 1 for i in letter_ids), letter_ids
    letter_ids = torch.tensor([i[0] for i in letter_ids], device="cuda")
    print(f"span shard={args.shard}/{args.nshards} jobs={len(jobs)} model={name} "
          f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    def build(j):
        msgs = [{"role": "user", "content": vuar.span_prompt(j["t1"], j["t2"], j["a"], j["b"])}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    order = sorted(range(len(jobs)), key=lambda i: len(jobs[i]["t1"]) + len(jobs[i]["t2"]))
    out = {}
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            enc = tok([build(jobs[i]) for i in idx], return_tensors="pt", padding=True).to("cuda")
            # Only the last position is needed: apply the LM head there instead of on every token.
            hidden = model.model(**enc).last_hidden_state[:, -1]
            logits = model.lm_head(hidden).float()[:, letter_ids]
            lp = torch.log_softmax(logits, dim=-1).cpu().tolist()
            for i, row in zip(idx, lp):
                out[jobs[i]["key"]] = dict(zip(letters, row))
            if (s // bs) % 20 == 0:
                print(f"{min(s + bs, len(order))}/{len(order)} scored, {time.time() - t0:.0f}s", flush=True)
    Path(args.out).write_text(json.dumps(out), encoding="utf-8")


if __name__ == "__main__":
    main()
