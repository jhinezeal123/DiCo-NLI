"""VUAR teacher: write unit-alignment traces for a shard of jobs on the single visible GPU.

Label-conditioned jobs (job["gold"] set) are retried with sampling until the deterministic
verifier accepts the trace (composition == gold, units grounded) or retries run out.
Label-free jobs (gold None) are decoded greedily once.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import transformers
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
    bs = int(cfg.get("teacher_batch_size", 24))
    retries = int(cfg.get("teacher_retries", 3))
    transformers.set_seed(0)

    name = cfg["teacher_model"]
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        name, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_use_double_quant=True,
                                               bnb_4bit_compute_dtype=torch.float16))
    model.eval()
    print(f"teacher shard={args.shard}/{args.nshards} jobs={len(jobs)} model={name} "
          f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    def prompt(job):
        msgs = [{"role": "user", "content": vuar.teacher_prompt(job["t1"], job["t2"], job.get("gold"))}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def generate(prompts, sample):
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        kw = dict(max_new_tokens=int(cfg.get("teacher_max_new_tokens", 110)), pad_token_id=tok.pad_token_id)
        kw.update(dict(do_sample=True, temperature=0.7, top_p=0.95) if sample else dict(do_sample=False))
        with torch.no_grad():
            out = model.generate(**enc, **kw)
        return tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    results = {}
    pending = sorted(range(len(jobs)), key=lambda i: len(jobs[i]["t1"]) + len(jobs[i]["t2"]))
    t0 = time.time()
    for attempt in range(retries + 1):
        retry = []
        for s in range(0, len(pending), bs):
            idx = pending[s:s + bs]
            texts = generate([prompt(jobs[i]) for i in idx], sample=attempt > 0)
            for i, text in zip(idx, texts):
                j = jobs[i]
                units = vuar.parse(text)
                ok = vuar.accept(units, j["t1"], j["t2"], j.get("gold"))
                results[i] = {"id": j["id"], "group": j["group"], "ref": j.get("ref"), "gold": j.get("gold"),
                              "text": text.strip().split("\n")[0], "units": vuar.format_units(units),
                              "label": vuar.compose(units), "grounded": bool(units) and vuar.grounded(units, j["t1"], j["t2"]),
                              "accepted": ok, "attempts": attempt + 1}
                if not ok and j.get("gold") and attempt < retries:
                    retry.append(i)
            print(f"attempt {attempt}: {min(s + bs, len(pending))}/{len(pending)} done, "
                  f"{time.time() - t0:.0f}s", flush=True)
        pending = retry
        print(f"after attempt {attempt}: {len(pending)} conditioned jobs still unexplained", flush=True)
        if not pending:
            break
    with open(args.out, "w", encoding="utf-8") as f:
        for i in sorted(results):
            f.write(json.dumps(results[i], ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
