"""Label-blind counterexample generation and independent NLI verification.

Only ``prepare`` loads pretrained models, when called by the remote experiment
runner. Parsing, canonicalization, and feature functions have no ML dependencies.
All features are uncertain evidence, never a proof of entailment. A failed or
missing witness is encoded as unknown, not as positive entailment evidence.
"""
import gc
import hashlib
import json
from pathlib import Path
import re
import time


GENERATOR_ID = "Qwen/Qwen2.5-1.5B-Instruct"
GENERATOR_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
VERIFIER_ID = "cross-encoder/nli-deberta-v3-small"
VERIFIER_REVISION = "fa2804872c3b4bd748f38c0185cc85775361e735"
COUNTEREXAMPLE_REVERSE = [1, 0, 4, 5, 2, 3, 7, 6]
RATIONALE_REVERSE = [3, 4, 5, 0, 1, 2, 6, 7]
_UNKNOWN = {"", "none", "null", "unknown", "no witness", "no counterexample",
            "not found", "n/a", "cannot determine", "not applicable"}


def normalize(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def canonical_pair(text1, text2):
    """Deterministic text-only order; no pair ID or label contributes evidence."""
    first, second = normalize(text1), normalize(text2)
    swapped = (first.casefold(), first) > (second.casefold(), second)
    return (second, first, True) if swapped else (first, second, False)


def present(value):
    if not isinstance(value, str):
        return False
    cleaned = normalize(value).lower().strip(" .:;!?")
    if cleaned in _UNKNOWN:
        return False
    # A generator may verbalize its abstention despite the empty-string prompt.
    # Do not encode such text as an available semantic counterexample.
    if re.match(r"^(?:no (?:valid )?(?:counterexample|witness)(?:\b|$)|"
                r"(?:i am |i'm )?unable to (?:find|construct)|"
                r"(?:i )?cannot (?:find|construct|determine))", cleaned):
        return False
    return True


def parse_generation(text):
    """Strictly extract the first complete JSON object; no prose recovery labels."""
    empty = {"a": "", "b": "", "r": "", "f": "", "g": "", "parsed": False}
    if not isinstance(text, str):
        return empty
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict) or any(k not in value for k in ("a", "b", "r", "f", "g")):
            continue
        result = {k: normalize(value[k])[:1200] if present(value[k]) else ""
                  for k in ("a", "b", "r", "f", "g")}
        if not result["a"] or not result["b"]:
            return empty
        result["parsed"] = True
        return result
    return empty


def build_prompt(a, b):
    # Deliberately no gold, prediction, source group, or dataset metadata.
    return (
        "Return only JSON with five string keys a,b,r,f,g. Rewrite phrase A as a "
        "minimal factual claim in a and phrase B in b, preserving number, modality "
        "and meaning without adding facts. In r give a short semantic comparison. "
        "In f give a concrete situation where a is true and b false; in g a situation "
        "where b is true and a false. Use an empty string for a counterexample you "
        "cannot construct. Keep every field short.\n"
        f"Phrase A: {a}\nPhrase B: {b}"
    )


def verification_requests(parsed):
    """Named premise/hypothesis pairs; contradictions are tested independently."""
    if not parsed.get("parsed"):
        return {}
    result = {}
    for premise_key, hypotheses in (("f", ("a", "b")),
                                    ("g", ("b", "a")),
                                    ("r", ("a", "b"))):
        if parsed[premise_key]:
            for hypothesis in hypotheses:
                result[premise_key + hypothesis] = (parsed[premise_key], parsed[hypothesis])
    return result


def features(parsed, scores):
    """Scores are normalized [contradiction, entailment, neutral] distributions."""
    zero = [0.0] * 8
    if not parsed.get("parsed"):
        return {"extra_counterexample": zero.copy(), "extra_rationale": zero.copy()}

    def probability(name, label):
        vector = scores.get(name)
        if vector is None or len(vector) != 3:
            return 0.0
        return float(vector[label])

    f_ok = bool(parsed["f"] and "fa" in scores and "fb" in scores)
    g_ok = bool(parsed["g"] and "ga" in scores and "gb" in scores)
    r_ok = bool(parsed["r"] and "ra" in scores and "rb" in scores)
    fe = probability("fa", 1) if f_ok else 0.0
    fc = probability("fb", 0) if f_ok else 0.0
    ge = probability("gb", 1) if g_ok else 0.0
    gc_ = probability("ga", 0) if g_ok else 0.0
    return {
        "extra_counterexample": [fe * fc, ge * gc_, fe, fc, ge, gc_, float(f_ok), float(g_ok)],
        "extra_rationale": ([probability("ra", 1), probability("ra", 0), probability("ra", 2),
                             probability("rb", 1), probability("rb", 0), probability("rb", 2),
                             1.0, 1.0] if r_ok else [0.0] * 7 + [1.0]),
    }


def reverse_features(values, field="extra_counterexample"):
    if len(values) != 8:
        raise ValueError("Witness features must have dimension eight")
    permutations = {"extra_counterexample": COUNTEREXAMPLE_REVERSE,
                    "counterexample": COUNTEREXAMPLE_REVERSE,
                    "extra_rationale": RATIONALE_REVERSE,
                    "rationale": RATIONALE_REVERSE}
    if field not in permutations:
        raise ValueError(f"Unknown witness feature field: {field}")
    return [values[index] for index in permutations[field]]


def attach_features(rows, cache):
    output = []
    for row in rows:
        a, b, swapped = canonical_pair(row["text1"], row["text2"])
        entry = cache[(a, b)]
        evidence = features(entry["parsed"], entry.get("scores", {}))
        copied = dict(row)
        for key, values in evidence.items():
            copied[key] = reverse_features(values, key) if swapped else values
        output.append(copied)
    return output


def prepare(train_rows, eval_rows, cfg, out_dir):
    """Build/reuse label-blind evidence cache once, then attach both arm features.

    Independent NLI verification concerns *generated expanded claims*, which may
    themselves contain generation errors. The audit explicitly exposes this
    limitation; claims are never treated as trusted gold or formal proofs.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

    start = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "witness-cache.jsonl"
    generation_tokens = int(cfg.get("witness_max_new_tokens", 160))
    generation_batch = int(cfg.get("witness_generation_batch_size", 8))
    verifier_batch = int(cfg.get("witness_verifier_batch_size", 32))
    max_length = int(cfg.get("witness_max_input_tokens", 256))
    seed = int(cfg.get("witness_seed", 1729))
    torch.manual_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Witness preparation requires the authorized remote CUDA compute")
    device = torch.device("cuda:0")
    specification = {
        "generator": GENERATOR_ID, "generator_revision": GENERATOR_REVISION,
        "verifier": VERIFIER_ID, "verifier_revision": VERIFIER_REVISION,
        "max_new_tokens": generation_tokens, "max_input_tokens": max_length,
        "seed": seed, "do_sample": False, "schema": "witness_v1",
        "prompt_sha256": hashlib.sha256(build_prompt("A", "B").encode()).hexdigest(),
    }
    fingerprint = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    pairs = sorted({canonical_pair(row["text1"], row["text2"])[:2]
                    for row in [*train_rows, *eval_rows]})
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("fingerprint") == fingerprint:
                cache[tuple(record["pair"])] = record
    missing = [pair for pair in pairs if pair not in cache]
    print("WITNESS_PREPARE " + json.dumps({**specification, "unique_pairs": len(pairs),
                                          "cache_hits": len(pairs) - len(missing)}), flush=True)
    if missing:
        tokenizer = AutoTokenizer.from_pretrained(GENERATOR_ID, revision=GENERATOR_REVISION,
                                                  padding_side="left")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            GENERATOR_ID, revision=GENERATOR_REVISION, torch_dtype=torch.float16
        ).to(device).eval()
        with cache_path.open("a", encoding="utf-8") as stream, torch.inference_mode():
            for offset in range(0, len(missing), generation_batch):
                batch_pairs = missing[offset:offset + generation_batch]
                prompts = [tokenizer.apply_chat_template(
                    [{"role": "user", "content": build_prompt(a, b)}],
                    tokenize=False, add_generation_prompt=True) for a, b in batch_pairs]
                inputs = tokenizer(prompts, padding=True, truncation=True,
                                   max_length=max_length, return_tensors="pt").to(device)
                generated = model.generate(**inputs, max_new_tokens=generation_tokens,
                                           do_sample=False, pad_token_id=tokenizer.pad_token_id)
                texts = tokenizer.batch_decode(generated[:, inputs["input_ids"].shape[1]:],
                                                skip_special_tokens=True)
                for pair, raw in zip(batch_pairs, texts):
                    entry = {"pair": list(pair), "fingerprint": fingerprint,
                             "parsed": parse_generation(raw), "raw": raw}
                    cache[pair] = entry
                    stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
                stream.flush()
                print("WITNESS_GENERATION " + json.dumps({"done": min(offset + generation_batch, len(missing)),
                                                           "total": len(missing),
                                                           "elapsed_seconds": round(time.time() - start, 1)}),
                      flush=True)
        del model, tokenizer, inputs, generated
        gc.collect()
        torch.cuda.empty_cache()

    requests = []
    for pair in pairs:
        entry = cache[pair]
        if entry.get("verified"):
            continue
        entry["scores"] = {}
        for key, (premise, hypothesis) in verification_requests(entry["parsed"]).items():
            requests.append((pair, key, premise, hypothesis))
    if requests:
        tokenizer = AutoTokenizer.from_pretrained(VERIFIER_ID, revision=VERIFIER_REVISION)
        model = AutoModelForSequenceClassification.from_pretrained(
            VERIFIER_ID, revision=VERIFIER_REVISION, torch_dtype=torch.float16
        ).to(device).eval()
        labels = {int(key): value.lower() for key, value in model.config.id2label.items()}
        if labels != {0: "contradiction", 1: "entailment", 2: "neutral"}:
            raise ValueError(f"Unexpected pinned verifier label mapping: {labels}")
        with torch.inference_mode():
            for offset in range(0, len(requests), verifier_batch):
                batch_requests = requests[offset:offset + verifier_batch]
                inputs = tokenizer([r[2] for r in batch_requests], [r[3] for r in batch_requests],
                                   padding=True, truncation=True, max_length=max_length,
                                   return_tensors="pt").to(device)
                probabilities = model(**inputs).logits.float().softmax(-1).cpu().tolist()
                for (pair, key, _, _), probabilities_ in zip(batch_requests, probabilities):
                    cache[pair]["scores"][key] = probabilities_
                if offset % (verifier_batch * 20) == 0:
                    print("WITNESS_VERIFICATION " + json.dumps({"done": min(offset + verifier_batch, len(requests)),
                                                                 "total": len(requests)}), flush=True)
        del model, tokenizer, inputs
        gc.collect()
        torch.cuda.empty_cache()
    with cache_path.open("w", encoding="utf-8") as stream:
        for pair in pairs:
            cache[pair]["verified"] = True
            stream.write(json.dumps(cache[pair], ensure_ascii=False) + "\n")
    validities = [features(cache[pair]["parsed"], cache[pair].get("scores", {}))["extra_counterexample"]
                  for pair in pairs]
    audit = {
        **specification, "fingerprint": fingerprint, "unique_pairs": len(pairs),
        "parse_success": sum(bool(cache[pair]["parsed"].get("parsed")) for pair in pairs),
        "forward_witnesses": sum(int(values[6]) for values in validities),
        "reverse_witnesses": sum(int(values[7]) for values in validities),
        "mean_forward_validity": sum(values[0] for values in validities) / max(len(pairs), 1),
        "mean_reverse_validity": sum(values[1] for values in validities) / max(len(pairs), 1),
        "elapsed_seconds": time.time() - start, "cache_path": str(cache_path),
        "claims_are_generated_and_not_independently_gold_verified": True,
        "missing_witness_is_unknown_not_entailment": True,
        "counterexample_and_rationale_share_generation_and_verification_cost": True,
    }
    print("WITNESS_AUDIT " + json.dumps(audit), flush=True)
    print("<<<WITNESS_CACHE_JSONL", flush=True)
    for pair in pairs:
        print(json.dumps(cache[pair], ensure_ascii=False), flush=True)
    print("WITNESS_CACHE_JSONL>>>", flush=True)
    return attach_features(train_rows, cache), attach_features(eval_rows, cache), audit
