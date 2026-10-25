"""Predict state meanings solely from public text; share them across subtasks."""
import hashlib
import json
import time

from causal_public_grounding import public_graph


SYSTEM = (
    "Define consistent binary states for the named variables in this hypothetical world. "
    "Return ONLY a JSON object mapping every exact supplied variable name to {\"one\":\"state phrase\",\"zero\":\"opposite state phrase\"}. "
    "Choose a clear pair of opposite states expressed in the PUBLIC premises or question. "
    "Bit one need NOT mean high or healthy: it can mean low cholesterol, for example. "
    "Choose meanings consistently; this glossary will define the bits for ALL later mechanisms/evidence/questions. "
    "Do not output causal rules, observations, interventions, probabilities or an answer."
)


def gold_development_glossary(program):
    return {value["name"].lower(): {"one": value["one"], "zero": value["zero"]}
            for value in program["meaning"].values()}


def glossary_prompt(text):
    names = list(public_graph(text)["parents"])
    return text + "\n\nExact variable names for the glossary: " + json.dumps(names)


def glossary_messages(row, demonstrations):
    messages = [{"role": "system", "content": SYSTEM}]
    for demo in demonstrations:
        messages.extend([{"role": "user", "content": glossary_prompt(demo["text"])},
                         {"role": "assistant", "content": json.dumps(gold_development_glossary(demo["program"]))}])
    messages.append({"role": "user", "content": glossary_prompt(row["text"])})
    return messages


def validate_glossary(value, names):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError("glossary must define exactly all public variables")
    for pair in value.values():
        if not isinstance(pair, dict) or set(pair) != {"one", "zero"}:
            raise ValueError("each glossary entry needs one/zero meanings")
        if any(not isinstance(v, str) or not v.strip() for v in pair.values()):
            raise ValueError("glossary meanings must be nonempty strings")
        if pair["one"].strip().lower() == pair["zero"].strip().lower():
            raise ValueError("glossary states must differ")
    return value


def parse_glossary(raw, names):
    def no_duplicates(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate glossary key")
            value[key] = item
        return value
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError("missing glossary JSON")
    return validate_glossary(json.loads(raw[start:end+1], object_pairs_hook=no_duplicates), names)


def glossary_context(value):
    return "\n\nEXPLICIT STATE GLOSSARY (defines ALL 0/1 bits, overriding any positive/high default):\n" + json.dumps(value, sort_keys=True)


def run_glossaries(model, tokenizer, dataset, cfg, emit):
    import torch
    glossaries, failures, started = {}, {}, time.monotonic()
    input_tokens, output_tokens = 0, 0
    for offset in range(0, len(dataset["rows"]), cfg["batch_size"]):
        batch = dataset["rows"][offset:offset+cfg["batch_size"]]
        prompts = [tokenizer.apply_chat_template(glossary_messages(row, dataset["demonstrations"]),
                   tokenize=False, add_generation_prompt=True) for row in batch]
        encoded = tokenizer(prompts, padding=True, truncation=False, return_tensors="pt").to(model.device)
        for device in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device)
        batch_started = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(**encoded, do_sample=False, max_new_tokens=cfg.get("glossary_max_new_tokens", 512),
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        for device in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device)
        elapsed = time.monotonic()-batch_started
        continuation = generated[:, encoded["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(continuation, skip_special_tokens=True)
        for i, (row, raw) in enumerate(zip(batch, texts)):
            record = {"question_id": row["question_id"], "raw_output": raw,
                      "input_tokens": int(encoded["attention_mask"][i].sum()),
                      "output_tokens": int((continuation[i] != tokenizer.pad_token_id).sum()),
                      "batch_seconds_per_case": elapsed/len(batch),
                      "prompt_sha256": hashlib.sha256(prompts[i].encode()).hexdigest()}
            try:
                value = parse_glossary(raw, public_graph(row["text"])["parents"])
                glossaries[row["question_id"]] = value
                record.update({"valid": True, "glossary": value})
            except (ValueError, TypeError, KeyError) as exc:
                failures[row["question_id"]] = str(exc)
                record.update({"valid": False, "error": str(exc)})
            input_tokens += record["input_tokens"]; output_tokens += record["output_tokens"]
            emit("PUBLIC_GLOSSARY_CASE", record)
        emit("PUBLIC_PROGRESS", {"method": "native_grounded_solver", "stage": "glossary",
                                 "done": min(offset+len(batch), len(dataset["rows"])), "total": len(dataset["rows"])})
    return glossaries, failures, {"elapsed_seconds": time.monotonic()-started,
                                 "input_tokens": input_tokens, "output_tokens": output_tokens,
                                 "glossary_failures": len(failures)}
