"""Local language extraction; all evaluation inputs come only from public text.

This explicit Rules/Observed document interface is narrower than unrestricted
English. No regular expression interprets causal operators or query outcomes.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time

from causal_dataset import render
from causal_protocol import decode_output


RULE_SYSTEM = (
    'Extract ONE base Boolean rule, ignoring its private noise. Return ONLY JSON '
    '{"name":"variable","op":"root|copy|not|and|or|xor|nand|nor","parents":["variable"]}. '
    'The parents are endogenous variables ONLY, never a noise bit or disturbance. '
    'A variable equal only to its private disturbance has op root and parents []. '
    'A single parent XOR private noise has op copy, NOT op xor. '
    'NOT a parent followed by private noise has op not. '
    'op xor requires TWO endogenous parents; every rule has a private-noise XOR that is NOT the base operation. '
    'Preserve every variable spelling exactly. Do not solve for a value.'
)
QUERY_SYSTEM = (
    'Read original observations and the final query only. Return ONLY JSON '
    '{"observed":{"variable":0},"intervene":{"variable":1},"target":"variable"}. '
    'Copy ALL original observations exactly; an intervention never rewrites original observed values. '
    'intervene contains only overrides still ACTIVE when the final question is answered. '
    'If an override is removed and the original equation is restored, intervene is {}. '
    'Without any intervention, intervene is {}. target is the variable being asked about. '
    'Do not calculate noise bits or answers.'
)


def split_document(text):
    prefix, remainder = text.split("Rules:\n", 1)
    rule_text, query = remainder.split("\nObserved in the original world:", 1)
    rules = [line.strip() for line in rule_text.splitlines() if line.strip()]
    if not rules:
        raise ValueError("document has no rules")
    return rules, "Observed in the original world:" + query


def json_object(text):
    decoder = json.JSONDecoder()
    for i, char in enumerate(text):
        if char == "{":
            try:
                value, _ = decoder.raw_decode(text[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    raise ValueError("subcall has no JSON object")


def demonstrations(training_rows):
    rules_by_op, queries = {}, []
    for index, demo in enumerate(training_rows):
        rules, _ = split_document(demo["text"])
        for line, node in zip(rules, demo["scm"]["variables"]):
            rules_by_op.setdefault(node["op"], (line, node))
        scm = copy.deepcopy(demo["scm"])
        cause, value = next(iter(scm["intervene"].items()))
        view = ("factual", "intervention", "restored", "intervention")[index % 4]
        if view != "intervention":
            scm["intervene"] = {}
        text = render(scm, view, cause, value)
        _, query = split_document(text)
        answer = {k: scm[k] for k in ("observed", "intervene", "target")}
        queries.append((query, answer))
    return list(rules_by_op.values()), queries


def build_requests(rows, training_rows):
    rule_examples, query_examples = demonstrations(training_rows)
    requests, cases = {}, {}

    def request(stage, text):
        system = RULE_SYSTEM if stage == "rule" else QUERY_SYSTEM
        examples = rule_examples if stage == "rule" else query_examples
        conversation = [{"role": "system", "content": system}]
        for question, answer in examples:
            conversation.extend([{"role": "user", "content": question},
                                 {"role": "assistant", "content": json.dumps(answer, separators=(",", ":"))}])
        conversation.append({"role": "user", "content": text})
        key = hashlib.sha256(json.dumps(conversation, sort_keys=True).encode()).hexdigest()
        requests.setdefault(key, {"request_id": key, "stage": stage, "messages": conversation})
        return key

    for row in rows:
        rules, query = split_document(row["text"])
        cases[row["case_id"]] = {"rules": [request("rule", line) for line in rules],
                                 "query": request("query", query)}
    return list(requests.values()), cases


def run_decomposed(model, tokenizer, dataset, cfg, emit):
    import torch
    requests, case_requests = build_requests(dataset["rows"], dataset["demonstrations"])
    by_id, started = {}, time.monotonic()
    for stage, cap in (("rule", 96), ("query", 192)):
        stage_requests = [r for r in requests if r["stage"] == stage]
        for offset in range(0, len(stage_requests), cfg["batch_size"]):
            batch = stage_requests[offset:offset + cfg["batch_size"]]
            prompts = [tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True)
                       for r in batch]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False).to(model.device)
            torch.cuda.synchronize()
            batch_start = time.monotonic()
            with torch.inference_mode():
                generated = model.generate(**encoded, do_sample=False, max_new_tokens=cap,
                                           pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - batch_start
            continuation = generated[:, encoded["input_ids"].shape[1]:]
            for i, (request, text) in enumerate(zip(batch, tokenizer.batch_decode(continuation, skip_special_tokens=True))):
                subcall = {"request_id": request["request_id"], "stage": stage, "raw_output": text,
                           "input_tokens": int(encoded["attention_mask"][i].sum()),
                           "output_tokens": int((continuation[i] != tokenizer.pad_token_id).sum()),
                           "seconds": elapsed / len(batch), "max_new_tokens": cap}
                by_id[request["request_id"]] = subcall
                emit("CAUSAL_SUBCALL", subcall)
            emit("CAUSAL_PROGRESS", {"method": "decomposed_solver", "stage": stage,
                                     "done": min(offset + len(batch), len(stage_requests)), "total": len(stage_requests)})
    usage_count = {}
    for ids in case_requests.values():
        for key in ids["rules"] + [ids["query"]]:
            usage_count[key] = usage_count.get(key, 0) + 1
    records = []
    for row in dataset["rows"]:
        ids = case_requests[row["case_id"]]
        all_ids = ids["rules"] + [ids["query"]]
        try:
            nodes = [json_object(by_id[key]["raw_output"]) for key in ids["rules"]]
            query = json_object(by_id[ids["query"]]["raw_output"])
            if set(query) != {"observed", "intervene", "target"}:
                raise ValueError("query subcall has incorrect fields")
            assembled = {"variables": nodes, **query}
            raw = json.dumps(assembled, separators=(",", ":"))
            decoded = decode_output(raw, "parser_solver", row["scm"])
        except (ValueError, TypeError, KeyError) as exc:
            raw = ""
            decoded = {"prediction": "invalid", "valid": False, "schema_exact": False, "error": str(exc)}
        record = {k: row[k] for k in ("case_id", "family_id", "split", "view", "answer", "changed")}
        record.update({**decoded, "method": "decomposed_solver", "correct": decoded["prediction"] == row["answer"],
                       "raw_output": raw, "subcall_ids": all_ids,
                       "input_tokens": sum(by_id[k]["input_tokens"] / usage_count[k] for k in all_ids),
                       "output_tokens": sum(by_id[k]["output_tokens"] / usage_count[k] for k in all_ids),
                       "batch_seconds_per_case": sum(by_id[k]["seconds"] / usage_count[k] for k in all_ids),
                       "token_cost_note": "shared identical-public-text subcalls amortized across their uses"})
        emit("CAUSAL_CASE", record)
        records.append(record)
    return records, {"subcalls": len(requests), "elapsed_seconds": time.monotonic() - started,
                     "input_tokens": sum(v["input_tokens"] for v in by_id.values()),
                     "output_tokens": sum(v["output_tokens"] for v in by_id.values())}
