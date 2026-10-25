"""Literal grounding plus constrained semantic choices for the Rules interface.

Assumptions: fully observed named bits, output-first rule statements, at most
two endogenous parents, and at most one proposed constant override. These are
explicit document-format assumptions, not an unrestricted English parser.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import re
import time

from causal_decomposition import split_document
from causal_protocol import covered_demonstrations, decode_output

LETTERS = "ABCDEFGH"
OP_OPTIONS = ["AND/conjunction (including NAND modulo private output noise)",
              "OR/inclusive disjunction (including NOR modulo private output noise)",
              "XOR/exclusive disjunction of TWO endogenous parents"]
OP_CODES = ["and", "or", "xor"]


def literal_document(text):
    rules, query = split_document(text)
    lines = query.splitlines()
    assignments = re.findall(r"\b([A-Za-z]\w*)\s*=\s*([01])\b", lines[0])
    if not assignments or len({name for name, _ in assignments}) != len(assignments):
        raise ValueError("unsupported observation assignments")
    observed = {name: int(bit) for name, bit in assignments}
    names = sorted(observed)
    if len(names) > len(LETTERS):
        raise ValueError("too many variables for closed choices")
    pattern = re.compile(r"\b(?:" + "|".join(re.escape(n) for n in names) + r")\b")
    nodes = []
    for line in rules:
        mentions = pattern.findall(line)
        if not mentions:
            raise ValueError("rule mentions no declared variable")
        subject = mentions[0]
        parents = sorted(set(mentions) - {subject})
        if len(parents) > 2:
            raise ValueError("rule exceeds the supported binary interface")
        # Unary output negations are absorbed by inferred private XOR noise.
        op = "root" if not parents else "copy" if len(parents) == 1 else None
        nodes.append({"name": subject, "op": op, "parents": parents, "public_rule": line})
    if len(nodes) != len(names) or {n["name"] for n in nodes} != set(names):
        raise ValueError("rules must be output-first and define each observed variable once")
    question = " ".join(line for line in lines[1:] if not line.startswith("The answer is"))
    mentions = pattern.findall(question)
    bits = re.findall(r"\b[01]\b", question)
    if not mentions or len(bits) > 1:
        raise ValueError("unsupported query or multiple proposed constants")
    proposal = {mentions[0]: int(bits[0])} if bits else {}
    return {"nodes": nodes, "observed": observed, "names": names,
            "question": question, "proposed_override": proposal}


def options_text(question, options):
    return question + "\nOptions:\n" + "\n".join(f"{LETTERS[i]}. {v}" for i, v in enumerate(options))


def build_grounded_requests(rows, training_rows):
    covered = covered_demonstrations(training_rows, ["factual", "intervention", "restored", "intervention"])
    op_examples, query_examples = {}, {"target": [], "active": []}
    for row in covered:
        document = literal_document(row["text"])
        gold_nodes = {n["name"]: n for n in row["scm"]["variables"]}
        for node in document["nodes"]:
            if len(node["parents"]) == 2:
                op = gold_nodes[node["name"]]["op"]
                code = 0 if op in ("and", "nand") else 1 if op in ("or", "nor") else 2
                op_examples.setdefault(code, (node["public_rule"], LETTERS[code]))
        query_examples["target"].append((options_text(document["question"], document["names"]),
                                         LETTERS[document["names"].index(row["scm"]["target"])]))
        query_examples["active"].append((options_text(document["question"], ["No override remains active", "An override remains active"]),
                                         "B" if row["scm"]["intervene"] else "A"))
    systems = {
        "op": ("Choose the base Boolean mechanism relating TWO endogenous parent variables. "
               "Ignore the separate XOR/flip of the variable's private noise. "
               "A is conjunction/AND or output-negated NAND; B is inclusive OR or output-negated NOR; "
               "C is exclusive disjunction/XOR of the two parent variables. "
               "A private-noise XOR alone is not reason to choose C. Reply with one option letter."),
        "target": "Which named variable is the FINAL question asking a value for? Reply with one option letter.",
        "active": ("After ALL instructions in the question, does any constant override remain ACTIVE? "
                   "Removing an override and restoring an original equation leaves no active override. "
                   "Reply with one option letter."),
    }
    requests, plans = {}, {}

    def add(stage, question, options):
        examples = ([(options_text(q, OP_OPTIONS), answer) for q, answer in op_examples.values()]
                    if stage == "op" else query_examples[stage])
        conversation = [{"role": "system", "content": systems[stage]}]
        for q, answer in examples:
            conversation.extend([{"role": "user", "content": q}, {"role": "assistant", "content": answer}])
        conversation.append({"role": "user", "content": options_text(question, options)})
        key = hashlib.sha256(json.dumps(conversation, sort_keys=True).encode()).hexdigest()
        requests.setdefault(key, {"request_id": key, "stage": stage, "messages": conversation,
                                  "choice_count": len(options)})
        return key

    for row in rows:
        document = literal_document(row["text"])
        for node in document["nodes"]:
            if node["op"] is None:
                node["request_id"] = add("op", node["public_rule"], OP_OPTIONS)
        document["target_request"] = add("target", document["question"], document["names"])
        document["active_request"] = (add("active", document["question"],
                                           ["No override remains active", "An override remains active"])
                                      if document["proposed_override"] else None)
        plans[row["case_id"]] = document
    return list(requests.values()), plans


def assemble_grounded(plan, decisions, operator_blind=False):
    nodes = []
    for node in plan["nodes"]:
        op = node["op"]
        if op is None:
            op = "and" if operator_blind else OP_CODES[decisions[node["request_id"]]["choice_index"]]
        nodes.append({"name": node["name"], "op": op, "parents": node["parents"]})
    target = plan["names"][decisions[plan["target_request"]]["choice_index"]]
    active = plan["active_request"] and decisions[plan["active_request"]]["choice_index"] == 1
    return {"variables": nodes, "observed": plan["observed"],
            "intervene": plan["proposed_override"] if active else {}, "target": target}


def run_grounded(model, tokenizer, dataset, cfg, emit):
    import torch
    requests, plans = build_grounded_requests(dataset["rows"], dataset["demonstrations"])
    decisions, started = {}, time.monotonic()
    for stage in ("op", "target", "active"):
        stage_requests = [r for r in requests if r["stage"] == stage]
        for offset in range(0, len(stage_requests), cfg["batch_size"]):
            batch = stage_requests[offset:offset + cfg["batch_size"]]
            prompts = [tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True) for r in batch]
            count = batch[0]["choice_count"]
            assert all(r["choice_count"] == count for r in batch)
            candidate_ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS[:count]]
            if any(len(ids) != 1 for ids in candidate_ids):
                raise ValueError("option letters must each be one tokenizer token")
            encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False).to(model.device)
            positions = encoded["attention_mask"].long().cumsum(-1) - 1
            encoded["position_ids"] = positions.clamp_min(0)
            for device in range(torch.cuda.device_count()):
                torch.cuda.synchronize(device)
            batch_started = time.monotonic()
            with torch.inference_mode():
                logits = model(**encoded, use_cache=False, num_logits_to_keep=1).logits[:, -1, :]
                indices = torch.tensor([ids[0] for ids in candidate_ids], device=logits.device)
                probabilities = logits.index_select(-1, indices).float().softmax(-1).cpu().tolist()
            for device in range(torch.cuda.device_count()):
                torch.cuda.synchronize(device)
            elapsed = time.monotonic() - batch_started
            for i, request in enumerate(batch):
                chosen = max(range(count), key=lambda j: probabilities[i][j])
                decision = {"request_id": request["request_id"], "stage": "grounded_" + stage,
                            "choice_index": chosen, "raw_output": LETTERS[chosen],
                            "conditional_choice_probs": probabilities[i],
                            "input_tokens": int(encoded["attention_mask"][i].sum()), "output_tokens": 1,
                            "seconds": elapsed / len(batch), "max_new_tokens": 1,
                            "note": "one constrained next-token decision; probabilities conditional on valid options"}
                decisions[request["request_id"]] = decision
                emit("CAUSAL_SUBCALL", decision)
            emit("CAUSAL_PROGRESS", {"method": "grounded_solver", "stage": stage,
                                     "done": min(offset + len(batch), len(stage_requests)), "total": len(stage_requests)})
    uses = Counter()
    for plan in plans.values():
        uses.update([n["request_id"] for n in plan["nodes"] if "request_id" in n])
        uses.update([plan["target_request"]])
        if plan["active_request"]:
            uses.update([plan["active_request"]])
    results = {"grounded_solver": [], "grounded_operator_blind": []}
    for row in dataset["rows"]:
        plan = plans[row["case_id"]]
        for name, blind in (("grounded_solver", False), ("grounded_operator_blind", True)):
            program = assemble_grounded(plan, decisions, operator_blind=blind)
            raw = json.dumps(program, separators=(",", ":"))
            decoded = decode_output(raw, "parser_solver", row["scm"])
            keys = [plan["target_request"]]
            if plan["active_request"]:
                keys.append(plan["active_request"])
            if not blind:
                keys.extend(n["request_id"] for n in plan["nodes"] if "request_id" in n)
            record = {k: row[k] for k in ("case_id", "family_id", "split", "view", "answer", "changed")}
            record.update({**decoded, "method": name, "correct": decoded["prediction"] == row["answer"],
                           "raw_output": raw, "subcall_ids": keys,
                           "input_tokens": sum(decisions[k]["input_tokens"] / uses[k] for k in keys),
                           "output_tokens": sum(1 / uses[k] for k in keys),
                           "batch_seconds_per_case": sum(decisions[k]["seconds"] / uses[k] for k in keys),
                           "literal_grounding": True, "unary_private_noise_quotient": True})
            results[name].append(record)
            emit("CAUSAL_CASE", record)
    return results, {"subcalls": len(requests), "elapsed_seconds": time.monotonic() - started,
                     "input_tokens": sum(r["input_tokens"] for r in decisions.values()),
                     "output_tokens": len(requests), "output_tokens_note": "constrained option decisions"}
