"""Public-only atomic semantics and a matched whole-program state control."""
import hashlib
import json
import re

from causal_action_algebra import PROGRAM_SEMANTICS, ACTION_SEMANTICS, action_descriptions, fold_actions, intervention_mask
from causal_choice import LETTERS
from causal_v3_compiler import build_grounded_v3, document_v3, with_options


def program_document(text):
    doc = document_v3(text)
    block = re.search(r"(?ms)^Program:\n(.*?)\nReadout:", doc["query_text"])
    if block is None:
        raise ValueError("missing public program block")
    lines = block.group(1).splitlines()
    matches = [re.fullmatch(r"(\d+)\. (.+)", line) for line in lines]
    if len(matches) != 8 or any(m is None or int(m.group(1)) != i+1 for i, m in enumerate(matches)):
        raise ValueError("expected eight ordered public instructions")
    mentions = [name for name in doc["names"] if re.search(r"\b" + re.escape(name) + r"\b", doc["final_question"])]
    if len(mentions) != 1:
        raise ValueError("final readout must bind exactly one declared public target")
    doc["target"] = mentions[0]
    doc["instructions"] = [m.group(2) for m in matches]
    return doc


def state_options(doc):
    first, second = doc["proposed"]
    return ["Neither proposed clamp is active",
            f"Only {first}'s proposed clamp is active",
            f"Only {second}'s proposed clamp is active",
            "Both proposed clamps are active"]


def build_sequence_requests(rows, demonstrations):
    # Existing six-world operator examples; discard unused target/status requests.
    old_requests, old_plans = build_grounded_v3(rows, demonstrations)
    requests = {r["request_id"]: r for r in old_requests if r["stage"] in ("binary", "ternary")}
    state_examples = []
    for demo in demonstrations:
        doc = program_document(demo["text"])
        state_examples.append((with_options(doc["query_text"], state_options(doc)),
                               intervention_mask(demo["scm"]["intervene"], doc["proposed"])))

    def add(stage, system, prompt, choices, examples=()):
        messages = [{"role": "system", "content": system}]
        for text, label in examples:
            messages.extend([{"role": "user", "content": text}, {"role": "assistant", "content": LETTERS[label]}])
        messages.append({"role": "user", "content": with_options(prompt, choices)})
        key = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
        requests.setdefault(key, {"request_id": key, "stage": stage, "messages": messages, "choice_count": len(choices)})
        return key

    plans = {}
    for row in rows:
        doc = program_document(row["text"])
        first, second = doc["proposed"]
        plan = {**doc, "nodes": old_plans[row["case_id"]]["nodes"]}
        plan["action_requests"] = []
        for instruction in doc["instructions"]:
            prompt = (f"First proposal: {first}={doc['proposed'][first]}. "
                      f"Second proposal: {second}={doc['proposed'][second]}.\n"
                      f"One instruction: {instruction}\nChoose its state-transition meaning, not a final state or a bit value.")
            key = add("action", ACTION_SEMANTICS + " Classify ONE instruction as a function on clamp statuses. "
                      "Do not execute the whole program. Reply with one option letter.",
                      prompt, action_descriptions(first, second))
            plan["action_requests"].append(key)
        plan["state_request"] = add("program_state", PROGRAM_SEMANTICS +
            " Determine which clamps remain active after the ENTIRE program. Do not solve the causal target. Reply with one option letter.",
            doc["query_text"], state_options(doc), state_examples)
        plans[row["case_id"]] = plan
    return list(requests.values()), plans


def assemble_sequence(plan, decisions, variant="sequence"):
    nodes = []
    for node in plan["nodes"]:
        op = node["op"]
        if op is None:
            choices = ["and", "or", "xor"] if node["stage"] == "binary" else ["majority3", "parity3"]
            op = choices[0 if variant == "operator_blind" else decisions[node["request_id"]]["choice_index"]]
        nodes.append({"name": node["name"], "parents": node["parents"], "op": op})
    actions = []
    if variant == "state_choice":
        mask = decisions[plan["state_request"]]["choice_index"]
    elif variant == "action_blind":
        mask = 0
    elif variant in ("sequence", "operator_blind"):
        actions = [decisions[key]["choice_index"] for key in plan["action_requests"]]
        mask = fold_actions(actions)[-1]
    else:
        raise ValueError("unknown sequence compiler variant")
    scm = {"variables": nodes, "observed": plan["observed"], "target": plan["target"],
           "intervene": {name: value for i, (name, value) in enumerate(plan["proposed"].items()) if mask & (1 << i)}}
    return scm, {"predicted_action_choices": actions, "predicted_clamp_states": fold_actions(actions) if actions else [],
                 "final_intervention_mask": mask, "target_binding": "unique declared identifier in public final question"}


def used_requests(plan, variant="sequence"):
    operators = [node["request_id"] for node in plan["nodes"] if "request_id" in node]
    if variant == "operator_blind":
        return plan["action_requests"]
    if variant == "action_blind":
        return operators
    return operators + ([plan["state_request"]] if variant == "state_choice" else plan["action_requests"])
