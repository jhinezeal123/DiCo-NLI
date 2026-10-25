"""Alternative zero-shot semantic decoders and a whole-program CoT competitor."""
import hashlib
import json
import math
import re

from causal_action_algebra import ACTION_SEMANTICS, PROGRAM_SEMANTICS, TRANSITIONS, intervention_mask
from causal_sequence_compiler import build_sequence_requests, program_document, state_options
from causal_v3_compiler import with_options

BASE_EFFECTS = tuple(range(9)) + (15,)


def slot_transition(effect, guard):
    if effect not in range(10) or guard not in range(3):
        raise ValueError("invalid semantic slot")
    table = TRANSITIONS[BASE_EFFECTS[effect]]
    return tuple(table[state] if guard == 0 or state & (1 << (guard-1)) else state for state in range(4))


def project_transition(probabilities):
    if len(probabilities) != 4 or any(len(row) != 4 or any(p < 0 or not math.isfinite(p) for p in row) for row in probabilities):
        raise ValueError("four states require four probability distributions")
    scores = [sum(math.log(max(probabilities[state][table[state]], 1e-12)) for state in range(4)) for table in TRANSITIONS]
    chosen = max(range(16), key=lambda index: scores[index])
    return TRANSITIONS[chosen], chosen, scores


def fold_tables(tables):
    states = [0]
    for table in tables:
        if len(table) != 4 or any(value not in range(4) for value in table):
            raise ValueError("invalid learned transition table")
        states.append(table[states[-1]])
    return states


def parse_cot_state(raw):
    matches = re.findall(r"<state>\s*([A-D])\s*</state>", raw, flags=re.I)
    if not matches:
        raise ValueError("missing strict final state marker")
    return "ABCD".index(matches[-1].upper())


def build_semantic_requests(rows, demonstrations, strategy):
    if strategy not in ("slots", "probes", "state_cot"):
        raise ValueError("unknown semantic strategy")
    old, plans = build_sequence_requests(rows, demonstrations)
    operators = [r for r in old if r["stage"] in ("binary", "ternary")]
    requests = {}

    def add(stage, messages, count=None):
        key = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
        request = {"request_id": key, "stage": stage, "messages": messages}
        if count is not None:
            request["choice_count"] = count
        else:
            request["max_new_tokens"] = 512
        requests.setdefault(key, request)
        return key

    for plan in plans.values():
        first, second = plan["proposed"]
        context = f"First proposal: {first}={plan['proposed'][first]}. Second proposal: {second}={plan['proposed'][second]}.\n"
        plan["semantic_requests"] = {}
        if strategy == "state_cot":
            system = (PROGRAM_SEMANTICS + " Work through the entire program, explicitly tracking the clamps after each step. "
                      "State A means neither, B only the first proposal, C only the second, D both. "
                      "Finish ONLY with one final marker <state>A</state>, <state>B</state>, <state>C</state>, or <state>D</state>.")
            messages = [{"role": "system", "content": system}]
            for demo in demonstrations:
                doc = program_document(demo["text"])
                mask = intervention_mask(demo["scm"]["intervene"], doc["proposed"])
                messages.extend([{"role": "user", "content": with_options(doc["query_text"], state_options(doc))},
                                 {"role": "assistant", "content": f"<state>{'ABCD'[mask]}</state>"}])
            messages.append({"role": "user", "content": with_options(plan["query_text"], state_options(plan)) +
                             "\nShow the clamp status after each numbered instruction, then give the final state marker."})
            plan["cot_request"] = add("program_cot", messages)
            continue
        for old_key, instruction in zip(plan["action_requests"], plan["instructions"]):
            prompt = context + "Instruction: " + instruction
            if strategy == "slots":
                effects = [f"Make {first} obey its proposed constant", f"Make {second} obey its proposed constant",
                           f"Make {first}'s original equation operate", f"Make {second}'s original equation operate",
                           "Make both proposed constants operate", "Make both original equations operate",
                           f"Switch {first} between original equation and proposed constant",
                           f"Switch {second} between original equation and proposed constant",
                           "Exchange the two clamp statuses", "Leave the current state unchanged"]
                effect_system = (ACTION_SEMANTICS + " Assume any stated prerequisite holds. Identify ONLY the resulting update. "
                                 "Do not identify the prerequisite or evaluate its current truth. Reply with one option letter.")
                guard_system = (ACTION_SEMANTICS + " Identify ONLY the external prerequisite for executing the update. "
                                "Distinguish a prerequisite from a toggle defining opposite updates in different states. "
                                "Reply with one option letter.")
                effect = add("effect", [{"role": "system", "content": effect_system},
                                       {"role": "user", "content": with_options(prompt, effects)}], 10)
                guard = add("guard", [{"role": "system", "content": guard_system},
                                      {"role": "user", "content": with_options(prompt, ["No external prerequisite",
                                        f"Only when {first}'s clamp is active", f"Only when {second}'s clamp is active"])}], 3)
                plan["semantic_requests"][old_key] = [effect, guard]
            else:
                keys = []
                for state in range(4):
                    current = "; ".join(f"{name} currently uses " + ("its proposed constant" if state & (1 << i) else "its original equation")
                                        for i, name in enumerate(plan["proposed"]))
                    system = ("Apply ONE instruction to the supplied current clamp state. Do not restart from an empty state. "
                              "An original equation means the variable is unclamped; its proposed constant means clamped. "
                              "Evaluate the condition before updating. A false condition leaves the state unchanged. "
                              "Toggle switches original equation versus proposed constant; it never flips the constant bit. "
                              "Reply with one option letter for the resulting clamp state.")
                    keys.append(add("state_probe", [{"role": "system", "content": system},
                              {"role": "user", "content": with_options(prompt + "\nBefore this instruction: " + current +
                                 ".\nAfter just this one instruction, which clamps are active?", state_options(plan))}], 4))
                plan["semantic_requests"][old_key] = keys
    return operators, list(requests.values()), plans


def infer_semantic_state(plan, outputs, strategy):
    if strategy == "state_cot":
        raw = outputs[plan["cot_request"]]["raw_output"]
        return parse_cot_state(raw), {"query_model_output": raw}
    tables, details = [], []
    for old_key in plan["action_requests"]:
        keys = plan["semantic_requests"][old_key]
        if strategy == "slots":
            effect, guard = (outputs[key]["choice_index"] for key in keys)
            table = slot_transition(effect, guard)
            details.append({"effect": effect, "guard": guard})
        else:
            table, chosen, _ = project_transition([outputs[key]["conditional_choice_probs"] for key in keys])
            details.append({"projected_action": chosen})
        tables.append(table)
    states = fold_tables(tables)
    return states[-1], {"predicted_clamp_states": states, "predicted_transition_tables": tables,
                        "semantic_decisions": details}


def semantic_request_keys(plan, strategy):
    if strategy == "state_cot":
        return [plan["cot_request"]]
    return [key for old_key in plan["action_requests"] for key in plan["semantic_requests"][old_key]]
