"""Independent short-program supervision and public-only semantic requests."""
from collections import Counter
import hashlib
import json
import random

from causal_action_algebra import PROGRAM_SEMANTICS, TRANSITIONS, fold_actions, render_action
from causal_choice import LETTERS
from causal_dataset import digest
from causal_sequence_compiler import action_question, build_sequence_requests, state_options
from causal_semantic_strategies import parse_cot_state
from causal_v3_compiler import with_options


def make_request(stage, messages, count=None):
    key = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
    request = {"request_id": key, "stage": stage, "messages": messages}
    request.update({"choice_count": count} if count is not None else {"max_new_tokens": 512})
    return request


def atomic_request(proposed, instruction):
    system, prompt, options = action_question(proposed, instruction)
    return make_request("action", [{"role": "system", "content": system},
                                  {"role": "user", "content": with_options(prompt, options)}], 16)


def probe_request(proposed, instruction, initial):
    first, second = proposed
    current = "; ".join(f"{name} currently uses " + ("its proposed constant" if initial & (1 << i)
                         else "its original equation") for i, name in enumerate(proposed))
    system = ("Apply ONE instruction to the supplied current clamp state. Do not restart from an empty state. "
              "An original equation means the variable is unclamped; its proposed constant means clamped. "
              "Evaluate the condition before updating. A false condition leaves the state unchanged. "
              "Toggle switches original equation versus proposed constant; it never flips the constant bit. "
              "Reply with one option letter for the resulting clamp state.")
    prompt = (f"First proposal: {first}={proposed[first]}. Second proposal: {second}={proposed[second]}.\n"
              f"Instruction: {instruction}\nBefore this instruction: {current}.\n"
              "After just this one instruction, which clamps are active?")
    return make_request("state_probe", [{"role": "system", "content": system},
                        {"role": "user", "content": with_options(prompt, state_options({"proposed": proposed}))}], 4)


def program_request(proposed, query, cot=False):
    system = PROGRAM_SEMANTICS + (
        " Track the ENTIRE program. State A means neither clamp, B only the first proposal, "
        "C only the second, D both. Write a compact trace, one entry per instruction: 1:A 2:B and so on. "
        "Finish with exactly one marker <state>A</state>, <state>B</state>, <state>C</state>, or <state>D</state>."
        if cot else " Determine which clamps remain active after the ENTIRE program. "
                    "Do not solve the causal target. Reply with one option letter.")
    prompt = with_options(query, state_options({"proposed": proposed}))
    return make_request("program_cot" if cot else "program_state", [
        {"role": "system", "content": system}, {"role": "user", "content": prompt}], None if cot else 4)


def build_supervised_requests(rows, demonstrations):
    old, plans = build_sequence_requests(rows, demonstrations)
    operators = [request for request in old if request["stage"] in ("binary", "ternary")]
    queries = {request["request_id"]: request for request in old if request["stage"] == "action"}
    for plan in plans.values():
        for mode in ("program_state", "program_cot"):
            request = program_request(plan["proposed"], plan["query_text"], mode == "program_cot")
            queries.setdefault(request["request_id"], request)
            plan[mode + "_request"] = request["request_id"]
    return operators, list(queries.values()), plans


def infer_supervised_mask(plan, outputs, mode):
    if mode == "action":
        actions = [outputs[key]["choice_index"] for key in plan["action_requests"]]
        states = fold_actions(actions)
        return states[-1], {"predicted_action_choices": actions, "predicted_clamp_states": states}
    key = plan[mode + "_request"]
    if mode == "program_state":
        return outputs[key]["choice_index"], {}
    if mode == "program_cot":
        raw = outputs[key]["raw_output"]
        return parse_cot_state(raw), {"query_model_output": raw}
    raise ValueError("unknown supervised semantic decoder")


def render_training_action(action, first, second, style):
    if style == 0:
        return render_action(action, first, second, False)
    if style not in (1, 2):
        raise ValueError("training never uses the held-out language renderer")
    if style == 1:
        effects = [f"Set {first}'s override status to active", f"Set {second}'s override status to active",
                   f"Set {first}'s override status to inactive", f"Set {second}'s override status to inactive",
                   "Set both override statuses to active", "Set both override statuses to inactive",
                   f"Flip the active/inactive status of {first}'s override",
                   f"Flip the active/inactive status of {second}'s override",
                   f"Swap the override statuses of {first} and {second}"]
    else:
        effects = [f"Replace {first}'s equation by its proposed constant", f"Replace {second}'s equation by its proposed constant",
                   f"Use the original equation of {first} again", f"Use the original equation of {second} again",
                   "Use both proposed constants", "Use both original equations",
                   f"Switch {first} between its equation and its proposed constant",
                   f"Switch {second} between its equation and its proposed constant",
                   f"Exchange which of {first} and {second} have an override"]
    if action < 9:
        return effects[action] + "."
    if action == 15:
        return "Make no change to the current override statuses."
    effect = {9: 0, 10: 1, 11: 2, 12: 3, 13: 6, 14: 7}[action]
    guard = second if action in (9, 11, 13) else first
    return f"If {guard}'s override status is active, {effects[effect][0].lower() + effects[effect][1:]}; otherwise make no change."


def training_query(proposed, instructions, example, rng):
    names = list(proposed) + [f"train{example}extra{index}" for index in range(10)]
    observed = sorted(rng.sample(names, rng.choice((6, 7, 8))))
    return ("Declared variables: " + ", ".join(names) + ".\nObserved in the original world: " +
            ", ".join(f"{name}={rng.randrange(2)}" for name in observed) + ".\nProposed overrides: " +
            ", ".join(f"{name}={value}" for name, value in proposed.items()) + ".\n" + PROGRAM_SEMANTICS +
            "\nProgram:\n" + "\n".join(f"{i+1}. {line}" for i, line in enumerate(instructions)) +
            f"\nReadout: What is the value of {names[-1]} in this same individual?\n"
            "Answer: 0, 1, or unknown if compatible worlds disagree.\n")


def build_intervention_training(seed=20261032, contexts_per_action=16, programs_per_length_final=16):
    """No evaluation dataset argument: all labels come from independent programs."""
    if contexts_per_action < 1 or programs_per_length_final < 1:
        raise ValueError("supervision counts must be positive")
    rng, rows, serial = random.Random(seed), [], 0

    def append(request, completion, **metadata):
        rows.append({**request, "completion": completion, **metadata})

    for action in range(16):
        for context in range(contexts_per_action):
            serial += 1
            proposed = {f"train{seed}x{serial}": context % 2, f"train{seed}y{serial}": (context // 2) % 2}
            style = context % 3
            instruction = render_training_action(action, *proposed, style)
            append(atomic_request(proposed, instruction), LETTERS[action], actions=[action],
                   proposed=proposed, renderer=style, initial_state=None)
            for state in range(4):
                append(probe_request(proposed, instruction, state), LETTERS[TRANSITIONS[action][state]],
                       actions=[action], proposed=proposed, renderer=style, initial_state=state)
    for length in range(1, 5):
        for endpoint in range(4):
            for index in range(programs_per_length_final):
                serial += 1
                proposed = {f"train{seed}x{serial}": rng.randrange(2), f"train{seed}y{serial}": rng.randrange(2)}
                for attempt in range(10000):
                    actions = [rng.randrange(16) for _ in range(length)]
                    if length > 1 and actions[-1] in (4, 5, 15):
                        continue
                    states = fold_actions(actions)
                    if states[-1] == endpoint:
                        break
                else:
                    raise RuntimeError("short-program support exhausted")
                style = index % 3
                lines = [render_training_action(action, *proposed, style) for action in actions]
                query = training_query(proposed, lines, serial, rng)
                for cot in (False, True):
                    completion = (" ".join(f"{step}:{LETTERS[state]}" for step, state in enumerate(states[1:], 1)) +
                                  f"\n<state>{LETTERS[endpoint]}</state>" if cot else LETTERS[endpoint])
                    append(program_request(proposed, query, cot), completion, actions=actions,
                           proposed=proposed, renderer=style, initial_state=0)
    if len({row["request_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate supervision request")
    audit = {"seed": seed, "examples": len(rows), "training_sha256": digest(rows),
             "tasks": dict(Counter(row["stage"] for row in rows)), "program_lengths": [1, 2, 3, 4],
             "renderers": ["plain_v5", "independent_status_paraphrases", "independent_equation_paraphrases"],
             "heldout_language_renderer_used": False, "evaluation_labels_used": False,
             "names": sorted({name for row in rows for name in row["proposed"]}),
             "program_final_classes": dict(Counter(row["completion"] for row in rows if row["stage"] == "program_state"))}
    return rows, audit


def training_feature(tokenizer, row, max_length):
    prompt = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(row["messages"] + [{"role": "assistant", "content": row["completion"]}],
                                         tokenize=False, add_generation_prompt=False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("assistant completion boundary is not prefix aligned")
    if len(ids) > max_length:
        raise ValueError("supervision exceeds length budget; no target truncation allowed")
    labels = [-100] * len(prompt_ids) + ids[len(prompt_ids):]
    if not any(label != -100 for label in labels):
        raise ValueError("empty completion supervision")
    return {"input_ids": ids, "labels": labels, "stage": row["stage"]}
