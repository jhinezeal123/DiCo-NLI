"""Tracked compositional execution versus matched closed-state inference."""
from collections import Counter, defaultdict
import json
import time

from causal_choice import decide
from causal_sequence_compiler import build_sequence_requests, assemble_sequence, used_requests


def run_sequence_methods(model, tokenizer, dataset, cfg, emit, score):
    requests, plans = build_sequence_requests(dataset["rows"], dataset["demonstrations"])
    started = time.monotonic()
    decisions = decide(model, tokenizer, requests, cfg["batch_size"],
                       lambda tag, value: emit(tag.replace("PUBLIC_", "CAUSAL_"), value))
    stages = defaultdict(list)
    for key, value in decisions.items():
        stages[value["stage"]].append(key)
    emit("CAUSAL_SEQUENCE_COMPONENT_COSTS", {
        "elapsed_seconds_all_components": time.monotonic() - started,
        "components": {stage: {"subcalls": len(keys),
                              "input_tokens": sum(decisions[k]["input_tokens"] for k in keys),
                              "model_forward_seconds": sum(decisions[k]["seconds"] for k in keys)}
                       for stage, keys in stages.items()},
        "note": "operator requests shared; primary uses atomic action decisions, state control uses whole-program decisions",
    })
    methods = (("grounded_solver", "sequence"), ("grounded_state_choice", "state_choice"),
               ("grounded_operator_blind", "operator_blind"), ("grounded_action_blind", "action_blind"))
    for method, variant in methods:
        keys_by_case = {case: used_requests(plan, variant) for case, plan in plans.items()}
        uses = Counter(key for keys in keys_by_case.values() for key in keys)
        raw, costs, diagnostics = {}, {}, {}
        for case, plan in plans.items():
            scm, diagnostics[case] = assemble_sequence(plan, decisions, variant)
            raw[case] = json.dumps(scm, separators=(",", ":"))
            keys = keys_by_case[case]
            costs[case] = {"input_tokens": sum(decisions[k]["input_tokens"] / uses[k] for k in keys),
                           "output_tokens": sum(1 / uses[k] for k in keys),
                           "batch_seconds_per_case": sum(decisions[k]["seconds"] / uses[k] for k in keys),
                           "subcall_ids": keys}
        usage = {"subcalls": len(uses), "input_tokens": sum(decisions[k]["input_tokens"] for k in uses),
                 "output_tokens": len(uses), "model_forward_seconds": sum(decisions[k]["seconds"] for k in uses),
                 "cost_scope": "unique required requests; shared operators included for both methods, not additive GPU wall time",
                 "atomic_action_examples": 0, "world_demonstrations": len(dataset["demonstrations"])}
        derived = "grounded_solver" if variant in ("operator_blind", "action_blind") else None
        score(method, raw, costs, usage, derived=derived, diagnostics_by_case=diagnostics)
