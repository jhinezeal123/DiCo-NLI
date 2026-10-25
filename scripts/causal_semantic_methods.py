"""Compare semantic decoders while holding learned operators and controls fixed."""
from collections import Counter
import json
import time

from causal_choice import decide
from causal_operator_replay import load_operator_replay
from causal_semantic_strategies import build_semantic_requests, infer_semantic_state, semantic_request_keys
from causal_sequence_compiler import scm_for_mask


def run_semantic_methods(model, tokenizer, dataset, cfg, emit, score, generate):
    strategy = cfg["sequence_strategy"]
    operators, requests, plans = build_semantic_requests(dataset["rows"], dataset["demonstrations"], strategy)
    operator_outputs = load_operator_replay(cfg, dataset, operators)
    if operator_outputs:
        for value in operator_outputs.values():
            emit("CAUSAL_SUBCALL", value)
    else:
        operator_outputs = decide(model, tokenizer, operators, cfg["batch_size"],
                                 lambda tag, value: emit(tag.replace("PUBLIC_", "CAUSAL_"), value))
    started = time.monotonic()
    if strategy == "state_cot":
        outputs = generate(model, tokenizer, requests, cfg, True)
    else:
        outputs = decide(model, tokenizer, requests, cfg["batch_size"],
                         lambda tag, value: emit(tag.replace("PUBLIC_", "CAUSAL_"), value))
    elapsed = time.monotonic() - started
    by_case = {case: semantic_request_keys(plan, strategy) for case, plan in plans.items()}
    uses = Counter(key for keys in by_case.values() for key in keys)
    masks, diagnostics = {}, {}
    for case, plan in plans.items():
        try:
            masks[case], diagnostics[case] = infer_semantic_state(plan, outputs, strategy)
        except ValueError as exc:
            masks[case] = None
            diagnostics[case] = {"query_compilation_error": str(exc),
                                 "query_model_output": outputs.get(plan.get("cot_request"), {}).get("raw_output", "")}
    for method in ("grounded_solver", "grounded_operator_blind", "grounded_action_blind"):
        action_blind = method == "grounded_action_blind"
        operator_blind = method == "grounded_operator_blind"
        raw, costs, info = {}, {}, {}
        for case, plan in plans.items():
            mask = 0 if action_blind else masks[case]
            raw[case] = (json.dumps(scm_for_mask(plan, operator_outputs, mask, operator_blind), separators=(",", ":"))
                         if mask is not None else "QUERY_COMPILATION_ERROR: missing final state")
            info[case] = {"sequence_strategy": strategy, "final_intervention_mask": mask,
                          **({} if action_blind else diagnostics[case])}
            keys = [] if action_blind else by_case[case]
            costs[case] = {"subcall_ids": keys,
                           "input_tokens": sum(outputs[k]["input_tokens"] / uses[k] for k in keys),
                           "output_tokens": sum(outputs[k]["output_tokens"] / uses[k] for k in keys),
                           "reused_operator_subcall_ids": [] if operator_blind else
                           [n["request_id"] for n in plan["nodes"] if "request_id" in n]}
        usage = {"sequence_strategy": strategy, "new_query_subcalls": 0 if action_blind else len(requests),
                 "input_tokens": 0 if action_blind else sum(v["input_tokens"] for v in outputs.values()),
                 "output_tokens": 0 if action_blind else sum(v["output_tokens"] for v in outputs.values()),
                 "elapsed_seconds": 0 if action_blind else elapsed,
                 "operator_predictions_reused": 0 if operator_blind else len(operator_outputs) if cfg.get("operator_replay") else 0,
                 "operator_source_run": cfg.get("operator_replay", {}).get("source_run"),
                 "cost_scope": "new query work only; frozen learned operator work reported separately; no test gold operators"}
        score(method, raw, costs, usage, diagnostics_by_case=info,
              derived="grounded_solver" if method != "grounded_solver" else None)
