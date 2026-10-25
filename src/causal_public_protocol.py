"""Original CLadder text inputs, model-grouped splits, and native decoding."""
import hashlib
import json
import random
import re

from causal_native import native_counterfactual, parse_native_params


def stable_rank(value):
    return hashlib.sha256(("cladder-native-public-v1:" + str(value)).encode()).hexdigest()


def public_text(question, model):
    # This is the complete allowlist for evaluation prompt content.
    return "\n\n".join([model["background"], question["given_info"], question["question"]])


def demonstration_program(question, model):
    """Hidden annotations are permitted ONLY for development demonstrations."""
    nodes = parse_native_params(model["params"])
    aliases = {n["name"]: chr(65 + i) for i, n in enumerate(nodes)}
    meaning, params = {}, {}
    for node in nodes:
        old, new = node["name"], aliases[node["name"]]
        mapping = model["variable_mapping"]
        meaning[new] = {"name": mapping[old + "name"], "one": mapping[old + "1"], "zero": mapping[old + "0"]}
        parents = ",".join(aliases[p] for p in node["parents"])
        key = f"p({new} | {parents})" if parents else f"p({new})"
        # All audited CLadder roots have full support; probabilities are immaterial
        # to point-identification. Retain the original value in demonstrations.
        params[key] = node["table"]
    meta = question["meta"]
    return {"meaning": meaning, "params": params,
            "observed": {aliases[k]: v for k, v in meta["given_info"].items()},
            "intervene": {aliases[meta["treatment"]]: meta["action"]},
            "outcome": aliases[meta["outcome"]], "polarity": meta["polarity"]}


def build_public_split(questions, models, models_per_graph=4):
    lookup = {m["model_id"]: m for m in models}
    groups = {}
    for q in questions:
        if q["meta"]["query_type"] == "det-counterfactual":
            groups.setdefault(q["meta"]["graph_id"], {}).setdefault(q["meta"]["model_id"], []).append(q)
    demonstrations, evaluation, audit = [], [], {}
    for graph, worlds in sorted(groups.items()):
        ranked = sorted(worlds, key=stable_rank)
        # 3220 was inspected during source audit, so its entire world is development.
        development = 3220 if 3220 in worlds else ranked[0]
        eligible = [w for w in ranked if w != development]
        if len(eligible) < models_per_graph:
            raise ValueError("insufficient distinct native models")
        demo_q = min(worlds[development], key=lambda q: stable_rank(q["question_id"]))
        if development == 3220:
            demo_q = next((q for q in worlds[development] if q["question_id"] == 73040), demo_q)
        demonstrations.append({"text": public_text(demo_q, lookup[development]),
                               "program": demonstration_program(demo_q, lookup[development]),
                               "answer": demo_q["answer"].lower(), "model_id": development, "graph": graph})
        selected = eligible[:models_per_graph]
        for world in selected:
            q = min(worlds[world], key=lambda q: stable_rank(q["question_id"]))
            evaluation.append({"text": public_text(q, lookup[world]), "answer": q["answer"].lower(),
                               "model_id": world, "question_id": q["question_id"], "graph": graph})
        audit[graph] = {"development_model": development, "evaluation_models": selected}
    demo_order = ["mediation", "chain", "fork", "diamond"]
    chosen = [d for graph in demo_order for d in demonstrations if d["graph"] == graph]
    if len(chosen) < 4:
        chosen += [d for d in demonstrations if d not in chosen][:4-len(chosen)]
    assert not {d["model_id"] for d in demonstrations} & {r["model_id"] for r in evaluation}
    return {"demonstrations": chosen, "rows": evaluation,
            "audit": {"groups": audit, "cases": len(evaluation),
                      "models": len({r["model_id"] for r in evaluation}),
                      "input_fields": ["model.background", "question.given_info", "question.question"],
                      "scope": "hash-selected public development pilot; no population-level claim"}}


SYSTEMS = {
    "direct": "Answer this hypothetical causal question. Reply only <answer>yes</answer> or <answer>no</answer>.",
    "structured": ("Solve the hypothetical causal question by abduction from factual evidence, surgical intervention, "
                   "and prediction using unchanged mechanisms. Explain briefly, then finish with "
                   "<answer>yes</answer> or <answer>no</answer>. Do not assume unmentioned causes."),
    "native_parser_solver": (
        "Translate the public hypothetical world and FINAL question into one JSON program, without answering it. "
        "Use short variable aliases A,B,C,... consistently. Include meaning (name, one, zero), params, observed, "
        "intervene, outcome, polarity. params keys are p(A) for roots and p(C | A,B) for children. "
        "A binary child table lists probabilities of child=1 in parent order: [[f(0,0),f(0,1)],[f(1,0),f(1,1)]]. "
        "Unary tables are [f(0),f(1)]. Child cells must be 0 or 1. Roots without specified probability may use 0.5. "
        "Use ONLY explicitly given factual evidence in observed; do not insert the proposed intervention there. "
        "intervene replaces the treatment equation with its proposed constant. outcome is the variable asked about. "
        "polarity is 1 when asking whether its positive state holds, 0 when asking its negative state. "
        "There is no extra private XOR noise. Output JSON only.")}


def native_messages(row, method, demonstrations):
    conversation = [{"role": "system", "content": SYSTEMS[method]}]
    for demo in demonstrations:
        response = (json.dumps(demo["program"], separators=(",", ":")) if method == "native_parser_solver"
                    else "<answer>" + demo["answer"] + "</answer>")
        conversation.extend([{"role": "user", "content": demo["text"]},
                             {"role": "assistant", "content": response}])
    conversation.append({"role": "user", "content": row["text"]})
    return conversation


def decode_native(text, method):
    try:
        if method == "native_parser_solver":
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end < start:
                raise ValueError("missing JSON program")
            program = json.loads(text[start:end + 1])
            if program["polarity"] not in (0, 1):
                raise ValueError("invalid outcome polarity")
            if len(program["intervene"]) != 1:
                raise ValueError("native single-treatment query requires one intervention")
            result = native_counterfactual(program["params"], program["observed"], program["intervene"], program["outcome"])
            answer = ("unknown" if result["value"] is None else
                      "yes" if result["value"] == program["polarity"] else "no")
            return {"prediction": answer, "valid": True, "parsed_program": program, **result}
        answers = re.findall(r"<answer>\s*(yes|no)\s*</answer>", text.lower())
        if len(answers) != 1:
            raise ValueError("require one answer tag")
        return {"prediction": answers[0], "valid": True}
    except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
        return {"prediction": "invalid", "valid": False, "error": str(exc)}


def public_metrics(records):
    return {"cases": len(records), "accuracy": sum(r["correct"] for r in records)/len(records),
            "valid_rate": sum(r["valid"] for r in records)/len(records),
            "by_graph": {g: sum(r["correct"] for r in records if r["graph"] == g)/sum(r["graph"] == g for r in records)
                         for g in sorted({r["graph"] for r in records})}}


def public_bootstrap(candidate, control):
    paired = {r["question_id"]: r for r in control}
    if set(paired) != {r["question_id"] for r in candidate}:
        raise ValueError("unpaired public evaluation")
    differences = [int(r["correct"]) - int(paired[r["question_id"]]["correct"]) for r in candidate]
    assert len({r["model_id"] for r in candidate}) == len(candidate)
    rng = random.Random(1729)
    samples = sorted(sum(rng.choices(differences, k=len(differences)))/len(differences) for _ in range(2000))
    return {"accuracy_delta": sum(differences)/len(differences), "ci95": [samples[49], samples[1949]],
            "unit": "model (one question per model)", "resamples": 2000}
