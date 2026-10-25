"""Literal public graph mentions plus learned deterministic mechanisms/query."""
from collections import Counter
import hashlib
import itertools
import json
import re
import time

from causal_native import parse_native_params
from causal_public_protocol import decode_native


def public_graph(text):
    # The author's standard background ends before the first blank line.
    background = text.split("\n\n", 1)[0]
    if ":" not in background:
        raise ValueError("unsupported public graph header")
    parents, unobserved = {}, set()
    for sentence in background.split(":", 1)[1].split("."):
        sentence = sentence.strip().lower()
        if not sentence:
            continue
        if sentence.endswith(" is unobserved"):
            unobserved.add(sentence.removesuffix(" is unobserved"))
            continue
        if " has a direct effect on " not in sentence:
            raise ValueError("unsupported public graph sentence: " + sentence)
        cause, effects = sentence.split(" has a direct effect on ")
        parents.setdefault(cause, set())
        for effect in re.split(r",\s*(?:and\s+)?|\s+and\s+", effects):
            parents.setdefault(effect, set()).add(cause)
    if not parents or unobserved-set(parents):
        raise ValueError("public graph missing named variables")
    params = {}
    for name, upstream in sorted(parents.items()):
        if len(upstream) > 3:
            raise ValueError("public deterministic pilot supports at most three parents")
        key = "p(" + name + (" | " + ",".join(sorted(upstream)) if upstream else "") + ")"
        table = .5
        if upstream:
            table = 0
            for _ in upstream:
                table = [table, table]
        params[key] = table
    parse_native_params(params)  # DAG/grammar check, no hidden graph.
    return {"parents": {name: sorted(upstream) for name, upstream in sorted(parents.items())},
            "unobserved": sorted(unobserved)}


SYSTEMS = {
    "cell": ("Evaluate ONE deterministic causal mechanism from the stated hypothetical rules. "
             "All direct parents are fixed to specified bits; there are no extra causes/noise. "
             "Ignore factual observations when evaluating the mechanism, and ignore the separate proposed intervention. "
             "Bits 1/0 denote the positive/negative or high/low states expressed in the premises. "
             "Choose the child bit, not a probability. Reply with one option letter."),
    "evidence": ("Extract original FACTUAL evidence for one named variable. Distinguish an asserted observation "
                 "from a causal rule or a proposed hypothetical intervention. A variable mentioned in a mechanism "
                 "is not thereby observed. An 'if ... instead of ...' proposal is not factual evidence. "
                 "Reply with one option letter."),
    "treatment": "Which variable does the FINAL counterfactual proposal externally set? Reply with one option letter.",
    "value": "What bit is the treatment set to in the FINAL hypothetical proposal (before 'instead of')? Reply with one option letter.",
    "outcome": "Which variable's outcome is the FINAL question asking about? Reply with one option letter.",
    "polarity": "Does the FINAL question ask whether the outcome's positive/high/present state (1), or negative/low/absent state (0), holds? Reply with one option letter.",
}


def request_specs(text):
    graph = public_graph(text)
    names = list(graph["parents"])
    question = text.rsplit("\n\n", 1)[-1]
    premise = text.rsplit("\n\n", 1)[0]
    specs = []
    for name, parents in graph["parents"].items():
        if parents:
            for bits in itertools.product((0, 1), repeat=len(parents)):
                assignment = ", ".join(f"{p}={bit}" for p, bit in zip(parents, bits))
                specs.append({"stage": "cell", "name": name, "bits": list(bits), "parents": parents,
                    "prompt": premise + f'\n\nFor the mechanism of "{name}", set ALL its parents to {assignment}. What is the child bit?',
                    "options": ["child bit 0", "child bit 1"]})
        if name not in graph["unobserved"]:
            specs.append({"stage": "evidence", "name": name,
                "prompt": text + f'\n\nIs "{name}" explicitly observed in the original actual world?',
                "options": ["Not observed / unspecified", "Observed as bit 0", "Observed as bit 1"]})
    for stage in ("treatment", "value", "outcome", "polarity"):
        options = names if stage in ("treatment", "outcome") else ["bit 0", "bit 1"]
        specs.append({"stage": stage, "prompt": text + "\n\n" + SYSTEMS[stage], "options": options})
    return graph, specs


def development_label(spec, program):
    nodes = {n["name"]: n for n in parse_native_params(program["params"])}
    name_to_alias = {meaning["name"].lower(): alias for alias, meaning in program["meaning"].items()}
    stage = spec["stage"]
    if stage == "cell":
        assignment = {name_to_alias[k]: v for k, v in zip(spec["parents"], spec["bits"])}
        node = nodes[name_to_alias[spec["name"]]]
        value = node["table"]
        for parent in node["parents"]:
            value = value[assignment[parent]]
        return int(value)
    if stage == "evidence":
        value = program["observed"].get(name_to_alias[spec["name"]])
        return 0 if value is None else int(value)+1
    if stage == "value":
        return next(iter(program["intervene"].values()))
    if stage == "polarity":
        return program["polarity"]
    alias = next(iter(program["intervene"])) if stage == "treatment" else program["outcome"]
    return spec["options"].index(program["meaning"][alias]["name"].lower())


def format_options(spec):
    return spec["prompt"] + "\nOptions:\n" + "\n".join(f"{'ABCDEFGH'[i]}. {v}" for i, v in enumerate(spec["options"]))


def build_public_grounded_requests(rows, demonstrations):
    examples = {stage: [] for stage in SYSTEMS}
    for demo in demonstrations:
        _, specs = request_specs(demo["text"])
        for spec in specs:
            examples[spec["stage"]].append((format_options(spec), development_label(spec, demo["program"])))
    # Balance available development option labels, bounded to keep context modest.
    selected = {}
    for stage, candidates in examples.items():
        used, chosen = Counter(), []
        limit = 3 if stage == "cell" else 2 if stage == "evidence" else 1
        for prompt, answer in candidates:
            if used[answer] < limit:
                chosen.append((prompt, answer)); used[answer] += 1
        selected[stage] = chosen
    requests, plans = {}, {}
    for row in rows:
        graph, specs = request_specs(row["text"])
        for spec in specs:
            stage = spec["stage"]
            conversation = [{"role": "system", "content": SYSTEMS[stage]}]
            for prompt, answer in selected[stage]:
                conversation.extend([{"role": "user", "content": prompt},
                                     {"role": "assistant", "content": "ABCDEFGH"[answer]}])
            conversation.append({"role": "user", "content": format_options(spec)})
            key = hashlib.sha256(json.dumps(conversation, sort_keys=True).encode()).hexdigest()
            requests.setdefault(key, {"request_id": key, "stage": stage, "messages": conversation,
                                      "choice_count": len(spec["options"])})
            spec["request_id"] = key
        plans[row["question_id"]] = {"graph": graph, "specs": specs}
    return list(requests.values()), plans


def assemble_public(plan, decisions):
    graph, specs = plan["graph"], plan["specs"]
    params, observed, query = {}, {}, {}
    for name, parents in graph["parents"].items():
        if not parents:
            params[f"p({name})"] = .5  # Audited full-support roots; point identification only.
            continue
        cell_values = {}
        for spec in specs:
            if spec["stage"] == "cell" and spec["name"] == name:
                cell_values[tuple(spec["bits"])] = decisions[spec["request_id"]]["choice_index"]
        def table(prefix):
            return cell_values[prefix] if len(prefix) == len(parents) else [table(prefix+(0,)), table(prefix+(1,))]
        params["p(" + name + " | " + ",".join(parents) + ")"] = table(())
    for spec in specs:
        choice = decisions[spec["request_id"]]["choice_index"]
        stage = spec["stage"]
        if stage == "evidence" and choice:
            observed[spec["name"]] = choice-1
        elif stage in ("treatment", "outcome"):
            query[stage] = spec["options"][choice]
        elif stage in ("value", "polarity"):
            query[stage] = choice
    return {"params": params, "observed": observed,
            "intervene": {query["treatment"]: query["value"]},
            "outcome": query["outcome"], "polarity": query["polarity"]}


def run_public_grounded(model, tokenizer, dataset, cfg, emit):
    from causal_choice import decide
    requests, plans = build_public_grounded_requests(dataset["rows"], dataset["demonstrations"])
    started = time.monotonic()
    decisions = decide(model, tokenizer, requests, cfg["batch_size"], emit)
    uses = Counter(spec["request_id"] for plan in plans.values() for spec in plan["specs"])
    records = []
    for row in dataset["rows"]:
        plan = plans[row["question_id"]]
        program = assemble_public(plan, decisions)
        raw = json.dumps(program, separators=(",", ":"))
        decoded = decode_native(raw, "native_parser_solver")
        keys = [spec["request_id"] for spec in plan["specs"]]
        record = {k: row[k] for k in ("question_id", "model_id", "graph", "answer")}
        record.update({**decoded, "raw_output": raw, "method": "native_grounded_solver",
            "correct": decoded["prediction"] == row["answer"], "subcall_ids": keys,
            "input_tokens": sum(decisions[k]["input_tokens"]/uses[k] for k in keys),
            "output_tokens": sum(1/uses[k] for k in keys),
            "batch_seconds_per_case": sum(decisions[k]["seconds"]/uses[k] for k in keys),
            "literal_public_graph": True, "new_lm_calls": len(keys)})
        records.append(record); emit("PUBLIC_CASE", record)
    return records, {"subcalls": len(requests), "elapsed_seconds": time.monotonic()-started,
                     "input_tokens": sum(r["input_tokens"] for r in decisions.values()), "output_tokens": len(requests)}
