"""Prompt construction and strict decoding, separate from model and gold labels."""
import json
import re

from causal_scm import canonical, parse_scm, solve


SCHEMA = (
    'Return ONLY JSON with exactly these keys: '
    '{"variables":[{"name":"...","op":"root|copy|not|and|or|xor|nand|nor",'
    '"parents":["..."]}],"observed":{"name":0},"intervene":{"name":1},"target":"..."}. '
    'Use all and only the stated variables. Root has zero parents, copy/not one, others two. '
    'Every non-root operation is followed by XOR with that variable\'s private noise; do not represent noise as a variable. '
    'observed describes the ORIGINAL world, including observations on intervened variables. '
    'intervene describes ONLY overrides still in effect at query time. '
    'For restored original equations, intervene is empty. Do not add inferred noise, answers or explanations.'
)


def trace(row):
    # This oracle explanation is used only on pre-generated demonstration cases.
    scm = row["scm"]
    from causal_scm import mechanism
    noise = {v["name"]: scm["observed"][v["name"]] ^
             mechanism(v["op"], [scm["observed"][p] for p in v["parents"]])
             for v in scm["variables"]}
    return ("Abduction from the original observations gives noise " + json.dumps(noise, separators=(",", ":")) +
            ". Replace only " + json.dumps(scm["intervene"]) +
            "; retain all noise and propagate in causal order. " +
            f"<answer>{row['answer']}</answer>")


def covered_demonstrations(demonstrations, views):
    import copy
    from causal_dataset import render
    examples = []
    for demo, view in zip(demonstrations, views):
        example = copy.deepcopy(demo)
        cause, value = next(iter(example["scm"]["intervene"].items()))
        if view != "intervention":
            example["scm"]["intervene"] = {}
        example["view"] = view
        example["text"] = render(example["scm"], view, cause, value)
        example["answer"] = solve(example["scm"])["answer"]
        examples.append(example)
    return examples


def messages(row, method, demonstrations, shots=2, demo_views=None):
    if method == "parser_solver":
        system = "Translate a described binary structural causal model into JSON. " + SCHEMA
    elif method == "structured":
        system = ("Solve binary structural causal counterfactuals. First infer each noise bit from the ORIGINAL "
                  "observations, then replace intervened equations and propagate with the SAME noise. "
                  "Explain briefly. End with <answer>0</answer>, <answer>1</answer> or <answer>unknown</answer>.")
    elif method == "direct":
        system = ("Solve binary structural causal counterfactuals. Reply ONLY with <answer>0</answer>, "
                  "<answer>1</answer> or <answer>unknown</answer>, with no explanation.")
    else:
        raise ValueError(method)
    conversation = [{"role": "system", "content": system}]
    # All four strata balance original outcome, causal change and final answers.
    # Each protocol receives the same cases; only supervision format differs.
    examples = covered_demonstrations(demonstrations, demo_views) if demo_views else demonstrations
    for demo in examples[:shots]:
        answer = (json.dumps(demo["scm"], separators=(",", ":")) if method == "parser_solver" else
                  trace(demo) if method == "structured" else f"<answer>{demo['answer']}</answer>")
        conversation.extend([{"role": "user", "content": demo["text"]},
                             {"role": "assistant", "content": answer}])
    conversation.append({"role": "user", "content": row["text"]})
    return conversation


def decode_output(text, method, gold_scm=None, reduce_unary=False):
    if method == "parser_solver":
        try:
            parsed = parse_scm(text, reduce_unary=reduce_unary)
            result = solve(parsed)
        except (ValueError, TypeError, KeyError) as exc:
            return {"prediction": "invalid", "valid": False, "error": str(exc), "schema_exact": False}
        fields_wrong = []
        if gold_scm is not None:
            gold = json.loads(canonical(gold_scm))
            current = json.loads(canonical(parsed))
            fields_wrong = [k for k in gold if gold[k] != current[k]]
        return {"prediction": result["answer"], "valid": True,
                "schema_exact": not fields_wrong, "fields_wrong": fields_wrong,
                "parsed_scm": parsed, "compatible_worlds": result["compatible_worlds"]}
    answers = re.findall(r"<answer>\s*(0|1|unknown)\s*</answer>", text, flags=re.I)
    if answers:
        return {"prediction": answers[-1].lower(), "valid": True}
    if text.strip().lower() in ("0", "1", "unknown"):
        return {"prediction": text.strip().lower(), "valid": True}
    return {"prediction": "invalid", "valid": False, "error": "missing unambiguous answer"}


def metrics(records):
    result = {}
    for split in sorted({r["split"] for r in records}):
        group = [r for r in records if r["split"] == split]
        families = {}
        for record in group:
            families.setdefault(record["family_id"], {})[record["view"]] = record
        joint = [all(v[k]["correct"] for k in ("factual", "intervention", "restored")) for v in families.values()]
        result[split] = {
            "cases": len(group), "families": len(families),
            "accuracy": sum(r["correct"] for r in group) / len(group),
            "joint_accuracy": sum(joint) / len(joint),
            "valid_rate": sum(r["valid"] for r in group) / len(group),
            "dcc": sum(v["factual"]["prediction"] == v["restored"]["prediction"] and
                       v["factual"]["valid"] and v["restored"]["valid"] for v in families.values()) / len(families),
            "intervention_accuracy": sum(v["intervention"]["correct"] for v in families.values()) / len(families),
            "schema_exact": (sum(r.get("schema_exact", False) for r in group) / len(group)
                             if any("schema_exact" in r for r in group) else None),
        }
    return result


def paired_bootstrap(candidate, reference, seed=991, repetitions=2000):
    import random
    rng, output = random.Random(seed), {}
    for split in sorted({r["split"] for r in candidate}):
        groups = []
        for rows in (candidate, reference):
            families = {}
            for r in rows:
                if r["split"] == split:
                    families.setdefault(r["family_id"], []).append(r)
            groups.append({k: int(len(v) == 3 and all(r["correct"] for r in v)) for k, v in families.items()})
        assert set(groups[0]) == set(groups[1])
        differences = [groups[0][k] - groups[1][k] for k in sorted(groups[0])]
        n = len(differences)
        samples = sorted(sum(rng.choice(differences) for _ in range(n)) / n for _ in range(repetitions))
        output[split] = {"joint_delta": sum(differences) / n,
                         "ci95": [samples[int(.025 * repetitions)], samples[int(.975 * repetitions)]],
                         "families": n}
    return output
