"""Four-view causal evaluation; incomplete observations remain incomplete."""
import json
import random

from causal_scm import canonical, parse_scm
from causal_dataset_v3 import VIEWS, solve_conditioned
from causal_protocol import decode_output

SCHEMA_V3 = (
    'Return ONLY JSON with exactly variables, observed, intervene, target. '
    'variables is a list of {"name":"...","op":"...","parents":["..."]}. '
    'Allowed operations: root (0 parents), copy/not (1), and/or/xor/nand/nor (2), majority3/parity3 (3). '
    'majority3 is 1 iff at least two parents are 1; parity3 is 1 iff an odd number are 1. '
    'The independent private XOR noise is implicit, never a parent or a separate variable. '
    'observed contains ONLY explicitly stated original observations; do not fill missing bits. '
    'intervene contains ONLY proposed overrides still active AFTER all query instructions. '
    'Removing one override leaves the other active unless it too is removed. '
    'target is the variable in the final question. No answers or noise assignments in JSON.'
)


def messages_v3(row, method, demonstrations):
    if method == "parser_solver":
        system = "Compile the public binary SCM and final query. " + SCHEMA_V3
    else:
        system = ("Solve same-individual binary causal queries. Abduce ALL noise worlds compatible with incomplete "
                  "original observations. Apply only overrides still active after all instructions, retaining each world's "
                  "same noise. Return unknown if resulting target values disagree. ")
        system += ("Reply ONLY <answer>0</answer>, <answer>1</answer>, or <answer>unknown</answer>." if method == "direct" else
                   "Explain briefly, then finish with <answer>0</answer>, <answer>1</answer>, or <answer>unknown</answer>.")
    result = [{"role": "system", "content": system}]
    for demo in demonstrations:
        if method == "parser_solver":
            answer = json.dumps(demo["scm"], separators=(",", ":"))
        elif method == "direct":
            answer = f"<answer>{demo['answer']}</answer>"
        else:
            oracle = solve_conditioned(demo["scm"])
            answer = (f"Conditioning on the original observations leaves {oracle['compatible_worlds']} noise worlds. "
                "For each, keep its noise and replace only " + json.dumps(demo["scm"]["intervene"]) +
                "; propagate the remaining equations. The possible target values are " + json.dumps(oracle["outcomes"]) +
                f". <answer>{demo['answer']}</answer>")
        result.extend([{"role": "user", "content": demo["text"]}, {"role": "assistant", "content": answer}])
    result.append({"role": "user", "content": row["text"]})
    return result


def decode_v3(raw, method, gold=None, normalized=False):
    if method != "parser_solver":
        return decode_output(raw, method)
    try:
        parsed = parse_scm(raw, reduce_unary=normalized)
        result = solve_conditioned(parsed)
        decoded = {"prediction": result["answer"], "valid": True, "parsed_scm": parsed,
                   "compatible_worlds": result["compatible_worlds"]}
        if gold is not None:
            decoded["schema_exact"] = canonical(parsed) == canonical(gold)
            decoded["query_exact"] = all(parsed[k] == gold[k] for k in ("observed", "intervene", "target"))
        return decoded
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        return {"prediction": "invalid", "valid": False, "error": str(exc), "schema_exact": False, "query_exact": False}


def family_groups(rows):
    result = {}
    for row in rows:
        group = result.setdefault(row["family_id"], {})
        if row["view"] in group:
            raise ValueError("duplicate family/view prediction")
        group[row["view"]] = row
    if any(set(group) != set(VIEWS) for group in result.values()):
        raise ValueError("v3 scoring requires all FOUR views per family")
    return result


def metrics_v3(records):
    result = {}
    for split in sorted({r["split"] for r in records}):
        rows = [r for r in records if r["split"] == split]
        groups = family_groups(rows)
        interventions = [g["intervention"] for g in groups.values()]
        by_label = {label: sum(r["correct"] for r in interventions if r["answer"] == label)/sum(r["answer"] == label for r in interventions)
                    for label in ("0", "1", "unknown")}
        result[split] = {"cases": len(rows), "families": len(groups),
            "accuracy": sum(r["correct"] for r in rows)/len(rows),
            "joint_accuracy": sum(all(r["correct"] for r in g.values()) for g in groups.values())/len(groups),
            "valid_rate": sum(r["valid"] for r in rows)/len(rows), "intervention_by_label": by_label,
            "intervention_balanced_accuracy": sum(by_label.values())/3,
            "partial_release_accuracy": sum(g["partial_release"]["correct"] for g in groups.values())/len(groups),
            "dcc": sum(g["factual"]["valid"] and g["restored"]["valid"] and
                       g["factual"]["prediction"] == g["restored"]["prediction"] for g in groups.values())/len(groups)}
    return result


def paired_v3(candidate, control):
    output = {}; rng = random.Random(991)
    for split in sorted({r["split"] for r in candidate}):
        a = family_groups([r for r in candidate if r["split"] == split])
        b = family_groups([r for r in control if r["split"] == split])
        if set(a) != set(b):
            raise ValueError("unpaired v3 families")
        differences = [int(all(r["correct"] for r in a[k].values())) - int(all(r["correct"] for r in b[k].values())) for k in sorted(a)]
        n = len(differences)
        draws = sorted(sum(rng.choices(differences, k=n))/n for _ in range(2000))
        output[split] = {"families": n, "joint_delta": sum(differences)/n, "ci95": [draws[49], draws[1949]],
                         "unit": "four-view family"}
    return output
