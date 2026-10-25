"""Finite binary SCMs with exact same-individual counterfactual inference.

Every endogenous variable is f(parents) XOR U_name; roots equal U_name.
Noise bits are independent Bernoulli(1/2). Observation conditions the factual
world before an intervention replaces equations, without resampling noise.
"""
from __future__ import annotations

import itertools
import json


ARITIES = {"root": 0, "copy": 1, "not": 1, "and": 2, "or": 2,
           "xor": 2, "nand": 2, "nor": 2, "majority3": 3, "parity3": 3}


def normalize_scm(value, reduce_unary=False):
    if not isinstance(value, dict) or set(value) != {"variables", "observed", "intervene", "target"}:
        raise ValueError("SCM must have exactly variables, observed, intervene, target")
    variables = value["variables"]
    if not isinstance(variables, list) or not 1 <= len(variables) <= 8:
        raise ValueError("one to eight variables required")
    by_name = {}
    for node in variables:
        if not isinstance(node, dict) or set(node) != {"name", "op", "parents"}:
            raise ValueError("invalid variable schema")
        name, op, parents = node["name"], node["op"], node["parents"]
        # Exact variadic Boolean identities, never guessed missing operands.
        # AND(x)=OR(x)=XOR(x)=x; NAND(x)=NOR(x)=NOT(x).
        if reduce_unary and isinstance(parents, list) and len(parents) == 1:
            if op in ("and", "or", "xor"):
                op = "copy"
            elif op in ("nand", "nor"):
                op = "not"
        if not isinstance(name, str) or not name or name in by_name:
            raise ValueError("variable names must be unique nonempty strings")
        if op not in ARITIES or not isinstance(parents, list) or len(parents) != ARITIES[op]:
            raise ValueError("invalid operation or parent count")
        if any(not isinstance(p, str) for p in parents) or len(set(parents)) != len(parents):
            raise ValueError("parents must be unique strings")
        by_name[name] = {"name": name, "op": op, "parents": sorted(parents)}
    ordered, remaining = [], dict(by_name)
    while remaining:
        ready = sorted(name for name, node in remaining.items()
                       if all(p in {n["name"] for n in ordered} for p in node["parents"]))
        if not ready:
            raise ValueError("cycle or unknown parent")
        for name in ready:
            ordered.append(remaining.pop(name))
    result = {"variables": ordered, "target": value["target"]}
    if result["target"] not in by_name:
        raise ValueError("unknown query target")
    for field in ("observed", "intervene"):
        assignment = value[field]
        if not isinstance(assignment, dict) or set(assignment) - set(by_name):
            raise ValueError("unknown assignment variable")
        if any(type(bit) is not int or bit not in (0, 1) for bit in assignment.values()):
            raise ValueError("assignments must be integer binary values")
        result[field] = dict(sorted(assignment.items()))
    return result


def mechanism(op, values):
    if op == "root":
        return 0
    if op == "copy":
        return values[0]
    if op == "not":
        return 1 - values[0]
    if op == "and":
        return values[0] & values[1]
    if op == "or":
        return values[0] | values[1]
    if op == "xor":
        return values[0] ^ values[1]
    if op == "nand":
        return 1 - (values[0] & values[1])
    if op == "nor":
        return 1 - (values[0] | values[1])
    if op == "majority3":
        return int(sum(values) >= 2)
    if op == "parity3":
        return sum(values) % 2
    raise ValueError(op)


def forward(variables, noise, intervention=None):
    state, intervention = {}, intervention or {}
    for node in variables:
        name = node["name"]
        state[name] = (intervention[name] if name in intervention else
                       mechanism(node["op"], [state[p] for p in node["parents"]]) ^ noise[name])
    return state


def solve(value):
    scm = normalize_scm(value)
    names = [v["name"] for v in scm["variables"]]
    outcomes, worlds = set(), 0
    for bits in itertools.product((0, 1), repeat=len(names)):
        noise = dict(zip(names, bits))
        factual = forward(scm["variables"], noise)
        if any(factual[k] != v for k, v in scm["observed"].items()):
            continue
        worlds += 1
        outcomes.add(forward(scm["variables"], noise, scm["intervene"])[scm["target"]])
    if not worlds:
        raise ValueError("observations contradict all possible noise assignments")
    answer = str(next(iter(outcomes))) if len(outcomes) == 1 else "unknown"
    return {"answer": answer, "compatible_worlds": worlds, "outcomes": sorted(outcomes)}


def canonical(value):
    return json.dumps(normalize_scm(value), sort_keys=True, separators=(",", ":"))


def contextual_equation_signature(value):
    """Effective structural equations after abduction from complete observations.

    Equal signatures imply equal endogenous responses to every intervention in
    this observed individual's context. This is not global SCM identification.
    """
    scm = normalize_scm(value)
    names = sorted(n["name"] for n in scm["variables"])
    if set(scm["observed"]) != set(names):
        raise ValueError("contextual equation comparison requires complete observations")
    result = {}
    for node in scm["variables"]:
        noise = scm["observed"][node["name"]] ^ mechanism(
            node["op"], [scm["observed"][p] for p in node["parents"]])
        others = [name for name in names if name != node["name"]]
        table = []
        for bits in itertools.product((0, 1), repeat=len(others)):
            assignment = dict(zip(others, bits))
            table.append(mechanism(node["op"], [assignment[p] for p in node["parents"]]) ^ noise)
        result[node["name"]] = tuple(table)
    return result


def parse_scm(text, reduce_unary=False):
    decoder = json.JSONDecoder()
    for position, token in enumerate(text):
        if token != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return normalize_scm(parsed, reduce_unary=reduce_unary)
    raise ValueError("no decodable JSON object")
