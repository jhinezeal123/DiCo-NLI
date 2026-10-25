"""Exact point counterfactuals for deterministic CPTs with uncertain root bits.

Unlike the synthetic suite, non-root equations here have no private XOR noise.
Published probabilities on root variables determine support; point queries do
not require averaging when every compatible world yields the same outcome.
"""
import itertools
import re


def _cell(table, indices):
    value = table
    for bit in indices:
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError("CPT dimensions must each have size two")
        value = value[bit]
    if value not in (0, 1):
        raise ValueError("conditional CPT is not deterministic")
    return int(value)


def parse_native_params(params):
    nodes = {}
    for key, table in params.items():
        match = re.fullmatch(r"p\(([^|()]+?)(?:\s*\|\s*([^()]+?))?\)", key)
        if not match:
            raise ValueError("unsupported parameter key: " + key)
        name = match.group(1).strip()
        parents = [p.strip() for p in match.group(2).split(",")] if match.group(2) else []
        if name in nodes or len(parents) != len(set(parents)):
            raise ValueError("duplicate variable or parent")
        if not parents:
            if not isinstance(table, (int, float)) or not 0 <= table <= 1:
                raise ValueError("invalid root probability")
        else:
            for bits in itertools.product((0, 1), repeat=len(parents)):
                _cell(table, bits)
        nodes[name] = {"name": name, "parents": parents, "table": table}
    ordered, remaining = [], dict(nodes)
    while remaining:
        known = {n["name"] for n in ordered}
        ready = sorted(name for name, n in remaining.items() if set(n["parents"]) <= known)
        if not ready:
            raise ValueError("cyclic CPTs or unknown parent")
        for name in ready:
            ordered.append(remaining.pop(name))
    return ordered


def native_forward(nodes, root_world, intervention):
    state = {}
    for node in nodes:
        name = node["name"]
        state[name] = (intervention[name] if name in intervention else
                       root_world[name] if not node["parents"] else
                       _cell(node["table"], [state[p] for p in node["parents"]]))
    return state


def native_counterfactual(params, observed, intervention, outcome):
    nodes = parse_native_params(params)
    names = {n["name"] for n in nodes}
    if outcome not in names or set(observed) - names or set(intervention) - names:
        raise ValueError("unknown query/evidence variable")
    if any(v not in (0, 1) for v in list(observed.values()) + list(intervention.values())):
        raise ValueError("evidence/interventions must be binary")
    roots = [n for n in nodes if not n["parents"]]
    supports = [[0] if n["table"] == 0 else [1] if n["table"] == 1 else [0, 1] for n in roots]
    outcomes, count = set(), 0
    for bits in itertools.product(*supports):
        root_world = dict(zip([n["name"] for n in roots], bits))
        original = native_forward(nodes, root_world, {})
        if any(original[k] != v for k, v in observed.items()):
            continue
        count += 1
        changed = native_forward(nodes, root_world, intervention)
        outcomes.add(changed[outcome])
    if not count:
        raise ValueError("factual evidence has no compatible root world")
    return {"value": next(iter(outcomes)) if len(outcomes) == 1 else None,
            "outcomes": sorted(outcomes), "compatible_worlds": count}
