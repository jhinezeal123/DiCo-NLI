"""Literal graph extraction from the original public CLadder background."""
import re
from causal_native import parse_native_params


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

