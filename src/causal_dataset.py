"""Fresh grouped causal cases; no model, benchmark labels, or gold in prompts."""
from __future__ import annotations

import copy
import hashlib
import json
import random
from collections import Counter

from causal_scm import forward, normalize_scm, solve


TOPOLOGIES = {
    "chain": [[], [0], [1], [2], [3]],
    "fork": [[], [0], [0], [0], [0]],
    "diamond": [[], [0], [0], [1, 2], [3]],
    "collider": [[], [], [0, 1], [1], [2, 3]],
    "skip": [[], [0], [0, 1], [1, 2], [2, 3]],
}
NAMES = "amber birch cedar dune elm flint grove hazel iris jade kelp lime maple nettle oak pine quartz reed sage thorn umber vine willow yarrow".split()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def ancestors(index, parents):
    found = set(parents[index])
    for p in parents[index]:
        found |= ancestors(p, parents)
    return found


def render(scm, view, cause=None, set_value=None, language="standard"):
    lines = [
        "All variables are binary, 0 or 1. Every variable has its own independent noise bit.",
        "Noise is fixed for this individual. Infer it from the observations; do not resample it.",
        "XOR means exclusive OR. AND, OR, NOT, NAND and NOR have their usual Boolean meanings.",
        "An intervention replaces only the named equation by a constant. Other equations and noise stay fixed.",
        "Rules:",
    ]
    phrases = {"and": "the logical conjunction of", "or": "the inclusive disjunction of",
               "xor": "the exclusive disjunction of", "nand": "the negated conjunction of",
               "nor": "the negated inclusive disjunction of"}
    for node in scm["variables"]:
        name, op, parents = node["name"], node["op"], node["parents"]
        if language == "heldout":
            if op == "root":
                line = f"The value of {name} is exactly its private disturbance."
            else:
                base = (parents[0] if op == "copy" else
                        f"the negation of {parents[0]}" if op == "not" else
                        f"{phrases[op]} {parents[0]} and {parents[1]}")
                line = f"To compute {name}, take {base}, then flip the result if {name}'s private disturbance is 1."
        else:
            if op == "root":
                line = f"{name} equals its own noise bit."
            else:
                base = (parents[0] if op == "copy" else f"NOT {parents[0]}" if op == "not" else
                        f"({parents[0]} {op.upper()} {parents[1]})")
                line = f"{name} equals {base}, XOR its own noise bit."
        lines.append(line)
    observed = ", ".join(f"{k}={v}" for k, v in scm["observed"].items())
    lines.append("Observed in the original world: " + (observed or "nothing") + ".")
    if view == "factual":
        question = f"Without any intervention, what is the value of {scm['target']}?"
    elif view == "restored":
        question = (f"First force {cause} to {set_value}. Then REMOVE that override and RESTORE its original equation. "
                    f"After restoration, what is the value of {scm['target']} in the same individual?")
    else:
        question = (f"In this same individual, replace the equation for {cause} by the constant {set_value}. "
                    f"After this intervention, what is the value of {scm['target']}?")
    lines.extend([question, "The answer is 0, 1, or unknown if both values remain possible."])
    return "\n".join(lines)


def family(rng, split, seen, index):
    topology_pool = ["collider", "skip"] if split == "ood_topology" else ["chain", "fork", "diamond"]
    wanted = ((index // 2) % 2, index % 2)  # factual value x whether intervention changes target
    for _ in range(20000):
        topology = rng.choice(topology_pool)
        parents = TOPOLOGIES[topology]
        names = rng.sample(NAMES, len(parents))
        ops = ["root" if not p else rng.choice(["copy", "not"]) if len(p) == 1 else
               rng.choice(["nand", "nor"] if split == "ood_mechanism" else ["and", "or", "xor"])
               for p in parents]
        if split == "ood_mechanism" and not any(op in ("nand", "nor") for op in ops):
            continue
        nodes = [{"name": names[i], "op": ops[i], "parents": [names[p] for p in ps]}
                 for i, ps in enumerate(parents)]
        noise = {name: rng.randrange(2) for name in names}
        observed = forward(nodes, noise)
        target = len(names) - 1
        cause_index = rng.choice(sorted(ancestors(target, parents)))
        cause = names[cause_index]
        set_value = 1 - observed[cause]
        cf = forward(nodes, noise, {cause: set_value})
        changed = int(cf[names[target]] != observed[names[target]])
        if (observed[names[target]], changed) != wanted:
            continue
        signature = digest({"parents": parents, "ops": ops,
                            "observed": [observed[n] for n in names],
                            "cause": cause_index, "value": set_value})
        if signature in seen:
            continue
        seen.add(signature)
        scm = normalize_scm({"variables": nodes, "observed": observed,
                             "intervene": {}, "target": names[target]})
        return {"family_id": signature[:20], "signature": signature, "split": split,
                "topology": topology, "scm": scm, "cause": cause,
                "set_value": set_value, "changed": bool(changed)}
    raise RuntimeError(f"could not sample stratum {split} {wanted}; enlarge generator support")


def expand(item):
    rows = []
    for view in ("factual", "intervention", "restored"):
        scm = copy.deepcopy(item["scm"])
        if view == "intervention":
            scm["intervene"] = {item["cause"]: item["set_value"]}
        scm = normalize_scm(scm)
        answer = solve(scm)["answer"]
        row = {k: item[k] for k in ("family_id", "signature", "split", "topology", "changed")}
        row.update({"case_id": item["family_id"] + ":" + view, "view": view,
                    "scm": scm, "answer": answer,
                    "text": render(scm, view, item["cause"], item["set_value"],
                                   "heldout" if item["split"] == "ood_language" else "standard")})
        rows.append(row)
    return rows


def build_dataset(seed, families_per_split):
    if families_per_split % 4:
        raise ValueError("families_per_split must be divisible by four")
    rng, seen, families = random.Random(seed), set(), []
    # Demonstrations are generated first and excluded structurally from all evaluation splits.
    demos = [family(rng, "demonstration", seen, i) for i in range(4)]
    for split in ("id", "ood_topology", "ood_mechanism", "ood_language"):
        families.extend(family(rng, split, seen, i) for i in range(families_per_split))
    rows = [row for item in families for row in expand(item)]
    by_split = {}
    for split in sorted({r["split"] for r in rows}):
        group = [r for r in rows if r["split"] == split]
        by_split[split] = {"families": len(group) // 3, "cases": len(group),
                           "labels": dict(Counter(r["answer"] for r in group)),
                           "changed_families": sum(r["changed"] for r in group if r["view"] == "factual")}
    assert len({f["signature"] for f in families + demos}) == len(families + demos)
    return {"seed": seed, "rows": rows, "demonstrations": [expand(d)[1] for d in demos],
            "audit": {"splits": by_split, "signature_overlap": 0,
                      "dataset_sha256": digest(rows), "generator_version": "binary-xor-noise-v1"}}
