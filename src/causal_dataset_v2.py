"""Six-variable causal validation suite with genuine held-out XOR mechanisms.

Every graph includes a directed backbone chain, so its topological order is
unique and independent of names. Family signatures collapse NOT/COPY and
NAND/AND, NOR/OR equivalences induced by private XOR noise and full observation.
"""
from __future__ import annotations

import random
from collections import Counter

from causal_dataset import NAMES, digest, expand
from causal_scm import forward, normalize_scm

VERSION = "binary-xor-noise-v2.1-world-groups"
CONFIRMATION_SEEDS = [271828, 314159, 161803]


def family_signature(parents, ops, observed_in_order, cause_index, set_value):
    equivalent = {"not": "copy", "nand": "and", "nor": "or"}
    return digest({
        "version": VERSION,
        "parents": [sorted(p) for p in parents],
        "operations_modulo_private_xor_noise": [equivalent.get(op, op) for op in ops],
        "observed": observed_in_order,
        # A world stays in one group regardless of which cause/value is queried.
    })


def sample_family(rng, split, seen, index, forced_binary_op=None):
    wanted = ((index // 2) % 2, index % 2)
    for _ in range(50000):
        n = 6
        parents = [[]] + [[i - 1] for i in range(1, n)]
        merges = rng.choice([2, 3]) if split == "ood_topology" else 1
        for node in rng.sample(list(range(2, n)), merges):
            parents[node].append(rng.randrange(node - 1))
            parents[node].sort()
        names = rng.sample(NAMES, n)
        ops = [
            "root" if not p else
            rng.choice(["copy", "not"]) if len(p) == 1 else
            "xor" if split == "ood_mechanism" else (forced_binary_op or rng.choice(["and", "or"]))
            for p in parents
        ]
        nodes = [{"name": names[i], "op": ops[i], "parents": [names[p] for p in ps]}
                 for i, ps in enumerate(parents)]
        noise = {name: rng.randrange(2) for name in names}
        observed = forward(nodes, noise)
        cause_index = rng.randrange(n - 1)
        cause = names[cause_index]
        set_value = 1 - observed[cause]
        counterfactual = forward(nodes, noise, {cause: set_value})
        changed = int(counterfactual[names[-1]] != observed[names[-1]])
        if (observed[names[-1]], changed) != wanted:
            continue
        signature = family_signature(parents, ops, [observed[name] for name in names], cause_index, set_value)
        if signature in seen:
            continue
        seen.add(signature)
        scm = normalize_scm({"variables": nodes, "observed": observed, "intervene": {}, "target": names[-1]})
        return {
            "family_id": signature[:20], "signature": signature, "split": split,
            "topology": f"chain_skip_{merges}", "scm": scm, "cause": cause,
            "set_value": set_value, "changed": bool(changed),
        }
    raise RuntimeError(f"insufficient unique support for {split}, stratum {wanted}")


def build_dataset_v2(seed, families_per_split, seen_signatures=None):
    if families_per_split <= 0 or families_per_split % 4:
        raise ValueError("families_per_split must be positive and divisible by four")
    rng = random.Random(seed)
    seen = set() if seen_signatures is None else seen_signatures
    count_before = len(seen)
    demos = [sample_family(rng, "demonstration", seen, i, "and" if i % 2 == 0 else "or")
             for i in range(4)]
    families = []
    for split in ("id", "ood_topology", "ood_mechanism", "ood_language"):
        families.extend(sample_family(rng, split, seen, i) for i in range(families_per_split))
    rows = [row for f in families for row in expand(f)]
    audit = {}
    for split in sorted({r["split"] for r in rows}):
        group = [r for r in rows if r["split"] == split]
        originals = [r for r in group if r["view"] == "factual"]
        audit[split] = {
            "families": len(originals), "cases": len(group),
            "labels": dict(Counter(r["answer"] for r in group)),
            "changed_families": sum(r["changed"] for r in originals),
            "topologies": dict(Counter(r["topology"] for r in originals)),
            "operators": dict(Counter(node["op"] for r in originals for node in r["scm"]["variables"])),
        }
    assert len(seen) - count_before == len(families) + len(demos)
    return {
        "seed": seed, "rows": rows,
        "demonstrations": [expand(d)[1] for d in demos],
        "audit": {"splits": audit, "signature_overlap": 0,
                  "dataset_sha256": digest(rows), "generator_version": VERSION,
                  "signature_equivalence": "renaming invariant unique topo order; output-negation modulo private XOR noise",
                  "prior_excluded_families": count_before},
    }

