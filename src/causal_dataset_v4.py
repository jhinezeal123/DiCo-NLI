"""Twelve-variable stress suite: longer paths, denser merges, larger posterior."""
from collections import Counter
import random

from causal_dataset import NAMES, digest
from causal_dataset_v3 import expand_v3, solve_conditioned, VIEWS
from causal_scm import forward, normalize_scm

VERSION = "partial-dense-twelve-variable-v4"
VARIABLE_COUNT = 12
RESERVED_SEEDS = [590001, 590002, 590003]


def sample_v4(rng, split, seen, index, forced_binary=None, wanted_partial=None):
    desired = ((index // 3) % 2, ("0", "1", "unknown")[index % 3])
    for _ in range(30000):
        parents = [[]] + [[i - 1] for i in range(1, VARIABLE_COUNT)]
        if split == "ood_mechanism":
            for node in rng.sample(list(range(3, VARIABLE_COUNT)), 3):
                parents[node] = sorted([node - 1] + rng.sample(list(range(node - 1)), 2))
        else:
            merges = rng.choice([6, 7, 8]) if split == "ood_topology" else 2
            for node in rng.sample(list(range(2, VARIABLE_COUNT)), merges):
                parents[node] = sorted([node - 1, rng.randrange(node - 1)])
        names = rng.sample(NAMES, VARIABLE_COUNT)
        ops = ["root" if not p else rng.choice(["copy", "not"]) if len(p) == 1 else
               forced_binary or rng.choice(["and", "or", "xor"]) if len(p) == 2 else
               rng.choice(["majority3", "parity3"]) for p in parents]
        nodes = [{"name": names[i], "op": ops[i], "parents": [names[p] for p in ps]}
                 for i, ps in enumerate(parents)]
        factual = forward(nodes, {name: rng.randrange(2) for name in names})
        if factual[names[-1]] != desired[0]:
            continue
        observed_indices = {VARIABLE_COUNT - 1} | set(rng.sample(list(range(VARIABLE_COUNT - 1)), rng.choice([5, 6, 7])))
        observed = {names[i]: factual[names[i]] for i in sorted(observed_indices)}
        causes = rng.sample(names[:-1], 2)
        proposals = {name: rng.randrange(2) for name in causes}
        released = rng.choice(causes)
        scm = normalize_scm({"variables": nodes, "observed": observed,
                             "intervene": proposals, "target": names[-1]})
        if solve_conditioned(scm)["answer"] != desired[1]:
            continue
        if wanted_partial is not None:
            partial = {**scm, "intervene": {k: v for k, v in proposals.items() if k != released}}
            if solve_conditioned(partial)["answer"] != wanted_partial:
                continue
        signature = digest({"version": VERSION, "parents": parents,
                            "ops_mod_output_noise": ["copy" if op == "not" else op for op in ops],
                            "observed_in_chain_order": [observed.get(name) for name in names]})
        if signature in seen:
            continue
        seen.add(signature)
        scm["intervene"] = {}
        return {"family_id": signature[:20], "signature": signature, "scm": scm,
                "split": split, "proposals": proposals, "released": released}
    raise RuntimeError(f"insufficient v4 support for {split} / {desired}")


def build_dataset_v4(seed, families_per_split, seen_signatures=None):
    if families_per_split <= 0 or families_per_split % 6:
        raise ValueError("v4 family count must be positive and divisible by six")
    seen = set() if seen_signatures is None else seen_signatures
    previous = len(seen)
    rng = random.Random(seed)
    demo_views = ["factual", "partial_release", "intervention", "restored", "partial_release", "intervention"]
    demos = [sample_v4(rng, "demonstration", seen, i, ["and", "or", "xor"][i % 3],
                       "0" if i == 1 else "1" if i == 4 else None) for i in range(6)]
    demonstrations = [next(row for row in expand_v3(family) if row["view"] == view)
                      for family, view in zip(demos, demo_views)]
    assert Counter(row["answer"] for row in demonstrations) == {"0": 2, "1": 2, "unknown": 2}
    families = [sample_v4(rng, split, seen, index)
                for split in ("id", "ood_topology", "ood_mechanism", "ood_language")
                for index in range(families_per_split)]
    rows = [row for family in families for row in expand_v3(family)]
    splits = {}
    for split in ("id", "ood_topology", "ood_mechanism", "ood_language"):
        selected = [row for row in rows if row["split"] == split]
        factual = [row for row in selected if row["view"] == "factual"]
        splits[split] = {
            "families": families_per_split, "cases": len(selected),
            "intervention_labels": dict(Counter(row["answer"] for row in selected if row["view"] == "intervention")),
            "observed_counts": dict(Counter(len(row["scm"]["observed"]) for row in factual)),
            "compatible_world_counts": dict(Counter(row["compatible_worlds"] for row in factual)),
            "binary_gate_counts": dict(Counter(sum(len(n["parents"]) == 2 for n in row["scm"]["variables"]) for row in factual)),
            "ternary_gate_counts": dict(Counter(sum(len(n["parents"]) == 3 for n in row["scm"]["variables"]) for row in factual)),
        }
    return {"seed": seed, "rows": rows, "demonstrations": demonstrations,
            "audit": {"generator_version": VERSION, "variable_count": VARIABLE_COUNT,
                      "dataset_sha256": digest(rows), "demonstrations_sha256": digest(demonstrations),
                      "splits": splits, "prior_excluded_families": previous,
                      "new_distinct_worlds": len(seen) - previous, "signature_overlap": 0,
                      "world_size_disjoint_from_all_v2_v3_worlds": True,
                      "views": list(VIEWS), "reserved_confirmation_seeds": RESERVED_SEEDS}}
