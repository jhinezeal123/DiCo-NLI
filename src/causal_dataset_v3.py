"""Harder eight-variable, partially observed, multi-intervention SCM suite."""
import copy
import itertools
import random
from collections import Counter

from causal_dataset import NAMES, digest
from causal_scm import normalize_scm, forward, mechanism

VERSION = "partial-multi-intervention-v3"
VIEWS = ("factual", "intervention", "partial_release", "restored")
RESERVED_SEEDS = [390001, 390002, 390003]


def solve_conditioned(value):
    """Exact enumeration via the bijection between endogenous values and U.

    With independent fair private XOR bits every endogenous assignment is
    equiprobable; enumerate only unobserved endogenous values, infer its U,
    and retain those same U under the intervention.
    """
    scm = normalize_scm(value)
    missing = [n["name"] for n in scm["variables"] if n["name"] not in scm["observed"]]
    outcomes = set()
    for bits in itertools.product((0, 1), repeat=len(missing)):
        factual = {**scm["observed"], **dict(zip(missing, bits))}
        noise = {n["name"]: factual[n["name"]] ^ mechanism(n["op"], [factual[p] for p in n["parents"]])
                 for n in scm["variables"]}
        outcomes.add(forward(scm["variables"], noise, scm["intervene"])[scm["target"]])
    return {"answer": str(next(iter(outcomes))) if len(outcomes) == 1 else "unknown",
            "compatible_worlds": 2**len(missing), "outcomes": sorted(outcomes)}


def render_v3(scm, proposals, released, view, language=False):
    names = [n["name"] for n in scm["variables"]]
    lines = ["Declared variables: " + ", ".join(names) + ".",
        "Every variable is binary and has an independent fair private noise bit, fixed for this individual.",
        "A root equals its own noise bit. Other equations are base operation XOR the variable's noise.",
        "Observations may be incomplete. Condition on ALL compatible original worlds; never resample noise.",
        "An override replaces only that variable's equation. Restoring it restores its original equation.",
        "majority3 returns 1 iff at least two of its three parents are 1; parity3 returns 1 iff an odd number are 1.",
        "Rules:"]
    for n in scm["variables"]:
        name, op, parents = n["name"], n["op"], n["parents"]
        if op == "root":
            rule = f"{name} equals its own noise bit." if not language else f"For {name}, the output is its private disturbance."
        else:
            base = (parents[0] if op == "copy" else "NOT " + parents[0] if op == "not" else
                    f"{op.upper()}(" + ", ".join(parents) + ")")
            if language:
                phrase = {"copy": "a copy of", "not": "the negation of", "and": "the conjunction of",
                          "or": "the inclusive disjunction of", "xor": "the exclusive disjunction of",
                          "majority3": "the majority vote over", "parity3": "the odd parity over"}[op]
                rule = f"For {name}, evaluate {phrase} " + ", ".join(parents) + f", then toggle it if {name}'s private disturbance is 1."
            else:
                rule = f"{name} equals {base}, XOR its own noise bit."
        lines.append(rule)
    lines.append("Observed in the original world: " + ", ".join(f"{k}={v}" for k, v in scm["observed"].items()) + ".")
    lines.append("Proposed overrides: " + ", ".join(f"{k}={v}" for k, v in proposals.items()) + ".")
    other = next(n for n in proposals if n != released)
    if not language:
        question = {
            "factual": "Do not apply either proposed override. ",
            "intervention": "Apply both proposed overrides and keep both active. ",
            "partial_release": f"Apply both proposed overrides. Then REMOVE the override for {released} and RESTORE its original equation. Keep the override for {other} active. ",
            "restored": f"Apply both proposed overrides. Restore the original equation for {released}; then also remove the override for {other} and restore its equation. Neither override remains. ",
        }[view] + f"What is the value of {scm['target']} in this same individual?"
    else:
        question = {
            "factual": "Leave both suggested clamps unused. ",
            "intervention": "Enforce both suggested clamps through the readout. ",
            "partial_release": f"Enforce both clamps, then undo the clamp on {released}. The clamp on {other} stays in force. ",
            "restored": f"Enforce both clamps, undo the one on {released}, and finally undo the one on {other}. Both ordinary mechanisms now operate again. ",
        }[view] + f"Which bit would {scm['target']} show?"
    lines.extend([question, "Answer: 0, 1, or unknown if compatible worlds disagree."])
    return "\n".join(lines)


def expand_v3(family):
    rows = []
    original = family["scm"]["observed"][family["scm"]["target"]]
    for view in VIEWS:
        scm = copy.deepcopy(family["scm"])
        scm["intervene"] = (dict(family["proposals"]) if view == "intervention" else
                            {k: v for k, v in family["proposals"].items() if k != family["released"]}
                            if view == "partial_release" else {})
        result = solve_conditioned(scm)
        rows.append({"case_id": family["family_id"]+":"+view, "family_id": family["family_id"],
            "split": family["split"], "view": view, "scm": scm, "answer": result["answer"],
            "compatible_worlds": result["compatible_worlds"],
            "changed": None if result["answer"] == "unknown" else int(result["answer"]) != original,
            "text": render_v3(scm, family["proposals"], family["released"], view, family["split"] == "ood_language")})
    return rows


def sample_v3(rng, split, seen, index, forced_binary=None, wanted_partial=None):
    desired = ((index//3) % 2, ("0", "1", "unknown")[index % 3])
    for _ in range(30000):
        parents = [[]] + [[i-1] for i in range(1, 8)]
        if split == "ood_mechanism":
            node = rng.randrange(3, 8)
            parents[node] = sorted([node-1]+rng.sample(list(range(node-1)), 2))
        else:
            merges = rng.choice([3, 4]) if split == "ood_topology" else 1
            for node in rng.sample(list(range(2, 8)), merges):
                parents[node] = sorted([node-1, rng.randrange(node-1)])
        names = rng.sample(NAMES, 8)
        ops = ["root" if not p else rng.choice(["copy", "not"]) if len(p) == 1 else
               forced_binary or rng.choice(["and", "or", "xor"]) if len(p) == 2 else
               rng.choice(["majority3", "parity3"]) for p in parents]
        nodes = [{"name": names[i], "op": ops[i], "parents": [names[p] for p in ps]} for i, ps in enumerate(parents)]
        factual = forward(nodes, {n: rng.randrange(2) for n in names})
        if factual[names[-1]] != desired[0]:
            continue
        observed_indices = {7} | set(rng.sample(list(range(7)), rng.choice([3, 4, 5])))
        observed = {names[i]: factual[names[i]] for i in sorted(observed_indices)}
        causes = rng.sample(names[:-1], 2)
        # Action values are independent of unobserved factual values.
        proposals = {n: rng.randrange(2) for n in causes}
        released = rng.choice(causes)
        scm = normalize_scm({"variables": nodes, "observed": observed, "intervene": proposals, "target": names[-1]})
        if solve_conditioned(scm)["answer"] != desired[1]:
            continue
        if wanted_partial is not None:
            partial = {**scm, "intervene": {k: v for k, v in proposals.items() if k != released}}
            if solve_conditioned(partial)["answer"] != wanted_partial:
                continue
        signature = digest({"version": VERSION, "parents": parents,
            "ops_mod_output_noise": ["copy" if op == "not" else op for op in ops],
            "observed_in_chain_order": [observed.get(n) for n in names]})
        if signature in seen:
            continue
        seen.add(signature)
        scm["intervene"] = {}
        return {"family_id": signature[:20], "signature": signature, "scm": scm, "split": split,
                "proposals": proposals, "released": released}
    raise RuntimeError(f"insufficient v3 support for {split} / {desired}")


def build_dataset_v3(seed, families_per_split, seen_signatures=None):
    if families_per_split <= 0 or families_per_split % 6:
        raise ValueError("v3 family count must be positive and divisible by six")
    seen = set() if seen_signatures is None else seen_signatures
    previous = len(seen); rng = random.Random(seed)
    demo_views = ["factual", "partial_release", "intervention", "restored", "partial_release", "intervention"]
    demos = [sample_v3(rng, "demonstration", seen, i, ["and", "or", "xor"][i % 3],
                       "0" if i == 1 else "1" if i == 4 else None) for i in range(6)]
    demonstrations = [next(r for r in expand_v3(f) if r["view"] == view) for f, view in zip(demos, demo_views)]
    assert Counter(r["answer"] for r in demonstrations) == {"0": 2, "1": 2, "unknown": 2}
    families = [sample_v3(rng, split, seen, i) for split in ("id", "ood_topology", "ood_mechanism", "ood_language")
                for i in range(families_per_split)]
    rows = [r for f in families for r in expand_v3(f)]
    audit = {split: {"families": families_per_split, "cases": 4*families_per_split,
                     "intervention_labels": dict(Counter(r["answer"] for r in rows if r["split"] == split and r["view"] == "intervention")),
                     "partial_release_labels": dict(Counter(r["answer"] for r in rows if r["split"] == split and r["view"] == "partial_release")),
                     "observed_counts": dict(Counter(len(r["scm"]["observed"]) for r in rows if r["split"] == split and r["view"] == "factual"))}
             for split in ("id", "ood_topology", "ood_mechanism", "ood_language")}
    return {"seed": seed, "rows": rows, "demonstrations": demonstrations,
            "audit": {"generator_version": VERSION, "dataset_sha256": digest(rows), "splits": audit,
                      "prior_excluded_families": previous, "new_distinct_worlds": len(seen)-previous,
                      "signature_overlap": 0, "views": list(VIEWS), "reserved_confirmation_seeds": RESERVED_SEEDS}}
