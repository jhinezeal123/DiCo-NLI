"""Log predictions and pair-clustered summaries for the synthetic probes."""
import collections
import csv
import json
import statistics
import sys

import numpy as np


def _condition_side(hypothesis, condition):
    if hypothesis == "h1":
        return "directional" if condition in ("specific_to_general", "general_to_specific") else "equivalence"
    if hypothesis == "h2":
        return "related_negative" if "organization" in condition else "role_entailment"
    if hypothesis == "h3":
        return "synonym" if condition.startswith("synonym") else "unrelated"
    if hypothesis == "h4":
        return "context" if condition.startswith("context") else "bare_scenario"
    raise ValueError(hypothesis)


def summarize(hypothesis, rows, trained, seeds, labels, reverse):
    label_names = list(labels)
    per_seed = {}
    prob_vectors = {}
    for seed in seeds:
        orig = trained[seed]["probe_orig"]
        swap = trained[seed]["probe_swap"]
        assert orig.shape == swap.shape == (len(rows), len(labels))
        fused = 0.5 * (orig + swap[:, reverse])
        per_seed[seed] = [label_names[int(np.argmax(v))] for v in fused]
        prob_vectors[seed] = fused
        print(f"<<<PROBE_PREDICTIONS_CSV seed={seed} hypothesis={hypothesis}")
        fields = ["instance_id", "pair_id", "condition", "text1", "text2", "label", "prediction"]
        fields += ["prob_" + name for name in label_names]
        writer = csv.DictWriter(sys.stdout, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row, pred, probs in zip(rows, per_seed[seed], fused):
            out = dict(row, prediction=pred)
            out.update({"prob_" + name: f"{float(value):.7f}" for name, value in zip(label_names, probs)})
            writer.writerow(out)
        print("PROBE_PREDICTIONS_CSV>>>")
        metrics = {}
        for condition in sorted({r["condition"] for r in rows}):
            indices = [i for i, r in enumerate(rows) if r["condition"] == condition]
            correct = sum(per_seed[seed][i] == rows[i]["label"] for i in indices)
            metrics[condition] = {"n": len(indices), "correct": correct, "accuracy": correct / len(indices)}
        print("PROBE_SEED_METRICS " + json.dumps({"hypothesis": hypothesis, "seed": seed,
                                                  "conditions": metrics}, sort_keys=True), flush=True)

    # Majority vote reproduces the CSV-generation convention; ties follow seed order.
    vote = [collections.Counter(per_seed[seed][i] for seed in seeds).most_common(1)[0][0]
            for i in range(len(rows))]
    conditions = {}
    sides = collections.defaultdict(lambda: collections.defaultdict(list))
    for i, row in enumerate(rows):
        name = row["condition"]
        if name not in conditions:
            conditions[name] = {"n": 0, "correct": 0, "predictions": collections.Counter()}
        cell = conditions[name]
        cell["n"] += 1
        cell["correct"] += int(vote[i] == row["label"])
        cell["predictions"][vote[i]] += 1
        sides[row["pair_id"]][_condition_side(hypothesis, name)].append(int(vote[i] == row["label"]))
    for cell in conditions.values():
        cell["accuracy"] = cell["correct"] / cell["n"]
        cell["predictions"] = dict(cell["predictions"])

    side_keys = {"h1": ("directional", "equivalence"),
                 "h2": ("related_negative", "role_entailment"),
                 "h3": ("synonym", "unrelated"),
                 "h4": ("context", "bare_scenario")}[hypothesis]
    groups = sorted(sides)
    assert all(all(side in sides[group] for side in side_keys) for group in groups)
    left = np.array([statistics.mean(sides[group][side_keys[0]]) for group in groups])
    right = np.array([statistics.mean(sides[group][side_keys[1]]) for group in groups])
    rng = np.random.default_rng(20261028)
    sampled = rng.integers(0, len(groups), size=(5000, len(groups)))
    delta = (left[sampled] - right[sampled]).mean(axis=1)
    summary = {
        "hypothesis": hypothesis, "seeds": list(seeds), "aggregation": "seed majority vote, ties in seed order",
        "num_rows": len(rows), "num_families": len(groups), "conditions": conditions,
        "comparison": {"left": side_keys[0], "right": side_keys[1],
                       "left_accuracy": float(left.mean()), "right_accuracy": float(right.mean()),
                       "difference_left_minus_right": float((left - right).mean()),
                       "family_bootstrap_95pct": [float(x) for x in np.quantile(delta, [0.025, 0.975])],
                       "resamples": 5000, "bootstrap_seed": 20261028},
        "limitations": ["Synthetic probe designed after inspecting dev errors, not an independent population sample.",
                        "Bootstrap resamples probe families, not model-training seeds.",
                        "NEG identity in bidirectional averaging is a decoding heuristic, not official SoftCons."]}
    if hypothesis == "h4":
        summary["limitations"].append("Bare scenario targets are incompatible counterfactual contexts, not two gold annotations for a bare phrase.")
        summary["both_explicit_contexts_correct"] = sum(
            all(vote[i] == rows[i]["label"] for i in range(len(rows))
                if rows[i]["pair_id"] == group and rows[i]["condition"].startswith("context"))
            for group in groups)
    print("PROBE_FINAL_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    return summary
