"""Equal-length, order-sensitive intervention programs over finite SCM worlds."""
from functools import lru_cache
from collections import defaultdict
import random

from causal_action_algebra import PROGRAM_SEMANTICS, fold_actions, intervention_mask, render_action
from causal_dataset import digest
from causal_dataset_v4 import build_dataset_v4
from causal_v3_compiler import document_v3

VERSION = "conditional-intervention-program-v5"
RESERVED_SEEDS = [690001, 690002, 690003]
PROGRAM_LENGTH = 8


@lru_cache(maxsize=1)
def previous_worlds():
    signatures = set()
    build_dataset_v4(20261029, 6, signatures)
    return frozenset(signatures)


def deletion_relevant_steps(actions):
    final = fold_actions(actions)[-1]
    return [i for i in range(len(actions)) if fold_actions(actions[:i] + actions[i+1:])[-1] != final]


def valid_program(actions):
    if any(action in (4, 5, 15) for action in actions[-2:]):
        return False
    if fold_actions(actions)[-1] == fold_actions(list(reversed(actions)))[-1]:
        return False
    relevant = deletion_relevant_steps(actions)
    return len(relevant) >= 4 and sum(i < 4 for i in relevant) >= 2


def sample_program_family(rng, partial_mask):
    """Match lexical multisets; only order distinguishes the four final SCMs."""
    nonconditional = list(range(9)) + [15]
    for _ in range(10000):
        bag = rng.choices(list(range(9, 15)), k=4) + rng.choices(nonconditional, k=4)
        buckets = {0: [], 3: [], partial_mask: []}
        seen_orders = set()
        for _ in range(256):
            actions = rng.sample(bag, len(bag))
            key = tuple(actions)
            if key in seen_orders:
                continue
            seen_orders.add(key)
            final = fold_actions(actions)[-1]
            if final not in buckets or len(buckets[final]) >= (2 if final == 0 else 1):
                continue
            if not valid_program(actions):
                continue
            buckets[final].append(actions)
            if len(buckets[0]) == 2 and buckets[3] and buckets[partial_mask]:
                return {"factual": buckets[0][0], "intervention": buckets[3][0],
                        "partial_release": buckets[partial_mask][0], "restored": buckets[0][1]}
    raise RuntimeError("insufficient support for matched order-sensitive program families")


def build_dataset_v5(seed, families_per_split, seen_signatures=None):
    seen = set() if seen_signatures is None else seen_signatures
    seen.update(previous_worlds())
    data = build_dataset_v4(seed, families_per_split, seen)
    rng = random.Random(seed ^ 0x51A7E)
    # Cover all four answer classes for the stronger whole-state control.
    # Only training proposal order changes; values, SCMs and test inputs do not.
    partial_demos = [row for row in data["demonstrations"] if row["view"] == "partial_release"]
    for index, row in enumerate(partial_demos):
        doc = document_v3(row["text"])
        active = next(iter(row["scm"]["intervene"]))
        other = next(name for name in doc["proposed"] if name != active)
        order = [active, other] if index == 0 else [other, active]
        replacement = "Proposed overrides: " + ", ".join(f"{name}={doc['proposed'][name]}" for name in order) + "."
        row["text"] = "\n".join(replacement if line.startswith("Proposed overrides:") else line for line in row["text"].splitlines())
    all_rows = data["demonstrations"] + data["rows"]
    families = defaultdict(list)
    for row in all_rows:
        families[row["family_id"]].append(row)
    programs = {}
    for family, rows in families.items():
        doc = document_v3(rows[0]["text"])
        partial_row = next((row for row in rows if row["view"] == "partial_release"), None)
        partial = intervention_mask(partial_row["scm"]["intervene"], doc["proposed"]) if partial_row else rng.choice([1, 2])
        programs[family] = sample_program_family(rng, partial)
    for row in all_rows:
        doc = document_v3(row["text"])
        first, second = doc["proposed"]
        final = intervention_mask(row["scm"]["intervene"], doc["proposed"])
        actions = programs[row["family_id"]][row["view"]]
        if fold_actions(actions)[-1] != final:
            raise ValueError("program endpoint differs from final SCM")
        lines = row["text"].splitlines()
        end = next(i for i, line in enumerate(lines) if line.startswith("Proposed overrides:"))
        prefix = "\n".join(lines[:end+1])
        language = row["split"] == "ood_language"
        instructions = [f"{i+1}. {render_action(action, first, second, language)}" for i, action in enumerate(actions)]
        question = (f"Which bit would {row['scm']['target']} show?" if language else
                    f"What is the value of {row['scm']['target']} in this same individual?")
        row["text"] = (prefix + "\n" + PROGRAM_SEMANTICS + "\nProgram:\n" + "\n".join(instructions)
                       + "\nReadout: " + question + "\nAnswer: 0, 1, or unknown if compatible worlds disagree.")
        row["program_actions"] = actions
        row["program_true_states"] = fold_actions(actions)
        row["program_relevant_steps"] = deletion_relevant_steps(actions)
    data["audit"].update({
        "generator_version": VERSION, "base_world_generator": "partial-dense-twelve-variable-v4",
        "dataset_sha256": digest(data["rows"]), "demonstrations_sha256": digest(data["demonstrations"]),
        "program_length_all_views": PROGRAM_LENGTH, "conditional_steps_all_views": 4,
        "minimum_deletion_relevant_steps": 4, "minimum_early_relevant_steps": 2,
        "all_programs_order_sensitive_to_reversal": True,
        "public_view_markers": False, "fixed_clamp_values_not_toggled": True,
        "identical_instruction_multisets_within_each_test_family": True,
        "training_final_state_counts": {"0": 2, "1": 1, "2": 1, "3": 2},
        "previous_v4_pilot_worlds_excluded": len(previous_worlds()),
        "reserved_confirmation_seeds": RESERVED_SEEDS,
        "view_semantics": "four linked final SCMs; factual/restored programs end with neither clamp, intervention both, partial_release one; all have the same8step grammar and length",
    })
    return data
