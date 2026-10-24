"""Conservative, train-seeded lexical contrast sets and a count-matched control.

The lexicon below is a small manually reviewed source of *proposed* lexical
relations, not an assertion about PhrasIS annotation conventions.  We generate
standalone noun phrases rather than modifying arbitrary input contexts: deleting
a modifier, replacing under negation, or changing a quantifier need not preserve
entailment.  No model or evaluation labels participate in candidate selection.

Each tuple is (specific noun, synonym, broader noun, non-entailing sibling).
The synonym is deliberately NOT a broader zoological term (dog/canine, for
example).  Siblings need not be mutually exclusive: the relevant requirement is
that neither noun meaning entails the other.  All four labels remain directly
supervised; NEGATIVE_OTHER is never inferred from missing reverse metadata.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import random
import re
from typing import Any


LEXICON_VERSION = "manual-minimal-np-v1"
LEXICAL_RELATIONS = (
    ("car", "automobile", "vehicle", "bicycle"),
    ("automobile", "motorcar", "vehicle", "bicycle"),
    ("truck", "lorry", "vehicle", "motorcycle"),
    ("motorcycle", "motorbike", "vehicle", "car"),
    ("taxi", "taxicab", "vehicle", "bus"),
    ("airplane", "aeroplane", "aircraft", "helicopter"),
    ("sofa", "couch", "furniture", "table"),
    ("fridge", "refrigerator", "appliance", "oven"),
    ("cellphone", "mobile phone", "device", "laptop"),
    ("laptop", "laptop computer", "computer", "desktop computer"),
    ("photograph", "photo", "image", "painting"),
    ("father", "dad", "parent", "mother"),
    ("mother", "mom", "parent", "father"),
    ("grandfather", "grandpa", "grandparent", "grandmother"),
    ("grandmother", "grandma", "grandparent", "grandfather"),
    ("cop", "police officer", "person", "firefighter"),
    ("television", "TV", "device", "radio"),
)

LABELS = (
    "BACKWARD_ENTAILMENT", "EQUIVALENCE", "FORWARD_ENTAILMENT",
    "NEGATIVE_OTHER",
)


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def unordered_pair(row: dict) -> tuple[str, str]:
    return tuple(sorted((normalize_text(row["text1"]),
                         normalize_text(row["text2"]))))


def _contains_noun(text: str, noun: str) -> bool:
    # Only exact lexical matches.  Singularization would introduce sense and
    # number assumptions; e.g. 'arms' is not necessarily the body-part plural.
    return bool(re.search(r"(?<!\w)" + re.escape(noun.casefold()) + r"(?!\w)",
                          normalize_text(text)))


def _np(noun: str, determiner: str) -> str:
    if not determiner:
        return noun
    if determiner == "indefinite":
        if noun == "furniture":
            return "some furniture"
        article = "an" if noun[0].casefold() in "aeiou" else "a"
        # The only abbreviation in the reviewed lexicon is pronounced tee-vee.
        if noun == "TV":
            article = "a"
        return article + " " + noun
    return determiner + " " + noun


def _clean_pool(train_rows: list[dict], eval_rows: list[dict]):
    # Evaluation is used only as an input-only exclusion list, never as a source
    # of nouns, labels, reverse-pair metadata, or training examples.
    eval_text = {unordered_pair(r) for r in eval_rows}
    eval_groups = {str(r["pair_id"]) for r in eval_rows}
    excluded = {str(r["pair_id"]) for r in train_rows
                if unordered_pair(r) in eval_text
                or str(r["pair_id"]) in eval_groups}
    pool = [dict(r) for r in train_rows if str(r["pair_id"]) not in excluded]
    ids = [str(r["instance_id"]) for r in pool]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate training instance_id")
    if any(r["label"] not in LABELS for r in pool):
        raise ValueError("unknown training label")
    return pool, eval_text, excluded


def _families(pool: list[dict], eval_text: set[tuple[str, str]],
              cfg: dict):
    maximum = int(cfg.get("counterfactual_max_families", 120))
    if maximum < 0:
        raise ValueError("counterfactual_max_families must be non-negative")
    generated: list[dict] = []
    provenance: list[dict] = []
    rejected = Counter()
    seen: set[tuple[str, str]] = set()
    original_ids = {str(r["instance_id"]) for r in pool}
    for relation_id, (specific, synonym, broad, sibling) in enumerate(LEXICAL_RELATIONS):
        # Lexical seeds must occur in training inputs.  A sorted representative
        # source gives stable provenance independent of input row ordering.
        sources = sorted((r for r in pool if any(
            _contains_noun(r[k], noun)
            for k in ("text1", "text2") for noun in (specific, synonym))),
            key=lambda r: (str(r["pair_id"]), str(r["instance_id"])))
        if not sources:
            rejected["specific_or_synonym_absent_from_train"] += 1
            continue
        source = sources[0]
        for determiner in ("", "indefinite", "the"):
            if len(provenance) >= maximum:
                break
            s, eq, b, n = (_np(x, determiner)
                           for x in (specific, synonym, broad, sibling))
            examples = [(s, b, "FORWARD_ENTAILMENT"),
                        (b, s, "BACKWARD_ENTAILMENT"),
                        (s, eq, "EQUIVALENCE"),
                        (s, n, "NEGATIVE_OTHER")]
            keys = [tuple(sorted((normalize_text(a), normalize_text(b))))
                    for a, b, _ in examples]
            # FE/BE are intentionally two orders of the same semantic pair.
            unique_keys = set(keys)
            if unique_keys & eval_text:
                rejected["family_eval_text_overlap"] += 1
                continue
            if unique_keys & seen:
                rejected["duplicate_generated_family_pair"] += 1
                continue
            signature = json.dumps([source["pair_id"], relation_id, determiner],
                                   ensure_ascii=True, separators=(",", ":"))
            family_id = hashlib.sha256(signature.encode()).hexdigest()[:16]
            for index, (a, b, label) in enumerate(examples):
                row = dict(source)
                new_id = f"cf_{family_id}_{index}"
                if new_id in original_ids:
                    raise ValueError("synthetic ID collides with original data")
                row.update(instance_id=new_id, text1=a, text2=b, label=label)
                # Keep the original source group so later grouped splits can
                # never separate a derivative from its source.
                generated.append(row)
            provenance.append({
                "family_id": family_id,
                "source_instance_id": str(source["instance_id"]),
                "source_pair_id": str(source["pair_id"]),
                "relation_index": relation_id,
                "specific": specific, "synonym": synonym,
                "hypernym": broad, "non_entailing_sibling": sibling,
                "surface_form": determiner or "bare_noun",
                "validation": "manually_reviewed_lexical_relations_not_benchmark_gold",
            })
            seen.update(unique_keys)
    return generated, provenance, dict(rejected)


def prepare(train_rows: list[dict], eval_rows: list[dict], arm: str,
            seed: int, cfg: dict) -> tuple[list[dict], list[dict], dict]:
    """Return data for either semantic contrast sets or label-matched resampling.

    Both arms use the same input-only exclusion and candidate-generation policy.
    The resampling arm changes no labels: it samples original rows within each
    label with replacement, matching the appended class counts exactly.  Thus
    epochs, batch count and label priors are matched.  Derivatives retain the
    source pair_id; callers must group before calling this function if splitting
    train internally.  Evaluation labels are neither accessed nor modified.
    """
    if arm not in {"counterfactual", "resample"}:
        raise ValueError("arm must be 'counterfactual' or 'resample'")
    pool, eval_text, excluded = _clean_pool(train_rows, eval_rows)
    proposed, provenance, rejected = _families(pool, eval_text, cfg)
    target_counts = Counter(r["label"] for r in proposed)
    if not proposed:
        raise ValueError("no conservative train-seeded counterfactual families survived")
    if arm == "counterfactual":
        appended = proposed
    else:
        rng = random.Random(int(seed))
        by_label = {label: sorted((r for r in pool if r["label"] == label),
                                  key=lambda r: str(r["instance_id"]))
                    for label in LABELS}
        missing = [label for label, count in target_counts.items()
                   if count and not by_label[label]]
        if missing:
            raise ValueError(f"cannot match resampling labels: {missing}")
        appended = []
        for label in LABELS:
            for j in range(target_counts[label]):
                source = rng.choice(by_label[label])
                row = dict(source)
                row["instance_id"] = f"rs_{seed}_{LABELS.index(label)}_{j}"
                appended.append(row)
    output = pool + appended
    ids = [str(r["instance_id"]) for r in output]
    if len(ids) != len(set(ids)):
        raise ValueError("output instance IDs are not unique")
    if any(unordered_pair(r) in eval_text for r in output):
        raise AssertionError("train/eval text leakage after augmentation")
    audit = {
        "method": "train_seeded_relation_contrast_sets",
        "arm": arm, "seed": int(seed),
        "lexicon_version": LEXICON_VERSION,
        "lexicon_sha256": hashlib.sha256(json.dumps(
            LEXICAL_RELATIONS, separators=(",", ":")).encode()).hexdigest(),
        "lexicon_provenance": "explicit manually reviewed lexical list in committed source",
        "original_train_rows": len(train_rows), "clean_train_rows": len(pool),
        "excluded_source_pair_ids": sorted(excluded),
        "removed_train_rows": len(train_rows) - len(pool),
        "family_count": len(provenance), "appended_rows": len(appended),
        "appended_label_counts": dict(sorted(target_counts.items())),
        "final_label_counts": dict(sorted(Counter(r["label"] for r in output).items())),
        "rejected": rejected, "eval_labels_used": False,
        "semantic_validation_status": "conservative_lexical_rules_requires_empirical_validation",
        "family_provenance": provenance,
        "appended_examples": appended,
        "proposed_counterfactual_examples": proposed,
    }
    return output, [dict(r) for r in eval_rows], audit
