"""Small, controlled and deliberately synthetic tests for five DiCo-NLI hypotheses.

The challenge sets are constructed after inspecting dev errors.  They are
diagnostic probes, not an independent sample from the task distribution.
No official dev labels or user annotations are used to train a model.
"""
import csv
import hashlib
import json
from collections import Counter


LEXEMES = [
    ("car", "automobile", "bicycle"),
    ("bicycle", "bike", "car"),
    ("sofa", "couch", "table"),
    ("truck", "lorry", "bicycle"),
    ("airplane", "aeroplane", "helicopter"),
    ("motorcycle", "motorbike", "car"),
    ("taxi", "taxicab", "bus"),
    ("television", "TV", "radio"),
    ("cell phone", "mobile phone", "tablet"),
    ("trash can", "rubbish bin", "chair"),
    ("stroller", "pushchair", "bicycle"),
    ("backpack", "rucksack", "suitcase"),
    ("photograph", "photo", "painting"),
    ("apartment", "flat", "office"),
]

ROLES = [
    ("hospital", "nurse"), ("hospital", "doctor"),
    ("university", "professor"), ("university", "student"),
    ("school", "teacher"), ("school", "principal"),
    ("restaurant", "chef"), ("restaurant", "manager"),
    ("museum", "curator"), ("museum", "director"),
    ("company", "director"), ("company", "manager"),
    ("hotel", "receptionist"), ("hotel", "manager"),
    ("theater", "director"), ("theater", "actor"),
    ("orchestra", "conductor"), ("orchestra", "musician"),
    ("library", "librarian"), ("library", "visitor"),
    ("airport", "worker"), ("airport", "manager"),
    ("bakery", "baker"), ("bakery", "manager"),
]

# (ambiguous phrase, target phrase, target-sense completion, other-sense completion).
# Bare inputs are repeated with *scenario* targets, not treated as two true gold labels.
SENSES = [
    ("the bank", "the riverbank", "the bank of the river", "the bank that lends money"),
    ("the bat", "the baseball bat", "the bat used to hit baseballs", "the bat flying at night"),
    ("the crane", "the construction crane", "the crane used at a building site", "the crane flying through the sky"),
    ("the seal", "the ocean seal", "the seal that swims in the ocean", "the seal on an official document"),
    ("the mouse", "the computer mouse", "the mouse used with a computer", "the mouse that eats cheese"),
    ("the spring", "the metal spring", "the coiled spring made of metal", "the spring season"),
    ("the club", "the sports club", "the club for people who play sports", "the club used as a weapon"),
    ("the file", "the computer file", "the file stored on a computer", "the file used to shape metal"),
    ("the match", "the sporting match", "the match between competing teams", "the match used to light a candle"),
    ("the bark", "the tree bark", "the bark of a tree", "the bark made by a dog"),
    ("the port", "the shipping port", "the port where ships dock", "the port on a computer"),
    ("the jam", "the fruit jam", "the jam made from fruit", "the jam of stopped cars"),
]


def article(phrase):
    return ("an " if phrase[0].lower() in "aeiou" else "a ") + phrase


def _row(group, condition, text1, text2, label):
    return {"instance_id": group + "__" + condition, "pair_id": group,
            "condition": condition, "text1": text1, "text2": text2, "label": label}


def make_raw(hypothesis):
    rows = []
    if hypothesis in ("h1", "h3"):
        for i, (noun, synonym, contrast) in enumerate(LEXEMES):
            for adjective in ("small", "large"):
                # Treat both adjective variants as one lexical family for bootstrap.
                group = f"{hypothesis}_{i:02d}"
                specific = article(adjective + " " + noun)
                paraphrase = article(adjective + " " + synonym)
                if hypothesis == "h1":
                    broad = article(noun)
                    family_rows = [
                        _row(group, "specific_to_general", specific, broad, "FORWARD_ENTAILMENT"),
                        _row(group, "general_to_specific", broad, specific, "BACKWARD_ENTAILMENT"),
                        _row(group, "equiv_forward", specific, paraphrase, "EQUIVALENCE"),
                        _row(group, "equiv_backward", paraphrase, specific, "EQUIVALENCE"),
                    ]
                else:
                    other = article(adjective + " " + contrast)
                    family_rows = [
                        _row(group, "synonym_forward", specific, paraphrase, "EQUIVALENCE"),
                        _row(group, "synonym_backward", paraphrase, specific, "EQUIVALENCE"),
                        _row(group, "contrast_forward", specific, other, "NEGATIVE_OTHER"),
                        _row(group, "contrast_backward", other, specific, "NEGATIVE_OTHER"),
                    ]
                for row in family_rows:
                    row["instance_id"] = f"{group}_{adjective}__{row['condition']}"
                rows.extend(family_rows)
    elif hypothesis == "h2":
        for i, (organization, role) in enumerate(ROLES):
            # Roles from the same organization are not independent families.
            group = "h2_" + organization.replace(" ", "_")
            full, person, org = article(organization + " " + role), article(role), article(organization)
            family_rows = [
                _row(group, "person_to_role", full, person, "FORWARD_ENTAILMENT"),
                _row(group, "role_to_person", person, full, "BACKWARD_ENTAILMENT"),
                _row(group, "person_to_organization", full, org, "NEGATIVE_OTHER"),
                _row(group, "organization_to_person", org, full, "NEGATIVE_OTHER"),
            ]
            for row in family_rows:
                row["instance_id"] = f"h2_{i:02d}__{row['condition']}"
            rows.extend(family_rows)
    elif hypothesis == "h4":
        for i, (bare, target, same_sense, other_sense) in enumerate(SENSES):
            group = f"h4_{i:02d}"
            rows.extend([
                _row(group, "bare_target_eq", bare, target, "EQUIVALENCE"),
                _row(group, "bare_target_neg", bare, target, "NEGATIVE_OTHER"),
                _row(group, "context_target_eq", same_sense, target, "EQUIVALENCE"),
                _row(group, "context_target_neg", other_sense, target, "NEGATIVE_OTHER"),
            ])
    else:
        raise ValueError(f"unsupported diagnostic hypothesis: {hypothesis}")
    return rows


def norm(text):
    return " ".join(str(text).casefold().split())


def unordered(text1, text2):
    return tuple(sorted((norm(text1), norm(text2))))


def build(hypothesis, train_path, dev_path):
    """Remove *whole* families if any generated pair exactly overlaps train or dev."""
    existing = set()
    for path in (train_path, dev_path):
        with open(path, encoding="utf-8-sig", newline="") as handle:
            existing.update(unordered(r["text1"], r["text2"]) for r in csv.DictReader(handle))
    raw = make_raw(hypothesis)
    excluded = {r["pair_id"] for r in raw if unordered(r["text1"], r["text2"]) in existing}
    rows = [r for r in raw if r["pair_id"] not in excluded]
    ids = [r["instance_id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate probe instance ID"
    assert rows, "all diagnostic families overlapped official data"
    assert all(unordered(r["text1"], r["text2"]) not in existing for r in rows)
    if hypothesis != "h4":
        labels_by_order = {}
        for r in rows:
            key = (norm(r["text1"]), norm(r["text2"]))
            labels_by_order.setdefault(key, set()).add(r["label"])
        assert all(len(labels) == 1 for labels in labels_by_order.values()), "inconsistent authored probe labels"
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    audit = {"hypothesis": hypothesis, "synthetic": True, "informed_by_dev_error_analysis": True,
             "raw_rows": len(raw), "rows": len(rows),
             "raw_families": len({r["pair_id"] for r in raw}),
             "families": len({r["pair_id"] for r in rows}),
             "excluded_overlapping_families": sorted(excluded),
             "conditions": dict(Counter(r["condition"] for r in rows)), "sha256": digest}
    return rows, audit
