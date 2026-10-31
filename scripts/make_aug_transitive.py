"""Train-only augmentation: label-sound composition of train relations over shared phrases.

Rules (a->b, b->c => a->c): EQ*X = X, X*EQ = X, FE*FE = FE, BE*BE = BE (EQ*NEG = NEG).
FE*NEG / BE*NEG are not sound and are skipped. Pairs already in train, or whose unordered
normalized text pair occurs in dev, are excluded. Non-NEG pairs get both directions, as in
the official data.
Usage: python scripts/make_aug_transitive.py <upstream_dir> data/aug_transitive.csv
"""
import collections
import csv
import sys
from pathlib import Path

FLIP = {"EQUIVALENCE": "EQUIVALENCE", "FORWARD_ENTAILMENT": "BACKWARD_ENTAILMENT",
        "BACKWARD_ENTAILMENT": "FORWARD_ENTAILMENT", "NEGATIVE_OTHER": "NEGATIVE_OTHER"}


def norm(s):
    return " ".join(s.lower().split())


def compose(l1, l2):
    if l1 == "EQUIVALENCE":
        return l2
    if l2 == "EQUIVALENCE":
        return l1
    if l1 == l2 and l1 in ("FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT"):
        return l1
    return None


def main(upstream, out):
    d = Path(upstream) / "final_data"
    read = lambda p: list(csv.DictReader(open(p, encoding="utf-8")))
    train = read(d / "train/dico_nli_train_track1_participant_labeled.csv")
    dev = read(d / "dev/dico_nli_dev_track1_participant_labeled.csv")
    surface, rel, nb = {}, {}, collections.defaultdict(set)
    for r in train:
        a, b = norm(r["text1"]), norm(r["text2"])
        surface.setdefault(a, r["text1"].strip())
        surface.setdefault(b, r["text2"].strip())
        rel[(a, b)] = r["label"]
    for (a, b), l in list(rel.items()):
        rel.setdefault((b, a), FLIP[l])
        nb[a].add(b)
        nb[b].add(a)
    dev_pairs = {frozenset((norm(r["text1"]), norm(r["text2"]))) for r in dev}
    new = {}
    for b in sorted(nb):
        for a in sorted(nb[b]):
            for c in sorted(nb[b]):
                if a == c or (a, c) in rel or frozenset((a, c)) in dev_pairs:
                    continue
                l = compose(rel[(a, b)], rel[(b, c)])
                if l is not None:
                    new.setdefault(frozenset((a, c)), (a, c, l))
    rows = []
    for k, (a, c, l) in enumerate(sorted(new.values())):
        pid = f"aug_trans_{k:05d}"
        t1, t2 = surface[a], surface[c]
        rows.append([f"{pid}__en-en__original", pid, "en", "en", t1, t2, l])
        if l != "NEGATIVE_OTHER":
            rows.append([f"{pid}__en-en__flipped", pid, "en", "en", t2, t1, FLIP[l]])
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["instance_id", "pair_id", "text1_lang", "text2_lang", "text1", "text2", "label"])
        w.writerows(rows)
    print(f"pairs={len(new)} rows={len(rows)} labels={collections.Counter(r[-1] for r in rows)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
