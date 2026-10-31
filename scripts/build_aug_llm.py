"""Turn LLM-generated variants of train anchor phrases into Track 1 rows.

Each JSONL line {"x", "neg", "spec", "para"} yields NEG (x, neg) [one row], FE/BE (spec, x) and
EQ (x, para) [both directions, as in the official data]. Pair order is randomized with a fixed
seed. Pairs identical to x, or whose unordered normalized text pair occurs in train/dev, are dropped.
Usage: python scripts/build_aug_llm.py <upstream_dir> <out.csv> <gen1.jsonl> [...]
"""
import collections
import csv
import json
import random
import sys
from pathlib import Path

FLIP = {"EQUIVALENCE": "EQUIVALENCE", "FORWARD_ENTAILMENT": "BACKWARD_ENTAILMENT",
        "BACKWARD_ENTAILMENT": "FORWARD_ENTAILMENT"}


def norm(s):
    return " ".join(str(s).lower().split())


def main(upstream, out, *gens):
    d = Path(upstream) / "final_data"
    read = lambda p: list(csv.DictReader(open(p, encoding="utf-8")))
    seen = {frozenset((norm(r["text1"]), norm(r["text2"])))
            for split in ("train/dico_nli_train", "dev/dico_nli_dev")
            for r in read(d / f"{split}_track1_participant_labeled.csv")}
    rng = random.Random(0)
    rows, dropped = [], collections.Counter()
    for g in gens:
        for line in open(g, encoding="utf-8"):
            if not line.strip():
                continue
            e = json.loads(line)
            x = e["x"].strip()
            for key, label in (("neg", "NEGATIVE_OTHER"), ("spec", "FORWARD_ENTAILMENT"), ("para", "EQUIVALENCE")):
                v = (e.get(key) or "").strip()
                k = frozenset((norm(x), norm(v)))
                if not v or norm(v) == norm(x) or k in seen:
                    dropped[key] += 1
                    continue
                seen.add(k)
                pid = f"aug_llm_{len(rows):05d}"
                t1, t2 = (v, x) if key == "spec" else (x, v)  # spec entails x
                if rng.random() < 0.5:
                    t1, t2, label = t2, t1, FLIP.get(label, label)
                rows.append([f"{pid}__en-en__original", pid, "en", "en", t1, t2, label])
                if label in FLIP:
                    rows.append([f"{pid}__en-en__flipped", pid, "en", "en", t2, t1, FLIP[label]])
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["instance_id", "pair_id", "text1_lang", "text2_lang", "text1", "text2", "label"])
        w.writerows(rows)
    print(f"rows={len(rows)} labels={collections.Counter(r[-1] for r in rows)} dropped={dict(dropped)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
