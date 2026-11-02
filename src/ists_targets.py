"""Train-only auxiliary targets from the PhrasIS source annotation (iSTS alignment type and 0-5 similarity score).

Only DiCo TRAIN instances are joined (normalised text match in either order); the type is oriented to the instance
(FORW<->BACK when the instance is the reversed PhrasIS pair). Nothing from PhrasIS is used for dev or at inference.
"""
import csv
import re
import urllib.request

TYPES = ["EQUI", "FORW", "BACK", "SIMI", "REL", "OPPO"]
SWAP = {"FORW": "BACK", "BACK": "FORW"}
norm = lambda s: " ".join(re.sub(r"\s+", " ", str(s)).strip().split())


def write(cfg, train_path, out_path, emit):
    base = cfg["ists_aux"]["phrasis_raw"]
    src = {}
    for split in ("train", "test"):
        for dom in ("headlines", "images"):
            raw = urllib.request.urlopen(f"{base}/data/PhrasIS.{split}.{dom}.all.txt", timeout=120).read().decode("utf-8")
            for line in raw.splitlines():
                f = line.split("\t")
                if len(f) < 7 or f[1].strip() not in TYPES:
                    continue
                src.setdefault((norm(f[2]), norm(f[3])), set()).add((f[1].strip(), float(f[0])))
    n = hit = 0
    with open(train_path, encoding="utf-8") as fin, open(out_path, "w", newline="", encoding="utf-8") as fout:
        w = csv.writer(fout)
        w.writerow(["instance_id", "ists_type", "ists_score"])
        for r in csv.DictReader(fin):
            n += 1
            a, b = norm(r["text1"]), norm(r["text2"])
            fwd, rev = src.get((a, b)), src.get((b, a))
            m = fwd if fwd else ({(SWAP.get(t, t), s) for t, s in rev} if rev else None)
            if not m or len(m) != 1:
                continue
            t, s = next(iter(m))
            hit += 1
            w.writerow([r["instance_id"], t, s])
    emit(f"ists_aux: matched {hit}/{n} train instances")
