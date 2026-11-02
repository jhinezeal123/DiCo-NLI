"""M6 (analysis): attach the iSTS 2016 source sentence pair to DiCo train/dev rows.

Join: DiCo (text1, text2) -> PhrasIS chunk pair (either order) -> (split, domain, sentence id) -> iSTS 2016 sentence pair.
Only the raw sentences are used (no alignment types/scores). Sentence 1 is oriented to contain text1.
Analysis only: the shared task does not provide source sentences.
"""
import csv
import io
import re
import urllib.request

norm = lambda s: " ".join(re.sub(r"\s+", " ", str(s)).strip().split())


def _get(url):
    return urllib.request.urlopen(url, timeout=180).read()


def write(cfg, split_paths, out_path, emit):
    import pandas as pd
    sc = cfg["source_context"]
    sents = {}
    for sp, fname in sc["ists_files"].items():
        d = pd.read_parquet(io.BytesIO(_get(f"{sc['ists_base']}/{fname}")))
        for r in d.itertuples():
            sents[(sp, r.subset, str(r.id))] = (" ".join(r.tokens_a), " ".join(r.tokens_b))
    src = {}
    for sp in ("train", "test"):
        for dom in ("headlines", "images"):
            raw = _get(f"{sc['phrasis_raw']}/data/PhrasIS.{sp}.{dom}.all.txt").decode("utf-8")
            for line in raw.splitlines():
                f = line.split("\t")
                if len(f) < 7:
                    continue
                src.setdefault((norm(f[2]), norm(f[3])), set()).add((sp, dom, f[6].strip()))
    shuffle = sc.get("shuffle_seed")
    with open(out_path, "w", newline="", encoding="utf-8") as fout:
        w = csv.writer(fout)
        w.writerow(["instance_id", "sent1", "sent2"])
        for split, path in split_paths.items():
            n = hit = 0
            rows = []
            for r in csv.DictReader(open(path, encoding="utf-8")):
                n += 1
                a, b = norm(r["text1"]), norm(r["text2"])
                m, rev = src.get((a, b)), False
                if not m:
                    m, rev = src.get((b, a)), True
                if not m:
                    continue
                key = sorted(m)[0]  # deterministic pick when a chunk pair occurs in several sentences
                if key not in sents:
                    continue
                hit += 1
                rows.append((r["instance_id"], key, rev))
            if shuffle is not None:  # control: each source key gets another key's sentences (same domain)
                import random
                rng = random.Random(shuffle)
                remap = {}
                for dom in sorted({k[1] for _, k, _ in rows}):
                    keys = sorted({k for _, k, _ in rows if k[1] == dom})
                    order = keys[:]
                    rng.shuffle(order)
                    remap.update({order[i]: order[(i + 1) % len(order)] for i in range(len(order))})
                same = sum(remap[k] == k for k in remap)
                emit(f"source_context: shuffled control, {len(remap)} source keys remapped, fixed points {same}")
                rows = [(i, remap[k], rev) for i, k, rev in rows]
            for i, k, rev in rows:
                s = sents[k]
                w.writerow([i, *(s[::-1] if rev else s)])
            emit(f"source_context: {split} {hit}/{n} rows with source sentences")
