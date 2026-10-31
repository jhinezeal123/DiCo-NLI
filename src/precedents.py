"""Retrieved annotation precedents: for each query pair, the k most similar TRAIN source pairs with their gold labels.

Each train source pair is scored in both orientations, s = max(sim(q, [c; d]), sim(q, [d; c])), and shown in the
better-matching orientation (label FE<->BE swapped when flipped). A swapped query therefore sees exactly the swapped
precedents. Train queries never see their own source pair. Dev queries retrieve from train only.
"""
import hashlib

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

WORD = {"EQUIVALENCE": "equivalent", "FORWARD_ENTAILMENT": "forward", "BACKWARD_ENTAILMENT": "backward",
        "NEGATIVE_OTHER": "unrelated"}
SWAP = {"FORWARD_ENTAILMENT": "BACKWARD_ENTAILMENT", "BACKWARD_ENTAILMENT": "FORWARD_ENTAILMENT"}


def _pair(i):
    return i.split("__")[0]


def build(tr_ids, tr1, tr2, tr_lab, q_ids, q1, q2, k=4, mode="nearest"):
    # one canonical row per train source pair
    first = {}
    for j, i in enumerate(tr_ids):
        first.setdefault(_pair(i), j)
    keys = sorted(first)
    rows = [first[p] for p in keys]
    c, d = [tr1[j] for j in rows], [tr2[j] for j in rows]
    lab = [tr_lab[j] for j in rows]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1, sublinear_tf=True).fit(list(tr1) + list(tr2))
    emb = lambda a, b: normalize(hstack([vec.transform(a), vec.transform(b)]).tocsr())
    E1, E2 = emb(c, d), emb(d, c)
    Q = emb(q1, q2)
    S1, S2 = (Q @ E1.T).toarray(), (Q @ E2.T).toarray()
    keys = np.array(keys)
    out = []
    for n, qid in enumerate(q_ids):
        qp = _pair(qid)
        if mode == "nearest":
            score = np.maximum(S1[n], S2[n])
            score[keys == qp] = -1
            chosen = np.argsort(-score, kind="stable")[:k]
            s1, s2 = S1[n][chosen], S2[n][chosen]
            # exact ties (symmetric phrases) are broken by the query's orientation so swapped queries stay consistent
            flip = (s2 > s1 + 1e-12) | ((np.abs(s2 - s1) <= 1e-12) & qid.endswith("flipped"))
        else:  # random source pairs (deterministic per query pair); orientation follows the query's orientation
            rng = np.random.default_rng(int(hashlib.md5(qp.encode()).hexdigest()[:8], 16))
            cand = np.flatnonzero(keys != qp)
            chosen = rng.choice(cand, size=k, replace=False)
            base = rng.random(k) < 0.5
            flip = base ^ qid.endswith("flipped")
        ex = []
        for j, f in zip(chosen, flip):
            a, b, l = (d[j], c[j], SWAP.get(lab[j], lab[j])) if f else (c[j], d[j], lab[j])
            ex.append(f"Example: {a} => {b}: {WORD[l]}")
        out.append("\n".join(reversed(ex)))
    return out
