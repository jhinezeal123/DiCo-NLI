"""R2: the reversal-tied multi-view Dawid-Skene framework on SICK, plus data for external validation.

SICK (Marelli et al., 2014) annotates entailment in both directions (entailment_AB, entailment_BA), so each pair
has a 4-way directional label: EQ (A=>B and B=>A), FE (A=>B only), BE (B=>A only), NO (otherwise). Both orders are
used as instances (the swapped order has FE<->BE). DS parameters are fit on the SICK TRAIN split only.
Expert re-annotations (Kalouli et al.) are joined offline; they are never used here.
"""
import csv
import io
import urllib.request

import numpy as np
import torch
from sklearn.metrics import f1_score

import convention_probe as cp
import ds_probe as dsp
import lcf_trainer as lcf

Z = lcf.Z
LAB = ["BACKWARD_ENTAILMENT", "EQUIVALENCE", "FORWARD_ENTAILMENT", "NEGATIVE_OTHER"]
IDENT = np.array([1, 2, 0, 3, 3])


def directional(ab, ba):
    a, b = ab == "A_entails_B", ba == "B_entails_A"
    return "EQUIVALENCE" if a and b else "FORWARD_ENTAILMENT" if a else "BACKWARD_ENTAILMENT" if b else "NEGATIVE_OTHER"


def load(url):
    raw = urllib.request.urlopen(url, timeout=120).read().decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(raw), delimiter="\t"))
    out = []
    for r in rows:
        y = directional(r["entailment_AB"], r["entailment_BA"])
        sw = {"FORWARD_ENTAILMENT": "BACKWARD_ENTAILMENT", "BACKWARD_ENTAILMENT": "FORWARD_ENTAILMENT"}.get(y, y)
        out.append((f"sick_{r['pair_ID']}__original", r["sentence_A"], r["sentence_B"], y, r["SemEval_set"]))
        out.append((f"sick_{r['pair_ID']}__flipped", r["sentence_B"], r["sentence_A"], sw, r["SemEval_set"]))
    return out


def run(cfg, upstream, emit):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load(cfg["sick"]["url"])
    ids = [r[0] for r in rows]
    t1, t2 = [r[1] for r in rows], [r[2] for r in rows]
    y = np.array([LAB.index(r[3]) for r in rows])
    split = np.array([r[4] for r in rows])
    emit(f"sick rows={len(rows)} " + " ".join(f"{s}={int((split == s).sum())}" for s in sorted(set(split))) + " labels " +
         " ".join(f"{l[:2]}={np.mean(y == k):.3f}" for k, l in enumerate(LAB)))
    cues = cp.cue_features(t1, t2, lambda a, b: np.zeros(len(a), dtype=int))  # no headline/caption domain in SICK
    T, Q = [], []
    for teacher in cfg["probe"]["teachers"]:
        name, _, rev = teacher.partition("@")
        q = lcf.teacher_q({"lcf": {"teacher": name, "teacher_revision": rev or None}}, t1, t2, device)
        Q.append(q)
        T.append(q.argmax(1))
        zs = lcf.logic_to_label(q).argmax(1)
        emit(f"METRIC sick/teacher/{name.split('/')[-1]}/zeroshot_acc={np.mean(zs == y):.4f} "
             f"sick/teacher/{name.split('/')[-1]}/zeroshot_f1={f1_score(y, zs, average='weighted'):.4f}")
    T = np.stack(T, 1)
    tr = split == "TRAIN"
    ds = dsp.DS(T.shape[1])
    ds.fit(np.mean([q[tr] for q in Q], 0), T[tr], y[tr], cues[tr], emit=emit)
    C = dsp.c_of(ds.C, cues).mean(0)
    emit("ds_prior " + " ".join(f"{z}={v:.4f}" for z, v in zip(Z, ds.pi)))
    emit(f"METRIC sick/P_DIR_given_IND={C[4, 0] + C[4, 2]:.4f} sick/P_DIR_given_EXC={C[3, 0] + C[3, 2]:.4f} "
         f"sick/P_EQ_given_EQV={C[0, 1]:.4f} sick/P_FE_given_FWD={C[1, 2]:.4f} sick/P_NO_given_FWD={C[1, 3]:.4f} "
         f"sick/P_NO_given_EQV={C[0, 3]:.4f} sick/P_FE_given_EQV={C[0, 2]:.4f}")
    for k in range(T.shape[1]):
        emit(f"METRIC sick/view_acc/{k}={float((ds.pi * np.diag(ds.T[k])).sum()):.4f}")
    emit("<<<TABLE sick_channel")
    emit("z,P_BE,P_EQ,P_FE,P_NO")
    for zi, zn in enumerate(Z):
        emit(f"{zn}," + ",".join(f"{v:.4f}" for v in C[zi]))
    emit("TABLE>>>")
    r = ds.posterior(T, y, cues)
    pt = ds.posterior(T, None, cues)
    conv = IDENT[r.argmax(1)] != y
    tdis = IDENT[pt.argmax(1)] != y
    for s in sorted(set(split)):
        m = split == s
        emit(f"METRIC sick/{s}/convention_rate={conv[m].mean():.4f} sick/{s}/teacher_disagree_rate={tdis[m].mean():.4f}")
    for k, l in enumerate(LAB):
        m = y == k
        emit(f"sick convention_rate gold={l} n={int(m.sum())} rate={conv[m].mean():.4f}")
    # P(gold != identity) as a soft convention score
    soft = 1 - (r * (IDENT[None, :] == y[:, None])).sum(1)
    emit("<<<TABLE sick_posteriors")
    emit("instance_id,split,gold,conv_soft," + ",".join(f"r_{z}" for z in Z) + "," + ",".join(f"pt_{z}" for z in Z))
    for i in range(len(ids)):
        emit(f"{ids[i]},{split[i]},{LAB[y[i]]},{soft[i]:.4f}," + ",".join(f"{v:.4f}" for v in list(r[i]) + list(pt[i])))
    emit("TABLE>>>")
