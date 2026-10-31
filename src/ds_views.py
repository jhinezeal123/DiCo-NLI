"""R1: robustness of the Dawid-Skene convention estimates to the choice of logic views.

Views are (teacher, language track) pairs. Tracks 1/2/3 are the English/Spanish/Basque versions of the same pairs
(pair-aligned, identical labels), so a multilingual teacher on Spanish or Basque text is a view whose errors are
less correlated with the English MNLI teachers. An instruction LLM queried zero-shot adds a different model family.
"""
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GroupKFold
from transformers import AutoModelForCausalLM, AutoTokenizer

import convention_probe as cp
import dico_trainer as dt
import ds_probe as dsp
import lcf_trainer as lcf

Z = lcf.Z
IDENT = np.array([1, 2, 0, 3, 3])  # Z -> label index (BE, EQ, FE, NO)
LLM_PROMPT = ("Phrase A: {a}\nPhrase B: {b}\n\nIf phrase A is an accurate description, what follows for phrase B?\n"
              "A) Phrase B is also accurate (A implies B)\nB) Phrase B is incompatible with A\n"
              "C) Cannot tell: B may or may not be accurate\nAnswer with a single letter.")


def q_from(e12, e21, c):
    rest = (1 - e12) * (1 - e21)
    q = np.stack([e12 * e21, e12 * (1 - e21), e21 * (1 - e12), rest * c, rest * (1 - c)], 1)
    return q / q.sum(1, keepdims=True)


@torch.no_grad()
def llm_q(name, rev, t1, t2, device):
    tok = AutoTokenizer.from_pretrained(name, revision=rev)
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(name, revision=rev, torch_dtype=torch.float16).to(device).eval()
    ids = [tok.encode(x, add_special_tokens=False)[0] for x in ("A", "B", "C")]

    def probs(a, b):
        out = []
        texts = [tok.apply_chat_template([{"role": "user", "content": LLM_PROMPT.format(a=x, b=y)}], tokenize=False,
                                         add_generation_prompt=True) for x, y in zip(a, b)]
        for s in range(0, len(texts), 32):
            enc = tok(texts[s:s + 32], return_tensors="pt", padding=True).to(device)
            logits = m(**enc).logits[:, -1, ids].float()
            out.append(torch.softmax(logits, -1).cpu().numpy())
        return np.concatenate(out)  # [n, (entail, contradict, neutral)]

    p12, p21 = probs(t1, t2), probs(t2, t1)
    del m
    torch.cuda.empty_cache()
    return q_from(p12[:, 0], p21[:, 0], np.maximum(p12[:, 1], p21[:, 1]))


def load_track(upstream, split, track, ids_ref):
    ids, a, b, y = dt.load_split(upstream / "final_data" / split / f"dico_nli_{split}_track{track}_participant_labeled.csv")
    key = lambda i: i.split("__")[0] + "__" + i.split("__")[2]
    pos = {key(k): n for n, k in enumerate(ids)}
    order = [pos[key(k)] for k in ids_ref]
    return [a[i] for i in order], [b[i] for i in order], y[order]


def run(cfg, upstream, emit):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = upstream / "final_data"
    tr_ids, tr1, tr2, ytr = dt.load_split(data / "train" / "dico_nli_train_track1_participant_labeled.csv")
    dv_ids, dv1, dv2, ydv = dt.load_split(data / "dev" / "dico_nli_dev_track1_participant_labeled.csv")
    groups = np.array([i.split("__")[0] for i in tr_ids])
    dom_pred = lcf.domain_model(tr_ids, tr1, tr2)
    c_tr, c_dv = cp.cue_features(tr1, tr2, dom_pred), cp.cue_features(dv1, dv2, dom_pred)  # cues from English
    ctx_dv = (4 * c_dv[:, 0] + 2 * c_dv[:, 1] + c_dv[:, 2]).astype(int)
    names = [n.replace("|", "_").replace("=", "") for n in lcf.CTX_NAMES]

    views, T_tr, T_dv, Q_tr = [], [], [], []
    for v in cfg["views"]:
        name, _, rev = v["teacher"].partition("@")
        tr = 1 if v["track"] == 1 else v["track"]
        a_tr, b_tr, y2 = (tr1, tr2, ytr) if tr == 1 else load_track(upstream, "train", tr, tr_ids)
        a_dv, b_dv, y3 = (dv1, dv2, ydv) if tr == 1 else load_track(upstream, "dev", tr, dv_ids)
        assert (y2 == ytr).all() and (y3 == ydv).all(), "cross-track labels must be identical"
        if v["kind"] == "llm":
            q_tr, q_dv = llm_q(name, rev or None, a_tr, b_tr, device), llm_q(name, rev or None, a_dv, b_dv, device)
        else:
            tc = {"lcf": {"teacher": name, "teacher_revision": rev or None}}
            q_tr, q_dv = lcf.teacher_q(tc, a_tr, b_tr, device), lcf.teacher_q(tc, a_dv, b_dv, device)
        tag = f"{name.split('/')[-1]}@t{tr}"
        views.append(tag)
        T_tr.append(q_tr.argmax(1))
        T_dv.append(q_dv.argmax(1))
        zs = lcf.logic_to_label(q_dv).argmax(1)
        emit(f"METRIC view/{tag}/zeroshot_dev_acc={np.mean(zs == ydv):.4f} view/{tag}/dist=" +
             "|".join(f"{Z[k]}:{np.mean(T_tr[-1] == k):.3f}" for k in range(5)))
        Q_tr.append(q_tr)
    T_tr, T_dv = np.stack(T_tr, 1), np.stack(T_dv, 1)
    emit("<<<TABLE view_agreement_train")
    emit("view_a,view_b,agreement,kappa")
    for i in range(len(views)):
        for j in range(i + 1, len(views)):
            emit(f"{views[i]},{views[j]},{np.mean(T_tr[:, i] == T_tr[:, j]):.4f},{cohen_kappa_score(T_tr[:, i], T_tr[:, j]):.4f}")
    emit("TABLE>>>")

    folds = list(GroupKFold(n_splits=5).split(T_tr, ytr, groups))
    member, summ = {}, []
    for setname, idx in cfg["view_sets"].items():
        sel = [views.index(v) for v in idx] if idx != "all" else list(range(len(views)))
        oof = np.zeros((len(ytr), 4))
        for a, b in folds:
            ds = dsp.DS(len(sel))
            ds.fit(np.mean([Q_tr[s][a] for s in sel], 0), T_tr[a][:, sel], ytr[a], c_tr[a])
            oof[b] = ds.predict_y(T_tr[b][:, sel], c_tr[b])
        nll, acc, f1 = cp.scores(oof, ytr)
        ds = dsp.DS(len(sel))
        r_tr = ds.fit(np.mean([Q_tr[s] for s in sel], 0), T_tr[:, sel], ytr, c_tr, emit=emit)
        C = dsp.c_of(ds.C, c_tr).mean(0)
        r_dv = ds.posterior(T_dv[:, sel], ydv, c_dv)
        pt_dv = ds.posterior(T_dv[:, sel], None, c_dv)
        conv = IDENT[r_dv.argmax(1)] != ydv
        tdis = IDENT[pt_dv.argmax(1)] != ydv
        member[setname] = (conv, tdis)
        accs = [float((ds.pi * np.diag(ds.T[k])).sum()) for k in range(len(sel))]
        emit(f"METRIC set/{setname}/n_views={len(sel)} set/{setname}/cv_nll={nll:.4f} set/{setname}/cv_f1={f1:.4f} "
             f"set/{setname}/P_DIR_given_IND={C[4, 0] + C[4, 2]:.4f} set/{setname}/P_DIR_given_EXC={C[3, 0] + C[3, 2]:.4f} "
             f"set/{setname}/P_EQ_given_EQV={C[0, 1]:.4f} set/{setname}/P_FE_given_FWD={C[1, 2]:.4f} "
             f"set/{setname}/dev_convention_rate={conv.mean():.4f} set/{setname}/dev_teacher_disagree_rate={tdis.mean():.4f} "
             f"set/{setname}/train_identity_eq_gold={np.mean(IDENT[r_tr.argmax(1)] == ytr):.4f} "
             f"set/{setname}/mean_view_acc={np.mean(accs):.4f}")
        for k in range(8):
            m = ctx_dv == k
            if m.sum():
                emit(f"convention_rate_dev set={setname} context={names[k]} n={int(m.sum())} rate={conv[m].mean():.3f}")
        emit(f"<<<TABLE ds_dev_posteriors_{setname}")
        emit("instance_id,gold," + ",".join(f"r_{z}" for z in Z) + "," + ",".join(f"pt_{z}" for z in Z))
        for i, k in enumerate(dv_ids):
            emit(f"{k},{dt.LABELS[ydv[i]]}," + ",".join(f"{v:.4f}" for v in list(r_dv[i]) + list(pt_dv[i])))
        emit("TABLE>>>")
    emit("<<<TABLE convention_membership_agreement")
    emit("set_a,set_b,conv_kappa,conv_jaccard,tdis_kappa")
    ks = list(member)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            ca, cb = member[ks[i]][0], member[ks[j]][0]
            emit(f"{ks[i]},{ks[j]},{cohen_kappa_score(ca, cb):.4f},{(ca & cb).sum() / max(1, (ca | cb).sum()):.4f},"
                 f"{cohen_kappa_score(member[ks[i]][1], member[ks[j]][1]):.4f}")
    emit("TABLE>>>")
