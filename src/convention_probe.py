"""M1: fixed-logic convention probe.

The logic layer is fixed to a frozen bidirectional MNLI teacher, z := q(x) over
Z = [EQV, FWD, REV, EXC, IND]; only the convention channel C(y | z, x) is learned:

    p(y | x) = sum_z q(z | x) C(y | z, x)

C is swap-tied (REV row = FWD row with FE/BE swapped; EQV/EXC/IND rows give FE and BE equal mass), so
every model below is reversal-consistent by construction. C variants differ only in what they may
condition on. Reports pair-grouped 5-fold CV and train->dev scores, bootstrap CIs, and tables for W&B.
"""
import csv
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

import dico_trainer as dt
import lcf_trainer as lcf

LAB = dt.LABELS  # BE, EQ, FE, NO
BE, EQ, FE, NO = range(4)
Z = lcf.Z
HEDGE = re.compile(r"\b(may|might|could|would|possible|possibly|reportedly|likely|expected|plans?|set to|feared|"
                   r"suspected|alleged|purported|seeks?)\b", re.I)


def cue_features(t1, t2, dom_pred):
    """Swap-invariant surface cues, one row per pair."""
    d = dom_pred(t1, t2)
    rows = []
    for h, a, b in zip(d, t1, t2):
        ta, tb = set(lcf.toks(a)), set(lcf.toks(b))
        ca, cb = ta - lcf.FW, tb - lcf.FW
        nest = int(bool(ca and cb and ca != cb and (ca < cb or cb < ca)))
        both_unique = int(bool(ca - cb) and bool(cb - ca))
        no_overlap = int(bool(ca and cb) and not (ca & cb))
        jac = len(ca & cb) / max(1, len(ca | cb))
        lgap = min(abs(len(a.split()) - len(b.split())), 5) / 5
        fact = int(bool(HEDGE.search(a)) != bool(HEDGE.search(b)))
        rows.append([h, lcf.fw_asym(a, b), lcf.num_mismatch(a, b), fact, nest, both_unique, no_overlap, jac, lgap])
    return np.array(rows, dtype=np.float32)


CUES = ["headline", "fw_asym", "num_mismatch", "fact_mismatch", "nesting", "both_unique", "no_overlap", "jaccard", "len_gap"]


class TiedC(torch.nn.Module):
    """C(y|z,x) = softmax(A_z + B_z phi(x)) with swap tying. n_feat=0 -> context-free; one-hot phi -> per-context."""

    def __init__(self, n_feat, init=4.0):
        super().__init__()
        a_sym = torch.zeros(3, 3)  # rows EQV, EXC, IND ; cols DIR, EQ, NO
        a_sym[0, 1], a_sym[1, 2], a_sym[2, 2] = init, init, init
        a_fwd = torch.zeros(4)
        a_fwd[FE] = init
        self.a_sym, self.a_fwd = torch.nn.Parameter(a_sym), torch.nn.Parameter(a_fwd)
        self.b_sym = torch.nn.Parameter(torch.zeros(3, 3, n_feat))
        self.b_fwd = torch.nn.Parameter(torch.zeros(4, n_feat))

    def forward(self, phi):  # phi [n, f] -> C [n, z, y]
        n = phi.shape[0]
        sym = self.a_sym[None] + torch.einsum("rcf,nf->nrc", self.b_sym, phi)  # [n,3,3]
        fwd = self.a_fwd[None] + phi @ self.b_fwd.T  # [n,4]
        out = torch.empty(n, 5, 4)
        for row, k in ((0, 0), (3, 1), (4, 2)):
            s = sym[:, k]
            out[:, row, BE], out[:, row, FE], out[:, row, EQ], out[:, row, NO] = s[:, 0], s[:, 0], s[:, 1], s[:, 2]
        out[:, 1] = fwd
        out[:, 2] = fwd[:, [FE, EQ, BE, NO]]
        return torch.softmax(out, -1)


def fit_c(q, y, phi, l2=1e-2, steps=400):
    m = TiedC(phi.shape[1])
    qt, yt, pt = torch.tensor(q, dtype=torch.float32), torch.tensor(y), torch.tensor(phi)
    opt = torch.optim.LBFGS(m.parameters(), lr=0.5, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        py = torch.einsum("nz,nzy->ny", qt, m(pt))
        loss = F.nll_loss(torch.log(py + 1e-9), yt) + l2 * (m.b_sym.pow(2).sum() + m.b_fwd.pow(2).sum())
        loss.backward()
        return loss

    opt.step(closure)
    return m


def predict_c(m, q, phi):
    with torch.no_grad():
        return torch.einsum("nz,nzy->ny", torch.tensor(q, dtype=torch.float32), m(torch.tensor(phi))).numpy()


def onehot_ctx(g, k):
    return np.eye(k, dtype=np.float32)[g]


def variants(cues):
    dom = cues[:, 0].astype(int)
    ctx8 = (4 * cues[:, 0] + 2 * cues[:, 1] + cues[:, 2]).astype(int)
    return {
        "global": np.zeros((len(cues), 0), dtype=np.float32),
        "domain": onehot_ctx(dom, 2),
        "ctx8": onehot_ctx(ctx8, 8),
        "rich": cues,
    }


def scores(p, y):
    nll = float(-np.mean(np.log(p[np.arange(len(y)), y] + 1e-9)))
    pred = p.argmax(1)
    return nll, float(np.mean(pred == y)), float(f1_score(y, pred, average="weighted"))


def unstructured(q_tr, c_tr, y_tr, q_te, c_te):
    X = lambda q, c: np.hstack([np.log(q + 1e-6), c])
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(X(q_tr, c_tr), y_tr)
    return clf.predict_proba(X(q_te, c_te))


def run(cfg, upstream, emit):
    """emit(line) prints to the run log. Returns nothing; everything is logged."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = "track1"
    data = Path(upstream) / "final_data"
    tr_ids, tr1, tr2, ytr = dt.load_split(data / "train" / f"dico_nli_train_{t}_participant_labeled.csv")
    dv_ids, dv1, dv2, ydv = dt.load_split(data / "dev" / f"dico_nli_dev_{t}_participant_labeled.csv")
    groups = [i.split("__")[0] for i in tr_ids]
    dom_pred = lcf.domain_model(tr_ids, tr1, tr2)
    c_tr, c_dv = cue_features(tr1, tr2, dom_pred), cue_features(dv1, dv2, dom_pred)
    assert np.allclose(c_dv, cue_features(dv2, dv1, dom_pred)), "cues must be swap-invariant"
    emit("cue_means_train " + " ".join(f"{n}={v:.3f}" for n, v in zip(CUES, c_tr.mean(0))))
    rng = np.random.default_rng(20261031)

    for teacher in cfg["probe"]["teachers"]:
        name, _, rev = teacher.partition("@")
        tag = name.split("/")[-1]
        tcfg = {"lcf": {"teacher": name, "teacher_revision": rev or None}}
        q_tr = lcf.teacher_q(tcfg, tr1, tr2, device)
        q_dv = lcf.teacher_q(tcfg, dv1, dv2, device)
        p0 = lcf.logic_to_label(q_dv)
        nll0, acc0, f10 = scores(p0, ydv)
        emit(f"METRIC {tag}/identity/dev_nll={nll0:.4f} {tag}/identity/dev_acc={acc0:.4f} {tag}/identity/dev_f1={f10:.4f}")

        # pair-grouped 5-fold CV on train + train->dev
        V_tr, V_dv = variants(c_tr), variants(c_dv)
        cv_rows, cv_pred = [], {}
        folds = list(GroupKFold(n_splits=5).split(q_tr, ytr, groups))
        for name in list(V_tr) + ["unstructured"]:
            oof = np.zeros((len(ytr), 4))
            for a, b in folds:
                if name == "unstructured":
                    oof[b] = unstructured(q_tr[a], c_tr[a], ytr[a], q_tr[b], c_tr[b])
                else:
                    m = fit_c(q_tr[a], ytr[a], V_tr[name][a])
                    oof[b] = predict_c(m, q_tr[b], V_tr[name][b])
            cv_pred[name] = oof
            nll, acc, f1 = scores(oof, ytr)
            if name == "unstructured":
                pdv = unstructured(q_tr, c_tr, ytr, q_dv, c_dv)
            else:
                pdv = predict_c(fit_c(q_tr, ytr, V_tr[name]), q_dv, V_dv[name])
            dn, da, df = scores(pdv, ydv)
            # reversal consistency on dev for gold-reversible pairs (by construction 1.0 for tied C)
            cv_rows.append([tag, name, nll, acc, f1, dn, da, df])
            emit(f"METRIC {tag}/{name}/cv_nll={nll:.4f} {tag}/{name}/cv_f1={f1:.4f} {tag}/{name}/dev_nll={dn:.4f} "
                 f"{tag}/{name}/dev_f1={df:.4f} {tag}/{name}/dev_acc={da:.4f}")
        emit(f"<<<TABLE cv_{tag}")
        emit("teacher,variant,cv_nll,cv_acc,cv_f1,dev_nll,dev_acc,dev_f1")
        for r in cv_rows:
            emit(",".join([r[0], r[1]] + [f"{v:.4f}" for v in r[2:]]))
        emit("TABLE>>>")

        # paired bootstrap over source pairs: held-out NLL difference vs global
        ug = sorted(set(groups))
        gidx = {g: [] for g in ug}
        for i, g in enumerate(groups):
            gidx[g].append(i)
        ll = {k: np.log(v[np.arange(len(ytr)), ytr] + 1e-9) for k, v in cv_pred.items()}
        per_pair = {k: np.array([v[gidx[g]].sum() for g in ug]) for k, v in ll.items()}
        for name in ("domain", "ctx8", "rich", "unstructured"):
            d = per_pair[name] - per_pair["global"]
            boots = [d[rng.integers(0, len(ug), len(ug))].sum() / len(ytr) for _ in range(2000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            emit(f"METRIC {tag}/{name}_minus_global/cv_loglik_per_row={d.sum() / len(ytr):.4f} "
                 f"{tag}/{name}_minus_global/ci_lo={lo:.4f} {tag}/{name}_minus_global/ci_hi={hi:.4f}")

        # 8-context C on full train with pair-bootstrap CIs
        ctx8 = (4 * c_tr[:, 0] + 2 * c_tr[:, 1] + c_tr[:, 2]).astype(int)
        full = fit_c(q_tr, ytr, V_tr["ctx8"])
        C_hat = full(torch.eye(8)).detach().numpy()
        B = int(cfg["probe"].get("bootstrap", 300))
        boot = np.zeros((B, 8, 5, 4))
        for bi in range(B):
            pick = rng.integers(0, len(ug), len(ug))
            idx = np.concatenate([gidx[ug[j]] for j in pick])
            boot[bi] = fit_c(q_tr[idx], ytr[idx], V_tr["ctx8"][idx])(torch.eye(8)).detach().numpy()
        lo, hi = np.percentile(boot, 2.5, 0), np.percentile(boot, 97.5, 0)
        emit(f"<<<TABLE C8_{tag}")
        emit("context,train_rows,z,y,estimate,ci_lo,ci_hi")
        names = [n.replace("|", "_").replace("=", "") for n in lcf.CTX_NAMES]
        for k in range(8):
            for zi, zn in enumerate(Z):
                for yi, yn in enumerate(["BE", "EQ", "FE", "NO"]):
                    emit(f"{names[k]},{int((ctx8 == k).sum())},{zn},{yn},{C_hat[k, zi, yi]:.4f},{lo[k, zi, yi]:.4f},{hi[k, zi, yi]:.4f}")
        emit("TABLE>>>")
        # headline deviations: P(y = NO | z=FWD) and P(y in DIR | z=IND), per context with CI
        for k in range(8):
            if (ctx8 == k).sum() < 30:
                continue
            emit(f"METRIC {tag}/C8/{names[k]}/P_NO_given_FWD={C_hat[k, 1, NO]:.4f} "
                 f"{tag}/C8/{names[k]}/P_NO_given_FWD_ci=[{lo[k, 1, NO]:.3f},{hi[k, 1, NO]:.3f}] "
                 f"{tag}/C8/{names[k]}/P_DIR_given_IND={C_hat[k, 4, FE] + C_hat[k, 4, BE]:.4f} "
                 f"{tag}/C8/{names[k]}/P_EQ_given_EQV={C_hat[k, 0, EQ]:.4f} "
                 f"{tag}/C8/{names[k]}/P_EQ_given_EQV_ci=[{lo[k, 0, EQ]:.3f},{hi[k, 0, EQ]:.3f}]")

        # rich-cue C coefficients (interpretability): effect of each cue on key transitions
        rich = fit_c(q_tr, ytr, V_tr["rich"])
        with torch.no_grad():
            emit(f"<<<TABLE rich_coef_{tag}")
            emit("transition," + ",".join(CUES))
            bf, bs = rich.b_fwd.numpy(), rich.b_sym.numpy()
            emit("FWD->NO (logit)," + ",".join(f"{v:.3f}" for v in bf[NO]))
            emit("FWD->EQ (logit)," + ",".join(f"{v:.3f}" for v in bf[EQ]))
            emit("EQV->NO (logit)," + ",".join(f"{v:.3f}" for v in bs[0, 2]))
            emit("EQV->DIR (logit)," + ",".join(f"{v:.3f}" for v in bs[0, 0]))
            emit("IND->DIR (logit)," + ",".join(f"{v:.3f}" for v in bs[2, 0]))
            emit("IND->EQ (logit)," + ",".join(f"{v:.3f}" for v in bs[2, 1]))
            emit("EXC->DIR (logit)," + ",".join(f"{v:.3f}" for v in bs[1, 0]))
            emit("TABLE>>>")
        # save teacher q for downstream nodes
        emit(f"<<<TABLE teacher_q_dev_{tag}")
        emit("instance_id," + ",".join(f"q_{z}" for z in Z))
        for i, k in enumerate(dv_ids):
            emit(k + "," + ",".join(f"{v:.4f}" for v in q_dv[i]))
        emit("TABLE>>>")
