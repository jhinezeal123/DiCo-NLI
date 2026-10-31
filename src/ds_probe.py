"""M1b: multi-view latent-class (Dawid-Skene) separation of teacher error from annotation convention.

Latent logic relation z in Z = [EQV, FWD, REV, EXC, IND]. Views, conditionally independent given z:
  * each NLI teacher k: its bidirectional relation t_k = argmax q_k(x) in Z, with confusion T_k(t | z)
  * the gold label y, with the swap-tied, cue-conditioned convention channel C(y | z, phi(x))
Everything is reversal-symmetric: prior pi(FWD)=pi(REV), T_k commutes with the FWD<->REV swap, C is swap-tied.
With >= 3 views the latent-class model is generically identifiable (Kruskal), so the convention channel no
longer has to absorb teacher error, which is what made the single-teacher C of M1/M2 flat.
"""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

import convention_probe as cp
import dico_trainer as dt
import lcf_trainer as lcf

Z = lcf.Z
SW = [0, 2, 1, 3, 4]  # FWD<->REV permutation of Z
BE, EQ, FE, NO = range(4)


def sym_counts(N):  # N [z, t]
    return (N + N[SW][:, SW]) / 2


def fit_c_soft(r, y, phi, l2=1e-2, steps=300, init=None):
    """M-step for C: maximise sum_i sum_z r_iz log C(y_i | z, phi_i)."""
    m = cp.TiedC(phi.shape[1])
    if init is not None:
        m.load_state_dict(init.state_dict())
    rt, yt, pt = torch.tensor(r, dtype=torch.float32), torch.tensor(y), torch.tensor(phi)
    opt = torch.optim.LBFGS(m.parameters(), lr=0.5, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        C = m(pt)  # [n, z, y]
        ll = torch.log(C[torch.arange(len(yt)), :, yt] + 1e-9)  # [n, z]
        loss = -(rt * ll).sum(1).mean() + l2 * (m.b_sym.pow(2).sum() + m.b_fwd.pow(2).sum())
        loss.backward()
        return loss

    opt.step(closure)
    return m


def c_of(m, phi):
    with torch.no_grad():
        return m(torch.tensor(phi)).numpy()


class DS:
    def __init__(self, n_views, smooth=1.0):
        self.K, self.a = n_views, smooth

    def teacher_lik(self, t):  # t [n, K] ints -> [n, z]
        L = np.tile(np.log(self.pi), (len(t), 1))
        for k in range(self.K):
            L += np.log(self.T[k][:, t[:, k]].T)
        return L

    def m_step(self, r, t, y, phi):
        pi = r.mean(0)
        self.pi = (pi + pi[SW]) / 2
        self.T = []
        for k in range(self.K):
            N = np.zeros((5, 5))
            np.add.at(N.T, t[:, k], r)  # N[z, t] += r_iz
            N = sym_counts(N) + self.a
            self.T.append(N / N.sum(1, keepdims=True))
        self.C = fit_c_soft(r, y, phi, init=getattr(self, "C", None))

    def posterior(self, t, y, phi):
        L = self.teacher_lik(t)
        if y is not None:
            L = L + np.log(c_of(self.C, phi)[np.arange(len(y)), :, y] + 1e-9)
        L -= L.max(1, keepdims=True)
        r = np.exp(L)
        return r / r.sum(1, keepdims=True)

    def fit(self, r0, t, y, phi, iters=100, emit=None):
        r = r0
        for it in range(iters):
            self.m_step(r, t, y, phi)
            r_new = self.posterior(t, y, phi)
            delta = np.abs(r_new - r).max()
            r = r_new
            if delta < 1e-4:
                break
        if emit:
            emit(f"ds_em iterations={it + 1} final_max_delta={delta:.2e}")
        return r

    def predict_y(self, t, phi):
        pz = self.posterior(t, None, phi)
        return np.einsum("nz,nzy->ny", pz, c_of(self.C, phi))


def scores(p, y):
    return cp.scores(p, y)


def run(cfg, upstream, emit):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = upstream / "final_data"
    tr_ids, tr1, tr2, ytr = dt.load_split(data / "train" / "dico_nli_train_track1_participant_labeled.csv")
    dv_ids, dv1, dv2, ydv = dt.load_split(data / "dev" / "dico_nli_dev_track1_participant_labeled.csv")
    groups = np.array([i.split("__")[0] for i in tr_ids])
    dom_pred = lcf.domain_model(tr_ids, tr1, tr2)
    c_tr, c_dv = cp.cue_features(tr1, tr2, dom_pred), cp.cue_features(dv1, dv2, dom_pred)
    ctx_tr = (4 * c_tr[:, 0] + 2 * c_tr[:, 1] + c_tr[:, 2]).astype(int)
    ctx_dv = (4 * c_dv[:, 0] + 2 * c_dv[:, 1] + c_dv[:, 2]).astype(int)

    tags, Q_tr, Q_dv = [], [], []
    for teacher in cfg["probe"]["teachers"]:
        name, _, rev = teacher.partition("@")
        tcfg = {"lcf": {"teacher": name, "teacher_revision": rev or None}}
        Q_tr.append(lcf.teacher_q(tcfg, tr1, tr2, device))
        Q_dv.append(lcf.teacher_q(tcfg, dv1, dv2, device))
        tags.append(name.split("/")[-1])
        zs = lcf.logic_to_label(Q_dv[-1]).argmax(1)
        emit(f"METRIC teacher/{tags[-1]}/zeroshot_dev_f1={f1_score(ydv, zs, average='weighted'):.4f}")
    K = len(tags)
    t_tr = np.stack([q.argmax(1) for q in Q_tr], 1)
    t_dv = np.stack([q.argmax(1) for q in Q_dv], 1)
    qbar_tr = np.mean(Q_tr, 0)
    # pairwise teacher agreement (conditional-independence sanity check: high agreement = correlated views)
    for a in range(K):
        for b in range(a + 1, K):
            emit(f"teacher_agreement {tags[a]} vs {tags[b]}: {np.mean(t_tr[:, a] == t_tr[:, b]):.4f}")

    # ---------------- pair-grouped CV: predict y from teachers + cues
    folds = list(GroupKFold(n_splits=5).split(t_tr, ytr, groups))
    variants = {"m1_rich_base": None, "m1_rich_meanq": None, "ds_all": list(range(K)), "unstructured_all": None}
    for k in range(K):
        variants[f"ds_drop_{tags[k]}"] = [j for j in range(K) if j != k]
    rows, oof_all = [], {}
    for vname, views in variants.items():
        oof = np.zeros((len(ytr), 4))
        for a, b in folds:
            if vname == "m1_rich_base":
                oof[b] = cp.predict_c(cp.fit_c(Q_tr[0][a], ytr[a], c_tr[a]), Q_tr[0][b], c_tr[b])
            elif vname == "m1_rich_meanq":
                oof[b] = cp.predict_c(cp.fit_c(qbar_tr[a], ytr[a], c_tr[a]), qbar_tr[b], c_tr[b])
            elif vname == "unstructured_all":
                X = np.hstack([np.log(np.concatenate(Q_tr, 1) + 1e-6), c_tr])
                oof[b] = LogisticRegression(max_iter=4000).fit(X[a], ytr[a]).predict_proba(X[b])
            else:
                ds = DS(len(views))
                r0 = np.mean([Q_tr[j][a] for j in views], 0)
                ds.fit(r0, t_tr[a][:, views], ytr[a], c_tr[a])
                oof[b] = ds.predict_y(t_tr[b][:, views], c_tr[b])
        oof_all[vname] = oof
        nll, acc, f1 = scores(oof, ytr)
        rows.append([vname, nll, acc, f1])
        emit(f"METRIC cv/{vname}/nll={nll:.4f} cv/{vname}/acc={acc:.4f} cv/{vname}/f1={f1:.4f}")
    emit("<<<TABLE ds_cv")
    emit("variant,cv_nll,cv_acc,cv_f1")
    for r in rows:
        emit(f"{r[0]},{r[1]:.4f},{r[2]:.4f},{r[3]:.4f}")
    emit("TABLE>>>")
    rng = np.random.default_rng(20261031)
    ug = sorted(set(groups))
    gidx = {g: np.where(groups == g)[0] for g in ug}
    ll = {k: np.log(v[np.arange(len(ytr)), ytr] + 1e-9) for k, v in oof_all.items()}
    for vname in ("ds_all", "m1_rich_meanq", "unstructured_all"):
        d = np.array([(ll[vname][gidx[g]] - ll["m1_rich_base"][gidx[g]]).sum() for g in ug])
        boots = [d[rng.integers(0, len(ug), len(ug))].sum() / len(ytr) for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        emit(f"METRIC cv/{vname}_minus_m1_rich_base/loglik_per_row={d.sum() / len(ytr):.4f} "
             f"cv/{vname}_minus_m1_rich_base/ci_lo={lo:.4f} cv/{vname}_minus_m1_rich_base/ci_hi={hi:.4f}")

    # ---------------- full fit on train; dev evaluation and interpretation
    ds = DS(K)
    r_tr = ds.fit(qbar_tr, t_tr, ytr, c_tr, emit=emit)
    p_dv = ds.predict_y(t_dv, c_dv)
    nll, acc, f1 = scores(p_dv, ydv)
    emit(f"METRIC dev/ds_all/nll={nll:.4f} dev/ds_all/acc={acc:.4f} dev/ds_all/f1={f1:.4f}")
    emit("ds_prior " + " ".join(f"{z}={v:.4f}" for z, v in zip(Z, ds.pi)))
    emit("<<<TABLE ds_teacher_confusion")
    emit("teacher,z," + ",".join(f"t_{z}" for z in Z))
    for k in range(K):
        for zi, zn in enumerate(Z):
            emit(f"{tags[k]},{zn}," + ",".join(f"{v:.4f}" for v in ds.T[k][zi]))
    emit("TABLE>>>")
    for k in range(K):
        acc_k = float((ds.pi * np.diag(ds.T[k])).sum())
        emit(f"METRIC ds/teacher_acc/{tags[k]}={acc_k:.4f}")
    # convention channel per context (mean over train rows), compared with the M1 single-teacher rich C
    C_ds = c_of(ds.C, c_tr)
    C_m1 = c_of(cp.fit_c(Q_tr[0], ytr, c_tr), c_tr)
    emit("<<<TABLE ds_vs_m1_convention")
    emit("context,n,source,z,P_BE,P_EQ,P_FE,P_NO")
    names = [n.replace("|", "_").replace("=", "") for n in lcf.CTX_NAMES]
    for k in range(8):
        m = ctx_tr == k
        if m.sum() < 30:
            continue
        for src, Cx in (("ds", C_ds), ("m1", C_m1)):
            M = Cx[m].mean(0)
            for zi, zn in enumerate(Z):
                emit(f"{names[k]},{int(m.sum())},{src},{zn}," + ",".join(f"{v:.4f}" for v in M[zi]))
    emit("TABLE>>>")
    for src, Cx in (("ds", C_ds), ("m1", C_m1)):
        M = Cx.mean(0)
        emit(f"METRIC conv/{src}/P_EQ_given_EQV={M[0, EQ]:.4f} conv/{src}/P_FE_given_FWD={M[1, FE]:.4f} "
             f"conv/{src}/P_DIR_given_IND={M[4, FE] + M[4, BE]:.4f} conv/{src}/P_DIR_given_EXC={M[3, FE] + M[3, BE]:.4f} "
             f"conv/{src}/P_NO_given_IND={M[4, NO]:.4f} conv/{src}/P_NO_given_EXC={M[3, NO]:.4f} "
             f"conv/{src}/mean_row_entropy={float(-(Cx * np.log(Cx + 1e-9)).sum(-1).mean()):.4f}")
    # how often does the identified logic disagree with the label? (estimated convention rate)
    ident = lcf.logic_to_label(r_tr).argmax(1)
    emit(f"METRIC ds/train_identity_of_posterior_eq_gold={np.mean(ident == ytr):.4f} "
         f"ds/train_posterior_argmax_eq_teacher_mean={np.mean(r_tr.argmax(1) == qbar_tr.argmax(1)):.4f}")
    r_dv = ds.posterior(t_dv, ydv, c_dv)
    gap = lcf.logic_to_label(r_dv).argmax(1) != ydv
    for k in range(8):
        m = ctx_dv == k
        if m.sum():
            emit(f"convention_rate_dev context={names[k]} n={int(m.sum())} P(gold != identity(logic posterior))={gap[m].mean():.3f}")
    emit(f"METRIC ds/dev_convention_rate={gap.mean():.4f}")
    # dev posteriors for downstream inspection: logic posterior with and without the label view
    pz_t = ds.posterior(t_dv, None, c_dv)
    emit("<<<TABLE ds_dev_posteriors")
    emit("instance_id,gold," + ",".join(f"r_{z}" for z in Z) + "," + ",".join(f"pt_{z}" for z in Z))
    for i, k in enumerate(dv_ids):
        emit(f"{k},{dt.LABELS[ydv[i]]}," + ",".join(f"{v:.4f}" for v in list(r_dv[i]) + list(pz_t[i])))
    emit("TABLE>>>")



def write_hints(cfg, upstream, out_path, emit):
    """Label-free logic hints z_hat = argmax DS teacher-only posterior (DS parameters fit on train only)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = upstream / "final_data"
    tr_ids, tr1, tr2, ytr = dt.load_split(data / "train" / "dico_nli_train_track1_participant_labeled.csv")
    dv_ids, dv1, dv2, ydv = dt.load_split(data / "dev" / "dico_nli_dev_track1_participant_labeled.csv")
    dom_pred = lcf.domain_model(tr_ids, tr1, tr2)
    c_tr, c_dv = cp.cue_features(tr1, tr2, dom_pred), cp.cue_features(dv1, dv2, dom_pred)
    Q_tr, Q_dv = [], []
    for teacher in cfg["probe"]["teachers"]:
        name, _, rev = teacher.partition("@")
        tcfg = {"lcf": {"teacher": name, "teacher_revision": rev or None}}
        Q_tr.append(lcf.teacher_q(tcfg, tr1, tr2, device))
        Q_dv.append(lcf.teacher_q(tcfg, dv1, dv2, device))
    t_tr = np.stack([q.argmax(1) for q in Q_tr], 1)
    t_dv = np.stack([q.argmax(1) for q in Q_dv], 1)
    ds = DS(len(Q_tr))
    ds.fit(np.mean(Q_tr, 0), t_tr, ytr, c_tr, emit=emit)
    z_tr = ds.posterior(t_tr, None, c_tr).argmax(1)
    z_dv = ds.posterior(t_dv, None, c_dv).argmax(1)
    # swap check: hints of swapped dev inputs must be the FWD<->REV image
    z_dv_sw = ds.posterior(np.array(SW)[t_dv], None, c_dv).argmax(1)  # teacher relations of swapped inputs
    emit(f"hint_swap_consistency_dev={np.mean(np.array(SW)[z_dv] == z_dv_sw):.4f}")
    if cfg["logic_hint"] == "shuffled":
        rng = np.random.default_rng(20261031)
        z_tr, z_dv = rng.permutation(z_tr), rng.permutation(z_dv)
    for nm, z, y in (("train", z_tr, ytr), ("dev", z_dv, ydv)):
        pred = lcf.logic_to_label(np.eye(5)[z]).argmax(1)
        emit(f"METRIC hint/{nm}/identity_label_acc={np.mean(pred == y):.4f} hint/{nm}/identity_label_f1="
             f"{f1_score(y, pred, average='weighted'):.4f} hint/{nm}/dist=" + "|".join(f"{Z[k]}:{np.mean(z == k):.3f}" for k in range(5)))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("instance_id,z\n")
        for ids, z in ((tr_ids, z_tr), (dv_ids, z_dv)):
            for k, v in zip(ids, z):
                f.write(f"{k},{Z[v]}\n")

