"""Logic-Convention Factorization (LCF).

    p(y | x) = sum_z p(z | x) * C(y | z, g(x))

z: latent natural-logic relation between the ordered phrases, Z = [EQV, FWD, REV, EXC, IND]
   (FWD: text1 entails text2 only; REV: text2 entails text1 only; EXC: mutually exclusive; IND: neither).
   Predicted by the usual LoRA sequence classifier with a 5-way head, and anchored with
   lambda * KL(q || p(z|x)) to q built from a frozen MNLI teacher run in both directions.
C: learned row-stochastic "convention" matrix mapping logic to DiCo labels, one per context g.
   Contexts are swap-invariant surface cues: predicted domain (headline/caption), one-sided function
   word around nested content, number mismatch. C is tied so that swapping the inputs (FWD<->REV)
   swaps FE<->BE in the output: reversal consistency holds by construction.

Returns 4-way label probabilities in dico_trainer.LABELS order (pipeline-compatible) and logs the
learned convention matrices plus diagnostics.
"""
import csv
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import dico_trainer as dt

Z = ["EQV", "FWD", "REV", "EXC", "IND"]
BE, EQ, FE, NO = (dt.LABEL2ID[l] for l in ("BACKWARD_ENTAILMENT", "EQUIVALENCE", "FORWARD_ENTAILMENT", "NEGATIVE_OTHER"))
REPO = Path(__file__).resolve().parents[1]
FW = set("a an the in on at of for to from with by over under into onto after before during as and or 's , . - "
         "near along about against amid among between than".split())
NUM = re.compile(r"\b(\d[\d,.]*|one|two|three|four|five|six|seven|eight|nine|ten|dozens?|hundreds?|thousands?|"
                 r"millions?|billions?)\b", re.I)


# ----------------------------------------------------------------------------- contexts g(x)
def toks(s):
    return re.findall(r"[a-z0-9']+|'s", s.lower())


def fw_asym(a, b):
    ta, tb = toks(a), toks(b)
    ca, cb = set(ta) - FW, set(tb) - FW
    if not ca or not cb or ca == cb or not (ca < cb or cb < ca):
        return 0
    gen, spec = (ta, tb) if ca < cb else (tb, ta)
    return int(bool((set(gen) & FW) - (set(spec) & FW)))


def num_mismatch(a, b):
    na, nb = set(NUM.findall(a.lower())), set(NUM.findall(b.lower()))
    return int(bool(na or nb) and na != nb)


def domain_model(train_ids, t1, t2):
    """Text-only headline-vs-caption classifier fit on train rows whose PhrasIS domain is known."""
    dom = {r["instance_id"]: r["domain"] for r in csv.DictReader(open(REPO / "data/eval/train_domain_track1.csv", encoding="utf-8"))}
    text = [f"{a} || {b}" for a, b in zip(t1, t2)]
    idx = [i for i, k in enumerate(train_ids) if dom.get(k) in ("headlines", "images")]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, lowercase=False)
    X = vec.fit_transform([text[i] for i in idx])
    clf = LogisticRegression(max_iter=2000, C=4.0).fit(X, [dom[train_ids[i]] == "headlines" for i in idx])
    def predict(a, b):  # averaged over both orders so the context is exactly swap-invariant
        pa = clf.predict_proba(vec.transform([f"{x} || {y}" for x, y in zip(a, b)]))[:, 1]
        pb = clf.predict_proba(vec.transform([f"{y} || {x}" for x, y in zip(a, b)]))[:, 1]
        return ((pa + pb) / 2 > 0.5).astype(int)
    return predict


def contexts(t1, t2, dom_pred, n_ctx):
    if n_ctx == 1:
        return np.zeros(len(t1), dtype=int)
    d = dom_pred(t1, t2)
    return np.array([4 * h + 2 * fw_asym(a, b) + num_mismatch(a, b) for h, a, b in zip(d, t1, t2)])


CTX_NAMES = [f"{'headline' if h else 'caption'}|fw_asym={f}|num_mismatch={n}" for h in (0, 1) for f in (0, 1) for n in (0, 1)]


# ----------------------------------------------------------------------------- convention layer
class Convention(torch.nn.Module):
    """C(y | z, g) with swap tying. Free logits: EQV/EXC/IND rows over (BE=FE tied, EQ, NO) and FWD row over 4."""

    def __init__(self, n_ctx, init=4.0):
        super().__init__()
        sym = torch.zeros(n_ctx, 3, 3)  # rows EQV, EXC, IND ; cols DIR (shared FE/BE), EQ, NO
        sym[:, 0, 1] = init  # EQV -> EQ
        sym[:, 1, 2] = init  # EXC -> NO
        sym[:, 2, 2] = init  # IND -> NO
        fwd = torch.zeros(n_ctx, 4)
        fwd[:, FE] = init  # FWD -> FE
        self.sym, self.fwd = torch.nn.Parameter(sym), torch.nn.Parameter(fwd)

    def forward(self):
        n = self.sym.shape[0]
        out = torch.empty(n, len(Z), 4, device=self.sym.device)
        for row, k in ((0, 0), (3, 1), (4, 2)):
            s = self.sym[:, k]
            out[:, row, BE], out[:, row, FE], out[:, row, EQ], out[:, row, NO] = s[:, 0], s[:, 0], s[:, 1], s[:, 2]
        out[:, 1] = self.fwd
        out[:, 2] = self.fwd[:, [FE, EQ, BE, NO]]  # REV row = FWD row with FE<->BE (LABELS order BE,EQ,FE,NO)
        return torch.softmax(out, dim=-1)  # [ctx, z, y]


# ----------------------------------------------------------------------------- teacher anchor
@torch.no_grad()
def teacher_q(cfg, t1, t2, device):
    name = cfg["lcf"]["teacher"]
    tok = AutoTokenizer.from_pretrained(name, revision=cfg["lcf"].get("teacher_revision"))
    m = AutoModelForSequenceClassification.from_pretrained(name, revision=cfg["lcf"].get("teacher_revision")).to(device).eval()
    lab = {v.lower(): int(k) for k, v in m.config.id2label.items()}
    ient, icon = lab["entailment"], lab["contradiction"]

    def probs(a, b):
        out = []
        for s in range(0, len(a), 128):
            enc = tok(a[s:s + 128], b[s:s + 128], truncation=True, max_length=128, padding=True, return_tensors="pt").to(device)
            out.append(torch.softmax(m(**enc).logits.float(), -1).cpu().numpy())
        return np.concatenate(out)

    p12, p21 = probs(t1, t2), probs(t2, t1)
    del m
    torch.cuda.empty_cache()
    e12, e21 = p12[:, ient], p21[:, ient]
    c = np.maximum(p12[:, icon], p21[:, icon])
    rest = (1 - e12) * (1 - e21)
    q = np.stack([e12 * e21, e12 * (1 - e21), e21 * (1 - e12), rest * c, rest * (1 - c)], 1)
    return q / q.sum(1, keepdims=True)


def logic_to_label(q):
    """Identity convention: EQV->EQ, FWD->FE, REV->BE, EXC/IND->NO (for the teacher zero-shot baseline)."""
    p = np.zeros((len(q), 4))
    p[:, EQ], p[:, FE], p[:, BE], p[:, NO] = q[:, 0], q[:, 1], q[:, 2], q[:, 3] + q[:, 4]
    return p


# ----------------------------------------------------------------------------- training
def run(cfg, seed, train_path, dev_path, device):
    g = dt.set_seed(seed)
    tr_ids, tr1, tr2, ytr = dt.load_split(train_path)
    dv_ids, dv1, dv2, ydv = dt.load_split(dev_path)
    lc = cfg["lcf"]
    n_ctx = int(lc["n_contexts"])
    amp = bool(cfg.get("amp", False)) and device.type == "cuda"

    q_tr = teacher_q(cfg, tr1, tr2, device)
    q_dv = teacher_q(cfg, dv1, dv2, device)
    zs = logic_to_label(q_dv).argmax(1)
    print(f"teacher={lc['teacher']} zero-shot dev (identity convention): acc {np.mean(zs == ydv):.4f} "
          f"weighted_f1 {f1_score(ydv, zs, average='weighted'):.4f}", flush=True)

    dom_pred = domain_model(tr_ids, tr1, tr2)
    dev_dom = {r["instance_id"]: r["domain"] for r in csv.DictReader(open(REPO / "data/eval/dev_domain_track1.csv", encoding="utf-8"))}
    dp = dom_pred(dv1, dv2)
    known = [i for i, k in enumerate(dv_ids) if dev_dom.get(k) in ("headlines", "images")]
    print(f"domain classifier dev accuracy on PhrasIS-matched rows: "
          f"{np.mean([dp[i] == (dev_dom[dv_ids[i]] == 'headlines') for i in known]):.4f} (n={len(known)})", flush=True)
    g_tr, g_dv = contexts(tr1, tr2, dom_pred, n_ctx), contexts(dv1, dv2, dom_pred, n_ctx)
    g_dv_swap = contexts(dv2, dv1, dom_pred, n_ctx)
    names = CTX_NAMES if n_ctx == 8 else ["global"]
    print("train rows per context:", {names[k]: int(v) for k, v in enumerate(np.bincount(g_tr, minlength=n_ctx))})
    assert np.array_equal(g_dv, g_dv_swap), "contexts must be swap-invariant"

    tok, model = dt.build_model(cfg, len(Z), device)
    conv = Convention(n_ctx).to(device)
    L, tpl = cfg["max_length"], cfg.get("pair_template")
    f_tr = dt.encode(tok, tr1, tr2, L, tpl)
    f_dv, f_dv_swap = dt.encode(tok, dv1, dv2, L, tpl), dt.encode(tok, dv2, dv1, L, tpl)

    bs, epochs = int(cfg["batch_size"]), int(cfg["num_epochs"])
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": params, "lr": float(cfg["lr"]), "weight_decay": float(cfg["wd"])},
                             {"params": conv.parameters(), "lr": float(lc["conv_lr"]), "weight_decay": 0.0}],
                            betas=(0.9, 0.999), eps=1e-6)
    steps = (len(f_tr) + bs - 1) // bs
    sched = transformers.get_linear_schedule_with_warmup(opt, int(steps * epochs * float(cfg["warmup_pcrt"])), steps * epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    y_t, q_t, g_t = torch.tensor(ytr, device=device), torch.tensor(q_tr, dtype=torch.float32, device=device), torch.tensor(g_tr, device=device)
    ls, lam = float(cfg.get("label_smoothing", 0.0)), float(lc["anchor_lambda"])

    @torch.no_grad()
    def predict(feats, gg):
        model.eval()
        C = conv()
        pz = torch.tensor(dt.predict_probs(model, tok, feats, bs, device, amp), dtype=torch.float32, device=device)
        py = torch.einsum("nz,nzy->ny", pz, C[torch.tensor(gg, device=device)])
        return pz.cpu().numpy().astype(np.float64), py.cpu().numpy().astype(np.float64)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(f_tr), generator=g).tolist()
        tot = [0.0, 0.0]
        for s in range(0, len(perm), bs):
            idx = perm[s:s + bs]
            b = dt.batch(tok, f_tr, idx, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = model(**b).logits
            logpz = F.log_softmax(logits.float(), -1)
            C = conv()[g_t[idx]]
            logpy = torch.logsumexp(logpz.unsqueeze(-1) + torch.log(C + 1e-9), dim=1)
            target = F.one_hot(y_t[idx], 4).float() * (1 - ls) + ls / 4
            nll = -(target * logpy).sum(-1).mean()
            kl = (q_t[idx] * (torch.log(q_t[idx] + 1e-9) - logpz)).sum(-1).mean()
            loss = nll + lam * kl
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot[0] += nll.item()
            tot[1] += kl.item()
        _, py = predict(f_dv, g_dv)
        print(f"Epoch {epoch + 1} nll {tot[0] / steps:.4f} kl_anchor {tot[1] / steps:.4f} | dev acc "
              f"{np.mean(py.argmax(1) == ydv):.4f} weighted_f1 {f1_score(ydv, py.argmax(1), average='weighted'):.4f}", flush=True)

    pz, p = predict(f_dv, g_dv)
    _, q = predict(f_dv_swap, g_dv_swap)

    # ---------------- interpretability diagnostics
    C = conv().detach().cpu().numpy()
    counts = np.bincount(g_tr, minlength=n_ctx)
    print("<<<CONVENTION_MATRICES (rows z, cols y in order BE,EQ,FE,NO)")
    for k in range(n_ctx):
        print(f"context={names[k]} train_rows={counts[k]}")
        for zi, zn in enumerate(Z):
            print(f"  {zn}: " + " ".join(f"{v:.3f}" for v in C[k, zi]))
    print("CONVENTION_MATRICES>>>")
    zt = q_dv.argmax(1)
    print(f"z_vs_teacher dev argmax agreement {np.mean(pz.argmax(1) == zt):.4f}; "
          f"z distribution {dict(zip(Z, np.round(pz.mean(0), 4)))}; teacher {dict(zip(Z, np.round(q_dv.mean(0), 4)))}")
    # convention gap: predicted label differs from the identity mapping of the model's own logic
    ident = logic_to_label(pz).argmax(1)
    for k in range(n_ctx):
        m = g_dv == k
        if m.sum():
            print(f"convention_gap context={names[k]} n={int(m.sum())} "
                  f"P(label != logic-identity label)={np.mean(p.argmax(1)[m] != ident[m]):.3f} "
                  f"acc={np.mean(p.argmax(1)[m] == ydv[m]):.3f}")
    print("<<<DEV_Z_PROBS")
    print("instance_id,context," + ",".join(f"pz_{z}" for z in Z) + "," + ",".join(f"q_{z}" for z in Z))
    for i, k in enumerate(dv_ids):
        print(f"{k},{g_dv[i]}," + ",".join(f"{v:.4f}" for v in list(pz[i]) + list(q_dv[i])))
    print("DEV_Z_PROBS>>>", flush=True)
    return {"orig": p, "swap": q}
