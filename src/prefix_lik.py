"""Training-free prefix-weighted continuation likelihood (speculative-decoding-inspired).

For a direction a -> b we build "{a}<connective> {b}" and ask a base causal LM how
likely each word-level prefix of b is: "a, so w1", "a, so w1 w2", ... One
teacher-forced forward pass over the full string yields exactly the per-prefix
numbers a batch of separate prefix prompts would give (causal masking), so we
score the whole batch in one pass.

Per-prefix score s_k (k = 1..n words of b):
  next : log P(w_k | a, conn, w_<k)                      (conditional next-word score)
  cum  : mean_{j<=k} log P(w_j | a, conn, w_<j)          (normalised prefix likelihood)
Optional PMI: subtract the same quantity under a null context (connective without a),
which removes the frequency/length prior of b itself.
Aggregate S(a->b) = sum_k w_k s_k / sum_k w_k with weights that grow toward the end
of b (later prefixes are closer to the complete hypothesis, so they are more
constrained than early ones, which admit many continuations).

Decision rules (thresholds fit on TRAIN only):
  twodir: e_f = S_f > th, e_b = S_b > th -> EQ / FE / BE / NEG
  diff  : max(S_f, S_b) <= th -> NEG; |S_f - S_b| <= delta -> EQ; else sign -> FE / BE
The variant (template x score x pmi x weighting x rule) is selected on train
weighted F1. A logistic regression on the same features is reported as a labelled
departure from the threshold rule.
"""
import itertools
import json
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from dico_trainer import LABEL2ID, LABELS, REV, load_split

BE, EQ, FE, NEG = (LABEL2ID[l] for l in ("BACKWARD_ENTAILMENT", "EQUIVALENCE",
                                           "FORWARD_ENTAILMENT", "NEGATIVE_OTHER"))

# name -> (conditional context, null context); target b is appended after a space.
TEMPLATES = {
    "so": ("{a}, so", "So,"),
    "which_means": ("{a}, which means", "Which means"),
    "in_other_words": ("{a}. In other words,", "In other words,"),
    "that_is": ("{a}, that is,", "That is,"),
}
WEIGHTS = {
    "uniform": lambda k: np.ones_like(k),
    "linear": lambda k: k,
    "quadratic": lambda k: k ** 2,
    "exp2": lambda k: 2.0 ** k,
    "last": lambda k: (k == k.max()).astype(float),
}


def lower_first(s):
    # "A woman" -> "a woman"; leave acronyms / all-caps tokens ("UN", "US") alone.
    w = s.split(" ")[0]
    if len(w) > 1 and w[0].isupper() and w[1:].islower() or len(w) == 1 and w in "A":
        return s[0].lower() + s[1:]
    return s


def build_model(cfg, device):
    tok = AutoTokenizer.from_pretrained(cfg["model_id"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"  # keep absolute positions identical to unpadded scoring
    kw = {}
    if cfg.get("load_in_4bit"):
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        kw["device_map"] = {"": 0}
    else:
        kw["torch_dtype"] = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], **kw)
    if not cfg.get("load_in_4bit"):
        model = model.to(device)
    return tok, model.eval()


@torch.no_grad()
def word_logprobs(tok, model, contexts, targets, device, bs=64):
    """For each (context, target) return per-word log P of target's words given context."""
    texts = [c + " " + t for c, t in zip(contexts, targets)]
    out = [None] * len(texts)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tok([texts[i] for i in idx], return_offsets_mapping=True, padding=True,
                  return_tensors="pt")
        offsets = enc.pop("offset_mapping")
        enc = enc.to(device)
        logits = model(**enc).logits[:, :-1].float()
        # log p(next token) without materialising a second vocab-sized tensor
        lp = logits.gather(-1, enc["input_ids"][:, 1:, None])[..., 0] - torch.logsumexp(logits, -1)
        lp = lp.cpu().numpy()
        del logits
        mask = enc["attention_mask"].cpu().numpy()
        for row, i in enumerate(idx):
            start = len(contexts[i]) + 1  # first char of the target
            words, pos = targets[i].split(" "), start
            spans = []
            for w in words:
                spans.append((pos, pos + len(w)))
                pos += len(w) + 1
            wl = np.zeros(len(words))
            for t in range(1, offsets.shape[1]):
                if not mask[row, t]:
                    continue
                a, b = offsets[row, t].tolist()
                if b <= a:
                    continue
                last = b - 1  # a token belongs to the word holding its last character
                for k, (ws, we) in enumerate(spans):
                    if ws <= last < we:
                        wl[k] += lp[row, t - 1]
                        break
            out[i] = wl
    return out


def score_all(tok, model, t1, t2, device):
    """Per-word conditional and null logprobs for both directions and every template."""
    feats = {}
    for name, (ctx, null) in TEMPLATES.items():
        for d, (A, B) in (("f", (t1, t2)), ("b", (t2, t1))):
            tg = [lower_first(b) for b in B]
            feats[(name, d, "cond")] = word_logprobs(tok, model, [ctx.format(a=a) for a in A], tg, device)
            feats[(name, d, "null")] = word_logprobs(tok, model, [null] * len(A), tg, device)
    return feats


def aggregate(cond, null, score, pmi, weight):
    vals = []
    for c, n in zip(cond, null):
        x = c - n if pmi else c
        if score == "cum":
            x = np.cumsum(x) / np.arange(1, len(x) + 1)
        k = np.arange(1, len(x) + 1, dtype=float)
        w = WEIGHTS[weight](k)
        vals.append(float((w * x).sum() / w.sum()))
    return np.array(vals)


def wf1(y, p, k=4):
    """Support-weighted F1 (same definition as sklearn average='weighted'), fast for grid search."""
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp, sup, pc = np.diag(cm), cm.sum(1), cm.sum(0)
    f = np.where(sup + pc > 0, 2 * tp / np.maximum(sup + pc, 1), 0.0)
    return float((f * sup).sum() / sup.sum())


def rule_twodir(sf, sb, th):
    ef, eb = sf > th, sb > th
    return np.where(ef & eb, EQ, np.where(ef, FE, np.where(eb, BE, NEG)))


def rule_diff(sf, sb, th, delta):
    d = sf - sb
    lab = np.where(np.abs(d) <= delta, EQ, np.where(d > 0, FE, BE))
    return np.where(np.maximum(sf, sb) <= th, NEG, lab)


def fit_rule(rule, sf, sb, y):
    """Grid-search thresholds on train weighted F1."""
    allv = np.concatenate([sf, sb])
    ths = np.quantile(allv, np.linspace(0.02, 0.98, 49))
    best = (-1, None)
    if rule == "twodir":
        for th in ths:
            f = wf1(y, rule_twodir(sf, sb, th))
            best = max(best, (f, (th,)), key=lambda z: z[0])
    else:
        ad = np.abs(sf - sb)
        deltas = np.quantile(ad, np.linspace(0.02, 0.98, 33))
        for th, de in itertools.product(ths, deltas):
            f = wf1(y, rule_diff(sf, sb, th, de))
            best = max(best, (f, (th, de)), key=lambda z: z[0])
    return best


def apply_rule(rule, sf, sb, params):
    return rule_twodir(sf, sb, *params) if rule == "twodir" else rule_diff(sf, sb, *params)


def local_cons(pred, ids, ref):
    """Local (non-official) soft/hard consistency over gold reversible pairs."""
    if ref is None:
        return float("nan"), float("nan")
    pos = {i: k for k, i in enumerate(ids)}
    soft = hard = n = 0
    seen = set()
    for i, (rev, gold) in ref.items():
        if not rev or rev not in pos or (rev, i) in seen or gold == NEG:
            continue
        seen.add((i, rev))
        a, b = pred[pos[i]], pred[pos[rev]]
        n += 1
        soft += a != NEG and REV[a] == b
        hard += a == gold and b == ref[rev][1]
    return soft / max(n, 1), hard / max(n, 1)


def load_ref(dev_path):
    import csv
    p = str(dev_path).replace("participant_labeled", "reference")
    try:
        with open(p, newline="", encoding="utf-8") as f:
            return {r["instance_id"]: (r.get("reverse_pair_id") or "", LABEL2ID[r["label"]])
                    for r in csv.DictReader(f)}
    except (OSError, KeyError):
        return None


def recalls(y, p):
    return " ".join(f"{LABELS[c][:5]}={float((p[y == c] == c).mean()):.3f}" for c in range(len(LABELS)))


def run(cfg, seed, train_path, dev_path, device):
    t0 = time.time()
    _, tr1, tr2, ytr = load_split(train_path)
    dv_ids, dv1, dv2, ydv = load_split(dev_path)
    ref = load_ref(dev_path)
    tok, model = build_model(cfg, device)
    print(f"prefix_lik model={cfg['model_id']} 4bit={bool(cfg.get('load_in_4bit'))} "
          f"templates={list(TEMPLATES)}", flush=True)
    ftr = score_all(tok, model, tr1, tr2, device)
    fdv = score_all(tok, model, dv1, dv2, device)
    print(f"scoring_seconds={time.time() - t0:.1f}", flush=True)

    nw = np.array([len(x) for x in fdv[("so", "f", "cond")]] + [len(x) for x in fdv[("so", "b", "cond")]])
    print(f"dev target words per direction: mean={nw.mean():.2f} share_single_word={(nw <= 1).mean():.3f} "
          f"share_<=2={(nw <= 2).mean():.3f}  (single-word targets make all weightings identical)")

    # Raw word-level logprobs for offline re-analysis without GPU.
    print("<<<WORD_LOGPROBS_DEV")
    for i in range(len(dv_ids)):
        rec = {"id": dv_ids[i]}
        for (name, d, kind), v in fdv.items():
            rec[f"{name}|{d}|{kind}"] = [round(float(z), 3) for z in v[i]]
        print(json.dumps(rec, separators=(",", ":")))
    print("WORD_LOGPROBS_DEV>>>")
    print("<<<WORD_LOGPROBS_TRAIN")
    for i in range(len(tr1)):
        rec = {"i": i, "y": int(ytr[i])}
        for (name, d, kind), v in ftr.items():
            rec[f"{name}|{d}|{kind}"] = [round(float(z), 3) for z in v[i]]
        print(json.dumps(rec, separators=(",", ":")))
    print("WORD_LOGPROBS_TRAIN>>>")

    rows, feat_tr, feat_dv = [], [], []
    for name, score, pmi, weight in itertools.product(TEMPLATES, ("next", "cum"), (False, True), WEIGHTS):
        S = {}
        for split, F in (("tr", ftr), ("dv", fdv)):
            for d in ("f", "b"):
                S[(split, d)] = aggregate(F[(name, d, "cond")], F[(name, d, "null")], score, pmi, weight)
        feat_tr += [S[("tr", "f")], S[("tr", "b")]]
        feat_dv += [S[("dv", "f")], S[("dv", "b")]]
        for rule in ("twodir", "diff"):
            f_tr, params = fit_rule(rule, S[("tr", "f")], S[("tr", "b")], ytr)
            pdv = apply_rule(rule, S[("dv", "f")], S[("dv", "b")], params)
            pdv_swap = apply_rule(rule, S[("dv", "b")], S[("dv", "f")], params)
            soft, hard = local_cons(pdv, dv_ids, ref)
            rows.append(dict(template=name, score=score, pmi=pmi, weight=weight, rule=rule,
                             params=[round(float(p), 4) for p in params], train_wf1=f_tr,
                             dev_wf1=f1_score(ydv, pdv, average="weighted"), dev_soft=soft, dev_hard=hard,
                             pred=pdv, pred_swap=pdv_swap))

    section = lambda t: print(f"\n----- {t} -----", flush=True)
    section("ABLATION (all variants; thresholds fit on train; dev columns are LOCAL diagnostics, "
            "not used for selection)")
    print(f"{'template':15s} {'score':5s} {'pmi':5s} {'weight':9s} {'rule':6s} train_wF1 dev_wF1 "
          f"dev_soft dev_hard params")
    for r in sorted(rows, key=lambda r: -r["train_wf1"]):
        print(f"{r['template']:15s} {r['score']:5s} {str(r['pmi']):5s} {r['weight']:9s} {r['rule']:6s} "
              f"{r['train_wf1']:.4f}    {r['dev_wf1']:.4f}  {r['dev_soft']:.4f}   {r['dev_hard']:.4f}   "
              f"{r['params']}")

    section("WEIGHTING EFFECT (mean over templates/score/pmi/rule)")
    for w in WEIGHTS:
        sel = [r for r in rows if r["weight"] == w]
        print(f"weight={w:9s} train_wF1={np.mean([r['train_wf1'] for r in sel]):.4f} "
              f"dev_wF1={np.mean([r['dev_wf1'] for r in sel]):.4f}")
    for key in ("pmi", "score", "rule", "template"):
        for v in sorted({r[key] for r in rows}, key=str):
            sel = [r for r in rows if r[key] == v]
            print(f"{key}={str(v):15s} train_wF1={np.mean([r['train_wf1'] for r in sel]):.4f} "
                  f"dev_wF1={np.mean([r['dev_wf1'] for r in sel]):.4f} "
                  f"best_train={max(r['train_wf1'] for r in sel):.4f}")

    best = max(rows, key=lambda r: r["train_wf1"])
    section("SELECTED VARIANT (by train weighted F1)")
    print(json.dumps({k: v for k, v in best.items() if k not in ("pred", "pred_swap")}, default=str))
    print(f"dev per-class recall: {recalls(ydv, best['pred'])}")
    print(f"dev pred dist: {np.bincount(best['pred'], minlength=4).tolist()} (order {LABELS})")

    # Labelled departure: 4-way logistic regression over every score as features.
    Xtr, Xdv = np.stack(feat_tr, 1), np.stack(feat_dv, 1)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    lr = LogisticRegression(max_iter=2000, C=1.0, class_weight=None).fit((Xtr - mu) / sd, ytr)
    p_lr = lr.predict((Xdv - mu) / sd)
    # Swapped-input features: exchange each (f, b) column pair.
    perm = np.arange(Xdv.shape[1]).reshape(-1, 2)[:, ::-1].reshape(-1)
    pr_lr = lr.predict_proba((Xdv - mu) / sd)
    pr_lr_swap = lr.predict_proba((Xdv[:, perm] - mu) / sd)
    bid = (0.5 * (pr_lr + pr_lr_swap[:, REV])).argmax(1)
    section("DEPARTURE: logistic regression on all prefix scores (LOCAL metrics)")
    for tag, p in (("standard", p_lr), ("bidir", bid)):
        soft, hard = local_cons(p, dv_ids, ref)
        print(f"LR_{tag}: dev_wF1={f1_score(ydv, p, average='weighted'):.4f} soft={soft:.4f} hard={hard:.4f} "
              f"recall: {recalls(ydv, p)}")
    print(f"prefix_lik total_seconds={time.time() - t0:.1f}", flush=True)

    eye = np.eye(len(LABELS))
    return {"orig": eye[best["pred"]], "swap": eye[best["pred_swap"]]}
