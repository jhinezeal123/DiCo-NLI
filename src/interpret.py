"""Post-training interpretability for the decoder seq-classification model.

Everything is computed on the dev rows in the original direction (text1, text2) and printed
as one JSON line per row (INTERP_ROW) plus per-layer summaries (INTERP_LAYER / INTERP_PROBE),
so the analysis can be rebuilt from the run log alone.

Methods
- Integrated Gradients on input embeddings. Baseline keeps the template tokens and replaces
  every text1/text2 token by the pad-token embedding. Two targets: log p(pred) and the
  contrastive margin logit[c1] - logit[c2] (errors: c1=pred, c2=gold; correct: c1=gold,
  c2=runner-up). Completeness (sum of attributions vs f(x) - f(baseline)) is reported.
- Grad-CAM for transformers: per layer l, token relevance relu(sum_d h_l[t,d] * mean_t' g_l[t',d])
  with g = d log p(pred) / d h_l; plus gradient x activation sum_d h*g.
- Occlusion: delete one word of text1 or text2 at a time; change in p(pred) and new label.
- Attention of the decision (last) token per layer, mean over heads (eager attention).
- Logit lens: trained head applied to the final-normed last-token state of every layer.
- Linear probes: logistic regression per layer on last-token states of train rows, evaluated
  on dev; fine-tuned model vs the same model with the LoRA adapter (and head) disabled.
"""
import json
import re

import numpy as np
import torch
import torch.nn.functional as F

IG_STEPS = 24
CAM_TOKEN_LAYERS = (7, 14, 21, 28)  # hidden_states indices kept at token level
PROBE_LAYERS = tuple(range(0, 29, 2))


def _r(x, k=4):
    return [round(float(v), k) for v in x]


def _spans(template, t1, t2):
    pre, rest = template.split("{t1}")
    mid, post = rest.split("{t2}")
    s1 = len(pre)
    s2 = s1 + len(t1) + len(mid)
    return pre + t1 + mid + t2 + post, (s1, s1 + len(t1)), (s2, s2 + len(t2))


def _words(text, offset):
    return [(m.group(), m.start() + offset, m.end() + offset) for m in re.finditer(r"\S+", text)]


def _norm(w):
    return re.sub(r"[^\w]", "", w.lower())


def layout(tok, template, t1, t2, max_length):
    """Token ids plus, per token, role (0 template, 1 text1, 2 text2) and word index in its text."""
    text, sp1, sp2 = _spans(template, t1, t2)
    enc = tok(text, return_offsets_mapping=True, truncation=True, max_length=max_length)
    w1, w2 = _words(t1, sp1[0]), _words(t2, sp2[0])
    roles, widx = [], []
    for s, e in enc["offset_mapping"]:
        role, wi = 0, -1
        for r, (a, b), ws in ((1, sp1, w1), (2, sp2, w2)):
            if e > a and s < b:
                role = r
                for k, (_, ws_, we_) in enumerate(ws):
                    if e > ws_ and s < we_:
                        wi = k
                        break
                break
        roles.append(role)
        widx.append(wi)
    n1, n2 = {_norm(w) for w, _, _ in w1}, {_norm(w) for w, _, _ in w2}
    return {
        "ids": enc["input_ids"], "roles": roles, "widx": widx,
        "words1": [w for w, _, _ in w1], "words2": [w for w, _, _ in w2],
        "shared1": [_norm(w) in n2 for w, _, _ in w1], "shared2": [_norm(w) in n1 for w, _, _ in w2],
        "tokens": tok.convert_ids_to_tokens(enc["input_ids"]),
    }


def _logits(model, amp, **kw):
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        out = model(**kw)
    return out


def integrated_gradients(model, emb_layer, ids, roles, pad_id, target_fn, amp, device):
    x = emb_layer(torch.tensor([ids], device=device)).detach().float()
    base = x.clone()
    content = torch.tensor([r > 0 for r in roles], device=device)
    base[0, content] = emb_layer(torch.tensor([pad_id], device=device)).detach().float()[0]
    alphas = (torch.arange(IG_STEPS, device=device, dtype=torch.float32) + 0.5) / IG_STEPS
    path = (base + alphas[:, None, None] * (x - base)).requires_grad_(True)
    mask = torch.ones(path.shape[:2], dtype=torch.long, device=device)
    logits = _logits(model, amp, inputs_embeds=path, attention_mask=mask).logits.float()
    target_fn(logits).sum().backward()
    attr = ((x - base)[0] * path.grad.mean(0)).sum(-1)
    with torch.no_grad():
        ends = _logits(model, amp, inputs_embeds=torch.cat([x, base]),
                       attention_mask=torch.ones((2, x.shape[1]), dtype=torch.long, device=device)).logits.float()
        fx, fb = target_fn(ends).tolist()
    return attr.detach().cpu().numpy(), fx - fb


def gradcam(model, emb_layer, ids, target, amp, device):
    # Parameters are frozen, so start the graph at the input embeddings.
    x = emb_layer(torch.tensor([ids], device=device)).detach().float().requires_grad_(True)
    out = _logits(model, amp, inputs_embeds=x, attention_mask=torch.ones(x.shape[:2], dtype=torch.long, device=device),
                  output_hidden_states=True)
    hs = out.hidden_states
    for h in hs:
        h.retain_grad()
    F.log_softmax(out.logits.float(), -1)[0, target].backward()
    cam, gxa = [], []
    for h in hs:
        a, g = h[0].float().detach(), h.grad[0].float()
        cam.append(torch.relu((a * g.mean(0)).sum(-1)).cpu().numpy())
        gxa.append((a * g).sum(-1).cpu().numpy())
    return np.stack(cam), np.stack(gxa)


def _role_mass(v, roles, shared_tok):
    v = np.abs(np.asarray(v, dtype=np.float64))
    tot = v.sum() + 1e-12
    roles = np.asarray(roles)
    content = v[roles > 0].sum() + 1e-12
    return [v[roles == 1].sum() / tot, v[roles == 2].sum() / tot, v[roles == 0].sum() / tot,
            v[(roles > 0) & shared_tok].sum() / content]


@torch.no_grad()
def last_token_states(model, tok, feats, device, amp, bs=32):
    """[N, n_layers+1, D] float16 last-token hidden states (index 0 = embeddings)."""
    out = []
    for s in range(0, len(feats), bs):
        b = tok.pad(feats[s:s + bs], return_tensors="pt").to(device)
        hs = _logits(model, amp, **b, output_hidden_states=True).hidden_states
        if tok.padding_side == "left":
            idx = torch.full((b["input_ids"].shape[0],), -1, device=device)
        else:
            idx = b["attention_mask"].sum(1) - 1
        ar = torch.arange(len(idx), device=device)
        out.append(torch.stack([h[ar, idx] for h in hs], 1).half().cpu().numpy())
    return np.concatenate(out)


def run_probes(seed, Htr, ytr, Hdv, ydv, err_mask, tag):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler
    for l in PROBE_LAYERS:
        sc = StandardScaler().fit(Htr[:, l].astype(np.float32))
        clf = LogisticRegression(max_iter=3000, C=0.5)
        clf.fit(sc.transform(Htr[:, l].astype(np.float32)), ytr)
        p = clf.predict(sc.transform(Hdv[:, l].astype(np.float32)))
        print("INTERP_PROBE " + json.dumps({
            "seed": seed, "model": tag, "layer": l,
            "dev_acc": round(float((p == ydv).mean()), 4),
            "dev_wf1": round(float(f1_score(ydv, p, average="weighted")), 4),
            "acc_on_head_errors": round(float((p[err_mask] == ydv[err_mask]).mean()), 4),
            "acc_on_head_correct": round(float((p[~err_mask] == ydv[~err_mask]).mean()), 4),
            "train_acc": round(float(clf.score(sc.transform(Htr[:, l].astype(np.float32)), ytr)), 4),
        }), flush=True)


def interpret(cfg, seed, model, tok, train, dev, device, amp, labels):
    """train/dev: (ids, t1, t2, y). Prints INTERP_* lines."""
    import time
    t0 = time.time()
    global IG_STEPS
    tpl, L = cfg["pair_template"], cfg["max_length"]
    # fp16 autocast overflowed eager attention (NaN) and made IG paths noisy; analysis can run in fp32.
    amp = bool(cfg.get("interpret_amp", amp))
    IG_STEPS = int(cfg.get("ig_steps", IG_STEPS))
    print(f"INTERP_CONFIG amp={amp} ig_steps={IG_STEPS}", flush=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    base_model = model.get_base_model()
    emb_layer = model.get_input_embeddings()
    head, final_norm = base_model.score, base_model.model.norm
    _, d1, d2, ydv = dev
    lays = [layout(tok, tpl, a, b, L) for a, b in zip(d1, d2)]
    pad_id = tok.pad_token_id

    # Standard predictions for the original direction (same as the pipeline's p).
    with torch.no_grad():
        probs = []
        for ly in lays:
            lo = _logits(model, amp, input_ids=torch.tensor([ly["ids"]], device=device)).logits.float()
            probs.append(torch.softmax(lo, -1)[0].cpu().numpy())
    probs = np.array(probs)
    pred = probs.argmax(1)
    err = pred != ydv
    print(f"INTERP_START seed={seed} rows={len(lays)} errors={int(err.sum())} "
          f"acc={float((~err).mean()):.4f}", flush=True)

    # Attention needs the eager implementation; switch only for that pass.
    cfg_obj = base_model.config
    attn_ok = True
    comp = []

    for i, ly in enumerate(lays):
        ids, roles = ly["ids"], ly["roles"]
        g, p_ = int(ydv[i]), int(pred[i])
        order = np.argsort(-probs[i])
        c1, c2 = (p_, g) if p_ != g else (g, int(order[1]))
        shared_tok = np.array([(r == 1 and w >= 0 and ly["shared1"][w]) or (r == 2 and w >= 0 and ly["shared2"][w])
                               for r, w in zip(roles, ly["widx"])])
        ig_p, d_p = integrated_gradients(model, emb_layer, ids, roles, pad_id,
                                         lambda lo: F.log_softmax(lo, -1)[:, p_], amp, device)
        ig_c, d_c = integrated_gradients(model, emb_layer, ids, roles, pad_id,
                                         lambda lo: lo[:, c1] - lo[:, c2], amp, device)
        comp.append((float(ig_p.sum()), d_p, float(ig_c.sum()), d_c))
        cam, gxa = gradcam(model, emb_layer, ids, p_, amp, device)

        # Logit lens with the trained head on the last-token state of every layer.
        with torch.no_grad():
            out = _logits(model, amp, input_ids=torch.tensor([ids], device=device), output_hidden_states=True)
            hs = out.hidden_states
            lens = []
            for l, h in enumerate(hs):
                v = h[0, -1].float()
                v = v if l == len(hs) - 1 else final_norm(v).float()
                lo = head(v[None]).float()
                lens.append(torch.softmax(lo, -1)[0].cpu().numpy())
            lens = np.array(lens)

        # Occlusion: delete one word at a time.
        variants, keys = [], []
        for side, words in ((1, ly["words1"]), (2, ly["words2"])):
            for k in range(len(words)):
                kept = " ".join(w for j, w in enumerate(words) if j != k)
                a, b = (kept, d2[i]) if side == 1 else (d1[i], kept)
                variants.append(tok(tpl.format(t1=a, t2=b), truncation=True, max_length=L))
                keys.append((side, k))
        occl = []
        if variants:
            with torch.no_grad():
                b = tok.pad([{"input_ids": v["input_ids"], "attention_mask": v["attention_mask"]} for v in variants],
                            return_tensors="pt").to(device)
                pv = torch.softmax(_logits(model, amp, **b).logits.float(), -1).cpu().numpy()
            occl = [[s, k, round(float(probs[i, p_] - pv[j, p_]), 4), int(pv[j].argmax()), round(float(pv[j, g]), 4)]
                    for j, (s, k) in enumerate(keys)]

        attn = None
        if attn_ok:
            try:
                prev = cfg_obj._attn_implementation
                cfg_obj._attn_implementation = "eager"
                with torch.no_grad():
                    att = _logits(model, amp, input_ids=torch.tensor([ids], device=device),
                                  output_attentions=True).attentions
                cfg_obj._attn_implementation = prev
                attn = [_role_mass(a[0, :, -1, :].float().mean(0).cpu().numpy(), roles, shared_tok) for a in att]
                attn = [_r(x) for x in attn]
            except Exception as exc:  # attention is optional; never kill the run for it
                attn_ok = False
                cfg_obj._attn_implementation = prev
                print(f"INTERP_WARN attention disabled: {exc.__class__.__name__}: {exc}", flush=True)

        row = {
            "seed": seed, "i": i, "id": dev[0][i], "gold": labels[g], "pred": labels[p_],
            "probs": _r(probs[i]), "c1": labels[c1], "c2": labels[c2],
            "tokens": ly["tokens"], "roles": roles, "widx": ly["widx"],
            "words1": ly["words1"], "words2": ly["words2"], "shared1": ly["shared1"], "shared2": ly["shared2"],
            "ig_pred": _r(ig_p, 5), "ig_pred_delta": round(d_p, 5),
            "ig_con": _r(ig_c, 5), "ig_con_delta": round(d_c, 5),
            "cam": {str(l): _r(cam[l], 5) for l in CAM_TOKEN_LAYERS},
            "cam_roles": [_r(_role_mass(cam[l], roles, shared_tok)) for l in range(cam.shape[0])],
            "gxa_roles": [_r(_role_mass(gxa[l], roles, shared_tok)) for l in range(gxa.shape[0])],
            "lens_pred": [int(x) for x in lens.argmax(1)], "lens_pgold": _r(lens[:, g]),
            "occl": occl, "attn_roles": attn,
        }
        print("INTERP_ROW " + json.dumps(row), flush=True)
        if i % 100 == 0:
            print(f"INTERP_PROGRESS seed={seed} row={i} elapsed={time.time() - t0:.0f}s", flush=True)

    c = np.array(comp)
    print(f"INTERP_IG_COMPLETENESS seed={seed} corr_pred={np.corrcoef(c[:, 0], c[:, 1])[0, 1]:.4f} "
          f"corr_con={np.corrcoef(c[:, 2], c[:, 3])[0, 1]:.4f} "
          f"median_abs_err_pred={np.median(np.abs(c[:, 0] - c[:, 1])):.4f} "
          f"median_abs_err_con={np.median(np.abs(c[:, 2] - c[:, 3])):.4f}", flush=True)

    # Linear probes per layer: fine-tuned vs adapter-disabled base.
    from dico_trainer import encode
    f_tr = encode(tok, train[1], train[2], L, tpl)
    f_dv = encode(tok, d1, d2, L, tpl)
    Htr, Hdv = last_token_states(model, tok, f_tr, device, amp), last_token_states(model, tok, f_dv, device, amp)
    run_probes(seed, Htr, train[3], Hdv, ydv, err, "finetuned")
    del Htr, Hdv
    with model.disable_adapter():
        Btr, Bdv = last_token_states(model, tok, f_tr, device, amp), last_token_states(model, tok, f_dv, device, amp)
    run_probes(seed, Btr, train[3], Bdv, ydv, err, "base")
    print(f"INTERP_DONE seed={seed} wall={time.time() - t0:.0f}s", flush=True)
