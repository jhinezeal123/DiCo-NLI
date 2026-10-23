"""Fixed-budget fitting. Dev labels are never read by this trainer."""
import json
import time
from pathlib import Path
import numpy as np
import torch
import transformers
import dico_trainer as dt
from distill_core import per_row_loss, targets


@torch.no_grad()
def predict_logits(model, tok, features, bs, device, amp):
    model.eval()
    result = []
    for offset in range(0, len(features), bs):
        b = dt.batch(tok, features, range(offset, min(offset + bs, len(features))), device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            z = model(**b).logits
        result.append(z.float().cpu().numpy())
    z = np.concatenate(result)
    assert np.isfinite(z).all(), "nonfinite inference logits"
    return z


def fit(cfg, seed, train_path, dev_inputs_path, output, arm=None, cache=None, kd_cfg=None):
    generator = dt.set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(cfg.get("amp")) and device.type == "cuda"
    ids, t1, t2, y = dt.load_split(train_path)
    _, d1, d2, _ = dt.load_split(dev_inputs_path)  # This file has no label column.
    tok, model = dt.build_model(cfg, 4, device)
    print("EFFECTIVE_CONFIG " + json.dumps(dict(model=cfg, seed=seed, arm=arm,
          labels=dt.LABELS, device=str(device), amp=amp)), flush=True)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"PARAMETERS total={all_params} trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)}", flush=True)
    length, template = cfg["max_length"], cfg.get("pair_template")
    f_train = dt.encode(tok, t1, t2, length, template)
    f_swap = dt.encode(tok, t2, t1, length, template)
    f_dev = dt.encode(tok, d1, d2, length, template)
    f_dev_swap = dt.encode(tok, d2, d1, length, template)
    bs, epochs = int(cfg["batch_size"]), int(cfg["num_epochs"])
    paired = bool(arm and arm["paired"])
    kd = bool(arm and arm["kd"])
    weight = float(kd_cfg["kd_weight"]) if kd else 0.
    temperature = float(kd_cfg["temperature"]) if kd else 1.
    reversible = y != dt.LABEL2ID["NEGATIVE_OTHER"]
    if kd:
        stored = np.load(cache, allow_pickle=False)
        np.testing.assert_array_equal(stored["instance_ids"], np.asarray(ids))
        p, q = targets(stored["train_orig_logits"], stored["train_swap_logits"],
                       temperature, reversible, projected=paired)
        teacher_p, teacher_q = torch.tensor(p, device=device), torch.tensor(q, device=device)
        entropy = -(p * np.log(np.maximum(p, 1e-30))).sum(-1).mean()
        print(f"TEACHER_TARGET_AUDIT split=train rows={len(ids)} entropy_nats={entropy:.6f} projected_reversible_only={paired}", flush=True)
    else:
        teacher_p = teacher_q = None
    y_t = torch.tensor(y, device=device)
    rev_t = torch.tensor(dt.REV, device=device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(parameters, lr=cfg["lr"], weight_decay=cfg["wd"],
                            betas=(0.9, 0.999), eps=1e-6)
    steps_per_epoch = (len(ids) + bs - 1) // bs
    steps = steps_per_epoch * epochs
    sched = transformers.get_linear_schedule_with_warmup(opt, int(steps * cfg["warmup_pcrt"]), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    started = time.perf_counter()
    successful_steps = skipped_steps = 0
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(ids), generator=generator).tolist()
        total_loss = 0.
        for offset in range(0, len(ids), bs):
            idx = permutation[offset:offset + bs]
            b = dt.batch(tok, f_train, idx, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                z = model(**b).logits
            loss_rows = per_row_loss(z, y_t[idx], teacher_p[idx] if kd else None, weight, temperature)
            if paired:
                positions = [j for j, i in enumerate(idx) if reversible[i]]
                if positions:
                    swap_idx = [idx[j] for j in positions]
                    b_swap = dt.batch(tok, f_swap, swap_idx, device)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                        z_swap = model(**b_swap).logits
                    swap_loss = per_row_loss(z_swap, rev_t[y_t[swap_idx]],
                                              teacher_q[swap_idx] if kd else None, weight, temperature)
                    # Normalize per original row, preserving the NEGATIVE_OTHER class weight.
                    loss_rows = loss_rows.index_copy(0, torch.tensor(positions, device=device),
                                                     0.5 * (loss_rows[positions] + swap_loss))
            loss = loss_rows.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss epoch={epoch + 1} offset={offset}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            old_scale = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                sched.step()
                successful_steps += 1
            else:
                skipped_steps += 1
            total_loss += loss.item()
            if (offset // bs + 1) % 100 == 0:
                print(f"TRAIN_STEP epoch={epoch + 1} batch={offset // bs + 1}/{steps_per_epoch} loss={loss.item():.6f}", flush=True)
        print(f"TRAIN_EPOCH epoch={epoch + 1}/{epochs} loss={total_loss / steps_per_epoch:.6f} "
              f"wall_seconds={time.perf_counter() - started:.2f} successful_steps={successful_steps} skipped_steps={skipped_steps}", flush=True)
    training_seconds = time.perf_counter() - started
    model.eval()
    z_dev = predict_logits(model, tok, f_dev, bs, device, amp)
    z_dev_swap = predict_logits(model, tok, f_dev_swap, bs, device, amp)
    output = Path(output)
    payload = dict(dev_orig_logits=z_dev, dev_swap_logits=z_dev_swap)
    if arm is None:
        payload.update(instance_ids=np.asarray(ids),
                       train_orig_logits=predict_logits(model, tok, f_train, bs, device, amp),
                       train_swap_logits=predict_logits(model, tok, f_swap, bs, device, amp))
    else:
        checkpoint = output.parent / "model"
        model.save_pretrained(checkpoint)
        tok.save_pretrained(checkpoint)
    np.savez(output, **payload)
    # Same GPU, batch=8, length cap and autocast; tokenization and model load excluded.
    b = dt.batch(tok, f_dev, range(min(bs, len(f_dev))), device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        for _ in range(5):
            model(**b)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(30):
            model(**b)
        if device.type == "cuda":
            torch.cuda.synchronize()
    info = dict(total_parameters=all_params, training_seconds=training_seconds,
                successful_steps=successful_steps, skipped_steps=skipped_steps,
                latency_ms_per_batch=(time.perf_counter() - start) * 1000 / 30,
                latency_batch_size=len(b["input_ids"]),
                latency_padded_tokens=int(b["input_ids"].shape[1]),
                device=str(device), amp=amp,
                gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")
    output.with_suffix(".json").write_text(json.dumps(info, indent=2))
    print("WORKER_SUMMARY " + json.dumps(info), flush=True)
