"""S1: cross-fitted parent probabilities on train (pair-grouped 5-fold) + teacher views, for an offline stacker.

Each fold model is trained on 4/5 of the train pairs and predicts (a) its held-out train fold and (b) the full dev
split. Out-of-fold train probabilities let a combiner be fit on train only; dev labels are never used for fitting.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

import dico_trainer as dt
import lcf_trainer as lcf

REV = [2, 1, 0, 3]  # BE<->FE in LABELS order (BE, EQ, FE, NO)


def bidir(p, q):
    return 0.5 * (p + q[:, REV])


def run(cfg, upstream, emit, config_path, out_root):
    data = upstream / "final_data"
    tr_path = data / "train" / "dico_nli_train_track1_participant_labeled.csv"
    dv_path = data / "dev" / "dico_nli_dev_track1_participant_labeled.csv"
    tr, dv = pd.read_csv(tr_path), pd.read_csv(dv_path)
    groups = tr["instance_id"].str.split("__").str[0].values
    folds = list(GroupKFold(n_splits=int(cfg["crossfit"]["folds"])).split(tr, groups=groups))
    worker = Path(__file__).resolve().parents[1] / "scripts" / "train_worker.py"
    n_gpu = max(1, torch.cuda.device_count())
    pending, running, free = list(range(len(folds))), {}, list(range(n_gpu))
    seed = int(cfg["crossfit"]["seed"])
    while pending or running:
        while pending and free:
            k, gpu = pending.pop(0), free.pop(0)
            fd = out_root / f"fold{k}"
            fd.mkdir(parents=True)
            a, b = folds[k]
            tr.iloc[a].to_csv(fd / "train.csv", index=False)
            pd.concat([tr.iloc[b], dv]).to_csv(fd / "eval.csv", index=False)
            log = open(fd / "train.log", "w")
            proc = subprocess.Popen([sys.executable, "-u", str(worker), "--config", str(config_path), "--seed", str(seed),
                                     "--train", str(fd / "train.csv"), "--dev", str(fd / "eval.csv"),
                                     "--out", str(fd / "probs.npz")], stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
            running[k] = (proc, gpu, log, time.time())
            emit(f"launched fold={k} gpu={gpu} train_rows={len(a)} heldout_rows={len(b)}")
        time.sleep(5)
        for k in list(running):
            proc, gpu, log, t0 = running[k]
            if proc.poll() is None:
                continue
            log.close()
            del running[k]
            free.append(gpu)
            emit(f"fold={k} finished rc={proc.returncode} wall={time.time() - t0:.0f}s")
            if proc.returncode != 0:
                emit((out_root / f"fold{k}" / "train.log").read_text(errors="replace")[-3000:])
                sys.exit(f"fold {k} failed")

    lab = {l: i for i, l in enumerate(dt.LABELS)}
    oof = np.zeros((len(tr), 4))
    dev_folds = []
    for k, (a, b) in enumerate(folds):
        z = np.load(out_root / f"fold{k}" / "probs.npz")
        P = bidir(z["orig"], z["swap"])
        oof[b] = P[:len(b)]
        dev_folds.append(P[len(b):])
        yb = tr["label"].map(lab).values[b]
        emit(f"METRIC crossfit/fold{k}/heldout_acc={np.mean(P[:len(b)].argmax(1) == yb):.4f}")
    ytr = tr["label"].map(lab).values
    ydv = dv["label"].map(lab).values
    pdv = np.mean(dev_folds, 0)
    emit(f"METRIC crossfit/oof_train_acc={np.mean(oof.argmax(1) == ytr):.4f} crossfit/dev_fold_ensemble_acc="
         f"{np.mean(pdv.argmax(1) == ydv):.4f}")
    emit("<<<TABLE oof_train_probs")
    emit("instance_id,fold," + ",".join(dt.LABELS))
    fold_of = np.zeros(len(tr), dtype=int)
    for k, (_, b) in enumerate(folds):
        fold_of[b] = k
    for i, iid in enumerate(tr["instance_id"]):
        emit(f"{iid},{fold_of[i]}," + ",".join(f"{v:.5f}" for v in oof[i]))
    emit("TABLE>>>")
    emit("<<<TABLE dev_fold_probs")
    emit("instance_id,fold," + ",".join(dt.LABELS))
    for k, P in enumerate(dev_folds):
        for i, iid in enumerate(dv["instance_id"]):
            emit(f"{iid},{k}," + ",".join(f"{v:.5f}" for v in P[i]))
    emit("TABLE>>>")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, tr1, tr2, _ = dt.load_split(tr_path)
    _, dv1, dv2, _ = dt.load_split(dv_path)
    for teacher in cfg["probe"]["teachers"]:
        name, _, rev = teacher.partition("@")
        tc = {"lcf": {"teacher": name, "teacher_revision": rev or None}}
        tag = name.split("/")[-1]
        for split, ids, a, b in (("train", tr["instance_id"], tr1, tr2), ("dev", dv["instance_id"], dv1, dv2)):
            q = lcf.teacher_q(tc, a, b, device)
            emit(f"<<<TABLE teacher_q_{split}_{tag}")
            emit("instance_id," + ",".join(f"q_{z}" for z in lcf.Z))
            for iid, row in zip(ids, q):
                emit(f"{iid}," + ",".join(f"{v:.5f}" for v in row))
            emit("TABLE>>>")
