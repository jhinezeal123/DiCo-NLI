"""Paired hidden-state training for a matched readout architecture pilot."""
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from sklearn.metrics import f1_score

import dico_trainer as dt
from joint_readout import Readout, REV


def last_state(backbone, inputs):
    states = backbone(**inputs, use_cache=False, return_dict=True).last_hidden_state
    positions = (inputs["attention_mask"].long() * torch.arange(states.shape[1], device=states.device)).argmax(-1)
    return states[torch.arange(states.shape[0], device=states.device), positions]


def run(cfg, seed, train_path, dev_path, device):
    generator = dt.set_seed(seed)
    _, tr1, tr2, ytr = dt.load_split(train_path)
    _, dv1, dv2, ydv = dt.load_split(dev_path)
    tok, model = dt.build_model(cfg, 4, device)
    base = model.get_base_model()
    assert base.config.model_type == "qwen2", base.config.model_type
    # Keep LoRA on the transformer; replace and freeze its unused original score module.
    for p in base.score.parameters():
        p.requires_grad_(False)
    backbone = base.model
    with torch.random.fork_rng(devices=[device.index or 0]):
        torch.manual_seed(seed + 2000003)
        head = Readout(base.config.hidden_size, cfg["joint_readout_arm"]).to(device)
    trainable_backbone = [p for p in model.parameters() if p.requires_grad]
    params = trainable_backbone + list(head.parameters())
    print("JOINT_HEAD_AUDIT", json.dumps({"arm": cfg["joint_readout_arm"], "seed": seed,
          "hidden_size": base.config.hidden_size, "head_parameters": sum(p.numel() for p in head.parameters()),
          "trainable_backbone_parameters": sum(p.numel() for p in trainable_backbone),
          "microbatch": 4, "pair_batch": cfg["batch_size"], "initialization_rng_isolated": True}), flush=True)
    L, tpl = cfg["max_length"], cfg.get("pair_template")
    train_a, train_b = dt.encode(tok,tr1,tr2,L,tpl), dt.encode(tok,tr2,tr1,L,tpl)
    dev_a, dev_b = dt.encode(tok,dv1,dv2,L,tpl), dt.encode(tok,dv2,dv1,L,tpl)
    bs, epochs = int(cfg["batch_size"]), int(cfg["num_epochs"])
    steps = (len(train_a)+bs-1)//bs
    total = steps*epochs
    assert total == 1143
    amp = bool(cfg["amp"]) and device.type == "cuda"
    opt = torch.optim.AdamW(params,lr=float(cfg["lr"]),weight_decay=float(cfg["wd"]),betas=(.9,.999),eps=1e-6)
    sched = transformers.get_linear_schedule_with_warmup(opt,int(total*float(cfg["warmup_pcrt"])),total)
    scaler = torch.amp.GradScaler("cuda",enabled=amp)
    labels = torch.tensor(ytr,device=device)
    torch.cuda.reset_peak_memory_stats(device)

    @torch.no_grad()
    def evaluate():
        model.eval();head.eval()
        orig,swap=[],[]
        for start in range(0,len(dev_a),64):
            idx=range(start,min(start+64,len(dev_a)))
            with torch.autocast("cuda",dtype=torch.float16,enabled=amp):
                a=last_state(backbone,dt.batch(tok,dev_a,idx,device))
                b=last_state(backbone,dt.batch(tok,dev_b,idx,device))
                z,w=head(a,b).float(),head(b,a).float()
            torch.testing.assert_close(z,w[:,REV],atol=1e-5,rtol=1e-5)
            orig.append(z.softmax(-1).cpu().numpy());swap.append(w.softmax(-1).cpu().numpy())
        return np.concatenate(orig).astype(np.float64),np.concatenate(swap).astype(np.float64)

    t0=time.time()
    for epoch in range(epochs):
        model.train();head.train()
        perm=torch.randperm(len(train_a),generator=generator).tolist()
        losses=0.
        for start in range(0,len(perm),bs):
            outer=perm[start:start+bs]
            opt.zero_grad(set_to_none=True)
            step_loss=0.
            for micro in range(0,len(outer),4):
                idx=outer[micro:micro+4]
                with torch.autocast("cuda",dtype=torch.float16,enabled=amp):
                    a=last_state(backbone,dt.batch(tok,train_a,idx,device))
                    b=last_state(backbone,dt.batch(tok,train_b,idx,device))
                    logits=head(a,b)
                loss=F.cross_entropy(logits.float(),labels[idx],label_smoothing=float(cfg["label_smoothing"]))
                fraction=len(idx)/len(outer)
                scaler.scale(loss*fraction).backward()
                step_loss+=loss.item()*fraction
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params,1.)
            scaler.step(opt);scaler.update();sched.step()
            losses+=step_loss
            if (start//bs+1)%100==0:
                print(f"JOINT_PROGRESS epoch={epoch+1} step={start//bs+1}/{steps} mean_loss={losses/(start//bs+1):.5f}",flush=True)
        p,q=evaluate()
        print(f"JOINT_EPOCH epoch={epoch+1} loss={losses/steps:.6f} dev_f1={f1_score(ydv,p.argmax(1),average='weighted'):.8f}",flush=True)
    print("JOINT_RUNTIME",json.dumps({"seconds":time.time()-t0,"optimizer_steps":total,
          "peak_memory_gb":torch.cuda.max_memory_allocated(device)/1e9}),flush=True)
    return {"orig":p,"swap":q}
