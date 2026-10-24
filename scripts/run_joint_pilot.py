"""Run the two matched readout arms, then score them without tuning."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run(cfg,upstream):
    import numpy as np
    import run_pipeline as rp
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
    from joint_readout import check_readouts,REV
    print("JOINT_CHECKS",json.dumps(check_readouts()),flush=True)
    rp.section("CONFIG");print(json.dumps(cfg,indent=1),flush=True)
    paths={"train":upstream/"final_data/train/dico_nli_train_track1_participant_labeled.csv",
           "dev":upstream/"final_data/dev/dico_nli_dev_track1_participant_labeled.csv",
           "dev_reference":upstream/"final_data/dev/dico_nli_dev_track1_reference.csv"}
    train,dev=rp.read_csv(paths["train"]),rp.read_csv(paths["dev"])
    rp.audit(train,dev,paths);rp.environment(cfg["model_id"])
    root=Path("outputs")/cfg["name"];root.mkdir(parents=True,exist_ok=False)
    queue=["independent","joint"];free=list(range(rp.gpu_count()));running={};trained={}
    worker=Path(__file__).resolve().parent/"train_worker.py"
    while queue or running:
        while queue and free:
            arm,gpu=queue.pop(0),free.pop(0)
            out=root/arm;out.mkdir()
            acfg={**cfg,"joint_readout_arm":arm}
            config_path=out/"config.json";config_path.write_text(json.dumps(acfg))
            log=open(out/"train.log","w")
            proc=subprocess.Popen([sys.executable,"-u",str(worker),"--config",str(config_path),
                "--seed","42","--train",str(paths["train"]),"--dev",str(paths["dev"]),"--out",str(out/"probs.npz")],
                stdout=log,stderr=subprocess.STDOUT,env={**os.environ,"CUDA_VISIBLE_DEVICES":str(gpu)})
            running[arm]=(proc,gpu,log,0)
            print(f"JOINT_LAUNCHED arm={arm} seed=42 gpu={gpu}",flush=True)
        time.sleep(5)
        for arm in list(running):
            proc,gpu,log,cursor=running[arm]
            path=root/arm/"train.log";lines=path.read_text(errors="replace").splitlines()
            for line in lines[cursor:]:
                if line.startswith("JOINT_"):print(f"WORKER arm={arm} {line}",flush=True)
            running[arm]=(proc,gpu,log,len(lines))
            if proc.poll() is None:continue
            log.close();del running[arm];free.append(gpu)
            if proc.returncode:
                print(path.read_text(errors="replace")[-5000:],flush=True)
                for p,_,_,_ in running.values():p.kill()
                raise RuntimeError(f"{arm} failed rc={proc.returncode}")
            trained[arm]=dict(np.load(root/arm/"probs.npz"))
    sys.path.insert(0,str(upstream))
    from evaluation_functions import compute_evaluation
    from evaluation_functions.schemas import GoldInstance
    ref=rp.read_csv(paths["dev_reference"])
    gold={r["instance_id"]:GoldInstance(instance_id=r["instance_id"],pair_id=r["pair_id"],label=r["label"],
          reverse_pair_id=r["reverse_pair_id"] or None) for r in ref}
    norm=lambda r:tuple(sorted((r["text1"].strip().lower(),r["text2"].strip().lower())))
    train_keys={norm(r) for r in train}
    clean_ids={r["instance_id"] for r in dev if norm(r) not in train_keys}
    assert len(clean_ids)==645
    scores={}
    labels=["BACKWARD_ENTAILMENT","EQUIVALENCE","FORWARD_ENTAILMENT","NEGATIVE_OTHER"]
    for arm in ("independent","joint"):
        p,q=trained[arm]["orig"],trained[arm]["swap"]
        assert np.allclose(p,q[:,REV],atol=1e-5)
        pred={r["instance_id"]:labels[int(v.argmax())] for r,v in zip(dev,.5*(p+q[:,REV]))}
        assert set(pred)==set(gold)
        scores[arm]=compute_evaluation(gold,pred).to_dict()
        clean=compute_evaluation({i:gold[i] for i in clean_ids},{i:pred[i] for i in clean_ids}).to_dict()
        print("JOINT_SCORES",json.dumps({"arm":arm,"seed":42,"full":scores[arm],"clean":clean}),flush=True)
        print(f"<<<JOINT_PREDICTIONS arm={arm} seed=42\ninstance_id,label")
        for r in dev:print(r["instance_id"]+","+pred[r["instance_id"]])
        print("JOINT_PREDICTIONS>>>")
        print(f"FINAL_JOINT_METRICS arm={arm} seed=42 F1={scores[arm]['weighted_f1']:.8f} NO={scores[arm]['classification']['per_label']['NEGATIVE_OTHER']['f1']:.8f} EQ={scores[arm]['classification']['per_label']['EQUIVALENCE']['f1']:.8f} Hard={scores[arm]['hard_cons']:.8f}",flush=True)
    parent={"weighted_f1":.8649572272905587,"NO_f1":.704225352112676,"EQ_f1":.8639053254437871,"hard_cons":.8953068592057761}
    s=scores["joint"];c=scores["independent"]
    advance=(s["weighted_f1"]-parent["weighted_f1"]>=.005 and s["weighted_f1"]-c["weighted_f1"]>=.005
             and s["classification"]["per_label"]["NEGATIVE_OTHER"]["f1"]-parent["NO_f1"]>=.025
             and s["classification"]["per_label"]["EQUIVALENCE"]["f1"]-parent["EQ_f1"]>=-.01
             and s["hard_cons"]>=parent["hard_cons"])
    print("JOINT_SCREENING",json.dumps({"advance":bool(advance),"parent_seed42":parent}),flush=True)
