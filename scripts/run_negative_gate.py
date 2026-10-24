"""Tracked binary NO-gate experiment. This staged file must not fit locally."""
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from expert_router import ExpertRouter,normalize,numeric_features,compose_negative_gate

LABELS=["BACKWARD_ENTAILMENT","EQUIVALENCE","FORWARD_ENTAILMENT","NEGATIVE_OTHER"]
REV=np.array([2,1,0,3])


def read(path):
    with path.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))


def key(row):
    return tuple(sorted(" ".join(row[k].lower().split()) for k in ("text1","text2")))


def base_probs(path,ids):
    table={r["instance_id"]:r for r in read(path)}
    assert set(table)==set(ids)
    p=np.array([[float(table[i]["p_"+l]) for l in LABELS] for i in ids])
    q=np.array([[float(table[i]["q_"+l]) for l in LABELS] for i in ids])
    return normalize(.5*(p+q[:,REV]))


def run(cfg,upstream):
    sys.path.insert(0,str(upstream))
    from evaluation_functions import compute_evaluation
    from evaluation_functions.schemas import GoldInstance
    root=Path("configs/gate_data")
    provenance=json.loads((root/"provenance.json").read_text())
    for name,meta in provenance["files"].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==meta["sha256"]
    print("CONFIG",json.dumps(cfg,sort_keys=True),flush=True)
    print("PROVENANCE",json.dumps(provenance,sort_keys=True),flush=True)
    print(f"ENV numpy={np.__version__} sklearn={sklearn.__version__}",flush=True)
    test_p=normalize(np.array([[.1,.2,.3,.4],[.2,.5,.1,.2]]))
    test_ev=np.array([[.6,.3,.1,.2,.6,.2],[.9,.05,.05,.8,.1,.1]])
    test_text=[("a person","a woman"),("runs","is running")]
    np.testing.assert_allclose(numeric_features(test_text,test_p,test_ev),
        numeric_features([(b,a) for a,b in test_text],test_p[:,REV],test_ev[:,[3,4,5,0,1,2]]),atol=1e-12)
    np.testing.assert_allclose(compose_negative_gate(test_p,test_p[:,3]),test_p,atol=1e-12)
    print("GATE_CHECKS synthetic feature reversal and chain-rule identity passed",flush=True)
    data=upstream/"final_data"
    train=read(data/"train/dico_nli_train_track1_participant_labeled.csv")
    dev=[{k:r[k] for k in ("instance_id","pair_id","text1","text2")} for r in read(data/"dev/dico_nli_dev_track1_participant_labeled.csv")]
    assert not({r["pair_id"] for r in train}&{r["pair_id"] for r in dev})
    dev_keys={key(r) for r in dev};train_keys={key(r) for r in train}
    retained=[r for r in train if key(r) not in dev_keys]
    assert len(retained)==3029
    all_train_ids=[r["instance_id"] for r in train]
    all_p=base_probs(root/"train_oof.csv",all_train_ids)
    train_index={i:j for j,i in enumerate(all_train_ids)}
    p=all_p[[train_index[r["instance_id"]] for r in retained]]
    labels=np.array([LABELS.index(r["label"]) for r in retained])
    oof_labels={r["instance_id"]:r["label"] for r in read(root/"train_oof.csv")}
    assert all(oof_labels[r["instance_id"]]==r["label"] for r in train)
    evidence={r["instance_id"]:r for r in read(root/"nli_evidence.csv")}
    evidence_columns=["e_fwd","n_fwd","c_fwd","e_bwd","n_bwd","c_bwd"]
    ev=lambda ids:np.array([[float(evidence[i][k]) for k in evidence_columns] for i in ids])
    train_ev=ev([r["instance_id"] for r in retained])
    train_texts=[(r["text1"],r["text2"]) for r in retained]
    models={}
    with threadpool_limits(limits=2):
        for arm,use_evidence in (("text_only_gate",False),("nli_gate",True)):
            model=ExpertRouter("boosted",policy="negative_gate",use_evidence=use_evidence)
            model.fit(train_texts,p,train_ev,labels)
            models[arm]=model
            print("FROZEN_GATE",json.dumps({"arm":arm,**model.diagnostics},sort_keys=True),flush=True)
    print("FIT_COMPLETE dev_labels_used_for_fit=False metadata_features=[]",flush=True)
    reference=read(data/"dev/dico_nli_dev_track1_reference.csv")
    gold={r["instance_id"]:GoldInstance(instance_id=r["instance_id"],pair_id=r["pair_id"],label=r["label"],reverse_pair_id=r["reverse_pair_id"] or None) for r in reference}
    truth={r["instance_id"]:r["label"] for r in reference}
    dev_ids=[r["instance_id"] for r in dev]
    dev_texts=[(r["text1"],r["text2"]) for r in dev]
    dev_ev=ev(dev_ids)
    clean={r["instance_id"] for r in dev if key(r) not in train_keys}
    assert len(clean)==645
    results={arm:[] for arm in ("parent","text_only_gate","nli_gate")}
    all_predictions={}
    for seed in cfg["seeds"]:
        base=base_probs(root/f"dev_seed{seed}.csv",dev_ids)
        arrays={"parent":base.argmax(1)};gates={}
        with threadpool_limits(limits=2):
            for arm,model in models.items():
                arrays[arm],gates[arm]=model.predict(dev_texts,base,dev_ev)
        parent=dict(zip(dev_ids,[LABELS[j] for j in arrays["parent"]]))
        for arm,labels_ in arrays.items():
            pred=dict(zip(dev_ids,[LABELS[j] for j in labels_]))
            all_predictions[(arm,seed)]=pred
            score=compute_evaluation(gold,pred).to_dict()
            clean_score=compute_evaluation({i:gold[i] for i in clean},{i:pred[i] for i in clean}).to_dict()
            results[arm].append(score)
            print("GATE_SCORES",json.dumps({"arm":arm,"seed":seed,**score},sort_keys=True),flush=True)
            print("CLEAN_GATE_SCORES",json.dumps({"arm":arm,"seed":seed,"n":len(clean),**{k:clean_score[k] for k in ("weighted_f1","soft_cons","hard_cons")}}),flush=True)
            print("GATE_CHANGES",json.dumps({"arm":arm,"seed":seed,
                  "fixed":sum(parent[i]!=truth[i] and pred[i]==truth[i] for i in truth),
                  "broken":sum(parent[i]==truth[i] and pred[i]!=truth[i] for i in truth)}),flush=True)
            print(f"<<<NO_GATE_PREDICTIONS arm={arm} seed={seed}")
            print("instance_id,label")
            for i in dev_ids:print(i+","+pred[i])
            print("NO_GATE_PREDICTIONS>>>",flush=True)
    means={}
    for arm,scores in results.items():
        means[arm]={k:statistics.mean(r[k] for r in scores) for k in ("weighted_f1","soft_cons","hard_cons")}
        for label in ("EQUIVALENCE","NEGATIVE_OTHER"):
            for metric in ("f1","precision","recall"):
                means[arm][label+"_"+metric]=statistics.mean(r["classification"]["per_label"][label][metric] for r in scores)
        print("FINAL_GATE_METRICS",json.dumps({"arm":arm,"n_seeds":len(scores),**means[arm]},sort_keys=True),flush=True)
    for arm in models:
        delta={k:means[arm][k]-means["parent"][k] for k in means["parent"]}
        passed=(delta["NEGATIVE_OTHER_f1"]>=.05 and delta["EQUIVALENCE_f1"]>=-.01 and delta["weighted_f1"]>=0 and delta["hard_cons"]>=0)
        print("GATE_CRITERION",json.dumps({"arm":arm,"passed":passed,"deltas":delta},sort_keys=True),flush=True)
