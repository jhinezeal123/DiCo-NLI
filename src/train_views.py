"""Parallel train views. Pair IDs only join/source-group views, never enter model inputs."""
import csv
import hashlib
import re
from collections import defaultdict
from pathlib import Path


def clean(text):
    return re.sub(r"\s+"," ",text).strip()


def prepare_spanish(train_path,ids,text1,text2,gold_labels):
    train_path=Path(train_path)
    spanish_path=train_path.with_name("dico_nli_train_track2_participant_labeled.csv")
    with train_path.open(encoding="utf-8-sig",newline="") as f:en=list(csv.DictReader(f))
    with spanish_path.open(encoding="utf-8-sig",newline="") as f:es=list(csv.DictReader(f))
    en_by_id={r["instance_id"]:r for r in en}
    es_by_id={r["instance_id"]:r for r in es}
    assert set(ids)==set(en_by_id)
    assert {i.replace("__en-en__","__es-es__") for i in ids}==set(es_by_id)
    labels_by_text=defaultdict(set)
    norm=lambda r:(clean(r["text1"]).lower(),clean(r["text2"]).lower())
    for row in es:labels_by_text[norm(row)].add(row["label"])
    unsafe=set()
    for row in es:
        a,b=norm(row)
        if len(labels_by_text[(a,b)])>1 or (a==b and row["label"]!="EQUIVALENCE"):
            unsafe.add(row["pair_id"])
    alternate1,alternate2,pairs,eligible=[],[],[],[]
    for i,a,b,label in zip(ids,text1,text2,gold_labels):
        row=es_by_id[i.replace("__en-en__","__es-es__")]
        assert row["label"]==label==en_by_id[i]["label"]
        assert row["pair_id"]==en_by_id[i]["pair_id"]
        allowed=row["pair_id"] not in unsafe
        alternate1.append(clean(row["text1"]) if allowed else a)
        alternate2.append(clean(row["text2"]) if allowed else b)
        pairs.append(row["pair_id"]);eligible.append(allowed)
    audit={"rows":len(ids),"eligible_rows":sum(eligible),"excluded_source_pairs":len(unsafe),
           "english_fallback_rows":len(ids)-sum(eligible),"labels_match":True,
           "source_en_lf_sha256":hashlib.sha256(train_path.read_bytes().replace(b"\r\n",b"\n")).hexdigest(),
           "source_es_lf_sha256":hashlib.sha256(spanish_path.read_bytes().replace(b"\r\n",b"\n")).hexdigest()}
    assert audit["eligible_rows"]==3026 and len(unsafe)==10
    return alternate1,alternate2,pairs,eligible,audit


def epoch_features(english,spanish,pairs,eligible,rng,probability):
    assert 0<=probability<=1
    choices={pair:rng.random()<probability for pair in dict.fromkeys(pairs)}
    selected=[choices[pair] and allowed for pair,allowed in zip(pairs,eligible)]
    features=[es if choose else en for en,es,choose in zip(english,spanish,selected)]
    return features,selected
