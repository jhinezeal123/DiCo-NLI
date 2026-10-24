"""Package only saved predictions for a future tracked NO-gate experiment."""
import argparse
import csv
import hashlib
import io
import json
import re
from pathlib import Path


def extract(raw, tag):
    match=re.search(r"<<<"+re.escape(tag)+r"\r?\n(.*?)\r?\n"+tag.split()[0]+r">>>",raw,re.S)
    assert match,tag
    return list(csv.DictReader(io.StringIO(match.group(1))))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base-log",type=Path,required=True)
    ap.add_argument("--base-commit",required=True)
    ap.add_argument("--oof-log",type=Path,required=True)
    ap.add_argument("--evidence",type=Path,required=True)
    ap.add_argument("--out",type=Path,default=Path("configs/gate_data"))
    args=ap.parse_args()
    dest=args.out;dest.mkdir(parents=True,exist_ok=True)
    base=args.base_log.read_text(encoding="utf-8",errors="replace")
    oof=args.oof_log.read_text(encoding="utf-8",errors="replace")
    tables={"train_oof.csv":extract(oof,"OOF_PROBS")}
    for seed in (42,13,7):tables[f"dev_seed{seed}.csv"]=extract(base,f"DEV_PROBS seed={seed}")
    with args.evidence.open(encoding="utf-8-sig",newline="") as f:tables["nli_evidence.csv"]=list(csv.DictReader(f))
    provenance={"base_run":args.base_log.stem,"base_commit":args.base_commit,
                "oof_run":args.oof_log.stem,"oof_commit":"807f7fd",
                "teacher_source_run":"fdf5820a-f650-4190-9b3f-9b2795f7de6e",
                "teacher_revision":"6f5cf0a2b59cabb106aca4c287eed12e357e90eb","files":{}}
    for name,rows in tables.items():
        assert len(rows)==(3042 if name=="train_oof.csv" else 3702 if name=="nli_evidence.csv" else 660)
        assert len(rows)==len({r["instance_id"] for r in rows})
        columns=[k for k in rows[0] if k!="label" or name=="train_oof.csv"]
        path=dest/name
        with path.open("w",encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=columns,lineterminator="\n",extrasaction="ignore")
            writer.writeheader();writer.writerows(rows)
        provenance["files"][name]={"rows":len(rows),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    (dest/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(provenance,indent=2))


if __name__=="__main__":main()
