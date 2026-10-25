"""Frozen native model-instance confirmation; public input allowlist only."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

from causal_public_protocol import build_public_split, public_text, stable_rank


def build_confirmation(questions, models, plan):
    lookup = {q["question_id"]:q for q in questions}
    model_lookup = {m["model_id"]:m for m in models}
    cases = plan["final_cases"]
    forbidden = set(plan["development_model_ids"] + plan["pilot_model_ids"] + plan["training_model_ids_if_sft"])
    rows = []
    for case in cases:
        q = lookup[case["question_id"]]
        mid = q["meta"]["model_id"]
        if mid != case["model_id"] or q["meta"]["graph_id"] != case["graph"] or mid in forbidden:
            raise ValueError("confirmation split identity or exclusion mismatch")
        if q["meta"]["query_type"] != "det-counterfactual":
            raise ValueError("unexpected query type")
        eligible = [x for x in questions if x["meta"].get("model_id") == mid and x["meta"]["query_type"] == "det-counterfactual"]
        if q["question_id"] != min(eligible,key=lambda x:stable_rank(x["question_id"]))["question_id"]:
            raise ValueError("confirmation question selection mismatch")
        rows.append({"question_id":q["question_id"],"model_id":mid,"graph":case["graph"],
                     "answer":q["answer"].lower(),"text":public_text(q,model_lookup[mid])})
    if len(rows) != 108 or len({r["model_id"] for r in rows}) != 108 or len({r["question_id"] for r in rows}) != 108:
        raise ValueError("confirmation requires108 distinct questions and models")
    graphs = {r["graph"] for r in rows}
    if len(graphs) != 9 or any(sum(r["graph"] == g for r in rows) != 12 for g in graphs):
        raise ValueError("confirmation requires12 models in each of9 graphs")
    demos = build_public_split(questions,models,4)["demonstrations"]
    return {"rows":rows,"demonstrations":demos,
            "audit":{"cases":108,"models":108,"graphs":9,"overlap_with_training_development_pilot":0,
                     "dataset_sha256":hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),
                     "input_fields":["model.background","question.given_info","question.question"],
                     "scope":plan["scope"]}}


def unpack_adapter(path, expected_sha, output):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("frozen adapter checksum mismatch")
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != {"README.md","adapter_config.json","adapter_model.safetensors"}:
            raise ValueError("unexpected adapter archive members")
        if sum(info.file_size for info in archive.infolist()) > 32*1024*1024:
            raise ValueError("adapter too large")
        for name in names:
            (output/name).write_bytes(archive.read(name))
    return output
