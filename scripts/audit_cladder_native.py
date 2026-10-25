"""Audit public deterministic labels with hidden-model CPTs; no LM inference."""
import argparse
from collections import Counter
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from causal_native import native_counterfactual


def emit(tag, value):
    print(tag + " " + json.dumps(value, sort_keys=True), flush=True)


def load_source(name, revision, expected_hash):
    directory = Path("external") / "cladder_native" / revision
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    url = f"https://raw.githubusercontent.com/causalNLP/cladder/{revision}/data/{name}"
    if not path.exists():
        path.write_bytes(urllib.request.urlopen(url, timeout=60).read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_hash:
        raise ValueError("source checksum mismatch: " + name)
    emit("NATIVE_SOURCE", {"name": name, "url": url, "sha256": digest, "bytes": path.stat().st_size})
    return json.loads(path.read_bytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg = json.loads(Path(parser.parse_args().config).read_text())
    emit("NATIVE_AUDIT_CONFIG", {**cfg, "run_id": os.environ.get("ORX_RUN_ID"),
                                 "commit": os.environ.get("ORX_COMMIT"), "python": sys.version.split()[0]})
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(load_source, item["name"], cfg["revision"], item["sha256"]) for item in cfg["files"]]
        questions, models = [future.result() for future in futures]
    model_lookup = {m["model_id"]: m for m in models}
    assert len(model_lookup) == len(models)
    rows = [q for q in questions if q["meta"]["query_type"] == "det-counterfactual"]
    records = []
    for row in rows:
        meta = row["meta"]
        record = {"question_id": row["question_id"], "model_id": meta["model_id"],
                  "graph_id": meta["graph_id"], "polarity": meta["polarity"], "official_answer": row["answer"].lower()}
        try:
            result = native_counterfactual(model_lookup[meta["model_id"]]["params"], meta["given_info"],
                                           {meta["treatment"]: meta["action"]}, meta["outcome"])
            answer = "unknown" if result["value"] is None else "yes" if result["value"] == meta["polarity"] else "no"
            record.update({**result, "oracle_answer": answer, "match": answer == record["official_answer"],
                           "status": "identified" if result["value"] is not None else "not_point_identified"})
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            record.update({"status": "unsupported_or_inconsistent", "error": str(exc), "match": False})
        records.append(record)
        emit("NATIVE_AUDIT_CASE", record)
    counts = Counter(r["status"] for r in records)
    groups = {}
    for graph in sorted({r["graph_id"] for r in records}):
        group = [r for r in records if r["graph_id"] == graph]
        groups[graph] = {"cases": len(group), "matches": sum(r["match"] for r in group)}
    summary = {"cases": len(records), "model_count": len({r["model_id"] for r in records}),
               "matches": sum(r["match"] for r in records), "status_counts": dict(counts),
               "match_rate_all_cases": sum(r["match"] for r in records) / len(records),
               "by_graph": groups, "new_lm_calls": 0,
               "note": "Oracle audit using published hidden CPTs/evidence; not LM performance or an input protocol."}
    output = Path("outputs") / cfg["name"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit.json").write_text(json.dumps({"config": cfg, "summary": summary, "cases": records}, indent=2))
    emit("NATIVE_AUDIT_SUMMARY", summary)


if __name__ == "__main__":
    main()
