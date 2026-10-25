"""Replay exact persisted generations under a semantics-preserving IR rule."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from causal_dataset import build_dataset
from causal_protocol import decode_output, metrics, paired_bootstrap


def emit(tag, value):
    print(tag + " " + json.dumps(value, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    source = Path(cfg["source_predictions"])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == cfg["source_predictions_sha256"]
    effective = {**cfg, "run_id": os.environ.get("ORX_RUN_ID"), "commit": os.environ.get("ORX_COMMIT")}
    emit("CAUSAL_CONFIG", effective)
    dataset = build_dataset(cfg["dataset_seed"], cfg["families_per_split"])
    assert dataset["audit"]["dataset_sha256"] == cfg["dataset_sha256"]
    gold = {r["case_id"]: r for r in dataset["rows"]}
    emit("DATASET_AUDIT", dataset["audit"])
    originals = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    grouped, seen = {}, set()
    for record in originals:
        key = (record["method"], record["case_id"])
        assert key not in seen
        seen.add(key)
        assert record["answer"] == gold[record["case_id"]]["answer"]
        record["correct"] = record["prediction"] == gold[record["case_id"]]["answer"]
        record["source_run_id"] = cfg["source_run_id"]
        record["source_log_line"] = record.pop("log_line", None)
        grouped.setdefault(record["method"], []).append(record)
    assert all(len(rows) == len(gold) for rows in grouped.values())
    candidates = []
    for method in ("parser_solver", "decomposed_solver"):
        started = time.monotonic()
        name = method + "_normalized"
        rows = []
        for original in grouped[method]:
            case = gold[original["case_id"]]
            decoded = decode_output(original["raw_output"], "parser_solver", case["scm"], reduce_unary=True)
            excluded = {"prediction", "valid", "error", "schema_exact", "fields_wrong",
                        "parsed_scm", "compatible_worlds", "correct"}
            base = {k: v for k, v in original.items() if k not in excluded}
            row = {**base, **decoded, "method": name,
                   "correct": decoded["prediction"] == case["answer"],
                   "normalization": "exact singleton Boolean reductions only", "new_lm_calls": 0}
            rows.append(row)
        grouped[name] = rows
        candidates.append(name)
        emit("CAUSAL_METHOD_SUMMARY", {"method": name, "metrics": metrics(rows),
             "replay_compute_seconds": time.monotonic() - started, "new_lm_calls": 0,
             "source_generation_input_tokens": sum(r["input_tokens"] for r in rows),
             "source_generation_output_tokens": sum(r["output_tokens"] for r in rows)})
    for rows in grouped.values():
        for row in rows:
            emit("CAUSAL_CASE", row)
    comparison = {candidate: {m: paired_bootstrap(grouped[candidate], rows)
                             for m, rows in grouped.items() if m not in candidates}
                  for candidate in candidates}
    final = {"config": cfg, "audit": dataset["audit"], "metrics": {m: metrics(v) for m, v in grouped.items()},
             "normalization_vs": comparison,
             "parser_solver_vs": {m: paired_bootstrap(grouped["parser_solver"], rows)
                                   for m, rows in grouped.items() if m != "parser_solver"},
             "primary_method": "decomposed_solver_normalized", "new_lm_calls": 0,
             "source_run_id": cfg["source_run_id"], "source_commit": cfg["source_commit"],
             "pilot_only": True, "gate": {"status": "v1 replay diagnostic; requires v2 and fresh confirmation"}}
    output = Path("outputs") / cfg["name"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(final, indent=2))
    emit("CAUSAL_FINAL_SUMMARY", final)


if __name__ == "__main__":
    main()
