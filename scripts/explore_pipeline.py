"""Five independent method pilots with matched controls and official scoring.

Every experiment is launched through the inherited bash run.sh contract. The
pipeline prepares label-blind evaluation inputs and independent arm/seed jobs;
workers never receive evaluation labels or official reverse-pair references.
"""
import copy
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from explore_data import LABELS, REV, clean_train, dump_rows, validate_prepared, without_labels


def prepare_arm(train, evaluation, cfg, arm, seed):
    if arm in ("counterfactual", "resample"):
        from explore_counterfactual import prepare
        return prepare(train, evaluation, arm, seed, cfg)
    if arm in ("edit_retrieval", "topic_retrieval"):
        from explore_retrieval import prepare
        return prepare(train, evaluation, arm, seed, cfg)
    return copy.deepcopy(train), copy.deepcopy(evaluation), {"arm": arm, "eval_labels_used": False}


def stream_workers(jobs, max_gpus):
    """Run at most one job per visible GPU; immediately persist each worker's logs."""
    waiting, active, complete = list(jobs), {}, []
    free = list(range(max_gpus))
    worker = Path(__file__).resolve().parent / "explore_worker.py"
    try:
        while waiting or active:
            while waiting and free:
                job, gpu = waiting.pop(0), free.pop(0)
                out = job["out"]
                out.mkdir(parents=True)
                writer = (out / "train.log").open("w", encoding="utf-8")
                reader = (out / "train.log").open("r", encoding="utf-8", errors="replace")
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1",
                       "TOKENIZERS_PARALLELISM": "false", "CUBLAS_WORKSPACE_CONFIG": ":16:8",
                       "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
                command = [sys.executable, "-u", str(worker), "--config", str(job["config"]),
                           "--seed", str(job["seed"]), "--train", str(job["train"]),
                           "--eval", str(job["eval"]), "--out", str(out)]
                process = subprocess.Popen(command, stdout=writer, stderr=subprocess.STDOUT, env=env)
                key = (job["arm"], job["seed"])
                active[key] = (job, gpu, process, writer, reader, time.time())
                print(f"WORKER_START arm={key[0]} seed={key[1]} gpu={gpu}", flush=True)
            time.sleep(3)
            for key, state in list(active.items()):
                job, gpu, process, writer, reader, started = state
                chunk = reader.read()
                if chunk:
                    print(f"<<<WORKER_LOG arm={key[0]} seed={key[1]}\n{chunk}", end="", flush=True)
                if process.poll() is None:
                    continue
                writer.close()
                remainder = reader.read()
                if remainder:
                    print(remainder, end="", flush=True)
                reader.close()
                del active[key]
                free.append(gpu)
                if process.returncode:
                    raise RuntimeError(f"worker failed arm={key[0]} seed={key[1]} rc={process.returncode}")
                print(f"WORKER_EXIT arm={key[0]} seed={key[1]} seconds={time.time()-started:.1f}", flush=True)
                complete.append(job)
    finally:
        for _, _, process, writer, reader, _ in active.values():
            if process.poll() is None:
                process.kill()
            process.wait()
            writer.close()
            reader.close()
    return complete


def score_job(job, cfg, evaluation_gold, reference_path, upstream):
    import numpy as np
    from run_pipeline import official_score, validate
    arrays = np.load(job["out"] / "probs.npz")
    p, q = arrays["orig"], arrays["swap"]
    assert p.shape == q.shape == (len(evaluation_gold), 4)
    assert np.isfinite(p).all() and np.isfinite(q).all()
    print(f"<<<DEV_PROBS arm={job['arm']} seed={job['seed']}")
    print("instance_id," + ",".join("p_"+l for l in LABELS) + "," + ",".join("q_"+l for l in LABELS))
    for row, a, b in zip(evaluation_gold, p, q):
        print(row["instance_id"] + "," + ",".join(f"{v:.8f}" for v in list(a)+list(b)))
    print("DEV_PROBS>>>", flush=True)
    outputs = []
    for mode, probs in (("standard", p), ("bidir", .5 * (p + q[:, REV]))):
        prediction_path = job["out"] / f"predictions_{mode}.csv"
        with prediction_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["instance_id", "label"])
            writer.writerows((r["instance_id"], LABELS[int(x.argmax())]) for r, x in zip(evaluation_gold, probs))
        validate(prediction_path, evaluation_gold)
        scores = official_score(upstream, reference_path, prediction_path.resolve(),
                                (job["out"] / f"scores_{mode}").resolve())
        result = {"arm": job["arm"], "seed": job["seed"], "decoding": mode,
                  **{k: scores[k] for k in ("weighted_f1", "soft_cons", "hard_cons")},
                  "classification": scores["classification"], "consistency": scores["consistency"]}
        outputs.append(result)
        print("SEED_METRICS " + json.dumps({k:v for k,v in result.items() if k not in ("classification","consistency")}), flush=True)
        print(f"<<<PREDICTIONS_CSV arm={job['arm']} seed={job['seed']} decoding={mode}")
        print(prediction_path.read_text(), end="")
        print("PREDICTIONS_CSV>>>", flush=True)
    return outputs


def run_exploration(cfg, upstream):
    from run_pipeline import audit, environment, gpu_count, read_csv, sha256
    print("EXPLORATION_CONFIG " + json.dumps(cfg, indent=2), flush=True)
    # Unit/integration tests use tiny random models, not inference/training on task examples.
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_explore*.py", "-v"], check=True)
    paths = {
        "train": upstream / "final_data/train/dico_nli_train_track1_participant_labeled.csv",
        "dev": upstream / "final_data/dev/dico_nli_dev_track1_participant_labeled.csv",
        "dev_reference": upstream / "final_data/dev/dico_nli_dev_track1_reference.csv",
    }
    train_gold, evaluation_gold = read_csv(paths["train"]), read_csv(paths["dev"])
    audit(train_gold, evaluation_gold, paths)
    evaluation = without_labels(evaluation_gold)
    train, cleaning = clean_train(train_gold, evaluation)
    validate_prepared(train, evaluation)
    print("CLEAN_DATA_AUDIT " + json.dumps(cleaning), flush=True)
    print("DATA_HASHES " + json.dumps({k:sha256(v) for k,v in paths.items()}), flush=True)
    environment(cfg["model_id"])
    output = Path("outputs") / cfg["name"]
    output.mkdir(parents=True, exist_ok=False)
    generation_audit = None
    if cfg["explore_method"] == "witness":
        from explore_witness import prepare
        train, evaluation, generation_audit = prepare(train, evaluation, cfg, output / "evidence")
        validate_prepared(train, evaluation)
        print("WITNESS_AUDIT " + json.dumps(generation_audit), flush=True)
    jobs, preparation = [], []
    for arm in cfg["arms"]:
        for seed in cfg["seeds"]:
            rows, eval_rows, arm_audit = prepare_arm(train, evaluation, cfg, arm, seed)
            validate_prepared(rows, eval_rows)
            # Protect evaluation order: scoring refers to this exact input order.
            assert [r["instance_id"] for r in eval_rows] == [r["instance_id"] for r in evaluation]
            assert all((a["text1"],a["text2"]) == (b["text1"],b["text2"]) for a,b in zip(eval_rows,evaluation))
            stem = output / "prepared" / f"{arm}_{seed}"
            stem.mkdir(parents=True)
            train_file, eval_file, config_file = stem / "train.jsonl", stem / "eval.jsonl", stem / "config.json"
            digests = {"train":dump_rows(train_file, rows), "eval":dump_rows(eval_file, eval_rows)}
            arm_cfg = {**cfg, "arm": arm}
            config_file.write_text(json.dumps(arm_cfg, indent=2), encoding="utf-8")
            record = {"arm":arm, "seed":seed, "hashes":digests, "audit":arm_audit}
            preparation.append(record)
            print("PREPARATION_AUDIT " + json.dumps(record, ensure_ascii=False), flush=True)
            jobs.append({"arm":arm, "seed":seed, "config":config_file.resolve(), "train":train_file.resolve(),
                         "eval":eval_file.resolve(), "out":(output / f"{arm}_seed{seed}").resolve()})
    started = time.time()
    n_gpu = min(gpu_count(), int(cfg.get("max_workers", 2)))
    completed = stream_workers(jobs, n_gpu)
    results = []
    for job in completed:
        results.extend(score_job(job, cfg, evaluation_gold, paths["dev_reference"], upstream))
    aggregates = []
    for arm in cfg["arms"]:
        for mode in ("standard", "bidir"):
            rows = [r for r in results if r["arm"] == arm and r["decoding"] == mode]
            assert len(rows) == len(cfg["seeds"])
            record = {"arm":arm, "decoding":mode, "seeds":cfg["seeds"]}
            for metric in ("weighted_f1", "soft_cons", "hard_cons"):
                values = [r[metric] for r in rows]
                record[metric] = {"mean":statistics.mean(values), "sample_sd":statistics.stdev(values) if len(values)>1 else 0., "min":min(values)}
            aggregates.append(record)
            print("FINAL_METRICS " + json.dumps(record), flush=True)
    deltas = []
    for method_arm, control_arm in cfg["comparisons"]:
        for mode in ("standard", "bidir"):
            for seed in cfg["seeds"]:
                method = next(r for r in results if (r["arm"],r["seed"],r["decoding"]) == (method_arm,seed,mode))
                control = next(r for r in results if (r["arm"],r["seed"],r["decoding"]) == (control_arm,seed,mode))
                delta = {"method":method_arm, "control":control_arm, "seed":seed, "decoding":mode,
                         **{m:method[m]-control[m] for m in ("weighted_f1","soft_cons","hard_cons")}}
                deltas.append(delta)
                print("MATCHED_DELTA " + json.dumps(delta), flush=True)
    summary = {"config":cfg, "data_audit":cleaning, "generation_audit":generation_audit,
               "results":results, "aggregates":aggregates, "matched_deltas":deltas,
               "wall_seconds_training":time.time()-started,
               "limitations":["Public dev was previously used for project selection; no unbiased test claim.",
                              "Three seed SD is not a confidence interval; use source-pair bootstrap for confirmation.",
                              "NEG identity in bidirectional averaging is an inference heuristic, not official reversal.",
                              "Architecture pilots use jointly contextualized phrase features, not global semantic guarantees."]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("EXPLORATION_SUMMARY_JSON " + json.dumps(summary), flush=True)
