"""Tracked harder-benchmark comparison with all four views and strong controls."""
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from causal_dataset_v3 import build_dataset_v3, solve_conditioned
from causal_dataset import digest
from causal_v3_protocol import messages_v3, decode_v3, metrics_v3, paired_v3
from causal_v3_compiler import build_local_v3, assemble_local_v3, build_grounded_v3, assemble_grounded_v3


def emit(tag, value):
    print(tag+" "+json.dumps(value,sort_keys=True),flush=True)


def request(messages, stage, cap):
    key=hashlib.sha256(json.dumps(messages,sort_keys=True).encode()).hexdigest()
    return {"request_id":key,"stage":stage,"messages":messages,"max_new_tokens":cap}


def load_control_replay(cfg, dataset):
    spec = cfg.get("control_replay")
    if not spec:
        return {}
    for key in ("model_id", "model_revision", "shots", "max_new_tokens"):
        if spec[key] != cfg[key]:
            raise ValueError(f"control replay configuration mismatch: {key}")
    if dataset["audit"]["dataset_sha256"] != spec["dataset_sha256"]:
        raise ValueError("control replay dataset mismatch")
    raw = Path(spec["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec["sha256"]:
        raise ValueError("control replay checksum mismatch")
    expected = {row["case_id"]:row for row in dataset["rows"]}
    grouped = {method:{} for method in spec["methods"]}
    for line in raw.decode().splitlines():
        row = json.loads(line)
        method, case = row["method"], row["case_id"]
        if method not in grouped or case not in expected or case in grouped[method]:
            raise ValueError("unexpected or duplicate control record")
        if any(row[key] != expected[case][key] for key in ("family_id","split","view","answer","changed")):
            raise ValueError("control record metadata mismatch")
        if row["correct"] != (row["prediction"] == row["answer"]):
            raise ValueError("control correctness mismatch")
        grouped[method][case] = row
    if any(set(rows) != set(expected) for rows in grouped.values()):
        raise ValueError("incomplete control predictions")
    return {method:[rows[row["case_id"]] for row in dataset["rows"]] for method,rows in grouped.items()}


def generate(model,tokenizer,requests,cfg,log_subcalls=False):
    import torch
    grouped=defaultdict(list); outputs={}
    for item in requests:
        grouped[(item["stage"],item["max_new_tokens"])].append(item)
    for (stage,cap),items in grouped.items():
        for offset in range(0,len(items),cfg["batch_size"]):
            batch=items[offset:offset+cfg["batch_size"]]
            prompts=[tokenizer.apply_chat_template(r["messages"],tokenize=False,add_generation_prompt=True) for r in batch]
            encoded=tokenizer(prompts,return_tensors="pt",padding=True,truncation=False).to(model.device)
            for device in range(torch.cuda.device_count()): torch.cuda.synchronize(device)
            started=time.monotonic()
            with torch.inference_mode():
                generated=model.generate(**encoded,do_sample=False,max_new_tokens=cap,
                    pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
            for device in range(torch.cuda.device_count()): torch.cuda.synchronize(device)
            elapsed=time.monotonic()-started
            continuation=generated[:,encoded["input_ids"].shape[1]:]
            texts=tokenizer.batch_decode(continuation,skip_special_tokens=True)
            for i,(item,text) in enumerate(zip(batch,texts)):
                result={"request_id":item["request_id"],"stage":stage,"raw_output":text,
                    "input_tokens":int(encoded["attention_mask"][i].sum()),
                    "output_tokens":int((continuation[i]!=tokenizer.pad_token_id).sum()),
                    "seconds":elapsed/len(batch),"max_new_tokens":cap,
                    "prompt_sha256":hashlib.sha256(prompts[i].encode()).hexdigest()}
                outputs[item["request_id"]]=result
                if log_subcalls: emit("CAUSAL_SUBCALL",result)
            emit("CAUSAL_PROGRESS",{"method":stage,"done":min(offset+len(batch),len(items)),"total":len(items)})
    return outputs


def main():
    cfg=json.loads(Path("configs/causal_v3.json").read_text())
    emit("CAUSAL_CONFIG",{**cfg,"commit":os.environ.get("ORX_COMMIT"),"run_id":os.environ.get("ORX_RUN_ID")})
    if "development_input" in cfg:
        raw=Path(cfg["development_input"]["path"]).read_bytes()
        if not cfg["pilot_only"] or hashlib.sha256(raw).hexdigest()!=cfg["development_input"]["sha256"]:
            raise ValueError("invalid opened-development dataset")
        dataset=json.loads(raw)
        if dataset["seed"]!=cfg["dataset_seed"] or digest(dataset["rows"])!=dataset["audit"]["dataset_sha256"]:
            raise ValueError("opened-development dataset identity mismatch")
    else:
        dataset=build_dataset_v3(cfg["dataset_seed"],cfg["families_per_split"])
    replay = load_control_replay(cfg, dataset)
    emit("DATASET_AUDIT",dataset["audit"])
    assert all(solve_conditioned(r["scm"])["answer"]==r["answer"] for r in dataset["rows"])
    emit("ORACLE_CONTROL",{"accuracy":1.,"cases":len(dataset["rows"]),"note":"by-construction oracle, not LM performance; independent brute-noise unit tests verify the kernel"})
    output=Path("outputs")/cfg["name"]; output.mkdir(parents=True,exist_ok=True)
    (output/"dataset.json").write_text(json.dumps(dataset,indent=2))
    import torch,transformers
    from transformers import AutoModelForCausalLM,AutoTokenizer
    if not torch.cuda.is_available(): raise RuntimeError("v3 LM evaluation requires GPU")
    torch.manual_seed(42)
    tokenizer=AutoTokenizer.from_pretrained(cfg["model_id"],revision=cfg["model_revision"],padding_side="left")
    tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(cfg["model_id"],revision=cfg["model_revision"],
        torch_dtype=torch.float16,attn_implementation="sdpa",device_map="balanced",low_cpu_mem_usage=True,
        max_memory={i:"13GiB" for i in range(torch.cuda.device_count())})
    model.eval()
    emit("ENVIRONMENT",{"python":sys.version.split()[0],"torch":torch.__version__,"transformers":transformers.__version__,
                         "gpu_count":torch.cuda.device_count(),"gpu":torch.cuda.get_device_name(0)})
    all_records={}
    for method, records in replay.items():
        all_records[method] = records
        for row in records:
            emit("CAUSAL_CASE", {**row,"source_run":row.get("source_run",cfg["control_replay"]["source_run"]),"new_lm_calls":0})
        emit("CAUSAL_METHOD_SUMMARY", {"method":method,"metrics":metrics_v3(records),
             "source_run":cfg["control_replay"]["source_run"],"new_lm_calls":0})

    def score(method,raw_by_case,usage_by_case=None,usage=None,normalized=False,kind="parser_solver",derived=None):
        records=[]
        for row in dataset["rows"]:
            raw=raw_by_case[row["case_id"]]
            decoded=decode_v3(raw,kind,row["scm"] if kind=="parser_solver" else None,normalized)
            record={k:row[k] for k in ("case_id","family_id","split","view","answer","changed")}
            record.update({**decoded,"method":method,"raw_output":raw,"correct":decoded["prediction"]==row["answer"]})
            if usage_by_case: record.update(usage_by_case[row["case_id"]])
            if derived: record.update({"generation_reused_from":derived,"new_lm_calls":0})
            records.append(record); emit("CAUSAL_CASE",record)
        all_records[method]=records
        emit("CAUSAL_METHOD_SUMMARY",{"method":method,"metrics":metrics_v3(records),**(usage or {}),
             **({"generation_reused_from":derived,"new_lm_calls":0} if derived else {})})
        (output/(method+"-predictions.json")).write_text(json.dumps(records,indent=2))

    for method in (() if replay else ("direct","structured","parser_solver")):
        started=time.monotonic(); requests=[]; keys={}
        for row in dataset["rows"]:
            item=request(messages_v3(row,method,dataset["demonstrations"]),method,cfg["max_new_tokens"])
            requests.append(item); keys[row["case_id"]]=item["request_id"]
        outputs=generate(model,tokenizer,requests,cfg)
        raw={case:outputs[key]["raw_output"] for case,key in keys.items()}
        costs={case:{"input_tokens":outputs[key]["input_tokens"],"output_tokens":outputs[key]["output_tokens"],
                     "batch_seconds_per_case":outputs[key]["seconds"],"prompt_sha256":outputs[key]["prompt_sha256"]} for case,key in keys.items()}
        usage={"elapsed_seconds":time.monotonic()-started,"input_tokens":sum(r["input_tokens"] for r in outputs.values()),
               "output_tokens":sum(r["output_tokens"] for r in outputs.values())}
        score(method,raw,costs,usage,kind=method)
        if method=="parser_solver":
            score(method+"_normalized",raw,costs,normalized=True,derived=method)

    if not replay:
        requests,plans=build_local_v3(dataset["rows"],dataset["demonstrations"])
        started=time.monotonic(); outputs=generate(model,tokenizer,requests,cfg,True)
        uses=Counter(key for keys in plans.values() for key in keys); raw={}; costs={}
        for case,keys in plans.items():
            try: raw[case]=assemble_local_v3(keys,outputs)
            except (ValueError,TypeError,KeyError) as exc: raw[case]="LOCAL_COMPILATION_ERROR: "+str(exc)
            costs[case]={"input_tokens":sum(outputs[k]["input_tokens"]/uses[k] for k in keys),
                         "output_tokens":sum(outputs[k]["output_tokens"]/uses[k] for k in keys),
                         "batch_seconds_per_case":sum(outputs[k]["seconds"]/uses[k] for k in keys),"subcall_ids":keys}
        usage={"subcalls":len(requests),"elapsed_seconds":time.monotonic()-started,
               "input_tokens":sum(r["input_tokens"] for r in outputs.values()),"output_tokens":sum(r["output_tokens"] for r in outputs.values())}
        score("decomposed_solver",raw,costs,usage)
        score("decomposed_solver_normalized",raw,costs,normalized=True,derived="decomposed_solver")

    from causal_choice import decide
    requests,plans=build_grounded_v3(dataset["rows"],dataset["demonstrations"])
    started=time.monotonic()
    decisions=decide(model,tokenizer,requests,cfg["batch_size"],lambda tag,value:emit(tag.replace("PUBLIC_","CAUSAL_"),value))
    uses=Counter(key for plan in plans.values() for key in
                 [plan["target_request"],*plan["active_requests"].values(),*[n["request_id"] for n in plan["nodes"] if "request_id" in n]])
    costs={}
    for case,plan in plans.items():
        keys=[plan["target_request"],*plan["active_requests"].values(),*[n["request_id"] for n in plan["nodes"] if "request_id" in n]]
        costs[case]={"input_tokens":sum(decisions[k]["input_tokens"]/uses[k] for k in keys),
                     "output_tokens":sum(1/uses[k] for k in keys),
                     "batch_seconds_per_case":sum(decisions[k]["seconds"]/uses[k] for k in keys),"subcall_ids":keys}
    usage={"subcalls":len(requests),"elapsed_seconds":time.monotonic()-started,
           "input_tokens":sum(v["input_tokens"] for v in decisions.values()),"output_tokens":len(requests),
           "cost_scope":"grounded system; operator-blind variant reuses query decisions"}
    for method,blind in (("grounded_solver",False),("grounded_operator_blind",True)):
        raw={case:json.dumps(assemble_grounded_v3(plan,decisions,blind),separators=(",",":")) for case,plan in plans.items()}
        score(method,raw,costs,usage,derived="grounded_solver" if blind else None)
    final={"config":cfg,"audit":dataset["audit"],"metrics":{m:metrics_v3(rows) for m,rows in all_records.items()},
           "primary_method":"grounded_solver","primary_vs":{m:paired_v3(all_records["grounded_solver"],rows)
                for m,rows in all_records.items() if m!="grounded_solver"},"parser_solver_vs":{},
           "pilot_only":True,"gate":{"status":"development_only","joint_minimum":.9,"delta_minimum":.05,
                "requires_positive_ci_lower":True,"confirm_dataset_seeds":[390001,390002,390003],"required_views":4}}
    (output/"summary.json").write_text(json.dumps(final,indent=2))
    emit("CAUSAL_FINAL_SUMMARY",final)


if __name__=="__main__": main()
