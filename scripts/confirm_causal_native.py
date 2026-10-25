"""Evaluate the exact selected adapter and prespecified controls on native108."""
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from audit_cladder_native import load_source, emit
from causal_native_confirmation import build_confirmation, unpack_adapter
from causal_public_protocol import native_messages, decode_native, public_metrics, public_bootstrap


def main():
    cfg=json.loads(Path("configs/causal_native_confirmation.json").read_text())
    emit("PUBLIC_CONFIG",{**cfg,"commit":os.environ.get("ORX_COMMIT"),"run_id":os.environ.get("ORX_RUN_ID")})
    questions,models=[load_source(f["name"],cfg["revision"],f["sha256"]) for f in cfg["files"]]
    plan_raw=Path(cfg["split_plan_path"]).read_bytes()
    if hashlib.sha256(plan_raw).hexdigest()!=cfg["split_plan_sha256"]:
        raise ValueError("confirmation plan checksum mismatch")
    plan=json.loads(plan_raw); dataset=build_confirmation(questions,models,plan)
    emit("PUBLIC_DATA_AUDIT",dataset["audit"])
    output=Path("outputs")/cfg["name"]; output.mkdir(parents=True,exist_ok=True)
    adapter_path=unpack_adapter(cfg["adapter_path"],cfg["adapter_sha256"],output/"frozen_adapter")
    import torch,transformers,peft
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    if not torch.cuda.is_available(): raise RuntimeError("native confirmation requires GPU")
    torch.manual_seed(cfg["seed"])
    emit("PUBLIC_ENVIRONMENT",{"torch":torch.__version__,"transformers":transformers.__version__,"peft":peft.__version__,
         "gpu_count":torch.cuda.device_count(),"gpu":torch.cuda.get_device_name(0)})
    all_records={}

    def evaluate(model,tokenizer,phase,method):
        model.eval(); records=[]; started=time.monotonic()
        demos=dataset["demonstrations"][:cfg["eval_shots"]]
        for offset in range(0,len(dataset["rows"]),cfg["eval_batch_size"]):
            batch=dataset["rows"][offset:offset+cfg["eval_batch_size"]]
            prompts=[tokenizer.apply_chat_template(native_messages(row,method,demos),tokenize=False,add_generation_prompt=True) for row in batch]
            encoded=tokenizer(prompts,padding=True,truncation=False,return_tensors="pt").to(model.device)
            for device in range(torch.cuda.device_count()): torch.cuda.synchronize(device)
            then=time.monotonic()
            with torch.inference_mode():
                generated=model.generate(**encoded,do_sample=False,max_new_tokens=cfg["max_new_tokens"],
                    pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
            for device in range(torch.cuda.device_count()): torch.cuda.synchronize(device)
            seconds=time.monotonic()-then; continuation=generated[:,encoded["input_ids"].shape[1]:]
            for i,(row,text) in enumerate(zip(batch,tokenizer.batch_decode(continuation,skip_special_tokens=True))):
                decoded=decode_native(text,method)
                record={k:row[k] for k in ("question_id","model_id","graph","answer")}
                record.update({**decoded,"method":phase+":"+method,"correct":decoded["prediction"]==row["answer"],
                    "raw_output":text,"input_tokens":int(encoded["attention_mask"][i].sum()),
                    "output_tokens":int((continuation[i]!=tokenizer.pad_token_id).sum()),"batch_seconds_per_case":seconds/len(batch),
                    "prompt_sha256":hashlib.sha256(prompts[i].encode()).hexdigest()})
                records.append(record); emit("PUBLIC_CASE",record)
            emit("PUBLIC_PROGRESS",{"method":phase+":"+method,"done":len(records),"total":len(dataset["rows"])})
        all_records[phase+":"+method]=records
        emit("PUBLIC_METHOD_SUMMARY",{"method":phase+":"+method,"metrics":public_metrics(records),"seconds":time.monotonic()-started})

    tokenizer=AutoTokenizer.from_pretrained(cfg["model_id"],revision=cfg["model_revision"],padding_side="left")
    tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(cfg["model_id"],revision=cfg["model_revision"],torch_dtype=torch.float16,
         attn_implementation="sdpa",low_cpu_mem_usage=True).to("cuda:0")
    evaluate(model,tokenizer,"base3b","direct")
    model=PeftModel.from_pretrained(model,str(adapter_path),is_trainable=False)
    emit("FROZEN_ADAPTER",{"source_run":cfg["adapter_source_run"],"sha256":cfg["adapter_sha256"],"training_updates":0})
    evaluate(model,tokenizer,"adapter3b","direct")
    del model,tokenizer; gc.collect(); torch.cuda.empty_cache()
    tokenizer=AutoTokenizer.from_pretrained(cfg["control7b_model_id"],revision=cfg["control7b_model_revision"],padding_side="left")
    tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(cfg["control7b_model_id"],revision=cfg["control7b_model_revision"],torch_dtype=torch.float16,
         attn_implementation="sdpa",low_cpu_mem_usage=True,device_map="balanced",max_memory={i:"13GiB" for i in range(torch.cuda.device_count())})
    evaluate(model,tokenizer,"base7b","direct")
    evaluate(model,tokenizer,"base7b","structured")
    metrics={name:public_metrics(rows) for name,rows in all_records.items()}
    strongest=max(cfg["controls"],key=lambda name:metrics[name]["accuracy"])
    comparisons={name:public_bootstrap(all_records[cfg["primary_method"]],all_records[name]) for name in cfg["controls"]}
    comparison=comparisons[strongest]; criteria=plan["gate"]
    passed=(metrics[cfg["primary_method"]]["accuracy"]>=criteria["overall_accuracy_minimum"] and
            comparison["accuracy_delta"]>=criteria["delta_over_strongest_matched_control_minimum"] and comparison["ci95"][0]>0)
    final={"config":cfg,"audit":dataset["audit"],"metrics":metrics,"primary_method":cfg["primary_method"],
           "comparisons":comparisons,"strongest_control":strongest,"gate":{"criteria":criteria,"passed":passed},
           "pilot_only":False,"scope":plan["scope"],"synthetic_three_seed_gate_separately_required":True}
    (output/"summary.json").write_text(json.dumps(final,indent=2))
    emit("PUBLIC_FINAL_SUMMARY",final)


if __name__=="__main__": main()
