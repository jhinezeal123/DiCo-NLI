"""Oracle-supervised native training with explicit model-group exclusions."""
import hashlib
import json

from causal_native import parse_native_params
from causal_public_protocol import demonstration_program, public_text, stable_rank, decode_native, native_messages


def build_supervision(questions, models, plan, limit, seed):
    allowed = set(plan["training_model_ids_if_sft"])
    forbidden = (set(plan["development_model_ids"]) | set(plan["pilot_model_ids"]) |
                 {r["model_id"] for r in plan["final_cases"]})
    if allowed & forbidden:
        raise ValueError("training/evaluation model overlap")
    selected = [q for q in questions if q["meta"]["query_type"] == "det-counterfactual"
                and q["meta"]["model_id"] in allowed]
    if len(selected) != plan["training_questions_if_sft"]:
        raise ValueError("training source count differs from preregistered split")
    selected.sort(key=lambda q: stable_rank(f"supervision:{seed}:{q['question_id']}"))
    selected = selected[:limit]
    lookup = {m["model_id"]: m for m in models}
    rows = []
    for question in selected:
        model = lookup[question["meta"]["model_id"]]
        program = demonstration_program(question, model)
        for node in parse_native_params(program["params"]):
            if not node["parents"]:
                program["params"][f"p({node['name']})"] = .5
        decoded = decode_native(json.dumps(program), "native_parser_solver")
        answer = question["answer"].lower()
        if decoded["prediction"] != answer:
            raise ValueError("training oracle does not match source label")
        rows.append({"question_id": question["question_id"], "model_id": model["model_id"],
                     "text": public_text(question, model), "answer": answer, "program": program})
    audit = {"questions_used": len(rows), "source_training_questions": plan["training_questions_if_sft"],
             "model_ids_used": sorted({r["model_id"] for r in rows}),
             "question_ids_used": [r["question_id"] for r in rows],
             "excluded_model_count": len(forbidden), "oracle_agreement": 1.0,
             "training_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
             "styles_per_question": ["direct", "native_parser_solver"],
             "scope": "model-instance split; graph and mechanism families may be shared"}
    return rows, audit


def training_pair(tokenizer, row, max_length):
    pair = []
    for method in ("direct", "native_parser_solver"):
        conversation = native_messages(row, method, [])
        answer = (json.dumps(row["program"], separators=(",", ":")) if method == "native_parser_solver"
                  else "<answer>" + row["answer"] + "</answer>")
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(conversation + [{"role": "assistant", "content": answer}],
                                              tokenize=False, add_generation_prompt=False)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        ids = tokenizer(full, add_special_tokens=False)["input_ids"]
        if ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError("chat template assistant boundary is not prefix aligned")
        if len(ids) > max_length:
            return None  # Drop BOTH styles rather than truncate a target program.
        labels = [-100]*len(prompt_ids) + ids[len(prompt_ids):]
        if not any(v != -100 for v in labels):
            raise ValueError("empty completion supervision")
        pair.append({"method": method, "input_ids": ids, "labels": labels})
    return pair
