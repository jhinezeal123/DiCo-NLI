"""Constrained next-token decisions, grouped by option count and prompt stage."""
from collections import defaultdict
import time


def decide(model, tokenizer, requests, batch_size, emit):
    import torch
    groups = defaultdict(list)
    for request in requests:
        groups[(request["stage"], request["choice_count"])].append(request)
    decisions = {}
    for (stage, count), group in sorted(groups.items()):
        candidates = [tokenizer.encode(letter, add_special_tokens=False) for letter in "ABCDEFGH"[:count]]
        if count > 8 or any(len(token) != 1 for token in candidates):
            raise ValueError("closed options require <=8 single-token letters")
        for offset in range(0, len(group), batch_size):
            batch = group[offset:offset+batch_size]
            prompts = [tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True) for r in batch]
            encoded = tokenizer(prompts, padding=True, truncation=False, return_tensors="pt").to(model.device)
            encoded["position_ids"] = (encoded["attention_mask"].long().cumsum(-1)-1).clamp_min(0)
            for device in range(torch.cuda.device_count()):
                torch.cuda.synchronize(device)
            started = time.monotonic()
            with torch.inference_mode():
                logits = model(**encoded, use_cache=False, num_logits_to_keep=1).logits[:, -1, :]
                indices = torch.tensor([v[0] for v in candidates], device=logits.device)
                probabilities = logits.index_select(-1, indices).float().softmax(-1).cpu().tolist()
            for device in range(torch.cuda.device_count()):
                torch.cuda.synchronize(device)
            elapsed = time.monotonic()-started
            for i, request in enumerate(batch):
                choice = max(range(count), key=lambda j: probabilities[i][j])
                value = {"request_id": request["request_id"], "stage": stage, "choice_index": choice,
                         "raw_output": "ABCDEFGH"[choice], "conditional_choice_probs": probabilities[i],
                         "input_tokens": int(encoded["attention_mask"][i].sum()), "output_tokens": 1,
                         "seconds": elapsed/len(batch)}
                decisions[request["request_id"]] = value
                emit("PUBLIC_SUBCALL", value)
            emit("PUBLIC_PROGRESS", {"method": "grounded_solver", "stage": stage,
                                     "choice_count": count, "done": min(offset+len(batch), len(group)), "total": len(group)})
    return decisions
