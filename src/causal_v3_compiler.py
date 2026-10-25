"""Public-text-only grounded and free local compilation for the harder suite."""
from collections import Counter
import hashlib
import json
import re

from causal_decomposition import split_document
from causal_v3_protocol import SCHEMA_V3, decode_v3
from causal_choice import LETTERS


def document_v3(text):
    match = re.search(r"^Declared variables: ([^.]+)\.", text)
    if not match:
        raise ValueError("v3 requires an explicit public variable declaration")
    names = sorted(n.strip() for n in match.group(1).split(","))
    if len(names) not in (8, 12) or len(set(names)) != len(names):
        raise ValueError("v3/v4 requires eight or twelve unique declared variables")
    rules, query = split_document(text)
    lines = query.splitlines()
    assignments = lambda line: {n: int(v) for n, v in re.findall(r"\b([A-Za-z]\w*)\s*=\s*([01])\b", line)}
    observed, proposed = assignments(lines[0]), assignments(lines[1])
    if set(observed)-set(names) or set(proposed)-set(names) or len(proposed) != 2:
        raise ValueError("unsupported v3 public assignments")
    pattern = re.compile(r"\b(?:"+"|".join(re.escape(n) for n in names)+r")\b")
    nodes = []
    for rule in rules:
        mentions = pattern.findall(rule)
        if not mentions:
            raise ValueError("public rule mentions no variable")
        name = mentions[0]; parents = sorted(set(mentions)-{name})
        if len(parents) > 3:
            raise ValueError("unsupported v3 parent count")
        nodes.append({"name": name, "parents": parents, "rule": rule,
                      "op": "root" if not parents else "copy" if len(parents) == 1 else None})
    if len(nodes) != len(names) or {n["name"] for n in nodes} != set(names):
        raise ValueError("every declared variable needs one output-first rule")
    question = " ".join(line for line in lines[2:] if not line.startswith("Answer:"))
    final = [s for s in re.split(r"(?<=[.!?])\s+", question) if "?" in s]
    if not final:
        raise ValueError("missing final interrogative")
    return {"names": names, "nodes": nodes, "observed": observed, "proposed": proposed,
            "question": question, "final_question": final[-1], "query_text": "Declared variables:"+text.split("Declared variables:",1)[1].split("\n",1)[0]+"\n"+query}


def with_options(text, options):
    return text+"\nOptions:\n"+"\n".join(f"{LETTERS[i]}. {value}" for i,value in enumerate(options))


def active_prompt_v3(doc, name):
    """Resolve plural action references using only public proposal context."""
    proposals = ", ".join(f"{key}={value}" for key, value in doc["proposed"].items())
    return (f"Specified variable: {name}.\nThe two proposed overrides: {proposals}.\n"
            f"Instructions and readout: {doc['question']}\n"
            f"Which rule computes {name} at the final readout?")


def equation_options_v3(doc, name):
    return [f"Use {name}'s original structural equation",
            f"Replace {name}'s equation by the constant {doc['proposed'][name]}"]


def build_grounded_v3(rows, demonstrations):
    systems = {
        "binary": "Choose the base operation of TWO endogenous parents, ignoring the separate private-noise XOR. Reply with one option letter.",
        "ternary": "Choose the base operation of THREE endogenous parents: majority vote or odd parity. Ignore private output-noise XOR. Reply with one option letter.",
        "target": "Which variable does the FINAL question ask a value for? Reply with one option letter.",
        "active": "Choose the RULE that computes the SPECIFIED variable at the final readout. A clamp replaces its structural equation with the publicly proposed constant, either 0 or 1. Forcing zero is still an equation replacement. Both refers to the two listed proposals. Unused or undone clamps use the original equation; enforced clamps use the constant until removed. Restoring one variable does not restore the other. Read all instructions in order. Choose the equation, not a bit value; reply with one option letter.",
    }
    options = {"binary": ["AND / conjunction", "OR / inclusive disjunction", "XOR / exclusive disjunction of the TWO parents"],
               "ternary": ["majority3: at least two true parents", "parity3: odd number of true parents"]}
    examples = {stage: [] for stage in systems}
    seen_ops = set()
    for demo in demonstrations:
        doc = document_v3(demo["text"])
        gold = {n["name"]: n for n in demo["scm"]["variables"]}
        for node in doc["nodes"]:
            if len(node["parents"]) == 2 and gold[node["name"]]["op"] not in seen_ops:
                op = gold[node["name"]]["op"]; seen_ops.add(op)
                examples["binary"].append((with_options(node["rule"], options["binary"]), ["and","or","xor"].index(op)))
        examples["target"].append((with_options(doc["final_question"], doc["names"]), doc["names"].index(demo["scm"]["target"])))
        for name in doc["proposed"]:
            prompt = active_prompt_v3(doc, name)
            examples["active"].append((with_options(prompt, equation_options_v3(doc,name)), int(name in demo["scm"]["intervene"])))
    requests, plans = {}, {}
    def add(stage, prompt, choices):
        messages = [{"role":"system", "content":systems[stage]}]
        for text, answer in examples[stage]:
            messages.extend([{"role":"user","content":text},{"role":"assistant","content":LETTERS[answer]}])
        messages.append({"role":"user","content":with_options(prompt, choices)})
        key = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
        requests.setdefault(key, {"request_id":key, "stage":stage, "messages":messages, "choice_count":len(choices)})
        return key
    for row in rows:
        doc = document_v3(row["text"])
        for node in doc["nodes"]:
            if node["op"] is None:
                stage = "binary" if len(node["parents"]) == 2 else "ternary"
                node["stage"] = stage; node["request_id"] = add(stage, node["rule"], options[stage])
        doc["target_request"] = add("target", doc["final_question"], doc["names"])
        doc["active_requests"] = {name:add("active",active_prompt_v3(doc,name),
                                         equation_options_v3(doc,name)) for name in doc["proposed"]}
        plans[row["case_id"]] = doc
    return list(requests.values()), plans


def assemble_grounded_v3(plan, decisions, blind=False):
    nodes = []
    for node in plan["nodes"]:
        op = node["op"]
        if op is None:
            choices = ["and","or","xor"] if node["stage"] == "binary" else ["majority3","parity3"]
            op = choices[0 if blind else decisions[node["request_id"]]["choice_index"]]
        nodes.append({"name":node["name"],"parents":node["parents"],"op":op})
    return {"variables":nodes,"observed":plan["observed"],
        "intervene":{name:plan["proposed"][name] for name,key in plan["active_requests"].items() if decisions[key]["choice_index"] == 1},
        "target":plan["names"][decisions[plan["target_request"]]["choice_index"]]}


def build_local_v3(rows, demonstrations):
    rules, queries, seen = [], [], set()
    for demo in demonstrations:
        doc = document_v3(demo["text"])
        gold = {n["name"]:n for n in demo["scm"]["variables"]}
        for node in doc["nodes"]:
            op = gold[node["name"]]["op"]
            if op not in seen:
                rules.append((node["rule"],json.dumps(gold[node["name"]]))); seen.add(op)
        queries.append((doc["query_text"],json.dumps({k:demo["scm"][k] for k in ("observed","intervene","target")})))
    requests, plans = {}, {}
    for row in rows:
        doc = document_v3(row["text"]); keys = []
        items = [("rule",n["rule"],rules) for n in doc["nodes"]]+[("query",doc["query_text"],queries)]
        for stage, text, examples in items:
            system = ("Extract ONLY one node as JSON with name,op,parents. Allowed op: root (0 parents), copy/not (1), "
                      "and/or/xor/nand/nor (2), majority3/parity3 (3). Majority3 needs at least two true parents; parity3 needs an odd number. "
                      "The separate private-noise XOR is not a parent or base XOR operation. Preserve all spellings. "
                      "Do not output a full SCM or an answer." if stage == "rule" else
                      "Extract ONLY JSON with observed,intervene,target. Copy only explicitly stated original observations, "
                      "never fill missing bits. Include only overrides still active after ALL instructions; removing one "
                      "does not remove another. Preserve names and identify the final queried target. Do not calculate an answer.")
            messages = [{"role":"system","content":system}]
            for question, answer in examples:
                messages.extend([{"role":"user","content":question},{"role":"assistant","content":answer}])
            messages.append({"role":"user","content":text})
            key = hashlib.sha256(json.dumps(messages,sort_keys=True).encode()).hexdigest()
            requests.setdefault(key,{"request_id":key,"stage":stage,"messages":messages,
                                     "max_new_tokens":96 if stage == "rule" else 256})
            keys.append(key)
        plans[row["case_id"]] = keys
    return list(requests.values()), plans


def assemble_local_v3(keys, outputs):
    def read(raw):
        decoder = json.JSONDecoder()
        for i,char in enumerate(raw):
            if char == "{":
                try:
                    value,_ = decoder.raw_decode(raw[i:]); return value
                except json.JSONDecodeError:
                    pass
        raise ValueError("missing local JSON")
    nodes = [read(outputs[key]["raw_output"]) for key in keys[:-1]]
    query = read(outputs[keys[-1]]["raw_output"])
    return json.dumps({"variables":nodes,**query},separators=(",",":"))
