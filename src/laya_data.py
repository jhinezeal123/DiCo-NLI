"""Canonical DiCo-NLI -> Laya choice schema; no model dependencies."""
import re

LABELS = ["BACKWARD_ENTAILMENT", "EQUIVALENCE", "FORWARD_ENTAILMENT", "NEGATIVE_OTHER"]
OPTION_KEYS = ["A", "B", "C", "D"]
REV = [2, 1, 0, 3]
QUESTION_ID = "relation"


def question():
    return {
        "type": "choice",
        "instructions": "Select the semantic relation from Text 1 to Text 2. Choose exactly one option.",
        "criteria": {
            "A": "Backward: Text 2 entails Text 1, but not conversely.",
            "B": "Equivalent: Text 1 and Text 2 entail each other.",
            "C": "Forward: Text 1 entails Text 2, but not conversely.",
            "D": "Other: neither directional entailment nor equivalence; a weaker, incompatible, unrelated or noisy relation.",
        },
    }


def state(row, swap=False):
    a, b = (row["text2"], row["text1"]) if swap else (row["text1"], row["text2"])
    clean = lambda text: re.sub(r"\s+", " ", str(text)).strip()
    return f"Text 1: {clean(a)}\nText 2: {clean(b)}"


def training_record(row):
    idx = LABELS.index(row["label"])
    return {"state": state(row), "questions": {QUESTION_ID: question()},
            "expected": {QUESTION_ID: OPTION_KEYS[idx]}}


def inference_record(row, swap=False):
    """Ignores any gold in row."""
    return {"state": state(row, swap), "questions": {QUESTION_ID: question()}}


def text_key(row):
    clean = lambda text: re.sub(r"\s+", " ", str(text)).strip().casefold()
    return tuple(sorted((clean(row["text1"]), clean(row["text2"]))))


def clean_train_groups(train, dev_inputs):
    dev_keys = {text_key(r) for r in dev_inputs}
    bad_groups = {r["pair_id"] for r in train if text_key(r) in dev_keys}
    cleaned = [r for r in train if r["pair_id"] not in bad_groups]
    assert not ({text_key(r) for r in cleaned} & dev_keys), "text leakage"
    assert not ({r["pair_id"] for r in cleaned} & {r["pair_id"] for r in dev_inputs}), "pair_id leakage"
    return cleaned, sorted(bad_groups)
