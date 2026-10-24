"""Train-only, direction-aware demonstration retrieval for the exploration pilot.

This module does not produce a prediction or select a context using evaluation
gold.  It compares the same TF-IDF retrieval machinery over two views: the full
ordered pair (topic control), and explicitly ordered differing spans (method).
Training queries exclude their entire source group and all unordered-text
duplicates, including reversed duplicates under unrelated identifiers.
"""

from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
import re

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


TOKEN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
LABELS = {"BACKWARD_ENTAILMENT", "EQUIVALENCE", "FORWARD_ENTAILMENT", "NEGATIVE_OTHER"}


def _norm(text) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def unordered_pair(row: dict) -> tuple[str, str]:
    return tuple(sorted((_norm(row["text1"]), _norm(row["text2"]))))


def differing_spans(text1: str, text2: str) -> tuple[list[str], list[str], list[str]]:
    """Return left-only, right-only and aligned-common tokens in phrase order."""
    a, b = TOKEN.findall(_norm(text1)), TOKEN.findall(_norm(text2))
    left, right, shared = [], [], []
    for tag, i, j, k, l in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            shared.extend(a[i:j])
        else:
            left.extend(a[i:j])
            right.extend(b[k:l])
    return left, right, shared


def edit_representation(text1: str, text2: str) -> str:
    a, b, common = differing_spans(text1, text2)
    # Role tags and adjacent bigrams distinguish dog->animal from animal->dog.
    # The shared part is capped so a long unchanged prefix cannot dominate edits.
    return ("leftchange " + (" ".join(a) or "emptyspan")
            + " rightchange " + (" ".join(b) or "emptyspan")
            + " sharedcontext " + (" ".join(common[:6]) or "emptyspan"))


def topic_representation(text1: str, text2: str) -> str:
    return "leftphrase " + _norm(text1) + " rightphrase " + _norm(text2)


class _Index:
    def __init__(self, rows: list[dict], view, cfg: dict):
        self.rows = sorted(rows, key=lambda r: str(r["instance_id"]))
        self.view = view
        self.keys = [unordered_pair(r) for r in self.rows]
        self.groups = [str(r["pair_id"]) for r in self.rows]
        self.documents = [view(r["text1"], r["text2"]) for r in self.rows]
        maximum = int(cfg.get("retrieval_max_features", 20000))
        self.vectorizers = [
            TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b",
                            sublinear_tf=True, max_features=maximum),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                            sublinear_tf=True, max_features=maximum),
        ]
        blocks, fitted = [], []
        for vectorizer in self.vectorizers:
            try:
                block = vectorizer.fit_transform(self.documents)
            except ValueError as exc:
                # Empty or punctuation-only corpora still have a deterministic
                # fallback; never consult a query label to choose a replacement.
                if "empty vocabulary" not in str(exc):
                    raise
                continue
            fitted.append(vectorizer)
            blocks.append(block)
        self.vectorizers = fitted
        self.matrix = normalize(hstack(blocks, format="csr")) if blocks else None
        self.k = int(cfg.get("retrieval_k", 2))
        if not 1 <= self.k <= 8:
            raise ValueError("retrieval_k must be between 1 and 8")

    def retrieve(self, row: dict, swapped: bool = False):
        a, b = (row["text2"], row["text1"]) if swapped else (row["text1"], row["text2"])
        key, group = unordered_pair(row), str(row["pair_id"])
        allowed = [i for i in range(len(self.rows))
                   if self.groups[i] != group and self.keys[i] != key]
        excluded_groups = sum(g == group for g in self.groups)
        excluded_text = sum(k == key for k in self.keys)
        diagnostic = {"excluded_source_group_rows": excluded_groups,
                      "excluded_unordered_text_rows": excluded_text,
                      "fallback": None}
        if not allowed:
            diagnostic["fallback"] = "no_eligible_train_examples"
            return [], diagnostic
        if self.matrix is None or not re.search(r"\w", str(a) + str(b)):
            diagnostic["fallback"] = "empty_lexical_query_or_vocabulary"
            return [], diagnostic
        query_text = self.view(a, b)
        query = normalize(hstack([v.transform([query_text]) for v in self.vectorizers], format="csr"))
        scores = (self.matrix @ query.T).toarray().ravel()
        # Stable corpus order provides deterministic tie-breaking, independent
        # of Python's hash randomization and query/evaluation labels.
        ranked = sorted(allowed, key=lambda i: (-float(scores[i]), str(self.rows[i]["instance_id"])))
        chosen = [i for i in ranked if float(scores[i]) > 1e-12][:self.k]
        if not chosen:
            diagnostic["fallback"] = "no_positive_similarity"
        return [(self.rows[i], float(scores[i])) for i in chosen], diagnostic


def _format_context(neighbors: list[tuple[dict, float]]) -> str:
    # Identifiers and group counts must never enter the model's context.
    return "\n\n".join(
        f"Example {i + 1}:\nText 1: {r['text1']}\nText 2: {r['text2']}\nRelation: {r['label']}"
        for i, (r, _) in enumerate(neighbors))


def prepare(train_rows: list[dict], eval_rows: list[dict], arm: str,
            seed: int, cfg: dict):
    """Attach matched-format contexts and independently retrieved swap contexts.

    Evaluation rows may omit label entirely.  Caller applies the same tokenizer
    context budget to both arms.  This function never augments, reorders or
    relabels the training pool; whole contaminated source groups are removed.
    """
    if arm not in {"topic_retrieval", "edit_retrieval"}:
        raise ValueError("unknown retrieval arm")
    eval_keys = {unordered_pair(r) for r in eval_rows}
    eval_groups = {str(r["pair_id"]) for r in eval_rows}
    blocked = {str(r["pair_id"]) for r in train_rows
               if unordered_pair(r) in eval_keys or str(r["pair_id"]) in eval_groups}
    pool = [dict(r) for r in train_rows if str(r["pair_id"]) not in blocked]
    if not pool:
        raise ValueError("no train-only retrieval pool remains")
    if any(r["label"] not in LABELS for r in pool):
        raise ValueError("unknown retrieval-index label")
    ids = [str(r["instance_id"]) for r in pool]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate training instance_id")
    view = edit_representation if arm == "edit_retrieval" else topic_representation
    index = _Index(pool, view, cfg)
    provenance, counts = [], Counter()
    outputs = []
    for split, rows in (("train", pool), ("eval", eval_rows)):
        output = []
        for row in rows:
            enriched = dict(row)
            for swapped in (False, True):
                neighbors, diagnostic = index.retrieve(row, swapped=swapped)
                field = "swap_context" if swapped else "context"
                enriched[field] = _format_context(neighbors)
                counts[f"{split}_queries"] += 1
                counts[f"{split}_nonempty_contexts"] += bool(neighbors)
                counts["excluded_source_group_rows"] += diagnostic["excluded_source_group_rows"]
                counts["excluded_unordered_text_rows"] += diagnostic["excluded_unordered_text_rows"]
                if diagnostic["fallback"]:
                    counts["fallback_" + diagnostic["fallback"]] += 1
                provenance.append({
                    "split": split, "query_instance_id": str(row["instance_id"]),
                    "orientation": "swap" if swapped else "original",
                    "query_representation": view(row["text2"], row["text1"]) if swapped
                                            else view(row["text1"], row["text2"]),
                    "neighbors": [{"instance_id": str(n["instance_id"]),
                                   "pair_id": str(n["pair_id"]),
                                   "similarity": round(score, 8)} for n, score in neighbors],
                    **diagnostic,
                })
            output.append(enriched)
        outputs.append(output)
    audit = {
        "method": "train_only_tfidf_demonstration_retrieval", "arm": arm,
        "seed": int(seed), "retrieval_k": index.k,
        "index_rows": len(index.rows), "removed_train_rows": len(train_rows) - len(pool),
        "excluded_source_pair_ids": sorted(blocked), "eval_labels_used": False,
        "context_format": "identical_for_both_arms_no_instance_or_pair_ids",
        "coverage_definition": "nonempty context; not a claim of semantic relevance",
        "counts": dict(counts),
        "index_input_sha256": hashlib.sha256(json.dumps(
            [(r["instance_id"], r["text1"], r["text2"], r["label"])
             for r in index.rows], ensure_ascii=True, separators=(",", ":")).encode()).hexdigest(),
        "neighbor_provenance": provenance,
    }
    return outputs[0], outputs[1], audit
