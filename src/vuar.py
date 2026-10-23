"""Verifiable Unit-Alignment Reasoning (VUAR): trace language, parser, verifier, prompts.

A trace explains a phrase pair as aligned information units, each with a relation:
  EQ    same meaning
  SPEC  text1 unit is more specific than the text2 unit (text1 unit implies text2 unit)
  GEN   text1 unit is more general than the text2 unit
  X1    extra information only in text1
  X2    extra information only in text2
  CONF  mutually exclusive / contradictory units
  LOOSE related or similar, but neither implies the other (hedges, different prepositions, ...)
The label is computed from the units by a fixed rule (compose), never generated.
Swapping the input mirrors the trace (mirror) and maps FE <-> BE exactly.
This module is torch-free so it can be unit-tested anywhere.
"""
import re

REL_SET = ("EQ", "SPEC", "GEN", "X1", "X2", "CONF", "LOOSE")
MIRROR = {"EQ": "EQ", "SPEC": "GEN", "GEN": "SPEC", "X1": "X2", "X2": "X1", "CONF": "CONF", "LOOSE": "LOOSE"}
LABEL_REV = {"EQUIVALENCE": "EQUIVALENCE", "FORWARD_ENTAILMENT": "BACKWARD_ENTAILMENT",
             "BACKWARD_ENTAILMENT": "FORWARD_ENTAILMENT", "NEGATIVE_OTHER": "NEGATIVE_OTHER"}
LABEL_DESC = {
    "EQUIVALENCE": "EQUIVALENCE (both phrases entail each other)",
    "FORWARD_ENTAILMENT": "FORWARD_ENTAILMENT (text1 entails text2, but not conversely)",
    "BACKWARD_ENTAILMENT": "BACKWARD_ENTAILMENT (text2 entails text1, but not conversely)",
    "NEGATIVE_OTHER": "NEGATIVE_OTHER (neither direction entails)",
}

_UNIT = re.compile(r'(?:"([^"]*)"|-)\s*~\s*(?:"([^"]*)"|-)\s*=\s*(EQ|SPEC|GEN|X1|X2|CONF|LOOSE)\b')


def parse(text):
    """Parse the first 'units:' line (or the whole text) into [(a, b, rel)], '' meaning empty."""
    if text is None:
        return []
    m = re.search(r"units:(.*)", text, flags=re.IGNORECASE)
    body = m.group(1) if m else text
    body = body.split("\n")[0]
    return [(a or "", b or "", r) for a, b, r in _UNIT.findall(body)]


def format_units(units):
    q = lambda s: '"%s"' % s.replace('"', "'") if s else "-"
    return " | ".join(f"{q(a)} ~ {q(b)} = {r}" for a, b, r in units)


def mirror(units):
    return [(b, a, MIRROR[r]) for a, b, r in units]


def compose(units):
    """Deterministic label from units; None if there are no units."""
    if not units:
        return None
    rels = {r for _, _, r in units}
    fwd = rels <= {"EQ", "SPEC", "X1"}
    bwd = rels <= {"EQ", "GEN", "X2"}
    if fwd and bwd:
        return "EQUIVALENCE"
    if fwd:
        return "FORWARD_ENTAILMENT"
    if bwd:
        return "BACKWARD_ENTAILMENT"
    return "NEGATIVE_OTHER"


def _norm(s):
    return re.sub(r"\s+", " ", s.lower()).strip()


def grounded(units, t1, t2):
    """Every non-empty unit side must be copied from its phrase and the relation must fit empty sides."""
    n1, n2 = _norm(t1), _norm(t2)
    for a, b, r in units:
        if a and _norm(a) not in n1:
            return False
        if b and _norm(b) not in n2:
            return False
        if not a and not b:
            return False
        if (not a or not b) and r in ("EQ", "SPEC", "GEN"):
            return False  # an empty side cannot be equal / more specific / more general
        if (r == "X1" and b) or (r == "X2" and a) or (r in ("X1", "X2") and not (a or b)):
            return False  # X1/X2 mean the other side is empty
        if r == "X1" and not a or r == "X2" and not b:
            return False
    return True


def accept(units, t1, t2, gold=None):
    if not units or not grounded(units, t1, t2):
        return False
    return gold is None or compose(units) == gold


INSTRUCTIONS = """You analyse the relation between two short English phrases (text1, text2) by splitting them into aligned information units.
For every unit write: "<unit copied from text1>" ~ "<unit copied from text2>" = RELATION, using - for an empty side. Join units with " | ".
Relations:
EQ    same meaning
SPEC  the text1 unit is more specific than the text2 unit (it implies it), e.g. "a dog" ~ "an animal"
GEN   the text1 unit is more general than the text2 unit
X1    extra information that exists only in text1 (text2 side is -)
X2    extra information that exists only in text2 (text1 side is -)
CONF  the units contradict each other or are different, mutually exclusive things (black vs blue, a carabiner vs a pulley)
LOOSE related or similar, but neither implies the other (hedges such as "reportedly", prepositions with different meaning, nearby but different concepts)
Rules: copy units exactly from the phrases; cover all content words; output a single line that starts with "units:" and nothing else.
The final relation is computed from your units: text1 entails text2 if every unit is EQ, SPEC or X1; text2 entails text1 if every unit is EQ, GEN or X2; any CONF or LOOSE unit blocks both directions."""

# (text1, text2, label, units): every example is checked by compose() in the unit tests.
FEW_SHOT = [
    ("The white sheep", "A sheep", "FORWARD_ENTAILMENT",
     '"The" ~ "A" = SPEC | "white" ~ - = X1 | "sheep" ~ "sheep" = EQ'),
    ("asylum", "humanitarian asylum", "BACKWARD_ENTAILMENT",
     '"asylum" ~ "asylum" = EQ | - ~ "humanitarian" = X2'),
    ("are seated", "sitting", "EQUIVALENCE", '"are seated" ~ "sitting" = EQ'),
    ("A brown and white dog", "A white and brown dog", "EQUIVALENCE",
     '"A" ~ "A" = EQ | "brown and white" ~ "white and brown" = EQ | "dog" ~ "dog" = EQ'),
    ("Black bird", "Blue bird", "NEGATIVE_OTHER", '"Black" ~ "Blue" = CONF | "bird" ~ "bird" = EQ'),
    ("reportedly dead", "dead", "NEGATIVE_OTHER", '"reportedly" ~ - = LOOSE | "dead" ~ "dead" = EQ'),
    ("some dirt bikes", "a bicycle", "FORWARD_ENTAILMENT", '"some dirt bikes" ~ "a bicycle" = SPEC'),
    ("of breast cancer", "about breast cancer", "NEGATIVE_OTHER",
     '"of" ~ "about" = LOOSE | "breast cancer" ~ "breast cancer" = EQ'),
    ("five cows", "cows", "FORWARD_ENTAILMENT", '"five" ~ - = X1 | "cows" ~ "cows" = EQ'),
    ("puts on notice", "warned", "EQUIVALENCE", '"puts on notice" ~ "warned" = EQ'),
]


def teacher_prompt(t1, t2, gold=None):
    parts = [INSTRUCTIONS, "", "Examples:"]
    for a, b, lab, units in FEW_SHOT:
        parts.append(f'text1: {a}\ntext2: {b}' + (f"\nannotated relation: {LABEL_DESC[lab]}" if gold else "")
                     + f"\nunits: {units}\n")
    parts.append("Now analyse:")
    parts.append(f"text1: {t1}\ntext2: {t2}")
    if gold:
        parts.append(f"annotated relation: {LABEL_DESC[gold]}\n(Choose units that explain this annotation.)")
    parts.append("units:")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Stage 1b: deterministic alignment; the teacher only judges replaced spans.
# ---------------------------------------------------------------------------
import difflib
import itertools

SPAN_RELS = ("EQ", "SPEC", "GEN", "CONF", "LOOSE")
LETTER = {"A": "EQ", "B": "SPEC", "C": "GEN", "D": "CONF", "E": "LOOSE"}
REL_LETTER = {v: k for k, v in LETTER.items()}


def _tok(w):
    return re.sub(r"[^\w]", "", w.lower()) or w.lower()


def diff_units(t1, t2):
    """Deterministic alignment into [(span1, span2, kind)], kind in EQ / X1 / X2 / REP.

    EQ  = words equal up to case and punctuation, X1 = only in text1, X2 = only in text2,
    REP = a replaced span whose relation must be judged.
    """
    a, b = t1.split(), t2.split()
    ops = difflib.SequenceMatcher(None, [_tok(w) for w in a], [_tok(w) for w in b], autojunk=False).get_opcodes()
    units = []
    for tag, i1, i2, j1, j2 in ops:
        sa, sb = " ".join(a[i1:i2]), " ".join(b[j1:j2])
        units.append((sa, sb, {"equal": "EQ", "delete": "X1", "insert": "X2"}.get(tag, "REP")))
    return _collapse_paraphrase(_merge_moves(units))


def _collapse_paraphrase(units):
    """Words only in text1 AND words only in text2 form a paraphrase region whose relation must be judged:
    collapse all X1/X2 units into one REP unit (placed at the first of them)."""
    x = [i for i, (_, _, k) in enumerate(units) if k in ("X1", "X2")]
    if not any(units[i][2] == "X1" for i in x) or not any(units[i][2] == "X2" for i in x):
        return units
    merged = (" ".join(units[i][0] for i in x if units[i][0]), " ".join(units[i][1] for i in x if units[i][1]), "REP")
    return [merged if i == x[0] else u for i, u in enumerate(units) if i not in x or i == x[0]]


def _merge_moves(units):
    """If the words only present in text1 are exactly the words only present in text2 (a pure reordering),
    replace those X1/X2 units by one EQ unit."""
    x1 = [i for i, (_, _, k) in enumerate(units) if k == "X1"]
    x2 = [i for i, (_, _, k) in enumerate(units) if k == "X2"]
    if not x1 or not x2 or any(k == "REP" for _, _, k in units):
        return units
    w1 = sorted(_tok(w) for i in x1 for w in units[i][0].split())
    w2 = sorted(_tok(w) for i in x2 for w in units[i][1].split())
    if w1 != w2:
        return units
    first = min(x1 + x2)
    merged = (" ".join(units[i][0] for i in x1), " ".join(units[i][1] for i in x2), "EQ")
    return [merged if i == first else u for i, u in enumerate(units) if i not in x1 + x2 or i == first]


def rep_indices(units):
    return [i for i, (_, _, k) in enumerate(units) if k == "REP"]


def build_trace(units, rels):
    """rels: one relation (from SPAN_RELS) per REP unit, in order. Returns [(a, b, rel)]."""
    it = iter(rels)
    return [(a, b, next(it) if k == "REP" else k) for a, b, k in units]


def best_assignment(units, scores, gold=None, max_reps=5):
    """Pick relations for the REP units.

    scores: list (one per REP unit, same order) of {rel: log-prob}.
    gold=None: per-span argmax. gold set: the highest-scoring assignment whose composition equals gold,
    or None if the gold label is infeasible for this alignment.
    """
    k = len(rep_indices(units))
    if k == 0:
        trace = build_trace(units, [])
        return trace if gold is None or compose(trace) == gold else None
    if gold is None:
        return build_trace(units, [max(sc, key=sc.get) for sc in scores])
    if k > max_reps:
        return None
    best, best_s = None, None
    for combo in itertools.product(SPAN_RELS, repeat=k):
        trace = build_trace(units, combo)
        if compose(trace) != gold:
            continue
        s = sum(scores[n][r] for n, r in enumerate(combo))
        if best_s is None or s > best_s:
            best, best_s = trace, s
    return best


SPAN_INSTRUCTIONS = """Two short English phrases differ in one part. Decide how the differing part of phrase 1 relates to the differing part of phrase 2, taking the phrases as context.
A  same meaning (equivalent)
B  the part of phrase 1 is more specific than the part of phrase 2 (it implies it), e.g. "a dog" vs "an animal"
C  the part of phrase 1 is more general than the part of phrase 2
D  they contradict each other or are different, mutually exclusive things (black vs blue, a carabiner vs a pulley)
E  related or similar, but neither implies the other (different prepositions, singular vs plural, nearby but different concepts, hedges)
Answer with a single letter."""

SPAN_FEW_SHOT = [
    ("Black bird", "Blue bird", "Black", "Blue", "D"),
    ("are standing", "is standing", "are", "is", "E"),
    ("a carabiner", "a pulley", "carabiner", "pulley", "D"),
    ("of breast cancer", "about breast cancer", "of", "about", "E"),
    ("some dirt bikes", "a bicycle", "some dirt bikes", "a bicycle", "B"),
    ("a bicycle", "some dirt bikes", "a bicycle", "some dirt bikes", "C"),
    ("puts on notice", "warned", "puts on notice", "warned", "A"),
    ("The white sheep", "A sheep", "The white", "A", "B"),
    ("are seated", "sitting", "are seated", "sitting", "A"),
]


def span_prompt(t1, t2, a, b):
    parts = [SPAN_INSTRUCTIONS, ""]
    for p1, p2, sa, sb, ans in SPAN_FEW_SHOT:
        parts.append(f'Phrase 1: {p1}\nPhrase 2: {p2}\nDiffering part of phrase 1: "{sa}"\n'
                     f'Differing part of phrase 2: "{sb}"\nAnswer: {ans}\n')
    parts.append(f'Phrase 1: {t1}\nPhrase 2: {t2}\nDiffering part of phrase 1: "{a}"\n'
                 f'Differing part of phrase 2: "{b}"\nAnswer:')
    return "\n".join(parts)
