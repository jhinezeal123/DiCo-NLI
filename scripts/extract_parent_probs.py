"""Extract per-seed bidirectional dev probabilities from a finished run log (DEV_PROBS blocks).

Output columns: instance_id, seed, then one probability per label in dico_trainer.LABELS order.
bidir = 0.5 * (p + q[:, REV]), exactly as run_pipeline.py decodes.
Usage: python scripts/extract_parent_probs.py <run.log> <out.csv>
"""
import csv
import io
import re
import sys

LABELS = ["BACKWARD_ENTAILMENT", "EQUIVALENCE", "FORWARD_ENTAILMENT", "NEGATIVE_OTHER"]
REV = {"BACKWARD_ENTAILMENT": "FORWARD_ENTAILMENT", "EQUIVALENCE": "EQUIVALENCE",
       "FORWARD_ENTAILMENT": "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER": "NEGATIVE_OTHER"}

log, out = sys.argv[1:3]
rows, seed, buf = [], None, None
for line in open(log, encoding="utf-8", errors="replace"):
    line = line.rstrip("\n")
    m = re.match(r"<<<DEV_PROBS seed=(\d+)", line)
    if m:
        seed, buf = int(m.group(1)), []
    elif line == "DEV_PROBS>>>" and buf is not None:
        for r in csv.DictReader(io.StringIO("\n".join(buf))):
            rows.append([r["instance_id"], seed] + [
                f"{0.5 * (float(r['p_' + l]) + float(r['q_' + REV[l]])):.6f}" for l in LABELS])
        buf = None
    elif buf is not None:
        buf.append(line)
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["instance_id", "seed"] + LABELS)
    w.writerows(rows)
print(f"wrote {len(rows)} rows")
