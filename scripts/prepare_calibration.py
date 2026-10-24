"""Package persisted base predictions without fitting models or reading dev labels."""
import argparse
import csv
import hashlib
import io
import json
import re
from pathlib import Path

OOF_RUN = "81fc6753-4916-40ae-950d-195e6ba11157"
PARENT_RUN = "13417727-b568-4efc-8fec-b67f7d0e3c03"


def extract(log, tag):
    pattern = r"<<<" + re.escape(tag) + r"\r?\n(.*?)\r?\n" + tag.split()[0] + r">>>"
    match = re.search(pattern, log, flags=re.S)
    if not match:
        raise ValueError(f"Missing prediction block: {tag}")
    return list(csv.DictReader(io.StringIO(match.group(1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", type=Path, required=True)
    args = ap.parse_args()
    dest = Path("configs/calibration")
    dest.mkdir(parents=True, exist_ok=True)
    provenance = {"oof_run": OOF_RUN, "oof_commit": "807f7fd",
                  "parent_run": PARENT_RUN, "parent_commit": "ccf60d6", "files": {}}
    logs = {}
    for run in (OOF_RUN, PARENT_RUN):
        raw = (args.logs / f"{run}.log").read_bytes()
        logs[run] = raw.decode("utf-8", errors="replace")
        provenance[f"log_sha256_{run}"] = hashlib.sha256(raw).hexdigest()
    blocks = {"train_oof.csv": extract(logs[OOF_RUN], "OOF_PROBS")}
    for seed in (42, 13, 7):
        blocks[f"dev_seed{seed}.csv"] = extract(logs[PARENT_RUN], f"DEV_PROBS seed={seed}")
    for fold in range(5):
        blocks[f"dev_fold{fold}.csv"] = extract(logs[OOF_RUN], f"DEV_PROBS fold={fold}")
    for name, rows in blocks.items():
        expected = 3042 if name == "train_oof.csv" else 660
        assert len(rows) == expected and len({r['instance_id'] for r in rows}) == expected
        # Labels are never transported with dev probabilities.
        keys = [k for k in rows[0] if k != "label" or name == "train_oof.csv"]
        path = dest / name
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        provenance["files"][name] = {"rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (dest / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
