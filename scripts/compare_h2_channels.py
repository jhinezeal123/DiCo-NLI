"""Score a pre-specified H2 comparison from three completed, logged runs.

The logs are the H3 repeat-control classifier, the same-data likelihood channel,
and the same-data contrastive channel. Both channels use the same seed as the
reference and the fusion is fixed at 50/50, without fitting on dev labels.
The scorer is the pinned organizer scorer, not a local metric implementation.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

LABELS = ["BACKWARD_ENTAILMENT", "EQUIVALENCE", "FORWARD_ENTAILMENT", "NEGATIVE_OTHER"]
REVERSE = [2, 1, 0, 3]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_run(path, seed):
    content = Path(path).read_text(encoding="utf-8", errors="replace")
    if "FINAL_METRICS decoding=bidir" not in content:
        raise ValueError(f"completed summary missing in {path}")
    hashes = re.findall(r"augmented_sha256=([0-9a-f]{64})", content)
    if len(hashes) != 1:
        raise ValueError(f"expected one augmented training hash in {path}, got {len(hashes)}")
    match = re.search(rf"^<<<DEV_PROBS seed={seed}\s*\n(.*?)^DEV_PROBS>>>\s*$",
                      content, re.MULTILINE | re.DOTALL)
    if not match:
        raise ValueError(f"seed {seed} probability block missing in {path}")
    rows = list(csv.DictReader(io.StringIO(match.group(1))))
    probs = {}
    for row in rows:
        key = row["instance_id"]
        if key in probs:
            raise ValueError(f"duplicate id {key} in {path}")
        p = [float(row['p_' + label]) for label in LABELS]
        q = [float(row['q_' + label]) for label in LABELS]
        if (not all(math.isfinite(x) and 0 <= x <= 1 for x in p + q)
                or abs(sum(p) - 1) > 1e-4 or abs(sum(q) - 1) > 1e-4):
            raise ValueError(f"invalid probabilities for {key} in {path}")
        probs[key] = [(p[j] + q[REVERSE[j]]) / 2 for j in range(4)]
    return probs, hashes[0]


def score(upstream, reference, probs, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    pred = output_dir / 'predictions.csv'
    with pred.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['instance_id', 'label'])
        for instance_id, p in probs.items():
            writer.writerow([instance_id, LABELS[max(range(4), key=lambda i: p[i])]])
    result = output_dir / 'scores'
    subprocess.run([sys.executable, '-m', 'evaluation_functions', '--gold', str(reference),
                    '--predictions', str(pred), '--output-dir', str(result)],
                   cwd=upstream, check=True, capture_output=True, text=True)
    data = json.loads((result / 'scores.json').read_text(encoding='utf-8'))
    return {name: data[name] for name in ('weighted_f1', 'soft_cons', 'hard_cons')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-log', required=True, type=Path)
    parser.add_argument('--likelihood-log', required=True, type=Path)
    parser.add_argument('--contrastive-log', required=True, type=Path)
    parser.add_argument('--upstream', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    paths = {'reference': args.reference_log, 'likelihood': args.likelihood_log,
             'contrastive': args.contrastive_log}
    parsed = {name: read_run(path, args.seed) for name, path in paths.items()}
    if len({item[1] for item in parsed.values()}) != 1:
        raise ValueError('the training data hashes differ across runs')
    arrays = {name: item[0] for name, item in parsed.items()}
    ids = set(arrays['reference'])
    if not ids or any(set(p) != ids for p in arrays.values()):
        raise ValueError('dev instance ids do not match across runs')
    upstream = args.upstream.resolve()
    reference = (upstream / 'final_data/dev/dico_nli_dev_track1_reference.csv').resolve()
    with reference.open(encoding='utf-8', newline='') as f:
        gold_ids = [r['instance_id'] for r in csv.DictReader(f)]
    if len(gold_ids) != len(ids) or set(gold_ids) != ids:
        raise ValueError('probability ids do not match the official reference')
    for name in ('likelihood', 'contrastive'):
        arrays[name + '_fusion'] = {
            key: [(a + b) / 2 for a, b in zip(arrays['reference'][key], arrays[name][key])]
            for key in gold_ids
        }
    with tempfile.TemporaryDirectory() as tmp:
        metrics = {name: score(upstream, reference, probs, Path(tmp) / name)
                   for name, probs in arrays.items()}
    result = {'seed': args.seed, 'channel_fusion_weight': 0.5, 'rows': len(ids),
              'train_augmented_sha256': parsed['reference'][1],
              'reference_sha256': digest(reference),
              'log_sha256': {name: digest(path) for name, path in paths.items()},
              'metrics': metrics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
