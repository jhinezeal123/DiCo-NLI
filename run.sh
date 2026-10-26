#!/usr/bin/env bash
# Fixed run contract for every experiment node: `bash run.sh`.
# Behaviour is driven only by committed code/config (configs/experiment.json).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [ -f configs/causal_dico_transfer.json ]; then
  PY="${PYTHON:-python3}"
  "$PY" -m pip install -q --only-binary=:all: -r requirements-causal-sft.txt
  exec "$PY" -u scripts/evaluate_causal_dico_transfer.py
fi

if [ -f configs/causal_replay.json ]; then
  exec "${PYTHON:-python3}" -u scripts/replay_causal_normalization.py --config configs/causal_replay.json
fi

if [ -f configs/causal_semantic_sft.json ]; then
  PY="${PYTHON:-python3}"
  "$PY" -m pip install -q --only-binary=:all: -r requirements-causal-sft.txt
  exec "$PY" -u scripts/train_intervention_semantics.py
fi

# Causal research uses the same fixed command and an independent committed recipe.
if [ -f configs/causal_v3.json ]; then
  PY="${PYTHON:-python3}"
  "$PY" -m pip install -q --only-binary=:all: -r requirements-causal.txt
  exec "$PY" -u scripts/causal_v3_pipeline.py
fi

if [ -f configs/causal_experiment.json ]; then
  PY="${PYTHON:-python3}"
  "$PY" --version
  "$PY" -m pip install -q --only-binary=:all: -r requirements-causal.txt
  exec "$PY" -u scripts/causal_pipeline.py --config configs/causal_experiment.json
fi

UPSTREAM_URL="https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI.git"
UPSTREAM_SHA="588968e610197ddc4c440314701cbc587afc4c1b"
UPSTREAM_DIR="$ROOT/external/dico_nli_official"

mkdir -p external outputs
if [ ! -d "$UPSTREAM_DIR/.git" ]; then
  git clone -q "$UPSTREAM_URL" "$UPSTREAM_DIR"
fi
git -C "$UPSTREAM_DIR" fetch -q origin "$UPSTREAM_SHA" 2>/dev/null || true
git -C "$UPSTREAM_DIR" checkout -q "$UPSTREAM_SHA"
echo "upstream_sha: $(git -C "$UPSTREAM_DIR" rev-parse HEAD)"

PY="${PYTHON:-python3}"
"$PY" --version
"$PY" -m pip install -q --only-binary=:all: -r requirements-run.txt

"$PY" -u scripts/run_pipeline.py --config configs/experiment.json --upstream "$UPSTREAM_DIR"
