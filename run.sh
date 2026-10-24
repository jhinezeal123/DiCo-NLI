#!/usr/bin/env bash
# Fixed run contract for every experiment node: `bash run.sh`.
# Behaviour is driven only by committed code/config (configs/experiment.json).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

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

# These pilots use FP16 LoRA, not torchao quantization. Some fresh Kaggle images
# ship torchao with an API incompatible with PEFT 0.14's optional dispatcher.
# Remove that unused optional backend before preflight and model construction.
"$PY" -m pip uninstall -y torchao

"$PY" -u scripts/run_pipeline.py --config configs/experiment.json --upstream "$UPSTREAM_DIR"
