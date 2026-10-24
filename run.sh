#!/usr/bin/env bash
# Fixed run contract for every experiment node: `bash run.sh`.
# Behaviour is driven only by committed code/config.
#
# This node trains two views of the same backbone (4-way sequence-classification head and
# LM-head label-word readout), each with the frozen LS 0.1 + NEG-0.8 recipe and 3 seeds,
# then fuses their bidirectional probability vectors. See scripts/fuse_views.py.
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

echo ""
echo "===== VIEW 1/2: 4-way sequence-classification head ====="
"$PY" -u scripts/run_pipeline.py --config configs/experiment.json --upstream "$UPSTREAM_DIR"

echo ""
echo "===== VIEW 2/2: LM-head label-word readout ====="
"$PY" -u scripts/run_pipeline.py --config configs/experiment_lmtokens.json --upstream "$UPSTREAM_DIR"

echo ""
echo "===== FUSE VIEWS ====="
"$PY" -u scripts/fuse_views.py --config configs/experiment.json --upstream "$UPSTREAM_DIR" \
  --out outputs/fused_two_views \
  --views "outputs/track1_ens_seqcls_ls0p1_negw0p8" "outputs/track1_ens_lmtokens_ls0p1_negw0p8"
