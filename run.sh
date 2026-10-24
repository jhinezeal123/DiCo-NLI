#!/usr/bin/env bash
# Probe run contract: caption -> generated image -> image embedding -> EQ/NO separation.
# No training, no official-data download needed: the fixed sample is in configs/probe_sample.csv.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
"$PY" --version
"$PY" -m pip install -q --upgrade \
  "transformers==4.48.3" "diffusers==0.31.0" "accelerate==1.0.1" "sentencepiece" "protobuf"

mkdir -p outputs
"$PY" -u scripts/image_probe.py
