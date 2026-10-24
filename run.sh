#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
python3 -m pip install -q --upgrade "transformers==4.48.3" "diffusers==0.31.0" "accelerate==1.0.1" "sentencepiece"
python3 -u scripts/probe_debug.py
