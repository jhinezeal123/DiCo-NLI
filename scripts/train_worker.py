"""Train one seed on the single GPU made visible to this process; save dev probabilities.

Launched by run_pipeline.py with CUDA_VISIBLE_DEVICES set, so each seed runs exactly as it
would alone on one GPU (same numerics) while several seeds share the machine's GPUs.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import dico_trainer as dt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"worker seed={args.seed} device={device}"
          + (f" gpu={torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""), flush=True)
    probs = dt.run(cfg, args.seed, args.train, args.dev, device)
    np.savez(args.out, **probs)


if __name__ == "__main__":
    main()
