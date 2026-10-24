"""One immutable arm/seed task; the supervisor assigns a single visible GPU."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if __name__ == "__main__":
    from explore_trainer import run_worker
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run_worker(json.loads(Path(args.config).read_text()), args.seed, args.train, args.eval, args.out)
