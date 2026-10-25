"""Build all preregistered world-disjoint confirmation sets in fixed order."""
from causal_dataset_v2 import build_dataset_v2


def build_confirmation_sets(plan):
    seeds = plan["seeds"]
    if len(seeds) != len(set(seeds)) or plan["development_seed"] in seeds:
        raise ValueError("confirmation seeds must be unique and distinct from development")
    seen = set()
    development = build_dataset_v2(plan["development_seed"], plan["development_families_per_split"], seen)
    datasets = {seed: build_dataset_v2(seed, plan["families_per_split"], seen) for seed in seeds}
    manifest = {"plan": plan, "development_dataset_sha256": development["audit"]["dataset_sha256"],
                "total_distinct_worlds_including_demonstrations": len(seen),
                "confirmation": {str(seed): {"dataset_sha256": data["audit"]["dataset_sha256"],
                                             "cases": len(data["rows"]),
                                             "prior_excluded_worlds": data["audit"]["prior_excluded_families"]}
                                 for seed, data in datasets.items()},
                "model_predictions_used": False}
    return datasets, manifest
