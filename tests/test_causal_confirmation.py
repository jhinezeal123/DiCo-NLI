from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from causal_confirmation import build_confirmation_sets


class ConfirmationSplitTests(unittest.TestCase):
    def test_all_sets_are_world_disjoint_with_ordered_exclusions(self):
        plan = {"seeds": [11, 22, 33], "development_seed": 44,
                "development_families_per_split": 4, "families_per_split": 4}
        datasets, manifest = build_confirmation_sets(plan)
        groups = [{r["family_id"] for r in d["rows"]} | {r["family_id"] for r in d["demonstrations"]}
                  for d in datasets.values()]
        for i, group in enumerate(groups):
            self.assertEqual(len(group), 20)
            for other in groups[i+1:]:
                self.assertFalse(group & other)
        self.assertEqual(manifest["total_distinct_worlds_including_demonstrations"], 80)
        self.assertEqual([d["audit"]["prior_excluded_families"] for d in datasets.values()], [20, 40, 60])
        self.assertEqual(manifest, build_confirmation_sets(plan)[1])

    def test_confirmation_rejects_reusing_development_seed(self):
        with self.assertRaises(ValueError):
            build_confirmation_sets({"seeds": [44, 22, 33], "development_seed": 44})
