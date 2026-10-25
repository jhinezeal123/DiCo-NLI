import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from causal_v3_confirmation import confirmation_datasets, shard_dataset, prepare_dataset


class ConfirmationIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.development,cls.datasets=confirmation_datasets()

    def test_all_shards_are_disjoint_and_reconstruct_exact_full_seed(self):
        seen={r["family_id"] for r in self.development["rows"]}
        for dataset in self.datasets.values():
            reconstructed=[]
            for index in range(4):
                shard=shard_dataset(dataset,index)
                self.assertEqual(len(shard["rows"]),192)
                self.assertEqual(shard["demonstrations"],dataset["demonstrations"])
                for split in ("id","ood_topology","ood_mechanism","ood_language"):
                    rows=[r for r in shard["rows"] if r["split"]==split]
                    self.assertEqual(len({r["family_id"] for r in rows}),12)
                    for label in ("0","1","unknown"):
                        self.assertEqual(sum(r["view"]=="intervention" and r["answer"]==label for r in rows),4)
                reconstructed.extend(shard["rows"])
            self.assertEqual(len({r["case_id"] for r in reconstructed}),768)
            self.assertEqual(sorted(reconstructed,key=lambda r:r["case_id"]),sorted(dataset["rows"],key=lambda r:r["case_id"]))
            current={r["family_id"] for r in reconstructed}
            self.assertFalse(current&seen); seen.update(current)

    def test_frozen_config_rejects_replay_or_prompt_changes(self):
        cfg=json.loads(Path("configs/causal_v3.json").read_text())
        for key,value in (("control_replay",{"invalid":True}),("shots",4),("max_new_tokens",256)):
            bad=copy.deepcopy(cfg); bad[key]=value
            with self.assertRaises(ValueError): prepare_dataset(bad)
