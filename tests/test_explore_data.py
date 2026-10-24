import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from explore_data import clean_train, validate_prepared, without_labels


def row(i, group, a, b, label="EQUIVALENCE"):
    return {"instance_id":i,"pair_id":group,"text1":a,"text2":b,"label":label}


class ExplorationDataTests(unittest.TestCase):
    def test_removes_entire_group_on_reversed_whitespace_overlap(self):
        train=[row("a","g1","  Red   CAR ","Vehicle"),row("b","g1","unrelated","member"),
               row("c","g2","a cat","a feline")]
        dev=[row("d","heldout","vehicle","red car")]
        cleaned,audit=clean_train(train,dev)
        self.assertEqual([r["instance_id"] for r in cleaned],["c"])
        self.assertEqual(audit["removed_rows"],2)
        self.assertEqual(len(train),3)

    def test_worker_input_allowlist_strips_all_gold_and_reversal_metadata(self):
        source=row("a","g","cat","feline")
        source.update(reverse_pair_id="b",score=1.,hidden_label="EQUIVALENCE")
        result=without_labels([source])
        self.assertEqual(set(result[0]),{"instance_id","pair_id","text1","text2"})
        validate_prepared([row("tr","train","dog","canine")],result)

    def test_prepared_rejects_eval_gold_duplicate_ids_and_text_leakage(self):
        train=[row("a","g","cat","feline")]
        with self.assertRaises(AssertionError):
            validate_prepared(train,[row("b","x","vehicle","car")])
        with self.assertRaises(AssertionError):
            validate_prepared(train,without_labels([row("b","x","feline","cat")]))
        with self.assertRaises(AssertionError):
            validate_prepared(train+train,without_labels([row("b","x","vehicle","car")]))


if __name__ == "__main__":
    unittest.main()
