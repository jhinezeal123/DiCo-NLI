"""Tiny real Qwen2/PEFT integration checks; no pretrained model downloads."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from peft import LoraConfig, TaskType, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForSequenceClassification

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from explore_data import clean_train, dump_rows, without_labels, validate_prepared
from explore_trainer import SemanticClassifier, collate, encode_row, run_worker


def tiny_tokenizer():
    words = ["<pad>", "<unk>", "Text", "1", "2", ":", "Relation", "a", "dog", "an", "animal",
             "large", "black", "retrieved", "example", "evidence", "canine", "cat"]
    tok = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>",
                                  pad_token="<pad>", eos_token="<pad>")


def tiny_backbone():
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                     num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                     max_position_embeddings=256, num_labels=4, pad_token_id=0,
                     attention_dropout=0.0, use_cache=False)
    cfg._attn_implementation = "sdpa"
    base = Qwen2ForSequenceClassification(cfg)
    return get_peft_model(base, LoraConfig(task_type=TaskType.SEQ_CLS, r=2,
                                         lora_alpha=4, lora_dropout=0.0,
                                         target_modules=["q_proj", "v_proj"]))


def config(arm="baseline"):
    return {"arm": arm, "pair_template": "Text 1: {t1}\nText 2: {t2}\nRelation:",
            "max_length": 64, "context_tokens": 12,
            "geometry_dim": 8, "geometry_rank": 2,
            "alignment_dim": 8, "alignment_states": 4}


def row(instance="one", text1="a dog", text2="an animal"):
    return {"instance_id": instance, "pair_id": instance, "text1": text1, "text2": text2}


class TrainerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(5)
        self.tok = tiny_tokenizer()

    def batch(self, rows, cfg):
        features = [encode_row(self.tok, r, cfg) for r in rows]
        return collate(features, range(len(features)), self.tok.pad_token_id, torch.device("cpu"))

    def test_backbone_logits_parity_and_right_padding(self):
        backbone = tiny_backbone().eval()
        model = SemanticClassifier(backbone, config()).eval()
        rows = [row(), row("two", "a large black dog", "a canine")]
        batch = self.batch(rows, config())
        with torch.no_grad():
            actual = model(**batch)
            expected = backbone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            individual = model(**self.batch(rows[:1], config()))
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.allclose(actual[:1], individual, atol=1e-6, rtol=1e-5))
        self.assertGreater(int((batch["attention_mask"] == 0).sum()), 0)
        self.assertTrue(torch.equal(batch["mask1"] * (1 - batch["attention_mask"]),
                                    torch.zeros_like(batch["mask1"])))

    def test_optimizer_reaches_lora_saved_score_and_semantic_head(self):
        for arm in ["geometry", "vector", "alignment", "uniform_alignment", "counterexample", "rationale"]:
            with self.subTest(arm=arm):
                model = SemanticClassifier(tiny_backbone(), config(arm)).train()
                batch = self.batch([row(), row("two", "a cat", "a dog")], config(arm))
                # Evidence heads require an actual signal to test all their layers.
                batch["extra"] = torch.randn(2, 8)
                opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
                before = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
                loss = torch.nn.functional.cross_entropy(model(**batch), torch.tensor([2, 3]))
                loss.backward()
                lora_grads = [p.grad for n, p in model.named_parameters() if "lora_" in n and p.grad is not None]
                score_grads = [p.grad for n, p in model.named_parameters()
                               if "score.modules_to_save" in n and p.grad is not None]
                head_grads = [p.grad for p in model.head.parameters() if p.grad is not None]
                self.assertGreater(sum(float(g.abs().sum()) for g in lora_grads), 0.)
                self.assertGreater(sum(float(g.abs().sum()) for g in score_grads), 0.)
                self.assertGreater(sum(float(g.abs().sum()) for g in head_grads), 0.)
                self.assertTrue(all(torch.isfinite(g).all() for g in lora_grads + score_grads + head_grads))
                opt.step()
                self.assertTrue(any(not torch.equal(before[n], p.detach()) for n, p in model.named_parameters()
                                    if n in before and "lora_" in n))
                self.assertTrue(any(not torch.equal(before[n], p.detach()) for n, p in model.named_parameters()
                                    if n in before and n.startswith("head.")))

    def test_context_and_swapping_preserve_phrase_masks(self):
        item = row()
        item.update(context="retrieved example evidence " * 30, swap_context="a canine example " * 20)
        cfg = config("relation")
        cfg.update(max_length=32, context_tokens=8)
        plain = encode_row(self.tok, row(), cfg)
        normal = encode_row(self.tok, item, cfg)
        swapped = encode_row(self.tok, item, cfg, swapped=True)
        self.assertLessEqual(len(normal["input_ids"]), 32)
        self.assertEqual(normal["input_ids"][-len(plain["input_ids"]):], plain["input_ids"])
        self.assertFalse(any(normal["mask1"][:8]))
        self.assertFalse(any(normal["mask2"][:8]))
        for encoded, expected1, expected2 in [(normal, "a dog", "an animal"),
                                               (swapped, "an animal", "a dog")]:
            first = [i for i, mask in zip(encoded["input_ids"], encoded["mask1"]) if mask]
            second = [i for i, mask in zip(encoded["input_ids"], encoded["mask2"]) if mask]
            self.assertEqual(first, self.tok(expected1, add_special_tokens=False)["input_ids"])
            self.assertEqual(second, self.tok(expected2, add_special_tokens=False)["input_ids"])

    def test_input_features_are_independent_of_labels_and_metadata(self):
        item = row()
        cfg = config()
        expected = encode_row(self.tok, item, cfg)
        enriched = dict(item, label="FORWARD_ENTAILMENT", reverse_pair_id="secret", hidden_gold="other")
        self.assertEqual(expected, encode_row(self.tok, enriched, cfg))
        enriched["label"] = "NEGATIVE_OTHER"
        self.assertEqual(expected, encode_row(self.tok, enriched, cfg))
        sanitized = without_labels([enriched])[0]
        self.assertEqual(set(sanitized), {"instance_id", "pair_id", "text1", "text2"})

    def test_eval_gold_fails_before_model_initialization(self):
        train = dict(row("train", "a cat", "an animal"), label="FORWARD_ENTAILMENT")
        evaluation = dict(row("eval"), label="FORWARD_ENTAILMENT")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dump_rows(root / "train.jsonl", [train])
            dump_rows(root / "eval.jsonl", [evaluation])
            with mock.patch("explore_trainer.build") as build:
                with self.assertRaises(AssertionError):
                    run_worker(config(), 42, root / "train.jsonl", root / "eval.jsonl", root / "out")
                build.assert_not_called()

    def test_text_overlap_removes_entire_source_group_without_eval_gold(self):
        first = dict(row("one", " A   DOG ", "An Animal"), pair_id="source", label="FORWARD_ENTAILMENT")
        mirror = dict(row("two", "An Animal", "A DOG"), pair_id="source", label="BACKWARD_ENTAILMENT")
        unrelated = dict(row("three", "a cat", "a canine"), label="NEGATIVE_OTHER")
        evaluation = [row("dev", "an animal", "a dog")]
        cleaned, audit = clean_train([first, mirror, unrelated], evaluation)
        self.assertEqual([r["instance_id"] for r in cleaned], ["three"])
        self.assertEqual(audit["removed_rows"], 2)
        validate_prepared(cleaned, evaluation)


if __name__ == "__main__":
    unittest.main()
