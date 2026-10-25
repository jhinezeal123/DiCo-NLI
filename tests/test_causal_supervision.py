import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from causal_supervision import build_supervision, training_pair
from train_causal_native import persist_adapter


class TinyTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        result = "".join("["+m["role"]+"]"+m["content"]+"[end]" for m in messages)
        return result+"[assistant]" if add_generation_prompt else result

    def __call__(self, text, add_special_tokens):
        return {"input_ids": list(text.encode())}


class SupervisionTests(unittest.TestCase):
    def fixture(self):
        models, questions = [], []
        for model_id in (1, 2, 3, 4):
            models.append({"model_id": model_id, "background": "A affects B.",
                "params": {"p(X)": .17, "p(Y | X)": [1, 0]},
                "variable_mapping": {"Xname": "A", "X1": "A on", "X0": "A off", "Yname": "B", "Y1": "B on", "Y0": "B off"}})
            for action in (0, 1):
                questions.append({"question_id": model_id*100+action, "given_info": "A on causes B off.",
                    "question": "Would B be on if A were " + str(action) + "?", "answer": "yes" if action == 0 else "no",
                    "meta": {"query_type": "det-counterfactual", "model_id": model_id,
                             "given_info": {}, "treatment": "X", "action": action, "outcome": "Y", "polarity": 1}})
        plan = {"training_model_ids_if_sft": [1], "development_model_ids": [2], "pilot_model_ids": [3],
                "final_cases": [{"model_id": 4}], "training_questions_if_sft": 2}
        return questions, models, plan

    def test_training_never_reads_heldout_programs_or_labels(self):
        questions, models, plan = self.fixture()
        expected = build_supervision(questions, models, plan, 2, 42)
        for q in questions:
            if q["meta"]["model_id"] != 1:
                q["answer"] = "SECRET_LABEL"; q["meta"]["given_info"] = "SECRET_EVIDENCE"
        for m in models:
            if m["model_id"] != 1:
                m["params"] = "SECRET_CPT"; m["variable_mapping"] = "SECRET_MAPPING"
        self.assertEqual(expected, build_supervision(questions, models, plan, 2, 42))
        self.assertEqual(expected[0][0]["program"]["params"]["p(A)"], .5)
        self.assertNotIn("SECRET", json.dumps(expected))

    def test_overlap_and_incorrect_teacher_labels_are_rejected(self):
        questions, models, plan = self.fixture()
        with self.assertRaises(ValueError):
            build_supervision(questions, models, {**plan, "training_model_ids_if_sft": [1, 3]}, 2, 42)
        questions[0]["answer"] = "no"
        with self.assertRaises(ValueError):
            build_supervision(questions, models, plan, 2, 42)

    def test_loss_masks_prompt_and_oversize_drops_whole_style_pair(self):
        questions, models, plan = self.fixture()
        row = build_supervision(questions, models, plan, 1, 42)[0][0]
        pair = training_pair(TinyTokenizer(), row, 5000)
        self.assertEqual([f["method"] for f in pair], ["direct", "native_parser_solver"])
        for feature in pair:
            first = next(i for i, label in enumerate(feature["labels"]) if label != -100)
            self.assertTrue(all(label == -100 for label in feature["labels"][:first]))
            completion = bytes(feature["input_ids"][first:]).decode()
            self.assertTrue(completion.startswith(("<answer>", "{")))
            self.assertEqual(feature["labels"][first:], feature["input_ids"][first:])
        self.assertIsNone(training_pair(TinyTokenizer(), row, 10))

    def test_persisted_adapter_chunks_round_trip_and_hash(self):
        emitted = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/"adapter_config.json").write_text('{"r":8}')
            (path/"adapter_model.safetensors").write_bytes(b"sample-adapter-only"*100)
            manifest = persist_adapter(path, lambda tag, value: emitted.append((tag, value)))
            chunks = [value for tag, value in emitted if tag == "SFT_ADAPTER_CHUNK"]
            self.assertEqual([c["index"] for c in chunks], list(range(manifest["chunks"])))
            raw = base64.b64decode("".join(c["base64"] for c in chunks))
            self.assertEqual(hashlib.sha256(raw).hexdigest(), manifest["sha256"])
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                self.assertEqual(archive.read("adapter_model.safetensors"), b"sample-adapter-only"*100)
