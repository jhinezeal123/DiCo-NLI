"""CPU-only checks for GPU safeguards and prompt isolation."""
import json
import sys
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gguf_pipeline as pipeline


class GgufChecks(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((Path(__file__).resolve().parents[1] / "configs/experiment.json").read_text())
        self.log = ("load_tensors: CUDA0 model buffer size = 4200.00 MiB\n"
                    "load_tensors: CUDA1 model buffer size = 4300.00 MiB\n"
                    "load_tensors: offloaded 65/65 layers to GPU\n")

    def test_both_devices_and_full_offload_required(self):
        result = pipeline.verify_sharding(self.log, self.cfg)
        self.assertEqual(set(result["model_buffer_MiB"]), {0, 1})
        # User clarified that using both GPUs for throughput takes priority over
        # equal memory use; unequal mixed-precision layers must remain valid.
        pipeline.verify_sharding(self.log.replace("4300.00", "8000.00"), self.cfg)
        for bad in (self.log.replace("CUDA1", "CPU"),
                    self.log.replace("4300.00", "0.00"),
                    self.log.replace("65/65", "64/65")):
            with self.subTest(log=bad), self.assertRaises(RuntimeError):
                pipeline.verify_sharding(bad, self.cfg)

    def test_prompts_cannot_include_target_gold_or_pair_metadata(self):
        target = {"instance_id": "heldout", "pair_id": "secret_group", "label": "SECRET_GOLD",
                  "reverse_pair_id": "secret_reverse", "text1": "a dog", "text2": "an animal"}
        messages = pipeline.messages_for(target, [])
        self.assertEqual(json.loads(messages[-1]["content"]), {"text1": "a dog", "text2": "an animal"})
        serialized = json.dumps(messages)
        for forbidden in ("SECRET_GOLD", "secret_group", "secret_reverse", "heldout"):
            self.assertNotIn(forbidden, serialized)

    def test_balanced_examples_are_fixed_and_exclude_reversed_duplicates(self):
        rows = []
        for label in pipeline.LABELS:
            for i in range(6):
                a, b = f"{label} a{i}", f"{label} b{i}"
                rows.extend([{"text1": a, "text2": b, "label": label},
                             {"text1": b, "text2": a, "label": label}])
        examples = pipeline.select_examples(rows, self.cfg)
        self.assertEqual(examples, pipeline.select_examples(rows, self.cfg))
        self.assertEqual(Counter(row["label"] for row in examples), {label: 2 for label in pipeline.LABELS})
        self.assertEqual(len({frozenset((row["text1"], row["text2"])) for row in examples}), 8)

    def test_launch_pins_both_gpus_and_equal_split(self):
        argv = pipeline.server_command(Path("llama-server"), Path("model.gguf"), self.cfg, 8123)
        for flag, value in (("--device", "CUDA0,CUDA1"), ("--tensor-split", "1,1"),
                            ("--gpu-layers", "999"), ("--split-mode", "layer"), ("--fit", "off"),
                            ("--verbosity", "4")):
            self.assertEqual(argv[argv.index(flag) + 1], value)

    def test_truncated_generation_is_not_silently_scored(self):
        class Response:
            def __enter__(self):
                import io
                return io.StringIO(json.dumps({"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]}))

            def __exit__(self, *args):
                return False

        with patch.object(pipeline.urllib.request, "urlopen", return_value=Response()):
            with self.assertRaisesRegex(RuntimeError, "Incomplete prediction"):
                pipeline.predict_one("http://localhost", self.cfg,
                                     {"instance_id": "x", "text1": "a", "text2": "b"}, [])


if __name__ == "__main__":
    unittest.main()
