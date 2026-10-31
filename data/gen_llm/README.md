LLM-generated augmentation (Aug B). Generator: Claude Sonnet subagents, prompt in the experiment
description; inputs were only `train_examples.json` (40 train rows) and 300 anchor phrases sampled
(seed 0) from train phrases of <=8 words that do not occur in dev. `out_*.jsonl` are raw generations;
`scripts/build_aug_llm.py` converts them to `data/aug_llm_contrast.csv`. Must be reported as LLM data
augmentation in the system description.
