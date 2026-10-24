# Laya task fine-tuning for DiCo-NLI Track 1

The fixed command is `bash run.sh`. The parent ModernBERT-large experiment
established the encoder baseline; this round uses Laya's pretrained typed
decision head and encoder, with task-specific supervised adaptation. CE and
RLCD are sibling hypotheses with the same data, optimizer and epoch budget.

- Checkpoint `convaiinnovations/laya` at revision
  `7b928d828b7b0e022f929d9bd2e44165aa270148`; SDK `laya==0.4.14`.
- Seeds 42, 13, 7; four fixed epochs; full encoder and head tuning.
- Micro-batch 8, accumulation 4 (effective batch up to 32). Encoder LR 2.5e-5,
  head LR 1e-4, weight decay .01, cosine decay, fp16 autocast and checkpointing.
- The RLCD sibling uses the canonical SDK loss: noisy-logit policy gradient
  plus soft CE, four samples, sigma .4 to .1. CE uses soft CE alone.
- Four neutral option IDs map to BACKWARD, EQUIVALENCE, FORWARD, NEGATIVE_OTHER.
  Option order is shuffled with the correct target permutation during training.
- Remove complete train pair_id groups whose normalized unordered text matches
  dev inputs. No teacher-generated labels, synthetic data or swapped NEG labels.
- Workers receive dev inputs with the gold-label column removed. No checkpoint
  selection or calibration on dev. Final epoch is fixed before launching.
- Pretrained and fine-tuned scores use raw logits with T=1. Clear inherited
  temperature settings in exported checkpoints. This does not fit confidence.
- Evaluate standard and bidirectional predictions with the pinned official
  scorer. NEG identity in the inference permutation is a heuristic, not an
  official reversal definition. Report mean/std/min over seeds and an ensemble.
- Historical ModernBERT/Qwen scores used unfiltered train. They are contextual,
  not perfectly matched controls. Judge by F1/HardCons, not model size or speed.

SDK APIs: items_from_rows, make_item, train_model, calibration_records,
save_checkpoint. Full checkpoints stay in the run's remote outputs directory
in the format laya.load reads. Metrics, predictions, probabilities and worker
summaries also go to persisted stdout for recovery. No SDK code is modified.

Sources: https://github.com/NandhaKishorM/laya/blob/main/laya/train.py and
https://huggingface.co/convaiinnovations/laya .
