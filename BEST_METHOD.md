# Best method tracker — Track 1 (English)

Single source of truth for the **current best Track 1 recipe**. Update this file (newest milestone on top)
whenever a run beats it. Every number below is on the official **dev** split, computed by the organizers'
scorer (`weighted_f1 / soft_cons / hard_cons`). Dev was used for model selection across the project, so these
numbers are optimistic; the hidden test set is the real check.

## Current best (milestone 2026-10-31)

**Qwen2.5-1.5B LoRA sequence classification + label smoothing 0.1 + bidirectional decoding + NEG decision weight 0.8, 3-seed ensemble.**

| Decode | F1 | soft_cons | hard_cons |
|---|---|---|---|
| bidir + NEG weight 0.8, **3-seed ensemble (submission)** | **0.8715** | **0.9639** | **0.9170** |
| bidir + NEG weight 0.8, mean ± sd over seeds 42/13/7 | 0.8636 ± 0.0061 | 0.9603 ± 0.0108 | 0.9158 ± 0.0146 |
| bidir (previous best), mean over seeds | 0.8676 ± 0.0039 | 0.9483 ± 0.0104 | 0.9073 ± 0.0127 |
| official starter kit (ModernBERT-base) | 0.713 | — | 0.702 |

- **Branch:** `orx/submission-recipe-ls-0-1-bidir-neg-decision-weig` (commit `e405f9c`); run `f0d2daaf-02e7-439d-a2cf-39958503d2c3`.
- **Run contract:** `bash run.sh` → `scripts/run_pipeline.py --config configs/experiment.json`.
- **Hardware:** Kaggle 2×T4, ~19 min for 3 seeds.

### Recipe

| Item | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B`, `AutoModelForSequenceClassification`, 4-way head |
| Input | `Text 1: {t1}\nText 2: {t2}\nRelation:`, max_length 128 |
| LoRA | r 16, alpha 32, dropout 0.05, all attention + MLP projections |
| Optimisation | AdamW lr 1e-4, wd 0.01, batch 8, 3 epochs, warmup 10 %, fp16 AMP, grad clip 1.0 |
| Loss | cross-entropy, **label smoothing 0.1** |
| Training data | official Track 1 train only (3,042 rows); no dev, trial or external data |
| Decoding | `p_bidir = 0.5 * (p(t1,t2) + reverse(p(t2,t1)))`, where reverse swaps FE↔BE |
| NEG weight | multiply P(NEGATIVE_OTHER) by **0.8** before argmax; chosen on **train 5-fold OOF** (max F1 + hard_cons), not on dev |
| Ensemble | mean of the 3 seeds' weighted probabilities |

## What did NOT beat it (do not retry without a new idea)

- **Scale/backbone:** Qwen2.5-3B/7B, Qwen3-1.7B/4B, Qwen3.5-2B, ModernBERT-large, DeBERTa-v3, NLI-pretrained cross-encoders all tie or lose.
- **Losses/heads:** consistency KL, paired CE, factorized entailment, NEG sub-type heads, SupCon, mean pooling.
- **NEG data augmentation (2026-10-31 round):** NEGATIVE_OTHER is *not* under-represented per pair (478 NEG pairs vs 401–445 for the other labels; reversible labels simply get two rows per pair). Results:
  - swapped/duplicated NEG: 0.859 / 0.905 / 0.871
  - transitive-closure pairs (+748 rows): 0.849 / 0.947 / 0.891
  - LLM-generated hard-NEG + matched FE/EQ (+1,473 rows): 0.857 / 0.941 / 0.899
  - LLM hard-NEG only (+297 rows): 0.856 / 0.937 / 0.893
  - Spanish views, random unrelated negatives, boundary up-weighting: no gain.
- **Original PhrasIS data:** NOT used. The ~1,180 unreleased non-UNR PhrasIS pairs are the likely hidden-test pool, so training on them would contaminate the test set.

## Open items

- Test input is released when evaluation opens (2027-01-10). The current pipeline predicts dev only and does not save LoRA adapters, so the submission run must retrain (deterministic; it reproduced the previous bidir numbers exactly) plus a test-prediction step.
- `pair_id` singleton = NEG leak (106/106 on dev). Never exploit it; raise it with the organizers.
