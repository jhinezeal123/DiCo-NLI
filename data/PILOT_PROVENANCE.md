# Boundary-upweight pilot — data provenance

**File:** `data/train_boundaryup.csv` (3,346 rows = 3,042 official train + 304 duplicated boundary rows)
**SHA256:** `930e4f6ed351e248928ff1d4ac498495c03d4f211c8c8647e3f8ad0c834397a9`
**Parent:** `orx/label-smoothing-0-1-qwen2-5-1-5b` (run `13417727`, dev bidir F1 0.8676)

## Intervention

For every label, 76 official train rows with the **highest TF-IDF cosine similarity to a
row of a different label** are duplicated (new `instance_id`, original `pair_id` and label).
The selection uses train text only; no dev text or label is used for scoring or selection.
The 15 train rows whose exact unordered normalized text pair also occurs in dev are excluded
from selection (they remain once in the base train block, as in the official train file).
The treatment therefore adds 304 rows (10%), exactly 76 per label, without changing the
label prior.

## Why this is the matched comparison

Sibling `orx/matched-control-duplicate-76-random-train-rows-p` duplicates 76 random rows
per label from the same filtered train pool. Both arms use the same code, data size,
label counts, seeds (42/13/7), batch size, and `max_steps=1143` (the parent's optimizer
budget: 3,042 rows / batch 8 × 3 epochs). The only difference is **which rows are
duplicated**: nearest cross-label rows vs random rows.

## Reproduction

The generator is a small deterministic script (TF-IDF word 1–2 grams, `min_df=1`,
sublinear TF, cosine similarity, `random.Random(20261030)` for the control sample).
The CSV itself is committed, so the training run is reproducible without the generator.
No external corpus is used in this arm.

## Gate (pre-registered)

- Promote if bidir dev F1 ≥ 0.8726 (+0.5 pp vs parent 0.8676) with NO F1 ≥ 0.685, **or**
- NO F1 ≥ 0.725 (+2 pp vs parent 0.7053) with EQ F1 ≥ 0.864.
- Otherwise report a null and do not sweep the duplication dose on dev.
