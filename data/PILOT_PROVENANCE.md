# Neighbor-consistency pruning pilot — data provenance

**File:** `data/train_consdrop.csv` (2,738 rows = 3,042 official train − 304 dropped rows)
**SHA256:** `887f1dbce6f42f6c5fb065cc9d9f08af6ee1f66e203ab5f7f4357e92aff4b1dd` (verify with `Get-FileHash`)
**Parent:** `orx/label-smoothing-0-1-qwen2-5-1-5b` (run `13417727`, dev bidir F1 0.8676)

## Intervention

For each train row, take the top-10 TF-IDF cosine neighbors that (a) come from a
different `pair_id`, (b) are not the reversed direction, and (c) are not the same
unordered normalized text pair. `neighbor_consistency` is the fraction of those
neighbors with the same label. For each label, drop the 76 rows with the lowest
consistency (304 rows, 10%; treatment mean consistency 0.053 vs 0.305 for the
random control). The 15 train rows whose text pair overlaps dev are not eligible
for dropping and stay in both arms.

## Matched control

Sibling `orx/matched-control-drop-10-random-train-rows` drops 76 random rows per
label from the same eligible pool. Both arms keep 2,738 rows with identical label
counts (EQ 726, FE 805, BE 805, NO 402), use `max_steps=1143` (same optimizer
budget as parent), seeds 42/13/7, and leave dev untouched.

## Gate

- Promote if NO F1 ≥ 0.720 (+1.5 pp vs parent 0.7053) or EQ F1 ≥ 0.888 (+1 pp vs
  0.8783), with HardCons not below parent 0.9073.
- Otherwise report a null; no dose sweep on dev.

## Reproduction

Deterministic TF-IDF (word 1–2 grams, `min_df=1`, sublinear TF), top-10
neighbors, `numpy.random.default_rng(20261030)` for the control sample. The CSV
is committed, so the training run is reproducible without the generator.
