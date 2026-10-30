# Random-drop control pilot — data provenance

**File:** `data/train_randdrop.csv` (2,738 rows = 3,042 official train − 304 dropped rows)
**SHA256:** `bbfd8bf38d15ccd40e691d8a95f4643e5647a12175bf55a109e9ea5f6286196d`
**Parent:** `orx/label-smoothing-0-1-qwen2-5-1-5b` (run `13417727`, dev bidir F1 0.8676)

## Intervention

Drop 76 random official train rows per label (304 rows, 10%) from the same
eligible pool as the neighbor-consistency treatment: excludes the 15 train rows
with dev text-pair overlap, but keeps every pair_id/direction. New random sample
`numpy.random.default_rng(20261030)`. The kept rows have identical label counts
to the treatment (EQ 726, FE 805, BE 805, NO 402).

## Matched comparison

Sibling `orx/neighbor-consistency-pruning-pilot-drop-10-least` drops the 76 rows
per label with the lowest top-10 neighbor label consistency instead. Both use
`max_steps=1143`, seeds 42/13/7, and leave dev untouched. The control isolates
the effect of *which* rows are dropped from the effect of dropping 10% of train.

## Gate

The control is not expected to beat the parent; compare it with the treatment on
bidir weighted F1, per-class F1 (especially NO/EQ), fixes/breaks, and hard-core
A/B/C. No dev fitting.
