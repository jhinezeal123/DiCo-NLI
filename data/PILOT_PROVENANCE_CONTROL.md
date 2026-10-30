# Repeat-control pilot — data provenance

**File:** `data/train_repeatctrl.csv` (3,346 rows = 3,042 official train + 304 duplicated random rows)
**SHA256:** `48ebd232dd04068fa9179a7e2073506a56643c2f03a34474c7f19c86ff2e7578`
**Parent:** `orx/label-smoothing-0-1-qwen2-5-1-5b` (run `13417727`, dev bidir F1 0.8676)

## Intervention

Duplicate 76 random official train rows per label (304 rows, 10%), using
`random.Random(20261030)`. The same 15 rows with exact/near-exact dev text-pair
overlap are excluded from selection, so the auxiliary rows do not add dev
duplicates. New `instance_id`; original `pair_id` and label are kept.

This is the matched control for sibling
`orx/boundary-upweight-pilot-duplicate-76-nearest-cro`: same code, data size,
label counts, seeds, batch size, and `max_steps=1143`. The only difference is
whether the duplicated rows are nearest cross-label rows or random rows.

## Gate

The control is not expected to beat the parent; it exists to isolate the effect
of boundary selection from extra data/optimizer steps. Compare with the
boundary arm on bidir F1, per-class F1 (especially NO/EQ), and fixes/breaks.
