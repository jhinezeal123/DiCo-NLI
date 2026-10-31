# Matched train-source paraphrase pilot

The source JSONL is copied from the completed Aug B experiment
`orx/aug-b-llm-generated-hard-neg-contrast-sets-with`, commit `99fb53f`,
`data/gen_llm/out_{0,1,2}.jsonl`. Those candidates were generated from train
anchors by Claude Sonnet; the `para` field alone is used. This pilot changes one
phrase of an existing source row and retains the source label; it does not use
the generated `spec` or `neg` fields or the previous Aug B training file.

At run time the pinned `cross-encoder/nli-deberta-v3-base` screens each source
phrase and candidate in both entailment directions, with fixed threshold 0.8.
This is automatic filtering, not proof of equivalence or a human audit.
Candidates are matched only against train; anchors appearing in participant
dev text are excluded and both original and modified pairs overlapping dev are
excluded. The procedure inspects no dev labels. Up to 80 source rows per label
are selected by a deterministic hash. The control retains precisely the same
selected rows and labels but duplicates the original instead of replacing the
phrase. Both runs report a source-ID hash and an explicit manifest; their
results are not a matched comparison unless selected source hashes agree.

Both arms use seeds 42/13/7, batch 8 and 1143 optimizer steps, which matches
the parent budget. Official dev is exploratory because it has been heavily
used; B/C subsets are diagnostic, not independent validation. Before any
promotion, evaluate a frozen candidate on a genuinely independent holdout and
have people review accepted and rejected paraphrases and ambiguous labels.
