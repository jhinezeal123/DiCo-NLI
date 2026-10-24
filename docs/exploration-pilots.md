# Five parallel exploration pilots

All five nodes descend from the Qwen2.5-1.5B + LS0.1 experiment. They retain
`bash run.sh`, model revision `8faed761d45a263340a0528343f099c05c9a4323`,
official dataset/scorer revision `588968e610197ddc4c440314701cbc587afc4c1b`,
seeds 42/13/7, fixed three epochs, and matched controls within each run.

The common runner removes complete train source groups with normalized
unordered text overlap against dev inputs, leaving 3,029 rows in this release.
The worker receives only evaluation identifiers and texts, without gold or
reverse-pair reference metadata. Context uses train examples only, excluding
the query's source group and all duplicate text pairs during train retrieval.

## Methods and controls

| Node | Main arm | Controls | Primary contrast |
|---|---|---|---|
| A | Conservative train-seeded lexical contrast sets | baseline; same-class resampling with identical appended counts | counterfactual vs resample |
| B | Retrieval using ordered differing spans | baseline; whole-pair topic retrieval | edit_retrieval vs topic_retrieval |
| C | Trainable soft subspace inclusion residual | baseline; exactly capacity-matched vector residual | geometry vs vector |
| D | Learned soft alignment and latent local relation residual | baseline; uniform alignment with identical active projections | alignment vs uniform_alignment |
| E | Generated directional counterexamples, independently NLI-scored | baseline; same-generator rationale evidence | counterexample vs rationale |

The ordinary classification head remains active. New residual heads backpropagate
through the same LoRA adapters and use a free supervised four-class output,
including NEGATIVE_OTHER. Unknown witnesses do not become entailment labels.
NEG identity in bidirectional averaging is an inference heuristic, not an
official reversal rule. Geometry uses contextualized phrase features and does
not guarantee a global partial order; alignment states have no human semantics.

Pilot A adds 120 conservative examples from 30 accepted lexical families on the
current release; it is a rule-based pilot rather than active OOF error selection.
No arbitrary modifier deletion or automatic three-class NLI label conversion is used.

Pilot E generates once per canonical unordered text pair before training, using
Qwen2.5-1.5B-Instruct revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
Its independent verifier is cross-encoder/nli-deberta-v3-small revision
`fa2804872c3b4bd748f38c0185cc85775361e735`. Verifier labels are explicitly
checked. Generated expanded claims can be wrong, so the signals are uncertain
features, not logically verified proofs. Cache contents and coverage are logged.

## Training and interpretation

The frozen backbone weights use FP16; trainable LoRA/readout/head parameters use
FP32 with AMP and gradient scaling. This differs from historical full-FP32-base
runs, so conclusions must use each pilot's contemporaneous matched controls.
The common schedule is LR1e-4 for adapters/readout, LR1e-3 for any residual head,
warmup0.1, weight decay0.01, three epochs, LS0.1. Retrieval uses max_length192
and batch4/accumulate2; other methods use max_length128 and batch8.

These are fixed-recipe, direct-to-public-dev pilots. There is no internal
holdout-based choice of thresholds or checkpoint, and no dev-driven fitting.
Because public dev was already used for research selection, these scores are
not unbiased hidden-test estimates. Future promotion should confirm on a
train-only grouped validation protocol and use source-pair bootstrap.

Each notebook runs up to two independent arm/seed jobs, one per GPU. All arms
use identical model/data/seed within a run except the named intervention.
Every arm prints standard and bidirectional official F1/SoftCons/HardCons,
per-label confusion, probabilities, and predictions. Final logs include
three-seed mean/sample-SD/min and matched per-seed differences, separately from
any ensemble (these pilots do not use ensembles). Final checkpoints contain
the LoRA adapter, tokenizer, effective config, and optional semantic head.

Witness JSON generation is bounded at 160 new tokens with no regeneration retries.

Run `python -m unittest discover -s tests -p 'test_explore*.py' -v` in the
declared run environment. Tests include tiny random Qwen integration models;
they are not task-data training. The same checks run remotely before models load.
