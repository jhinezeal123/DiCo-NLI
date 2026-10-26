# Causal checkpoints on DiCo-NLI Track 1

The user clarified that the causal-reasoning direction must be measured against
the DiCo-NLI project. Synthetic causal accuracy and CLadder scores do not establish
DiCo-NLI benefit. No additional private benchmark expansion is planned before
this direct transfer check. Previous causal results and their strict gate remain
recorded; they are not reinterpreted as NLI results.

## Fixed comparison

- Original Qwen2.5-3B-Instruct, revision aa8e72537993ba99e69dfaafa59ed015b17504d1.
- The same base with the frozen native CLadder adapter from run6f73ebc5.
- The same base with the frozen intervention-semantics adapter from run6750161c.

Adapters are small, checksum-pinned LoRA archives, not full model checkpoints.
Neither adapter was trained on DiCo-NLI. All three conditions use exactly the
same tokenizer,fp16 inference,batch4,prompt,eight demonstrations andA/B/C/D
next-token choice rule. The option mapping is fixed before any transfer output.
There is no generated reasoning,SCM parser or exact causal oracle on NLI inputs.
Predictions are independent by ordered phrase pair; reversal consistency is not
enforced in decoding. Identical public texts share an inference result while
every official instance ID retains a prediction.

## Data and scoring

Use the original organizer revision588968e610197ddc4c440314701cbc587afc4c1b,
the same pinned release as the project's NLI pipeline. Every downloaded dataset
and official scorer file is checked against its committed SHA256. The primary
evaluation is all660official Track1dev rows. The unchanged official scorer
produces WeightedF1,SoftCons,HardCons,per-label metrics and confusion matrices.
No replacement metric implementation is authoritative.

There are3042train rows and no train/dev ID or pair_id overlap. Fifteen dev rows
have a normalized unordered text-pair overlap with train. Report a secondary
645-row sensitivity subset after removing the complete eight affected pair_id
groups,including their reverse rows. Keep660as the primary denominator.

Demonstrations are selected by a fixed seeded SHA256 ordering of train rows:
two examples per official label,distinct source pair_id,excluding normalized
text pairs that appear in dev. This uses public input text for deduplication,
not dev labels for demonstration choice. All eight demo IDs and the complete
request hash are locked in the configuration. Dev labels are used for scoring
and audit only; poisoning tests verify they cannot enter the model prompt.

Report each causal adapter minus base3B with2000 paired cluster-bootstrap
resamples of official pair_id. Each resample retains both directions and calls
the unchanged official metric function. No claims about hidden test or leaderboard
performance. Public dev has already been used in project development,so this
is a development transfer check rather than a fresh confirmation.

## Interpretation and next decision

Report all three checkpoints and all three official metrics,including negative
transfer. Do not choose the adapter,prompt or label mapping after seeing results.
This is direct eight-shot transfer,not a matched NLI fine-tuning experiment and
not comparable as an architectural improvement over the project's supervised
NLI model. A negative result does not rule out causal auxiliary objectives during
NLI training. It does rule out claiming a DiCo-NLI gain from these frozen causal
checkpoints under this protocol. Further work must target the NLI training/eval
pipeline and be judged by actual NLI correctness and directional consistency.

Run command remains `bash run.sh`; committed `configs/causal_dico_transfer.json`
routes to the evaluator. Training is not repeated. Persist all1980per-instance
predictions,official score reports and fixed-checkpoint comparisons in the log.
