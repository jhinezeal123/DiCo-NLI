# Supervised intervention semantics: development pilot

Experiment 44601331-b0fb-4143-91cd-05599604247a builds on the retained
MC16 experiment 445ee135, not on any regressed decoder. The three previous
zero-shot alternatives are closed scientific negatives. Their implementation
and validated replay bundles are incorporated without rewriting their branches.

The hypothesis is that supervised action and short-program semantics improve
condition/toggle/restore grounding and compose to eight instructions. This pilot
changes supervision and query-model size, so the matched base3B comparison is
required. It does not isolate the contribution of intermediate-trace supervision.

## Fixed training and inference

- Fresh Qwen2.5-3B-Instruct revision aa8e72537993ba99e69dfaafa59ed015b17504d1,
  fp16 on cuda:0. This does not load or overwrite the native CLadder SFT adapter.
- LoRA q_proj/v_proj, rank8, alpha16, dropout0.05, fp32 trainable parameters.
  Completion-only loss; AdamW lr0.0002, weight_decay0, eps1e-6, norm clip1.
  Linear schedule, ten-percent warmup, checkpointing, fp16 gradient scaling.
- 1,792 independently generated examples, seed20261032:256 atomic16-way
  meanings,1024 one-step4-way transitions,256 short-program4-way endpoints,
  and256 compact short-program traces. Program lengths1/2/3/4; endpoint classes
  balanced at64each for the whole-program tasks. Plain v5 language plus two
  separately authored training renderers; never the held-out language renderer.
- Distinct synthetic variable names. Training construction accepts no test
  dataset, question labels, SCM answers or eight-step traces. Independent set
  interpretation verifies every generated transition and trace in unit checks.
- Two complete epochs, reshuffled once per epoch,3584 successful updates.
  Overflow retries use the same example; no silent dropping. Max length1536;
  fail before evaluation/training on an overlong target rather than truncate it.
- Fixed final checkpoint, no checkpoint selection based on test outputs.
- Query evaluation uses zero demonstrations, identically for base and adapter.
  Atomic MC16 prompts remain exactly the earlier prompts. Whole-program choices
  and compact CoT controls are tested using the same checkpoint. Generation cap512.

## Evaluation and attribution

Opened v5 development only:seed20261030,96queries,6families per split,four views.
The dataset and six world demonstrations retain their prior checksums. Operator
choices are82 actual 7B outputs from96d90d20, verified by checksum and request ID;
they are never replaced by gold. Eight frozen comparison methods are replayed
from768 records with provenance. This system combines a3B query model,a frozen7B
operator model,and exact private-XOR SCM inference. It is not a standalone3B model.

Primary is fixed as adapter3b:action; no switching to another decoder after seeing
results. Report base3B and adapter3B action/whole-state/whole-CoT results and two
adapter semantic ablations. Expected16methods x96 =1536 predictions. Expected
new query requests per checkpoint:162atomic +96whole-state +96CoT =354;
request IDs may repeat across checkpoints and are distinguished by phase.

Retain the seven original registered controls; also compare against the original
MC16 and matched base3B and adapter whole-program controls. Both operator-blind
and action-blind results are diagnostics, not gate controls. Replay costs and new
3B inference/training costs are reported separately.

The goal gate remains EACH OOD joint>=90%,>=5percentage points over the strongest
registered reasoning control,paired-family95%CI lower>0,on ALL three fresh seeds.
This pilot is not confirmation. Seeds690001/690002/690003 remain unopened.
Require larger independent development validation before freezing confirmation.
The previously locked192-query input bundle is reserved for that next check;
it needs fresh operators and all fresh controls, not this pilot's replay.

## Research context and limits

Related primary evidence motivates controls rather than guaranteeing transfer.
The selected reading was ranked from the connector search results: Execution-First
Synthetic Tool-Use Trace Generation2607.29175; Where Do CoT Training Gains Land
2606.26935; Beyond the Answer2610.23831; What Really Improves Mathematical
Reasoning2605.19762; How Reasoning Evolves from Post-Training Data2604.05134.
The first three original full texts were read; only those support statements here.

Execution-First describes its synthesis order: “Instead of starting from a
natural-language request, our approach first constructs and validates executable
tool traces in a controlled environment before synthesizing natural-language
tasks.” [Paper](https://www.alphaxiv.org/pdf/2607.29175).
Our finite set interpreter supplies exact labels; no teacher LM annotation is used.

Where Do CoT Training Gains Land cautions: “Although CoT reasoning still helps,
the relative advantage of CoT actions over prompt actions does not widen as
training progresses.” [Paper](https://www.alphaxiv.org/pdf/2606.26935).
This motivates evaluating both direct state predictions and generated traces at
the same checkpoint. Its agent setting differs from our finite SCM task.

Beyond the Answer states: “The answer-only and original-trace recipes therefore
rank differently on the two benchmarks: higher source-task accuracy does not
necessarily imply better cross-benchmark performance.”
[Paper](https://www.alphaxiv.org/pdf/2610.23831).
Our proposed OOD benefit is a hypothesis. Passing opened development cannot
establish it, and this joint-supervision recipe is not an ablation of trace loss.

## Execution

The fixed command remains `bash run.sh`; committed
`configs/causal_semantic_sft.json` selects the pilot. Install the pinned
`requirements-causal-sft.txt` on the managed run. Persist the adapter-only archive
to the run log (32MiB size cap), with SHA256 and a complete chunk manifest.
All training and LM inference run on configured remote compute. Local checks
use only generated data, fake unit responses and independent exact arithmetic.
