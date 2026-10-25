# Native supervised compiler pilot

The public inference pilots failed to improve over direct answering. This
experiment asks whether a small model can learn reliable program compilation
from oracle-checked supervision. It uses Qwen2.5-3B-Instruct at a pinned
revision, FP16 frozen base weights, and rank-8 LoRA on q_proj/v_proj with FP32
adapter parameters. This targets one T4 without a new quantization dependency;
the run must verify actual memory use and training stability.

The committed split manifest excludes nine development, 36 pilot and 108
reserved final model IDs from training. Of the remaining 391 model instances
and 1,046 deterministic-counterfactual questions, select 150 by a fixed hash
before observing adapter outputs. Model ID grouping does not imply unseen
graph/mechanism families or control pretraining exposure.

Every training question has two targets with equal per-style loss weight:
direct yes/no and a complete native program. The program specifies state
meanings, deterministic CPTs, evidence, action, outcome and query polarity.
The exact kernel must agree with the official training label. Root probabilities
are set to 0.5 because every audited root has full support and these are point
queries. Completion-only loss masks all prompt tokens; both styles are dropped
if either exceeds the sequence budget. No evaluation annotations enter training.

Explicit case-specific factual evidence takes precedence over a generic
background declaration that a variable is unobserved. The source dataset can
contain both for the same variable. The earlier constrained pilot hard-masked
such variables and therefore does not isolate the LM's extraction ability.
This program-prediction protocol imposes no such evidence mask.

Measure both inference formats on the same 36 development cases and four
development demonstrations before and after training. Optimize 150 successful
updates, each accumulating the two styles of one question; log overflows,
losses, tokens, gradients, runtime and VRAM. This tests the practical value of
joint supervision; it does not isolate the causal effect of a supervision style.
The reserved 108 cases remain unopened for model performance during selection.

Save only the adapter, configuration and model card. A size-limited checksummed
ZIP is chunked into the persisted log so it can be recovered to durable project
artifacts without using GitHub as compute transport. Full base weights and
optimizer state are not exported. All evaluation outputs remain in the log.

The full-coverage child expands the pilot to all 1,046 allowed training questions
from 391 model instances, with 1,046 successful optimizer updates. It keeps
rank, checkpoint, seed, per-update two-style weighting, learning-rate schedule
rule and original four-shot base/adapter evaluation. Thus it tests increased
training coverage and compute, not the isolated effect of one supervision
format. It uses the same strict exclusions for all reserved final worlds.
