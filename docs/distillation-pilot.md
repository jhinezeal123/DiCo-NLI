# Qwen3B to ModernBERT distillation pilot

Hypothesis: a compact ModernBERT-base can acquire some of the trained
Qwen2.5-3B teacher's directional accuracy using offline class probabilities.
The first run is a seed-42 screening experiment, not a multi-seed conclusion.

The fixed command remains `bash run.sh`. All changes are in committed code and
`configs/experiment.json`. The parent established the Qwen3B LoRA teacher recipe;
the teacher is refit here because the parent's trainer did not export train logits.

## Controlled comparisons

- `ce` versus `kd`: identical student, initialization, training order, optimizer,
  data and five-epoch budget; KD adds class probabilities at temperature 2 with
  weight 0.5 and the temperature-squared KL scaling.
- `paired_ce` versus `paired_projected_kd`: the same paired-forward budget.
  Both use original CE for every row and additional swapped loss only for
  reversible gold TRAIN labels. The paired loss is averaged per original row
  to preserve NEGATIVE_OTHER weighting. KD projects teacher probabilities for
  reversible rows onto `p = (p_teacher(x) + R p_teacher(swap(x))) / 2` and uses
  the reversed projected target for the swapped forward.
- `kd` versus `paired_projected_kd` is a combined intervention, not an isolated
  estimate of the projection's effect. Isolating projection requires a later
  unprojected paired-KD arm. Selection requires confirmation over three seeds.

Class order is BACKWARD, EQUIVALENCE, FORWARD, NEGATIVE_OTHER. NEGATIVE_OTHER has
no official reversal; it receives neither swapped supervision nor projected KD.
Standard inference is the deployment comparison. Bidirectional inference is
reported separately and costs two forwards. Its NEGATIVE_OTHER identity handling
is an inference heuristic, not part of the official reversal specification.

## Data and evaluation

Use the pinned organizer commit from `run.sh` and its official scorer. Remove
entire train pair_id groups if any normalized unordered text pair matches dev
inputs. Dev labels are removed from the file passed to all training workers.
There is no dev checkpoint selection and no external or synthetic training data.
Log the removed groups, hashes, revisions, effective hyperparameters, epoch
losses, overflow-skipped steps, final metrics and per-instance dev probabilities.
Within-run comparisons use the cleaned split; historical scores on the unfiltered
official train split are contextual and are not a paired control.

The offline teacher predicts the same train rows on which it was fine-tuned;
in-sample overconfidence is a limitation. A later grouped out-of-fold teacher or
teacher ensemble could address it. Latency uses the first eight dev inputs,
five warmup and thirty timed forwards on the allocated GPU. Different tokenizers
can produce different padded token counts, which are logged; it is a small-batch
diagnostic, not a production latency benchmark.

All student checkpoints and complete NPZ/JSON results are saved under the remote
run's `outputs/` directory. Predictions, probabilities, metadata, and compact
JSON summary are also printed to the persisted run log for local recovery.

## Literature used

- Revisiting Self-Distillation, arXiv:2206.08491: conventional CE plus
  temperature-scaled forward KL and self-distillation.
- Less is More: Task-aware Layer-wise Distillation for Language Model
  Compression, arXiv:2210.01351: task-filtered feature transfer as a later,
  more demanding alternative.
- Do Students Debias Like Teachers?, arXiv:2510.26038: NLI robustness and
  teacher agreement need independent checks after KD.
- Good Teachers Explain, arXiv:2402.03119: explanation alignment, evaluated
  in computer vision; transfer to text would require a separate hypothesis.

The reversal projection is our task-specific proposal; these papers do not
establish its effectiveness on DiCo-NLI.
