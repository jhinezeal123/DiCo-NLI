# Native public-text causal pilot

This child retains the independent CPT kernel that matched all 1,476 published
deterministic-counterfactual labels in its oracle audit. That audit used hidden
CPTs; the present language-model evaluation does not provide test CPTs,
variable mappings, formal query metadata, labels or reference reasoning.

Evaluation prompt content is allowlisted to the author's model background,
question given_info and question text. The LM predicts all program contents,
including its own aliases, deterministic truth tables, evidence, intervention,
outcome and polarity. There is no synthetic private XOR noise here.

Model IDs group the split. One development model per graph class is reserved;
previously inspected model 3220 is always development. Four demonstrations
(mediation, chain, fork, diamond) use development annotations. Per graph, four
other models and one question per model are selected by a fixed SHA-256 rank,
before observing model outputs. The complete selection audit is logged.
The 36-question pilot is not a final generalization result. In particular,
public benchmark exposure during model pretraining is not controlled.

Direct, structured and JSON-to-CPT protocols receive the same public cases and
four demonstration worlds. The JSON protocol receives structured development
solutions; the answer controls receive development answers. This is an explicit
supervision-format difference. Accuracy and valid-rate are reported with all
cases in the denominator. Unknown/invalid predictions are incorrect against
the official yes/no labels. Paired bootstrap resamples models, one question per
model. No class balancing or post-hoc exclusions use evaluation answers.

The committed configuration pins source hashes, model revision, generation
budget and batch size. All raw outputs and prompt hashes enter the run log.
Leakage-isolation, model grouping and analytic CPT tests run before launch.

## Grounded native child

The free JSON pilot obtained 36.11% accuracy (valid rate 61.11%), below direct
75%. This child keeps the exact kernel and all 36 pilot cases. It copies edges
from the original public background's explicit `has a direct effect on`
sentences and recognizes public `is unobserved` declarations. An input-only
audit matched 544/544 source graphs; maximum parent count is three and all root
probabilities have full support. This grammar constraint is explicit and does
not constitute unrestricted graph extraction.

The LM makes constrained next-token semantic decisions for every deterministic
CPT cell, observed state (unspecified/0/1), treatment, treatment bit, outcome
and question polarity. Original test annotations never enter these requests.
The mechanism requests see the public premises and parent assignment, while
the query/evidence requests see all three public fields. Root support is fixed
to {0,1}, sufficient for the audited point queries. No output probabilities
are rounded or repaired. Inconsistent evidence or unidentified outcomes still
count as failures. Conditional option probabilities and every decision are logged.

The structured_program_demos control now receives complete development
mechanisms and explicit abduction/intervention/prediction examples generated
from the same four development worlds. This improves supervision parity;
its performance remains an empirical question. The original direct protocol
is also rerun. This is still development, not a fresh held-out confirmation.
