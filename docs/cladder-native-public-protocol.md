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
