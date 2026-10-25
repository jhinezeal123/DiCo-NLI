# Harder causal benchmark v3 — before model outputs

The user retained the original per-seed superiority gate after v2 confirmation
reached 97.92–100% OOD accuracy but tied the normalized local parser on two
strata of seed 271828. Those opened seeds remain a failed superiority test.

V3 raises task demands: eight variables, four to six factual observations,
two independently sampled proposed action values, and four linked questions:
factual, both interventions active, one override removed, and both removed.
Joint accuracy now requires ALL FOUR answers. Incomplete observations require
conditioning on all compatible original worlds; unknown is a genuine target.
Full-intervention answers are balanced 0/1/unknown per stratum; factual target
values are balanced. Constant-unknown predictions fail the joint gate.

ID has one binary merge with AND/OR/XOR mechanisms. OOD topology has three or
four binary merges. OOD mechanisms use a held-out ternary majority or odd-parity
gate: this combines arity and mechanism shifts, not a pure mechanism-only
ablation. OOD language changes both rule phrasing and query instructions.
Private fair XOR noise is fixed across all four views. Output negation can be
absorbed by a bijective noise reparameterization even under partial observation;
the literal grounded unary COPY representative therefore remains exact.

World signatures ignore names, actions and hidden factual values. They retain
the unique backbone order, graph, mechanism classes modulo output noise, and
partial evidence. A shared exclusion set prevents cross-seed world overlap.
No label, hidden SCM, query view or sampled latent state is used by evaluation
prompt construction. Six demonstration worlds cover all four query views,
all ID binary classes, and two examples of each answer class. Ternary mechanisms
are not demonstrated, but their definitions are public and available to every
method. Factual/restored target values are explicitly observed; those views
are consistency checks, not the main computational difficulty.

Compare direct, structured, monolithic JSON and local JSON, including exact
unary-normalized versions of both JSON controls. All share the same checkpoint,
cases and six worlds. The grounded system derives declared names, parents,
observed values and proposals from the explicit public layout. The LM chooses
binary/ternary mechanism class, final target and active status of each action.
Validity is partly by construction and is not evidence of intrinsic reasoning.
An operator-blind ablation retains the query decisions and uses fixed gates.
Report actual tokens, time, cache reuse and any invalid/unknown outputs.

Development seed is 20261027, with 12 families per stratum (192 total questions).
Fresh confirmation seeds are 390001, 390002, 390003, reserved before v3 model
outputs. Use 48 families per stratum for confirmation: 768 questions per seed.
Freeze the selected method before opening these outputs. Keep the gate:
>=90% joint accuracy in every OOD stratum, >=5 percentage-point gain over the
strongest matched reasoning control with paired family-bootstrap 95% lower
bound >0, on each of three fresh seeds. Report blind ablations separately.
Native public-text validation is also required; DiCo-NLI transfer is optional
under the user's clarified priority.

Analytic ternary tests and independent brute-noise enumeration verify the
optimized conditioned-world solver. Tests reject missing views and isolate
all hidden evaluation annotations from both compilers' requests.
