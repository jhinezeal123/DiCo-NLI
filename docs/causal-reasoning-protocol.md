# Causal-reasoning experiment protocol

The first causal node inherits the working Qwen2.5-1.5B-Instruct recipe from
experiment a9dae341-dd9a-4d24-ad3d-3622a7371d86. The inherited achievement is
successful model execution on Kaggle T4, not an established causal improvement.
The fixed command remains `bash run.sh`; causal behavior is selected by the
committed `configs/causal_experiment.json`.

## First round

Establish direct, structured abduction/action/prediction, and LLM-to-SCM parser
plus exact solver baselines on the same fresh finite binary SCM families.
Each node's mechanism is f(parents) XOR a private independent uniform noise bit.
All original endogenous variables are observed in this initial pilot, so noise
is identifiable. Partial-observation unknown semantics is analytically tested;
the pilot does not yet measure abstention performance.

Each family has factual, intervened and restored-equation questions. Factual
and restored answers are explicitly observable; joint accuracy therefore tests
whether the intervention is also correct while preserving factual controls.
Do not describe this as three equally difficult reasoning problems. Half the
families change the target under intervention, half do not; original outcomes
are balanced too. A constant answer can reach DCC=100% but joint=25%.

ID uses chain/fork/diamond; topology holdout uses collider/skip with the same
number of variables. Mechanism holdout introduces NAND/NOR. Language holdout
uses a second verbalization. The language split is a narrow template shift,
not evidence of arbitrary natural-language understanding. All families and
demonstrations are deduplicated by equation/observation/intervention signature
before renaming variables. The topology frequencies may differ after outcome
stratification; report this limitation, rather than claiming a pure causal
effect of one OOD factor.

All protocols share checkpoint/revision, dataset, four demonstration families,
greedy decoding, and a 512-token output ceiling. Demonstration outputs differ:
answers for direct, oracle traces for structured, JSON for parsing. This is a
comparison of full protocols, not a clean architecture-only ablation. Report
actual input/output tokens and wall time; the same ceiling does not imply
equal realized compute. Add compute-matched sampling controls before a final
efficiency claim. The initial pilot used indices 0 and 3, whose counterfactual
answers were both 0; that node remains frozen. This child uses all four original
outcome-by-causal-change strata, giving two demonstration answers per class.

## Acceptance criteria, fixed before the first run

- Exact solver passes analytic same-individual intervention regression tests
  and matches generator labels for every case (agreement by construction).
- On each OOD stratum: joint accuracy at least 90%; at least 5 percentage point
  gain over the strongest matched reasoning control; paired family bootstrap
  95% lower bound strictly above zero.
- Confirm on three fresh dataset seeds after development. Use those seeds
  once for confirmation, not for tuning. Initial seed 20261025 is development.
- Independent public causal benchmark validation is required before a broad
  causal-reasoning claim. DiCo-NLI is downstream transfer, with its official
  scorer and separate labels, not causal gold.

The first 12-families-per-stratum run is diagnostic only and cannot pass this
gate. If exact solver plus gold schema is perfect but parsed schema is poor,
improve parsing before changing the solver. If parsed schema is correct but
answers are wrong, treat it as an oracle implementation bug. If the solver
protocol has no benefit over structured reasoning, do not infer improvement.

## Local checks and tracked execution

```powershell
python -m unittest discover -s tests -p test_causal_reasoning.py -v
```

Launch experiment compute only through the project experiment runner; commit
first. All effective configuration, model revision, dataset hash, per-case
outputs and final metrics are emitted into the persisted run log. Source code
never obtains the evaluation answer during prompt construction. The gold SCM
is passed to decoding only to compute parsing diagnostics, not predictions.
