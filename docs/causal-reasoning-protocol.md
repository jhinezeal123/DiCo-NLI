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

The local-parser child reads one public rule per subcall and separately parses
original observations/current question. All extraction still uses the LLM;
the document splitter recognizes only explicit Rules/Observed section markers.
Rule examples cover operations found in the same four ID demonstration
families; query examples cover factual, intervention and restored requests.
In this child's run, all direct/structured/monolithic controls receive the
same four query views (factual, intervention, restored, intervention), with
balanced answers. Comparisons within this run therefore match query coverage;
do not attribute cross-run changes in the old controls to decomposition.
Neither held-out language nor NAND/NOR rules enter the demonstration set.
Output supervision format still differs across protocols, so results measure
full-protocol utility rather than a change to the model architecture alone.

Identical public rule text is cached across the family's three query views.
Token/time costs are logged at subcall level and amortized across uses. Rule
generation has a 96-token ceiling and query generation 192; actual total cost
must be reported before claiming compute efficiency. Invalid extraction is a
failed prediction, with no gold-based fallback or automatic correction.

The unary-normalization child replays the exact persisted 3B outputs in a
CPU-only tracked run. It introduces only exact variadic Boolean identities:
AND(x)=OR(x)=XOR(x)=x, NAND(x)=NOR(x)=NOT(x). Missing/unknown parent names,
wrong observations, and unsupported arities remain errors. No label influences
normalization. Strict results are retained; mathematical equivalence is checked
on factual/interventional analytic fixtures. Source predictions and their hash
are committed, with LF-only normalization of exported JSONL line endings.

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

### Frozen three-seed confirmation

After selecting target focus on development, freeze the model revision and all
grounded operator/target/override prompts. Confirmation uses seeds 271828,
314159, 161803, each with 48 families per split (576 questions per seed).
Generate these sets in this fixed order with one shared world-exclusion set,
starting by excluding all 52 development worlds (including demonstrations).
Every confirmation run reconstructs the same full plan and logs its manifest.
The four demonstration worlds are independently sampled by each seed's fixed
generator; none belongs to any other evaluation/development set.

Controls selected before final outputs are direct and the free local JSON
parser, including its exact-unary-normalized candidate. These cover the
strongest observed development control across strata. The operator-blind
ablation is also retained and reported. Structured free reasoning was weaker
in this development round and is omitted to bound generation cost. No cached
development predictions are used for fresh confirmation.

For each of the three seeds, require joint accuracy >=90% in every OOD group,
>=5 percentage-point improvement against the strongest matched control, and
paired family-bootstrap 95% lower bound >0. Report comparisons to the blind
ablation separately as well. If confirmation fails, opened seeds become
development; any method revision requires new reserved confirmation seeds.
Public-text validation is still required before broad causal-reasoning claims.

The target-focus child changes only the target request: it takes the final
explicit interrogative sentence from the public query for demonstrations and
evaluation. The LM still chooses the target from all named variables. Binary
operator and override-status prompts remain byte-identical to the parent.
This addresses observed development errors that selected an earlier treatment
name. It is a development prompt change, not fresh confirmation.

The parent's matched predictions are reused only for scoring, from a committed
JSONL with SHA-256, dataset fingerprint and run/commit provenance. Reuse must
cover every case exactly once; it cannot overwrite newly generated methods.
No parent predictions enter model prompts. Generation costs for reused controls
are labeled as source costs, with zero new LM calls. The previous grounded
candidate is included as grounded_solver_previous for paired comparison.

The grounded compiler assumes the explicit output-first Rules/Observed
interface. It copies entity mentions and original bit assignments literally;
it does not infer them from hidden metadata. Root/unary rules are represented
in the private-XOR-noise quotient (unary NOT and COPY are contextually
equivalent after abduction). The LM makes closed next-token choices for binary
AND/OR/XOR class, target variable and whether a proposed override remains
active. Token choices are constrained to valid option letters; validity is
therefore by construction, not evidence of language understanding.

An operator-blind ablation uses AND at every binary node while reusing target
and query-intent decisions. This separates semantic mechanism selection from
literal grounding. Direct, structured and free-JSON local controls receive
the same v2.1 cases; the previously weak monolithic protocol is omitted to
save generation cost. Report actual tokens/time and exact-public-text caching.
These results do not establish unrestricted text parsing or intrinsic causal
reasoning in the LM.

The additional causal_program_exact diagnostic compares effective structural
equations after abduction from full observations. Equality implies the same
endogenous responses to every intervention in that individual's context, not
global population-level SCM identification. It accepts NOT/COPY and analogous
private-noise reparameterizations that literal JSON equality rejects. This is
an explanatory diagnostic added during development; the primary joint-accuracy
acceptance gate stays unchanged.

The v2 validation child replaces the five-variable suite with six variables
on a backbone chain. ID has one binary merge; held-out topology has two or
three. AND/OR occur in ID; binary XOR is withheld as a genuine new mechanism
class. Demonstrations contain both AND and OR. The wording holdout remains
one narrow template shift. Family signatures are invariant to names because
the backbone gives a unique topological order, and collapse output-negation
equivalences induced by private XOR noise. A shared exclusion set can prevent
overlap across pre-generated seed sets. Development seed is 20261026; the
reserved confirmation seeds remain 271828, 314159 and 161803.

The v2.1 compiler branch additionally excludes the chosen intervention from
the world-group signature, so identical SCM/observations cannot leak across
splits by being asked about a different cause. An input-only audit of the
earlier v2 development pilot found 52 distinct worlds among its 52 families;
its results remain usable, but future training/confirmation uses world groups.

Both strict and exact-unary-normalized parser candidates are scored from the
same generated text, with source generation costs retained. The v2 primary
candidate is monolithic parser plus unary normalization. This is still a
development pilot, not final confirmation.

```powershell
python -m unittest discover -s tests -p test_causal_reasoning.py -v
```

Launch experiment compute only through the project experiment runner; commit
first. All effective configuration, model revision, dataset hash, per-case
outputs and final metrics are emitted into the persisted run log. Source code
never obtains the evaluation answer during prompt construction. The gold SCM
is passed to decoding only to compute parsing diagnostics, not predictions.
