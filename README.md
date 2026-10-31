# DiCo-NLI — SemEval 2027 Task 2

> **Independent participant repository** for **Directional-Consistent Fine-Grained Natural Language Inference (DiCo-NLI)**.  
> **This is NOT the organizers' official repository.** This README is the project specification and working brief for human contributors and AI coding/research agents.  
> Last checked against official sources: **2026-10-23**. Competition dates/rules can change.

**Primary goal:** Build a valid, reproducible, competitive DiCo-NLI system; participate in the shared task; and prepare a scientifically defensible **SemEval system-description paper**. **Initial priority: Track 1 (English)**. Expand to other tracks only after the English baseline and local evaluation are reliable.

**Official references (source of truth):**

- [DiCo-NLI official website](https://inigolopezgazpio.net/SemEval-2027-Task-2-DiCo-NLI/)
- [Organizer GitHub: datasets, starter kit, scoring](https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI)
- [CodaBench development competition](https://www.codabench.org/competitions/18038/)
- [SemEval 2027 official schedule](https://semeval.github.io/SemEval2027/)
- [SemEval paper requirements](https://semeval.github.io/paper-requirements.html)
- [LREC 2026 pilot study](https://lrec.elra.info/lrec2026-main-423)

## 0. READ THIS FIRST — Instructions for AI Agents

**Do not treat this repository as a finished implementation.** At the time this README was written, this participant repository contained only a minimal README. Code, scripts, trained models, and results described below are **planned deliverables, not existing files**, unless subsequently created and verified.

> **Current best Track 1 method:** see [`BEST_METHOD.md`](BEST_METHOD.md) (dev F1 0.8715 / SoftCons 0.9639 / HardCons 0.9170, milestone 2026-10-31). Start new experiments from that recipe and update the file when a run beats it.

### Mandatory operating rules

1. **Check official sources before consequential decisions.** Organizer GitHub, the task website, and CodaBench override this README if dates, data policy, phases, file schemas, or evaluation procedures change.
2. **Start with Track 1 (English).** Produce a working end-to-end supervised baseline with the official metrics before researching improvements.
3. **Use official labels and names exactly.** There are **four** classes; the CSV output header must be exactly `instance_id,label`.
4. **Keep the official dev split independent of training.** Do not train on dev labels and then report dev scores as unbiased validation. Trial data may only be considered for augmentation **after** checking current rules and auditing duplicates/overlap.
5. **Prevent group leakage.** Do not put the same source `pair_id`, its reversed form, translations, or related mixed-language variants on opposite sides of a new train/validation split.
6. **Never use hidden official test labels.** No manual annotation of hidden gold, submission-based label extraction, or post-hoc information leakage.
7. **Use the organizers' official scorer as the authoritative evaluator.** Do not replace `weighted_f1`, `soft_cons`, or `hard_cons` with an unverified reimplementation.
8. **Log evidence for every result:** model ID/revision, data version and hash, track, code commit, seed, hyperparameters, hardware, training duration, validation scores, and prediction file. Do not invent metrics or claim leaderboard performance before a real submission.
9. **Preserve reproducibility.** Keep scripts/configurations under version control; do not commit credentials, large checkpoints, downloaded model caches, or competition hidden data.
10. **Publish responsibly.** Log external datasets, pretrained models, prompting, ensembles, and their licenses. A submitted run or high rank **does not automatically guarantee paper acceptance**.

### Definition of done (first milestone)

An agent may claim “Track 1 baseline complete” **only when**:

- The official Track 1 train/dev datasets have been downloaded and audited.
- A model has genuinely been fine-tuned **on train only**.
- Predictions cover **every dev `instance_id` exactly once**, with no unknown IDs and valid labels.
- The official scorer successfully produced `scores.json` and `scores.txt`.
- Metrics, configs, model/source revisions, and error analysis are saved in an experiment record.
- No training/development contamination or unsupported result claim is present.

---

## 1. What is the competition?

**DiCo-NLI** is **SemEval 2027 Task 2**, a multilingual **shared task**, not the ACL main-conference research track. SemEval 2027 is a workshop co-located with ACL 2027.

The model receives an **ordered pair of short phrases**, `text1` and `text2` (roughly premise and hypothesis), and predicts their fine-grained semantic inference relation. Unlike ordinary single-direction classification, this task also tests whether predictions **remain logically compatible when phrase order is reversed**.

Example (from the organizer's task description):

| `text1` | `text2` | Correct label |
| --- | --- | --- |
| Everyone is hungry | Someone is hungry | `FORWARD_ENTAILMENT` |
| Someone is hungry | Everyone is hungry | `BACKWARD_ENTAILMENT` |

A model can classify isolated examples reasonably well and still make logically inconsistent predictions after swapping inputs. **Our research objective is to improve correctness *and* reversal consistency** without trading one away unnecessarily.

### Four target labels and their meaning

| Label (exact spelling) | Meaning, from `text1` to `text2` | Reversal |
| --- | --- | --- |
| `EQUIVALENCE` | Both phrases entail one another | `EQUIVALENCE` |
| `FORWARD_ENTAILMENT` | `text1` entails `text2`, but not conversely | `BACKWARD_ENTAILMENT` |
| `BACKWARD_ENTAILMENT` | `text2` entails `text1`, but not conversely | `FORWARD_ENTAILMENT` |
| `NEGATIVE_OTHER` | Other/noise relations outside the three reversible classes | **Not part of official consistency scoring** |

**Important:** `NEGATIVE_OTHER` is a real fourth class for regular classification. The official SoftCons/HardCons calculations are restricted to gold reversible pairs. Do **not** assume the official rules define a semantic reversal transformation for arbitrary `NEGATIVE_OTHER` examples.

The source task builds on **PhrasIS** phrase-level inference examples (from aligned chunks of captions/headlines), not a conventional long-premise NLI dataset. It is related to the *Reversal Curse* literature but does **not** directly test factual-memory reversal.

## 2. Tracks

Participants may submit to one or more tracks; each is evaluated separately.

| Track | Language setting | Public train instances | Public dev instances |
| --- | --- | ---: | ---: |
| **1** | **English monolingual (initial focus)** | **3,042** | **660** |
| 2 | Spanish monolingual | 3,042 | 660 |
| 3 | Basque monolingual | 3,042 | 660 |
| 4 | Mixed multilingual (English/Spanish/Basque phrase combinations) | 18,252 | 3,960 |

Numbers reflect the official September 2026 public release, not the eventual hidden test size.

**Recommended path:** Track 1 → stable official scores → investigate reversal-aware methods → optionally extend to Track 4 with multilingual encoders (e.g., XLM-R). Avoid asserting Track 4 is equivalent to the concatenation of the three monolingual tracks.

## 3. Dataset splits, schemas, and leakage policy

**Download from the organizers' repository, not from this participant repository:** [official `final_data/`](https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI/tree/main/final_data) and [official `trial_data/`](https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI/tree/main/trial_data).

| Split | Purpose | Public labels? | Notes |
| --- | --- | --- | --- |
| **Trial** | Dry-run parsing, training/inference smoke tests, submission format | Yes, in trial labeled/gold files | **Not the official validation/test score** |
| **Train** | Parameter fitting; internal train-only cross-validation | **Yes** | Main training source |
| **Dev** (= validation) | Hyperparameter/model selection and local evaluation | **Yes** | Official public development holdout |
| **Official test** | Final CodaBench evaluation/ranking | **No during evaluation** | Test inputs are released during evaluation; gold remains hidden until after |

### Official released file patterns

Under `final_data/train/` and `final_data/dev/`, for each `track1` ... `track4`:

~~~text
dico_nli_<split>_<track>_participant_labeled.csv
dico_nli_<split>_<track>_submission_template.csv
dico_nli_<split>_<track>_reference.csv
~~~

For example, Track 1:

~~~text
final_data/train/dico_nli_train_track1_participant_labeled.csv
final_data/dev/dico_nli_dev_track1_participant_labeled.csv
final_data/dev/dico_nli_dev_track1_reference.csv
final_data/dev/dico_nli_dev_track1_submission_template.csv
~~~

- `participant_labeled.csv`: includes `instance_id`, `pair_id`, `text1_lang`, `text2_lang`, `text1`, `text2`, `label`. Feed `text1` and `text2` to the model; use `label` for allowed supervised learning/evaluation.
- `reference.csv`: official scorer's gold/reference view. Includes `instance_id`, `pair_id`, `reverse_pair_id` and `label` (plus other columns). `reverse_pair_id` links reversible counterparts.
- `submission_template.csv`: shows the required **two-column** output schema.
- Trial also has `participant_unlabeled.csv` (test-style), `hidden_gold.csv` (**public trial-only gold**) and trial baselines; these are **not official final-test answers**.

### Source-pair grouping is essential

The organizers construct splits at the underlying **source phrase pair** level. A source pair may occur in both phrase orders and be translated into several languages. Preserve this grouping whenever creating internal validation folds. Verify overlap across Train/Dev/Trial using `pair_id` *and* semantic/text audits where necessary. If trial overlaps held-out dev, **do not ingest overlapping examples into train**.

Trial source data includes **461 source pairs** in the organizer's trial release; do not confuse this source-pair count with the number of individual ordered rows in a track.

**Trial-as-extra-training-data policy:** Not the default baseline. Only experiment with `Train + cleaned Trial` after reviewing the final organizer rules, source-pair deduplication, overlap checks, and a reproducible data provenance report. Record any use in the paper.

## 4. Official evaluation: three metrics, not one

All competition metrics are in `[0,1]`, higher is better. The organizers' **official result is a three-score profile**, **not a single mandated composite metric**:

1. **Weighted F1 (`weighted_f1`):** Support-weighted per-label F1 over **all four** classes, measured on individual ordered examples.
2. **SoftCons (`soft_cons`):** Fraction of official **reversible pairs** where predicted labels are directionally compatible. For a pair `(x, reverse(x))`, the predictions must match the reversal mapping. **Can be high even if both predictions are wrong.**
3. **HardCons (`hard_cons`):** Fraction of reversible pairs where **both predictions are exactly correct**. This is stricter than isolated prediction accuracy.

The official scorer **excludes gold `NEGATIVE_OTHER` examples from SoftCons/HardCons**, while including them in Weighted F1. If a reversible gold pair is predicted as `NEGATIVE_OTHER`, it does **not** satisfy SoftCons.

Reversal mapping for the **reversible subset**:

~~~text
EQUIVALENCE          -> EQUIVALENCE
FORWARD_ENTAILMENT   -> BACKWARD_ENTAILMENT
BACKWARD_ENTAILMENT  -> FORWARD_ENTAILMENT
~~~

**Never improve SoftCons by blindly predicting `EQUIVALENCE` everywhere.** It may raise self-consistency but damage Weighted F1 and HardCons. Always report the full profile, along with confusion matrix/per-label F1 and pair-level errors.

### Valid submission file

Exactly these two columns, with **one unique row for every evaluation instance**:

~~~csv
instance_id,label
example_001,FORWARD_ENTAILMENT
example_002,EQUIVALENCE
~~~

The IDs above are illustrative; in real submissions, preserve the organizer-provided `instance_id` strings. No score columns, text columns, index column, missing rows, duplicate IDs, or extra headers.

## 5. Local workflow and official scorer

**Current repo status:** Documentation/plan only; there are no local training scripts yet. Agents should create and test the implementation instead of inventing commands for nonexistent modules.

### Download the official resources

From **this participant repo's root**, one reproducible setup option is:

~~~bash
mkdir -p external outputs
git clone https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI.git external/dico_nli_official
python3 --version   # Python >= 3.10 required for official scorer
~~~

Pin/log the upstream commit SHA for every experiment. The official repository provides a Hugging Face starter kit under `starter_kit/` and an official standalone scorer under `evaluation_functions/`.

### Starter kit example — supervised English baseline

This example is run from the **official starter kit directory**. Use compatible GPU/PyTorch packages for the actual environment; reduce batch size if necessary.

~~~bash
cd external/dico_nli_official/starter_kit
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 main.py \
  --model_path answerdotai/ModernBERT-base \
  --tokenizer_path answerdotai/ModernBERT-base \
  --do_train \
  --train_path ../final_data/train/dico_nli_train_track1_participant_labeled.csv \
  --do_dev \
  --dev_path ../final_data/dev/dico_nli_dev_track1_participant_labeled.csv \
  --no-do_test \
  --seed 42 \
  --dropout 0.1 \
  --warmup_pcrt 0.01 \
  --lr 5e-5 \
  --batch_size 8 \
  --wd 1e-5 \
  --num_epochs 3 \
  --no-is_optuna_trial \
  --max_length 128 \
  --results_dir ../../../outputs/track1_modernbert_seed42
~~~

This is a **starter configuration**, not a claim about achieved results. Check the [official starter kit README](https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI/blob/main/starter_kit/README.md) for the latest CLI and output filenames. ModernBERT is a convenient smoke/baseline choice; consider DeBERTa-v3-base as another Track 1 encoder baseline.

### Official local score on labeled Dev

Once an agent has produced a valid prediction CSV, run from `external/dico_nli_official/`:

~~~bash
python3 -m evaluation_functions \
  --gold final_data/dev/dico_nli_dev_track1_reference.csv \
  --predictions ../../outputs/track1_dev_predictions.csv \
  --output-dir ../../outputs/track1_dev_scores
~~~

The scorer writes `scores.json` and `scores.txt` to the output directory. The `--predictions` file **must already exist**; its name above is a target artifact, **not** a promise the starter kit writes that exact filename. Find/copy/rename the actual generated predictions before scoring.

The official validator rejects missing/unknown/duplicate IDs, wrong labels, unexpected columns, malformed CSVs, and inconsistent reference pair links. **A locally validated Dev score is not a final-test leaderboard score.**

### CodaBench: what is being scored?

The organizers announced a **development phase** for testing submission formats on **2026-09-08**. The public documentation **does not unambiguously identify which hidden CodaBench phase data are used for each development submission**; read the platform's current phase instructions rather than assuming it evaluates the published Dev file. **Official evaluation starts in January 2027.** Development scores must not be reported as final official ranks.

## 6. Research roadmap for agents

### Stage A — Reproducible baseline (P0)

**Task:** Implement an encoder classifier with a robust CSV input → tokenize ordered (`text1`, `text2`) → train → predict → validate flow.

**Suggested experiments:** ModernBERT-base / DeBERTa-v3-base, 3+ seeds if compute permits, class distribution checks, early stopping on public dev, and train-only grouped CV when needed.

**Deliverables:** parser + schema tests, baseline training/inference scripts, model config, predictions on Dev, official `scores.json`, and a short experiment report.

### Stage B — Direction-aware improvements (P1)

Try controlled ablations, one factor at a time:

| Candidate idea | Scientific question |
| --- | --- |
| **Reverse-pair augmentation** | Does showing both directions improve HardCons without hurting F1? |
| **Consistency regularization** | Can the forward and reversed predictive distributions be aligned under the reversible label mapping? |
| **Bidirectional inference/ensemble** | Does combining predictions from both input orders improve pair-level correctness? |
| **Loss weighting / calibration** | Can we improve hard cases, especially entailment direction versus equivalence? |
| **Group-aware validation / error buckets** | Which phrase types, lengths, or reversals remain problematic? |

A possible **research hypothesis** (not an established improvement): use a supervised cross-entropy objective plus a reversible-pair consistency regularizer:

~~~text
L_total = L_supervised + lambda * L_consistency
~~~

Use the **official reversible-label mapping** in the consistency term, handle `NEGATIVE_OTHER` separately, and compare against a classification-only baseline. Do not silently change the official evaluation to match a custom training loss.

### Stage C — Optional multilingual transfer (P2)

Only after Track 1 is reproducible, try Tracks 2/3/4 with multilingual encoders; report each track separately. Track 4 mixes languages within pairs and measures cross-lingual directional robustness. Make sure translated/source-group variants cannot leak into different validation folds.

### Stage D — Official test, paper, and release (P3)

Before official evaluation: finalize model selection using valid dev experiments, freeze config/checkpoint, verify inference can run without gold labels, and validate CSV output against organizer templates.

During official evaluation: use **only allowed test inputs**, submit exact predictions to the **correct CodaBench track/phase** and respect submission limits. Record the selected run/submission ID and platform scores.

For the system-description paper, write: task context; methods; data and external resources; validation strategy; official metrics; baseline comparisons; ablations; seeds/variance; error analysis; limitations; reproducibility. **A system-description paper is required for inclusion in the official DiCo-NLI SemEval ranking, per organizer documentation; paper publication still depends on peer review.** Accepted SemEval papers belong to **workshop proceedings**, not the ACL main-conference proceedings.

## 7. Planned project structure (NOT implemented yet)

Agents should prefer small, testable components. This is a **suggested target layout**, not a description of files already present:

~~~text
DiCo-NLI/
├── README.md
├── configs/                  # tracked hyperparameter/model configs
├── src/
│   ├── data/                 # schema, pair-group audits, loaders
│   ├── models/               # encoder and consistency models
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py           # thin wrapper around OFFICIAL scorer
├── tests/                    # parsing, reversal, submission-format tests
├── experiments/              # human-readable summaries/ablation tables
├── paper/                    # manuscript sources (when started)
├── external/                 # upstream checkout; generally untracked
└── outputs/                  # predictions, checkpoints, scores; generally untracked
~~~

**Before adding a new file:** check what the repository already contains; follow existing structure if one has emerged. Do not overwrite other agents' work.

### Experiment record requirements

For each run, log at least:

~~~yaml
run_id: track1_model_seed42
stage: baseline
track: 1
dataset_source: official final_data release
dataset_revision: <commit SHA>
train_file: <path>
validation_file: <path>
group_overlap_audit: <summary>
model_id: <Hugging Face ID or local path>
model_revision: <revision>
code_commit: <commit SHA>
seed: 42
hyperparameters: <config path or dictionary>
external_data: none
hardware: <GPU / RAM>
predictions_csv: <path>
official_scores_json: <path>
notes: <actual observations, not estimates>
~~~

Use the organizer's exact score names in results. Tables must clearly label **trial/dev/test** and **track** so no pilot, internal, or public-dev number is passed off as a held-out competition score.

## 8. Tentative competition timeline (as checked 2026-10-23)

| Milestone | Organizer timeline |
| --- | --- |
| Public train and dev data released | **2026-09-08** (done) |
| CodaBench development phase opened | **2026-09-08** (done) |
| Evaluation data ready internally | **2026-12-01** (**not** public test release) |
| **Official evaluation opens** | **2027-01-10** |
| **Official evaluation closes** | **By 2027-01-31** (confirm task-specific cutoff) |
| System paper submission | **February 2027** (exact day TBD) |
| Author notification | **March 2027** (tentative) |
| Camera-ready | **April 2027** (tentative) |
| SemEval workshop alongside ACL 2027 | **Summer 2027**, Kyoto |

Treat all future dates as **subject to organizer updates**. Final test ground truth stays hidden during competition and is expected to be released only afterward. Development submissions do not substitute for a final evaluation submission.

## 9. External resources and scientific background

- [Official DiCo-NLI repository](https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI) — authoritative README, datasets, scoring source code, rules.
- [Official final train/dev data](https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI/tree/main/final_data).
- [Official trial data](https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI/tree/main/trial_data).
- [Official evaluation functions](https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI/tree/main/evaluation_functions).
- [Official Hugging Face starter kit](https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI/tree/main/starter_kit).
- [SemEval 2027](https://semeval.github.io/SemEval2027/); [CodaBench](https://www.codabench.org/competitions/18038/).
- [LREC 2026 pilot: *Assessing Logical Consistency in Fine-Grained NLI*](https://lrec.elra.info/lrec2026-main-423). Pilot numbers are **not** directly comparable to official 2027 scores.
- [PhrasIS background](https://doi.org/10.1093/jigpal/jzae037).
- [Reversal Curse background](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5178b2f2d7c44aa390c0777dc77b3f0c-Abstract-Conference.html).

**Citation note:** The organizers' current official README still lists the DiCo-NLI task citation as TODO. Check for an official BibTeX entry before submitting a manuscript; do not invent one.

---

**Project success is defined by reproducible experiments, honest official validation, a valid competition submission, and a strong system-description paper — not leaderboard score alone.**
