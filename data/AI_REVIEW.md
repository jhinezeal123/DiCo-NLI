# Single-AI paraphrase review and matched pilot

The 320 original selected source rows and their NLI-screened paraphrases come
from the completed `f443159` pilot (run `66608dac-0862-4a38-ae39-3fe9f805886d`).
One AI reviewer, already familiar with the dev results and prior audit, examined
all 242 distinct source/replacement pairs without using dev rows as training
examples. It retained 141 pairs, covering 189 of the 320 selected train rows,
and marked the remainder uncertain or rejected. Review decisions and reasons
are in `cb-ai-paraphrase-review-all.csv`; the exact source rows and replacements
are frozen in `cb-ai-reviewed-selected.csv`. This is not human annotation, an
independent/blind review, or a guarantee of meaning preservation. Source labels
are inherited unchanged and were not independently validated.

The treatment trains on the original train set plus the 189 reviewed edits;
the control repeats the exact 189 source rows. Both retain the class distribution,
seeds 42/13/7, and 1143 optimizer steps. Each run checks source text and label,
excludes any selected original or modified text-pair matching participant dev,
and logs a hash of selected source IDs. The original train/dev overlap remains.
Both arms inherit the official `bash run.sh` contract and official scorer.

The comparison isolates replacing versus repeating these selected rows; it does
not isolate an AI-review effect against the previous 320-row pilot because the
selected rows and augmentation count differ. Dev and B/C subsets are exploratory.
