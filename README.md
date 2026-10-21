PepTox

Multi-task deep learning for joint prediction of antimicrobial activity, cytotoxicity, and hemolytic potency in therapeutic peptides.

Overview

PepTox predicts four endpoints for a therapeutic peptide from its sequence in a single pass:

Antimicrobial activity (binary classification)
Cytotoxicity against mammalian cells (binary classification)
Hemolytic classification (binary; auxiliary supervision)
Hemolytic potency — log10(HC50) regression

The model is trained on 36,729 database records representing 35,232 unique peptide sequences drawn from six public databases: DRAMP 3.0, HemoPI, Hemolytik 2.0, ToxinPred3, DBAASP v3, and UniProt, together with three synthetic AMP-negative decoy sets.

Repository contents

Top-level files

README.md — this file
LICENSE — MIT
requirements.txt — Python dependencies
.gitignore

data/

master_v6.csv — harmonized corpus (36,729 records; 35,232 unique peptide sequences)


src/

model.py — PepTox architecture
train.py — multi-task training (reproduces Table 3)
train_stl.py — single-task ablation trainer (reproduces Table 5)
hemopi2_baseline.py — HemoPI2 baseline comparison (Table 4)
predict.py — inference entry point

models/

peptox_best.pt — trained checkpoint (best of 5 seeds)
Installation

Create a conda environment and install dependencies:

conda create -n peptox python=3.10
conda activate peptox
pip install -r requirements.txt

Python 3.10+ and a CUDA-capable GPU are recommended for training and feature extraction.

Quick start — inference

Feature extraction uses the domain-adapted ESM-2 models described in the manuscript. Users wishing to predict on new peptides first run the full feature-extraction pipeline (see Methods), then:

python src/predict.py \
    --checkpoint models/peptox_best.pt \
    --sequences data/example_peptides.csv \
    --features-dir data/features \
    --output predictions.csv

The output is a CSV with per-peptide columns: p_antimicrobial, p_hemolytic, p_cytotoxic, log10_hc50_pred, and hc50_pred_uM.

Training

Prepare the five feature streams described in Methods (DA-35M, DA-650M, biophysical descriptors, mined motifs, cyto-specific descriptors) and then run:

python src/train.py --seed 0

Repeat for seeds 1–4 to produce the 5-seed ensemble used in the manuscript.

Reproducing paper results
Table 3 (main performance) — train 5 seeds via src/train.py and evaluate the ensemble
Table 4 (HemoPI2 comparison) — python src/hemopi2_baseline.py
Table 5 (STL ablation) — python src/train_stl.py --task cyto and --task potency, each over 5 seeds
Data

master_v6.csv contains the harmonized corpus with the following columns:

sequence — peptide amino-acid sequence
source — source database(s), comma-separated for cross-listed records
gen_label — binary antimicrobial label (1 / 0 / -1)
hemo_label — binary hemolytic label (1 / 0 / -1)
cyto_label — binary cytotoxicity label (1 / 0 / -1)
micromolar — HC50 in μM (-1 if absent; values ≥999 are non-hemolytic sentinels)

A sentinel value of -1 indicates no label is available for that task; such rows are excluded from the corresponding task’s loss during training.

HemoPI2 is used only as an external benchmark comparator (src/hemopi2_baseline.py). It is not part of the training corpus.

Checkpoint

Trained checkpoint — ## Download the model checkpoint

Before running inference, download peptox_best.pt from the Releases page and place it in models/:

https://github.com/deepdrugai/PepTox/releases/latest

The checkpoint corresponds to the best of 5 seeds (seed 1, validation composite 0.708); manuscript numbers come from the full 5-seed ensemble.


Citation

Citation will be added upon acceptance.

License

MIT — see LICENSE.
