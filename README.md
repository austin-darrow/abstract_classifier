# CIP Classifier

Classify research abstracts against the [CIP taxonomy](https://nces.ed.gov/ipeds/cipcode/) for field-of-science reporting. Built for TACC HPC allocation request abstracts.

Trains on LLM-generated synthetic abstracts + high-confidence pseudo-labels from real TACC data. Uses a single unified SciBERT model (315 detailed classes) with logit marginalization for major-field predictions.

## Current Results

| Model | Detailed Acc | Major Acc | Macro F1 |
|-------|-------------|-----------|----------|
| **Single unified model (D2c, Strategy C)** | **87.3%** | **93.0%** | **0.926** |
| Hierarchical two-model (D2b) | — | 94.1% | 0.936 |
| C4 sweep best (pre-D2 data) | — | 94.0% | 0.937 |

All metrics on synthetic test set (n=4,054). Real TACC DB labels have ~55% noise rate, making traditional accuracy metrics against them unreliable. See [docs/project_report.md](docs/project_report.md) for the full project report.

## Quick Start

```bash
# Install in editable mode
pip install -e .

# Predict with the best major-field model
cip-classifier finetune --model-path output/sweep/models/scibert_scivocab_uncased_lr3e-05_ep8_bs16_ls0.0_wd0.01_linear_freeze8/

# Run evaluation
cip-classifier evaluate

# Override config for Vista HPC
cip-classifier finetune -c configs/vista.yaml

# Generate synthetic abstracts (requires LLM inference server)
cip-classifier generate -c configs/generate.yaml
```

If `configs/default.yaml` exists, it's loaded automatically. Additional `-c` flags
override on top of it.

## Configuration

YAML files under `configs/`, composable via `-c`:

- **`configs/default.yaml`** — Base config (paths, models, evaluation params)
- **`configs/generate.yaml`** — LLM generation (DeepSeek-R1, server settings)
- **`configs/train.yaml`** — Classifier training (model type, hyperparams)
- **`configs/vista.yaml`** — TACC Vista HPC overrides (GPU, batch sizes)

## Running on TACC Vista

### First-time setup

```bash
idev -p gh-dev -N 1 -n 1 -t 00:30:00
bash slurm/setup_env.sh
```

### Submit jobs

```bash
# Hyperparameter sweep (SciBERT, 47 configs)
sbatch -A <alloc> slurm/run_sweep.sbatch

# Fine-tune SciBERT (single config)
sbatch -A <alloc> slurm/run_finetune.sbatch

# Train detailed-field model (315 classes)
sbatch -A <alloc> slurm/run_detailed_finetune.sbatch

# Hierarchical inference (major + detailed)
sbatch -A <alloc> slurm/run_hierarchical.sbatch
```

## Key Scripts

| Script | Purpose |
|--------|---------|
| `scripts/sweep_finetune.py` | Hyperparameter sweep over SciBERT configs |
| `scripts/run_detailed_finetune.py` | Train SciBERT on 315 detailed CIP fields |
| `scripts/run_hierarchical_inference.py` | Combine major + detailed models via constrained decoding |
| `scripts/generate_charts.py` | Generate publication-quality result charts |
| `scripts/build_silver_labels.py` | Build pseudo-labels from model consensus |
| `scripts/audit_labels.py` | Analyze DB label noise vs. model predictions |
| `scripts/eval_clean.py` | Evaluate on trusted (model+DB agreement) subset |
| `scripts/analyze_sweep.py` | Analyze and rank sweep results |
| `scripts/evaluate.py` | Run standard evaluation metrics |
| `scripts/prepare_data.py` | Prepare CIP taxonomy and training data |
| `scripts/generate_abstracts.py` | Generate synthetic training abstracts |
| `scripts/train_classifier.py` | Train classifier (legacy entry point) |

## Project Structure

```
├── configs/                     # YAML configuration
│   ├── default.yaml             # Base config (paths, models, eval)
│   ├── generate.yaml            # LLM generation settings
│   ├── train.yaml               # Classifier training params
│   └── vista.yaml               # TACC Vista HPC overrides
├── scripts/                     # Standalone entry points (for SLURM)
│   ├── sweep_finetune.py        # C4: hyperparameter sweep
│   ├── run_detailed_finetune.py # C5a: detailed-field training
│   ├── run_hierarchical_inference.py  # C5b: constrained decoding
│   ├── generate_charts.py       # Publication charts
│   └── ...                      # Data prep, evaluation, silver labels
├── src/cip_classifier/          # Library + CLI
│   ├── __main__.py              # Click CLI (cip-classifier command)
│   ├── config.py                # Pydantic config loading
│   ├── utils.py                 # Encoding, FAISS I/O, model loading
│   ├── data/                    # Taxonomy parsing, splitting
│   ├── generation/              # LLM abstract generation pipeline
│   ├── models/                  # SetFit model implementation
│   ├── baselines/               # FAISS retrieval + SciBERT fine-tune
│   └── evaluation/              # Metrics, comparison
├── slurm/                       # SLURM job scripts for Vista
│   ├── run_sweep.sbatch         # Hyperparameter sweep
│   ├── run_finetune.sbatch      # Single SciBERT training
│   ├── run_detailed_finetune.sbatch  # Detailed-field training
│   └── run_hierarchical.sbatch  # Hierarchical inference
├── data/
│   ├── raw/                     # Input Excel files (gitignored)
│   ├── processed/               # Taxonomy JSONs, sibling fields
│   └── generated/               # Synthetic abstracts (gitignored)
├── output/                      # All artifacts (gitignored)
│   ├── models/                  # Trained classifiers
│   ├── sweep/                   # Sweep results + best models
│   ├── index/                   # FAISS index + metadata
│   └── reports/                 # Charts, evaluation reports
├── archive/                     # Superseded code and outputs
│   ├── baselines/               # B2 TF-IDF, B3 embedding head, B6 zero-shot
│   ├── scripts/                 # B0 baseline, B1 kNN, old visualize
│   ├── slurm/                   # Old generation/pipeline scripts
│   └── output/                  # Old prediction files
├── docs/                        # Documentation
│   ├── project_report.md        # Full project report (methods, results, decisions)
│   ├── presentation.html        # Interactive presentation for TACC staff
│   └── sprint_review.html       # Sprint review presentation
└── pyproject.toml
```

## Experiment History

See [docs/project_report.md](docs/project_report.md) for the full project report. Summary:

| Phase | Description | Key Result |
|-------|-------------|------------|
| **B0–B6** | Baseline approaches (FAISS, kNN, TF-IDF, embedding head, SetFit, SciBERT, zero-shot LLM) | Best real TACC: B0 FAISS at 34.3% — synthetic-trained models overfit |
| **C0** | Repo cleanup, archive B2/B3/B6 | — |
| **C1** | SetFit bge-large upgrade | Underperformed SciBERT (90.6% vs 92.8%) |
| **C2** | Label quality audit | ~55% DB label noise rate discovered |
| **C3** | Silver labels + retraining | 5,327 pseudo-labels; SciBERT → 92.8% major |
| **C4** | Hyperparameter sweep (47 configs) | **93.96% major** (lr=3e-5, 8ep, freeze8) |
| **C5a** | Detailed-field SciBERT (315 classes) | 87.94% detailed accuracy |
| **C5b** | Hierarchical constrained decoding | **88.75% detailed, 94.06% major** |
