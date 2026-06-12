# CIP Classifier

Classify research abstracts against the [CIP taxonomy](https://nces.ed.gov/ipeds/cipcode/) for field-of-science reporting. Built for TACC HPC allocation request abstracts.

## Project Stages

| Stage | Command | Description |
|-------|---------|-------------|
| 0 | `cip-classifier baseline --step parse` | Parse CIP taxonomy from Excel → JSON |
| 1 | `cip-classifier generate` | Generate synthetic training data via LLM |
| 2 | `cip-classifier train` | Train classifier (embedding head or SetFit) |
| 3 | `cip-classifier baseline` | Embedding retrieval baseline (FAISS) |
| — | `cip-classifier evaluate` | Evaluate classification results |
| — | `cip-classifier visualize` | UMAP/t-SNE embedding plots |

## Quick Start

```bash
# Install in editable mode
pip install -e .

# Run full baseline pipeline (parse → index → classify → evaluate)
cip-classifier baseline

# Individual steps
cip-classifier baseline --step build-index
cip-classifier baseline --step classify
cip-classifier evaluate

# Override config for Vista HPC
cip-classifier baseline -c configs/vista.yaml

# Generate synthetic abstracts (requires LLM inference server)
cip-classifier generate -c configs/generate.yaml

# Train classifier on synthetic data
cip-classifier train -c configs/train.yaml
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
# Full baseline pipeline (single GH node)
sbatch -A <alloc> slurm/run_pipeline.sbatch

# Generate synthetic data (9 nodes, DeepSeek-R1 671B)
sbatch -A <alloc> slurm/generate_multinode.sbatch

# Train classifier (single node)
sbatch -A <alloc> slurm/train.sbatch
```

## Project Structure

```
├── configs/                     # YAML configuration
│   ├── default.yaml             # Base config (paths, models, eval)
│   ├── generate.yaml            # LLM generation settings
│   ├── train.yaml               # Classifier training params
│   └── vista.yaml               # TACC Vista HPC overrides
├── scripts/                     # Standalone entry points (for SLURM)
├── src/cip_classifier/          # Library + CLI
│   ├── __main__.py              # Click CLI (cip-classifier command)
│   ├── config.py                # Pydantic config loading
│   ├── utils.py                 # Encoding, FAISS I/O, model loading
│   ├── data/                    # Taxonomy parsing, splitting
│   ├── generation/              # LLM abstract generation pipeline
│   ├── models/                  # Classifier implementations
│   ├── baselines/               # FAISS retrieval baseline
│   └── evaluation/              # Metrics, comparison
├── slurm/                       # SLURM job scripts for Vista
├── data/
│   ├── raw/                     # Input Excel files (gitignored)
│   ├── processed/               # Taxonomy JSONs, sibling fields
│   └── generated/               # Synthetic abstracts (gitignored)
├── output/                      # All artifacts (gitignored)
│   ├── models/                  # Trained classifiers
│   ├── index/                   # FAISS index + metadata
│   ├── results/                 # Classification results
│   └── reports/                 # Evaluation reports
└── pyproject.toml
```

## Switching Models

```yaml
# configs/experiment_e5.yaml
models:
  index_encoder: intfloat/e5-mistral-7b-instruct
  query_encoder: intfloat/e5-mistral-7b-instruct
  query_prefix: "query: "
```

```bash
cip-classifier baseline -c configs/experiment_e5.yaml -c configs/vista.yaml
```
