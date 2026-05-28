# CIP Classifier

Classify research abstracts against the [CIP taxonomy](https://nces.ed.gov/ipeds/cipcode/) using embedding similarity and FAISS nearest-neighbor retrieval.

## Pipeline Steps

| Step | Command | Description |
|------|---------|-------------|
| 0 | `parse` | Parse CIP taxonomy from Excel → JSON |
| 1 | `build-index` | Embed taxonomy, build FAISS index |
| 2 | `classify` | Embed abstracts, retrieve top-K, majority vote |
| 3 | `evaluate` | Compare predictions to existing labels |
| — | `run-all` | Run steps 0–3 sequentially |

## Quick Start (Local)

```bash
# Install in editable mode
pip install -e .

# Run full pipeline with default config
python -m cip_classifier run-all

# Or run individual steps
python -m cip_classifier parse
python -m cip_classifier build-index
python -m cip_classifier classify
python -m cip_classifier evaluate

# Use a custom config
python -m cip_classifier classify --config config/default.yaml --config config/vista.yaml
```

## Configuration

All parameters live in YAML files under `config/`:

- **`config/default.yaml`** — Base configuration (model names, paths, hyperparameters)
- **`config/vista.yaml`** — TACC Vista overrides (larger batch sizes, bigger models)

Multiple `--config` flags are supported; later files override earlier ones.

### Key config options

```yaml
models:
  index_encoder: BAAI/bge-large-en-v1.5   # Model for taxonomy embeddings
  query_encoder: BAAI/bge-large-en-v1.5   # Model for abstract embeddings
  query_prefix: ""                         # Optional prefix for query model

classify:
  top_k: 10          # Neighbors for majority vote
  batch_size: 32     # Encoding batch size

runtime:
  device: auto       # auto | cuda | cpu
```

## Running on TACC Vista

### First-time setup

```bash
# Get an interactive node
idev -p gh-dev -N 1 -n 1 -t 00:30:00

# Run setup script (creates venv on $SCRATCH, installs deps + project)
bash slurm/setup_env.sh
```

### Submit jobs

```bash
# Full pipeline (single GH node, ~6 hrs)
sbatch -A <your_allocation> slurm/run_pipeline.sbatch

# Or individual steps
sbatch -A <your_allocation> slurm/build_index.sbatch
sbatch -A <your_allocation> slurm/classify.sbatch
```

### Data placement on Vista

Place input files where the config expects them:
```bash
cp SEDCIP24_TACC.xlsx   /path/to/project/data/raw/
cp database_abstracts.xlsx /path/to/project/data/raw/
```

## Project Structure

```
├── config/                  # Pipeline configuration (YAML)
│   ├── default.yaml
│   └── vista.yaml
├── src/cip_classifier/      # Python package
│   ├── __main__.py          # CLI entry point
│   ├── config.py            # Config loading & validation
│   ├── parse_fields.py      # Step 0
│   ├── build_index.py       # Step 1
│   ├── classify.py          # Step 2
│   ├── evaluate.py          # Step 3
│   └── utils.py             # Shared utilities
├── slurm/                   # SLURM job scripts for Vista
├── data/
│   ├── raw/                 # Input Excel files (gitignored)
│   └── processed/           # Parsed taxonomy JSONs
├── output/                  # All generated artifacts (gitignored)
│   ├── index/               # FAISS index + metadata
│   ├── results/             # Classification results
│   └── reports/             # Evaluation reports
└── pyproject.toml           # Package definition & dependencies
```

## Switching Models

To experiment with different embedding models, edit the config or create a new override:

```yaml
# config/experiment_e5.yaml
models:
  index_encoder: intfloat/e5-mistral-7b-instruct
  query_encoder: intfloat/e5-mistral-7b-instruct
  query_prefix: "query: "
```

Then run:
```bash
python -m cip_classifier run-all -c config/default.yaml -c config/experiment_e5.yaml
```
