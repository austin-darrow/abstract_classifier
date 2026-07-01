# CIP Classifier — Execution Plan

## Objective

Classify research abstracts into CIP taxonomy fields (74 major fields, 22 broad fields). Train on LLM-generated synthetic abstracts + high-confidence real TACC pseudo-labels, evaluate on manually-verified subset of real TACC abstracts.

## Success Criteria

| Metric | Target | Current Best | Notes |
|--------|--------|--------------|-------|
| Major field accuracy (synthetic test) | >70% | **93.96%** (C4) | SciBERT lr=3e-5, 8ep, freeze8 |
| Broad field accuracy (synthetic test) | >90% | **94.47%** (C4) | Same model |
| Detailed field accuracy (synthetic test) | — | **88.75%** (C5b) | Hierarchical constrained decoding |
| Silver label set size | >5,000 | **5,327** (C3) | B4+B5 consensus at conf ≥ 0.7 |

**Key insight:** The classifier is likely more accurate than existing TACC database labels. DB labels are NOT treated as ground truth.

---

## Data

| Dataset | Size | Purpose |
|---------|------|---------|
| Synthetic train | 16,183 | Base training data |
| Synthetic test | 4,054 | Development evaluation |
| Real TACC abstracts | 16,209 (46 of 74 fields) | Silver label source + final evaluation |
| Silver labels | 5,327 | High-confidence pseudo-labels (B4+B5 consensus) |
| Combined training set | 21,510 | Synthetic + silver (used by C3/C4/C5) |

- Generation: DeepSeek-R1 (671B, FP8) on Vista GH200 × 9 nodes
- Adversarial verification: same model verifies each abstract against sibling fields
- Train/test split: 80/20 stratified by major field
- Silver labels: model consensus (B4 + B5 agree) at confidence > threshold

---

## Phase A: Evaluation Framework ✅ DONE

Unified framework so every approach produces comparable results automatically.

| Step | Description | Status |
|------|-------------|--------|
| A1 | Standardized prediction format (`Prediction`, `PredictionSet`) | ✅ |
| A2 | Unified metrics (accuracy, F1, top-k, per-field, confusion) | ✅ |
| A3 | Multi-model comparison tables + agreement analysis | ✅ |
| A4 | Visualization (bar charts, confusion matrices, histograms) | ✅ |
| A5 | Training diagnostics (learning curves, error analysis) | ✅ |
| A6 | CLI: `cip-classifier compare` | ✅ |

---

## Phase B: Classifier Approaches ✅ DONE (archived)

All approaches evaluated. B4 (SetFit) and B5 (SciBERT fine-tune) selected as most promising. Others archived.

| # | Approach | Synth Major Acc | Real TACC Major Acc | Status |
|---|----------|-----------------|---------------------|--------|
| B0 | FAISS baseline (CIP defs) | 52.2% | **34.3%** | Archived (reference) |
| B1 | kNN on synthetic | 85.6% | 25.9% | Archived |
| B2 | TF-IDF + LogReg | 82.2% | 24.9% | Archived |
| B3 | Embedding head (MLP) | 88.6% | 19.6% | Archived |
| B4 | **SetFit** | 91.9% | 25.8% | **Active** |
| B5 | **SciBERT fine-tune** | 90.8% | **32.3%** | **Active** |
| B6 | Zero-shot LLM (32B) | 15.7% | 31.3% | Archived |

**Key findings:**
- Inverse correlation: more synthetic accuracy → worse real accuracy (distribution shift)
- B0 baseline beats all trained models on real data — problem is training distribution, not model capacity
- Same fields get 0% F1 across ALL approaches (ME, CS, Bio general) — likely mislabeled in DB
- Classifier appears more accurate than database labels based on manual review

---

## Phase C: Close the Synthetic→Real Gap

### C0. Repo Cleanup ✅ DONE

| Step | Action | Status |
|------|--------|--------|
| C0.1 | Archive B2 (tfidf), B3 (embedding_head), B6 (zeroshot) code → `archive/` | ✅ |
| C0.2 | Archive all prediction/metric files for B0-B3, B6 → `archive/output/predictions/` | ✅ |
| C0.3 | Archive completed generation SLURM scripts → `archive/slurm/` | ✅ |
| C0.4 | Update model dispatcher to remove embedding_head, add finetune | ✅ |

---

### C1. SetFit Upgrade (bge-large) ✅ DONE

Upgraded from `bge-base-en-v1.5` to `bge-large-en-v1.5`. Result: **underperformed** (90.6% vs SciBERT 92.8% on synthetic test). SetFit's pair-based contrastive training struggles to optimize the larger encoder at this scale.

| Step | Action | Status |
|------|--------|--------|
| C1.1 | Update `configs/train.yaml`: encoder → `BAAI/bge-large-en-v1.5` | ✅ |
| C1.2 | Re-run SetFit training on synthetic + silver data | ✅ |
| C1.3 | Evaluate on synthetic test + real TACC | ✅ |
| C1.4 | Compare to previous bge-base results | ✅ |

---

### C2. Label Quality Audit ✅ DONE

**Goal:** Quantify label noise in TACC data. Determine which 0% F1 fields are genuinely hard vs. mislabeled.

**Result:** ~55% of DB labels disagree with model consensus. DB labels are unreliable as ground truth.

| Step | Action | Status |
|------|--------|--------|
| C2.1 | Run `scripts/audit_labels.py` on B4/B5 predictions | ✅ |
| C2.2 | Review diagnostic charts (confidence vs. agreement, per-field plots) | ✅ |
| C2.3 | For problem fields (ME, CS, Bio general): read samples, determine if DB or model is right | ✅ |
| C2.4 | Establish per-field "label trust score" | ✅ |
| C2.5 | Determine confidence threshold for silver labels (informed by audit) | ✅ |

**Charts produced:**
- `confidence_vs_db_agreement.png` — At each threshold, what % of agreed predictions match DB?
- `agreement_rate_by_field.png` — Which fields do models agree on most?
- `problem_field_*.png` — What do models predict instead for 0% F1 fields?
- `confidence_distribution_agreement.png` — Confidence histogram split by DB match

**Script:** `scripts/audit_labels.py`

---

### C3. Silver Labels & Semi-Supervised Retraining ✅ DONE

**Goal:** Bridge synthetic→real gap by training on real TACC abstracts with pseudo-labels.

**Result:** 5,327 silver labels at confidence ≥ 0.7. SciBERT retrained to 92.8% major acc (+2.0% over synth-only). SetFit bge-large underperformed.

| Step | Action | Status |
|------|--------|--------|
| C3.1 | Run `scripts/build_silver_labels.py --threshold 0.7` | ✅ |
| C3.2 | Verify silver label quality (spot check, field distribution) | ✅ |
| C3.3 | Create combined training set: synthetic + silver (21,510 total) | ✅ |
| C3.4 | Retrain B5 (SciBERT) on combined data | ✅ |
| C3.5 | Retrain B4 (SetFit bge-large) on combined data | ✅ |
| C3.6 | Evaluate on synthetic test | ✅ |

---

### C4. Hyperparameter Tuning ✅ DONE

**Goal:** Squeeze more from SciBERT after silver labels are incorporated.

**Result:** 47-config sweep. Best: SciBERT lr=3e-5, 8ep, freeze8 at **93.96% major, 93.69% macro F1** (+1.2% over C3).

| Param | B5 SciBERT |
|-------|------------|
| Swept | lr={1e-5, 2e-5, 3e-5, 5e-5}, epochs={3,5,8,10}, schedulers, label_smoothing, freeze_layers, DeBERTa/BiomedBERT |
| Winner | lr=3e-5, 8ep, bs=16, freeze_layers=8, linear scheduler |

Best model: `output/sweep/models/scibert_scivocab_uncased_lr3e-05_ep8_bs16_ls0.0_wd0.01_linear_freeze8/`

---

### C5. Detailed-Field Classification & Hierarchical Inference ✅ DONE

**Goal:** Classify at the detailed CIP level (315 classes) and improve accuracy via hierarchical constrained decoding.

**Approach:** Instead of a cascade (broad→major as originally planned), we trained a flat detailed-field model and combined it with the C4 major-field model at inference time using constrained decoding.

| Step | Action | Status |
|------|--------|--------|
| C5.1 | Train SciBERT on 315 detailed CIP fields (synth only — silver labels lack detailed annotations) | ✅ |
| C5.2 | Evaluate flat detailed model (87.94% acc, 69.92% macro F1) | ✅ |
| C5.3 | Implement constrained decoding: major model constrains detailed model's output space | ✅ |
| C5.4 | Evaluate 4 strategies (top-1 mask, top-k max, combined, weighted) | ✅ |
| C5.5 | Select winner: Strategy C (combined = major_prob × detailed_prob) | ✅ |

**Results:**
| Model | Detailed Acc | Major Acc (rolled up) |
|-------|-------------|----------------------|
| Flat detailed (C5a) | 87.94% | 92.77% |
| **Hierarchical C (C5b)** | **88.75%** | **94.06%** |

**Scripts:** `scripts/run_detailed_finetune.py`, `scripts/run_hierarchical_inference.py`
**SLURM:** `slurm/run_detailed_finetune.sbatch`, `slurm/run_hierarchical.sbatch`
**Models:** Major at `output/sweep/models/scibert_...freeze8/`, Detailed at `output/models/detailed_finetune/`

---

## Execution Flow

```
C0 (cleanup) ✅
│
├── C1 (SetFit bge-large) ✅ ──┐
│                              ├──▶ C3 (silver labels + retrain) ✅
└── C2 (label audit) ✅ ─────┘          │
                                        ├──▶ C4 (hyperparam sweep) ✅
                                        │          │
                                        └──────────┴──▶ C5 (detailed + hierarchical) ✅
```

**All phases complete.** Best results: 93.96% major (flat), 88.75% detailed (hierarchical).

---

## Infrastructure

- **Training:** Vista single GH200 node (120GB GPU memory)
- **Active code:** `baselines/finetune.py` (B5), `baselines/setfit_classify.py` (B4), all evaluation
- **Archived:** `archive/` — retired approaches (B2, B3, B6), old predictions, generation scripts

---

## Key Decisions

1. **DB labels are not ground truth** — evaluate against manual audits + inter-model consensus
2. **Silver labels via model consensus** — B4+B5 agree at high confidence, not single-model confidence alone
3. **Focus on B4 (SetFit) + B5 (SciBERT) only** — other approaches archived
4. **No calibration against DB labels** — poisoned by label noise; use inter-model agreement instead
5. **Hierarchical is last** — validate flat-model improvements first (C3/C4), then try hierarchical (C5)
6. **Self-training capped at 2-3 rounds** — diminishing returns, risk of error propagation
7. **UNASSIGNED records (n=419) excluded** from all evaluation going forward
8. **Stop when good enough** — no over-engineering past success criteria
