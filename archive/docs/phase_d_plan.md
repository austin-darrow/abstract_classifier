# CIP Classifier — Phase D Plan

## Overview

All model experiments are complete (Phases A–C). Phase D focuses on presentation, targeted model improvements, and production deployment.

**Priority order:** D1 (presentation) → D2 (model improvements) → D3 (deployment)

---

## D1. Presentation ✅

Static HTML presentation for TACC devs and staff showing current capabilities.

- **File:** `docs/presentation.html`
- **Format:** Single-page HTML with Chart.js charts + embedded images
- **Content:** Key metrics, how-it-works diagram, accuracy progression, per-field F1, example predictions (2 synthetic + 2 real TACC), confusion analysis, training dynamics, next steps

---

## D2. Model Improvements

### D2.1 Generate Targeted Training Data

**Goal:** Improve the ~15 lowest-performing major fields where nearly all remaining error lives.

**Fields to target** (from per-field F1 chart, below 0.80 threshold):
- Engineering technologies
- Science and mathematics education
- Biological, biomedical, and biosystems engineering nec
- Public health
- Mechanical engineering
- Materials sciences
- Materials and mining engineering
- Electrical and computer engineering
- Pharmacy and pharmaceutical sciences
- Science-related technologies
- Clinical psychology
- Nursing and nursing science
- Interdisciplinary computer sciences
- Multidisciplinary/interdisciplinary sciences nec
- Psychology, other

**Method:** Generate 50–100 additional synthetic abstracts per field using DeepSeek-R1 on Vista. Use the existing generation pipeline (`scripts/generate_abstracts.py`) with field-specific targeting.

**Estimated cost:** ~1.5 hours on single Vista GH200 node.

### D2.2 Generate Detailed-Level Silver Labels

**Goal:** Create pseudo-labeled detailed-field annotations for real TACC abstracts.

**Method:**
1. Run hierarchical model (C5b Strategy C) on all 16,209 real TACC abstracts
2. Filter by confidence threshold (≥ 0.80 for detailed predictions)
3. Output: real abstracts with detailed-field pseudo-labels

**Script needed:** Extend `scripts/run_hierarchical_inference.py` with `--predict-real` flag or create `scripts/generate_detailed_silver.py`

**Expected yield:** ~5,000–8,000 detailed silver labels (based on C5b confidence distribution)

### D2.3 Train Single Unified Model

**Goal:** Test whether a single 315-class model trained on enriched data can replace the two-model approach.

**Architecture:**
- Single SciBERT fine-tune on 315 detailed classes
- Training data: original synthetic (16K) + targeted synthetic (D2.1) + detailed silver labels (D2.2)
- At inference: marginalize logits to get major-field probabilities (sum P(detailed_j) for all j ∈ major_i)
- Then apply Strategy C using marginalized major probs × detailed probs

**Comparison:** If single-model + marginalization matches or beats two-model approach:
- Simpler architecture (one model to serve, not two)
- Faster inference (one forward pass, not two)
- Easier deployment

**Hyperparams:** Start with C4 winner settings (lr=3e-5, 8ep, bs=16, freeze_layers=8)

### D2.4 Evaluation

- Compare single-model vs two-model on synthetic test (detailed + major accuracy)
- Generate calibration curve (confidence vs actual accuracy)
- If time permits: manually label 50–100 real abstracts for unbiased evaluation

---

## D3. Deployment Pipeline

### D3.1 FastAPI Server

**Endpoint:** `POST /classify`
```json
// Request
{"abstract": "...", "top_k": 3}

// Response
{
  "predictions": [
    {
      "detailed_field": "Computer Engineering, General",
      "major_field": "Electrical and computer engineering",
      "broad_field": "Engineering",
      "confidence": 0.94
    },
    ...
  ]
}
```

**Implementation:**
- Load model(s) once at startup
- CPU inference (ONNX export for 3–5x speedup over raw PyTorch)
- Health check endpoint: `GET /health`
- Batch endpoint: `POST /classify/batch` (accepts list of abstracts)

### D3.2 Web UI

- Single-page app served by FastAPI (`/` route)
- Paste abstract → see top-3 predictions with confidence bars
- Responsive, clean design (similar to presentation style)
- Show broad → major → detailed hierarchy in results
- No authentication required (internal tool)

### D3.3 Batch Classification Script

```bash
# Classify abstracts from file
python scripts/classify_batch.py --input abstracts.csv --output predictions.jsonl

# Or via API
curl -X POST https://vm.tacc.utexas.edu/classify/batch \
  -H "Content-Type: application/json" \
  -d @abstracts.json
```

### D3.4 GitLab CI/CD Pipeline

```yaml
stages:
  - test
  - build
  - deploy

build:
  stage: build
  script:
    - docker build -t cip-classifier .
    - docker push $REGISTRY/cip-classifier:$CI_COMMIT_SHORT_SHA

deploy:
  stage: deploy
  script:
    - ssh $VM_HOST "docker pull $REGISTRY/cip-classifier:$CI_COMMIT_SHORT_SHA"
    - ssh $VM_HOST "docker-compose up -d"
  only:
    - main
```

### D3.5 Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt
COPY src/ src/
COPY model/ model/   # Pre-exported ONNX model + tokenizer
COPY app/ app/       # FastAPI app + web UI
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### D3.6 Infrastructure Decisions (TBD)

| Decision | Options | Notes |
|----------|---------|-------|
| Repo structure | Same repo vs. separate deployment repo | Leaning separate — keeps ML experimentation separate from production |
| Model storage | Baked into Docker image vs. pulled from artifact storage | Baked in is simpler for CPU-only |
| ONNX export | Yes (3-5x faster CPU inference) vs. raw PyTorch | Recommend ONNX for production |
| VM spec | CPU only (likely) | SciBERT inference is fast on CPU (~1-2s/abstract without ONNX, ~0.3s with) |

---

## Execution Order

```
D1 (presentation) ✅
│
├── D2.1 (generate targeted data) ──┐
│                                    ├──► D2.3 (train single model)
├── D2.2 (detailed silver labels) ──┘         │
│                                             ├──► D2.4 (evaluate)
│                                             │
└─────────────────────────────────────────────┴──► D3 (deployment)
```

**D3 can start in parallel with D2** — the two-model approach is already good enough to deploy. Single-model improvements can be swapped in later via the CI/CD pipeline.

---

## Success Criteria

| Metric | Target | Notes |
|--------|--------|-------|
| Single-model detailed acc | ≥ 89% | Match or beat two-model (88.75%) |
| Single-model major acc (marginalized) | ≥ 94% | Match or beat C4 (93.96%) |
| API latency (CPU, p95) | < 2s | Acceptable for interactive use |
| Batch throughput | > 100 abstracts/min | For bulk classification jobs |
| Deployment | Zero-downtime redeploy | Via Docker + health checks |
