# Runpod Deployment Finish Guide

## Purpose

This guide covers the remaining steps to finish deploying the `business_prior`
backend using the artifacts that already exist in this repo.

Use this after moving to the next environment. Do not treat the old review-board
pipeline as the deployment surface. The deployable backend is the production
single-request runtime plus the FastAPI service wrapper.

For the public employer-facing demo stack, the primary deployment path is now:

- `Dockerfile.runpod.serverless` for the Runpod worker
- `function_app.py` for the Azure broker
- `website/` for the GitHub Pages frontend

This guide remains useful for local pod-style smoke validation of the inference
runtime before moving to the public serverless path.

## Artifacts Already In The Repo

Deployment-relevant code and docs:

- service wrapper: `src/product_campaign_pipeline/service.py`
- production runtime: `src/product_campaign_pipeline/production.py`
- startup script: `scripts/run_business_prior_service.py`
- smoke script: `scripts/run_business_prior_service_smoke.py`
- container spec: `Dockerfile.runpod`
- backend contract: `docs/context/BUSINESS_PRIOR_RUNTIME_CONTRACT.md`
- backend runbook: `docs/context/RUNPOD_BACKEND_RUNBOOK.md`
- accepted regression baseline: `outputs/final_benchmark_candidate_v5`

## What To Bring To The New Environment

Required:

- full repo at `/workspace/product_campaign_pipeline`
- raw dataset at `/workspace/data` if you want to keep the current data layout and rerun experiments
- Hugging Face token for gated model access

Strongly recommended:

- `/workspace/.hf_home` if you want to avoid re-downloading model weights
- `/root/.codex` if you want to preserve Codex global context and plugin state

Optional:

- project `.venv`
  It is reproducible and not the preferred artifact to preserve.

## Finish The Environment

1. Restore the project to the target path:

```bash
cd /workspace/product_campaign_pipeline
```

2. Rebuild the venv if you are not carrying one over:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
python -m pip install -e ".[generation,vision,storage,service,dev]"
```

3. Confirm the token and cache configuration:

```bash
echo "$HF_TOKEN"
python - <<'PY'
import os
print("HF_HOME =", os.getenv("HF_HOME"))
PY
```

4. Verify the retrieval asset path exists:

```bash
ls -lh data/creative_ranking/retrieval_index.train_top1024.json
```

## Build The Container

From repo root:

```bash
docker build -f Dockerfile.runpod -t pcp-business-prior:latest .
```

If you are using Runpod’s own image builder flow, the same file should still be the source of truth.

## Launch The Backend Locally First

Recommended local launch before Runpod:

```bash
source .venv/bin/activate
export HF_TOKEN=...
export PCP_OUTPUT_ROOT=/workspace/runtime_outputs
export PCP_RETRIEVAL_INDEX_PATH=/workspace/product_campaign_pipeline/data/creative_ranking/retrieval_index.train_top1024.json
export PCP_DEVICE=cuda
export PCP_ANALYSIS_DEVICE=cpu
export PCP_LOCALIZATION_DEVICE=cuda
python scripts/run_business_prior_service.py --host 0.0.0.0 --port 8000
```

## Verify Service Health

```bash
curl http://127.0.0.1:8000/healthz
curl -X POST "http://127.0.0.1:8000/warmup?include_generation=true"
curl http://127.0.0.1:8000/healthz
```

Expected:

- first `healthz` may show the service as ready without the generation pipeline loaded
- warmup should load the cached dependencies
- second `healthz` should show a loaded generation pipeline if generation warmup was requested

## Run The Smoke Test

Use a known valid-source image from the accepted benchmark set or another trusted test image:

```bash
python scripts/run_business_prior_service_smoke.py \
  --base-url http://127.0.0.1:8000 \
  --image /path/to/source.png \
  --product-title "Floral wallet" \
  --hint-phrase wallet
```

Expected:

- returns JSON with `status = "ok"`
- writes request artifacts into `PCP_OUTPUT_ROOT`
- final `output_path` exists

Also test one invalid-source input and confirm it returns:

- `status = "invalid_source"`
- `source_validity_issues`
- `invalid_reason`

## Deploy On Runpod

Use these as the minimum environment variables:

```bash
HF_TOKEN=...
PCP_OUTPUT_ROOT=/workspace/runtime_outputs
PCP_RETRIEVAL_INDEX_PATH=/workspace/product_campaign_pipeline/data/creative_ranking/retrieval_index.train_top1024.json
PCP_DEVICE=cuda
PCP_ANALYSIS_DEVICE=cpu
PCP_LOCALIZATION_DEVICE=cuda
PCP_WARMUP_ON_START=1
PCP_WARMUP_GENERATION_ON_START=0
```

Recommended startup command:

```bash
python3.12 scripts/run_business_prior_service.py --host 0.0.0.0 --port 8000
```

Recommendation:

- start with FLUX generation warmup disabled
- enable `include_generation=true` warmup only if cold-start latency is a real problem

## Final Deployment Signoff Checklist

Deployment is complete when all of these are true:

- container builds successfully from `Dockerfile.runpod`
- service starts on the target Runpod GPU
- `GET /healthz` succeeds
- `POST /warmup` succeeds
- one valid smoke request completes
- one invalid-source request is rejected correctly
- output artifacts are written where expected
- no local path assumptions from the old VM remain in the deployed backend configuration

## After Deployment

Once Runpod deployment is working:

- record the chosen Runpod GPU, disk, startup time, and warmup policy
- keep `outputs/final_benchmark_candidate_v5` as the regression baseline for backend changes
- do not reopen the exhausted local source-pool repair loop
