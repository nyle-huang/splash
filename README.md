# Product Campaign Pipeline

Offline pipeline for turning a single everyday product photo into campaign-style images while preserving product identity. The system supports two generation lines:

- `baseline`: product photo -> localization -> prompt composer -> FLUX
- `business-prior`: product photo -> localization -> CTR-aware planner -> prompt composer -> FLUX

The migrated workspace expects the CreativeRanking-style image corpus under `/workspace/data`.
The generation target is local `black-forest-labs/FLUX.2-klein-9B` on this VM through `diffusers`, not a hosted API.

## Repository Layout

- `scripts/`: host setup and environment verification
- `docs/context/`: maintained project context documents
- `src/product_campaign_pipeline/`: application code
- `tests/`: automated tests

## Host Setup

```bash
cd /workspace/product_campaign_pipeline
./scripts/bootstrap_debian13_gce_l4.sh
# reboot if the NVIDIA driver install requires it
./scripts/verify_environment.sh
```

## Generation Runtime

- Primary generator: local `black-forest-labs/FLUX.2-klein-9B`
- Access path: authenticated Hugging Face download on this VM
- Runtime expectation: memory-constrained inference on a single L4, with CPU offload or other memory-saving options if full-GPU loading does not fit
- The first human-review photo set should be curated from publicly viewable customer review photos in e-commerce comment sections and stored locally with source metadata

For a lighter CPU-only development install:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,data]"
pytest
```

## CLI Overview

```bash
pcp --help
pcp data-summary --data-root /workspace/data
pcp compose-baseline --title "Trail Runner" --category footwear --color white
pcp generate business-prior-photo --image /path/to/upload.png --product-title "Floral wallet" --retrieval-index /workspace/product_campaign_pipeline/data/creative_ranking/retrieval_index.train_top1024.json --output-dir /workspace/runtime_outputs
```

## Demo Deployment Surfaces

Deployment-facing assets now exist for the public `business_prior` demo stack:

- local/dev FastAPI smoke service: `src/product_campaign_pipeline/service.py`
- public Runpod Serverless worker: `src/product_campaign_pipeline/runpod_worker.py`
- Azure Functions broker: `function_app.py`
- GitHub Pages site: `website/`
- pod-style container spec for local smoke: `Dockerfile.runpod`
- serverless worker container spec: `Dockerfile.runpod.serverless`

See:

- `docs/context/BUSINESS_PRIOR_RUNTIME_CONTRACT.md`
- `docs/context/RUNPOD_BACKEND_RUNBOOK.md`
- `docs/context/DEMO_DEPLOYMENT_STACK.md`

## Maintained Context Docs

- `docs/context/ARCHITECTURE.md`
- `docs/context/ENVIRONMENT.md`
- `docs/context/DATA_CONTRACTS.md`
- `docs/context/EXPERIMENT_LOG.md`
- `docs/context/REVIEW_LOG.md`
- `docs/context/WRAP_UP_PLAN.md`
- `docs/context/BUSINESS_PRIOR_RUNTIME_CONTRACT.md`
- `docs/context/RUNPOD_BACKEND_RUNBOOK.md`
- `docs/context/RUNPOD_DEPLOYMENT_FINISH_GUIDE.md`
- `docs/context/FUTURE_OPTIMIZATION_AND_EXPERIMENT_GUIDE.md`
