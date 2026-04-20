# Runpod Backend Runbook

## Purpose

This runbook covers backend deployment of the `business_prior` line on Runpod.

It assumes:

- the future web interface is not built yet
- Runpod will host the backend service directly
- the current accepted regression baseline is `outputs/final_benchmark_candidate_v5`

## Backend Surface

Service module:

- `product_campaign_pipeline.service:app`

CLI runtime:

- `pcp generate business-prior-photo`

Current HTTP endpoints:

- `GET /healthz`
- `POST /warmup`
- `POST /generate/business-prior`

## Container Build

Build from repo root:

```bash
docker build -f Dockerfile.runpod -t pcp-business-prior:latest .
```

## Required Runtime Inputs

Required environment:

- `HF_TOKEN`

Expected mounted or copied project assets:

- repo code
- `data/creative_ranking/retrieval_index.train_top1024.json`

Not required in the deployment image:

- historical review outputs
- benchmark boards
- exhausted broad-tranche artifacts
- raw benchmark source pool outside the retrieval assets needed by the runtime

## Useful Environment Variables

- `PCP_RETRIEVAL_INDEX_PATH`
- `PCP_OUTPUT_ROOT`
- `PCP_MODEL_ID`
- `PCP_DEVICE`
- `PCP_ANALYSIS_DEVICE`
- `PCP_LOCALIZATION_DEVICE`
- `PCP_WIDTH`
- `PCP_HEIGHT`
- `PCP_NUM_INFERENCE_STEPS`
- `PCP_GUIDANCE_SCALE`
- `PCP_TOP_K`
- `PCP_WARMUP_ON_START`
- `PCP_WARMUP_GENERATION_ON_START`

Default behavior:

- service starts without full generation warmup
- `GET /healthz` reports readiness of the service wrapper
- `POST /warmup?include_generation=true` can be used to force model warmup before traffic

## Local Launch

```bash
source .venv/bin/activate
pip install -e ".[service]"
python scripts/run_business_prior_service.py --host 0.0.0.0 --port 8000
```

## Runpod Launch

Recommended container command:

```bash
python3.12 scripts/run_business_prior_service.py --host 0.0.0.0 --port 8000
```

Recommended initial environment:

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

Recommendation:

- start with localization/backbone warmup enabled
- leave FLUX generation warmup disabled unless cold-start latency proves too high
- if cold-start latency is unacceptable, enable generation warmup and budget for the longer startup time

## Health And Warmup

Check health:

```bash
curl http://127.0.0.1:8000/healthz
```

Force warmup:

```bash
curl -X POST "http://127.0.0.1:8000/warmup?include_generation=true"
```

## Smoke Test

With the service already running:

```bash
python scripts/run_business_prior_service_smoke.py \
  --base-url http://127.0.0.1:8000 \
  --image /path/to/source.png \
  --product-title "Floral wallet" \
  --hint-phrase wallet
```

Expected behavior:

- returns structured JSON
- writes one request directory under `PCP_OUTPUT_ROOT`
- selected final output exists on disk

## Invalid-Source Behavior

The backend is expected to reject some user uploads.

That is correct behavior when upstream validity fails.

Current invalid result shape:

- `status = "invalid_source"`
- `source_validity`
- `source_validity_score`
- `source_validity_issues`
- `invalid_reason`

The future web UI should map that to a user-visible retry / unsupported-input flow instead of pretending generation succeeded.

## Operational Notes

- The backend is single-request oriented and GPU-bound.
- FLUX loading and first-request latency can be high.
- Use the benchmarked runtime defaults before experimenting with higher steps or larger image sizes.
- Treat `final_benchmark_candidate_v5` as the regression baseline for backend changes.

## What Still Remains After This Runbook

- UI integration
- auth/rate limiting if exposed publicly
- async job orchestration if synchronous latency is too high for the planned web UX
