# Demo Deployment Stack

## Purpose

This document defines the public demo deployment surface for the `business_prior`
pipeline. The goal is low recurring cost, low expected traffic, and a clean public
entrypoint for employer review.

The deployed stack is:

- GitHub Pages static site: `website/`
- Azure Functions broker: `function_app.py`
- Runpod Serverless worker: `src/product_campaign_pipeline/runpod_worker.py`

## Runtime-Only Deployment Inventory

The public deployment should exclude the research workspace, large output bundles,
and the full `data/` tree.

Only these runtime artifacts are required for the serverless worker image:

- `pyproject.toml`
- `README.md`
- `src/product_campaign_pipeline/`
- `data/creative_ranking/retrieval_index.train_top1024.json`
- `requirements.runpod.txt`
- `Dockerfile.runpod.serverless`

Only these artifacts are required for the Azure broker:

- `function_app.py`
- `host.json`
- `requirements.txt`
- `src/product_campaign_pipeline/demo_broker.py`
- `src/product_campaign_pipeline/public_api.py`
- `.funcignore`

## Public Contract

Browser request payload:

- `image_base64`
- `mime_type`
- `product_title`
- optional `hint_phrases`
- optional `request_id`

Broker response / polled job result:

- `status`
- `job_id`
- `summary`
- optional `selected_candidate_mode`
- optional `final_image_base64`
- optional `final_image_mime_type`
- optional `invalid_source`
- optional `error_code`

The contract is implemented in `src/product_campaign_pipeline/public_api.py`.

## Azure Broker

Public endpoints:

- `POST /api/jobs`
- `GET /api/jobs/{job_id}`

Required environment variables:

- `PCP_DEMO_TOKEN`
- `PCP_RUNPOD_API_KEY`
- `PCP_RUNPOD_ENDPOINT_ID`
- `PCP_ALLOWED_ORIGIN`

Optional broker tuning variables:

- `PCP_RUNPOD_BASE_URL`
- `PCP_BROKER_MAX_IMAGE_BYTES`
- `PCP_BROKER_MAX_PAYLOAD_BYTES`
- `PCP_BROKER_REQUEST_TIMEOUT_SECONDS`

Deploy from the repo root so Azure can install `requirements.txt` and import the
editable package:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
func start
```

The committed `host.json` sets `routePrefix` to `""` so the external URLs remain
`/api/jobs` and `/api/jobs/{job_id}` instead of being prefixed twice.

## Runpod Serverless Worker

Build from the runtime-only Dockerfile:

```bash
docker build -f Dockerfile.runpod.serverless -t pcp-business-prior-serverless:latest .
```

Required worker environment variables:

- `HF_TOKEN`
- `PCP_WORKER_LOG_PATH`
- `PCP_OUTPUT_ROOT`
- `PCP_RETRIEVAL_INDEX_PATH`
- `PCP_DEVICE`
- `PCP_ANALYSIS_DEVICE`
- `PCP_LOCALIZATION_DEVICE`
- `PCP_GENERATED_LOCALIZATION_DEVICE`
- `PCP_RUNPOD_WORKER_MODE`

For the cost-optimized public demo, Runpod should attach a `100GB` network
volume at `/runpod-volume`. The serverless image defaults `HF_HOME` to
`/runpod-volume/hf_home` and `PCP_OUTPUT_ROOT` to
`/runpod-volume/runtime_outputs` so model downloads and runtime artifacts survive
worker cold starts. Verify the template `volumeMountPath` after every template
create or update; the Runpod CLI can silently leave it at `/workspace`, which
shadows the image workspace and prevents the intended `/runpod-volume` cache
layout from being used.

Worker startup logs go to stdout by default. Persistent file logging is disabled
unless `PCP_ENABLE_WORKER_FILE_LOG=1` is set. Leave it disabled for normal
serverless startup diagnostics so the worker can register before touching the
network volume. Enable it only for a focused debugging run that needs
`/runpod-volume/logs/runpod_worker.log`.

The image starts through `product_campaign_pipeline.runpod_entrypoint`.
`PCP_RUNPOD_WORKER_MODE=generation` runs the production worker.
`PCP_RUNPOD_WORKER_MODE=ping` runs a minimal dispatch/logging diagnostic worker
that imports only the Runpod SDK and returns immediately. Use `ping` mode before
warmup when Runpod queue state and exported logs disagree.

Direct Runpod API calls can verify worker dispatch without loading models with:

```json
{
  "input": {
    "_internal_ping": true
  }
}
```

Direct Runpod API calls can populate the cache with:

```json
{
  "input": {
    "_internal_warmup": true,
    "include_generation": true
  }
}
```

This internal warmup payload is not part of the browser-facing broker contract.

Recommended startup command:

```bash
python3.12 -m product_campaign_pipeline.runpod_entrypoint
```

## GitHub Pages Site

The public site lives in `website/`.

Before publishing, update `website/assets/config.js` with the Azure Function base URL.
The site intentionally does not store the demo token.

The included workflow publishes the static `website/` directory to GitHub Pages:

```bash
.github/workflows/deploy-demo-site.yml
```

If you configure a custom domain through GitHub Pages settings, GitHub's current
documentation says a `CNAME` file is not required when publishing from a custom
GitHub Actions workflow.

## Verified And Unverified

Current public deployment:

- GitHub Pages site: `https://nyle-huang.github.io/splash/`
- Azure broker: `https://splash-demo-broker-nh-y1.azurewebsites.net`
- Azure `PCP_RUNPOD_ENDPOINT_ID`: `7lurkouf1lpzfk`
- Runpod endpoint: `splash-business-prior-demo-h100-euris3-volume`
- Runpod endpoint id: `7lurkouf1lpzfk`
- Runpod template id: `c95jm8srb3`
- Runpod network volume id: `vb7l0nhag6`
- Runpod data center: `EUR-IS-3`
- Runpod GPU: `NVIDIA H100 80GB HBM3`
- Worker image: `ghcr.io/nyle-huang/splash-business-prior-serverless:f8757122ccaf1845cba3ba521f4558d2f8127c5b`

Deployment evidence from 2026-04-21:

- Cold H100 ping: `delayTime=368.976s`, `executionTime=0.064s`.
- Warm H100 ping: `delayTime=1.658s`, `executionTime=0.053s`.
- First valid full-quality H100 run on the new volume:
  `delayTime=85.698s`, `executionTime=460.792s`, selected mode `reveal`.
- Warm full-quality H100 run after model load:
  `delayTime=458.891s`, `executionTime=113.325s`, selected mode `reveal`.
- Broker invalid-source smoke after Azure endpoint switch:
  `status=invalid_source` with expected source-quality issues returned through
  the public broker.

Known operational caveat:

- `EUR-IS-3` H100 fixed execution latency once a worker is running, but Runpod
  can still throttle or delay dispatch. In the measured warm full-quality run,
  execution was `113.325s` but provider delay was `458.891s`.

Verified locally:

- request/result contract implementation exists
- Azure ASGI entrypoint exists
- serverless worker entrypoint exists
- static site exists
- focused broker/worker/site tests exist

Verified live:

- Azure broker submits to the H100 endpoint.
- H100 endpoint can run the full-quality path with current candidate count,
  source localizer, generated-output localizer, QA, ranking, and final image
  output enabled.

Unverified here:

- billed duration including model load
- end-to-end public browser session after the H100 endpoint switch
