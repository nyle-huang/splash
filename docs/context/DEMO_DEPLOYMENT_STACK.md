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
- Azure `PCP_RUNPOD_ENDPOINT_ID`: `3co74imgg53e3q`
- Runpod endpoint: `splash-business-prior-demo-pro6000-euris1-volume`
- Runpod endpoint id: `3co74imgg53e3q`
- Runpod template: `splash-business-prior-serverless-pro6000-329bb25-nooffload`
- Runpod template id: `foidwyis7u`
- Runpod network volume: `splash-business-prior-cache-euris1-pro6000-100gb`
- Runpod network volume id: `0k9tnlryio`
- Runpod data center: `EUR-IS-1`
- Runpod GPU: `NVIDIA RTX PRO 6000`
- Runpod workers max: `1`
- Worker image: `ghcr.io/nyle-huang/splash-business-prior-serverless:329bb25079127997415789bb6d0fe9e5a539dc93`

Historical H100 deployment evidence from 2026-04-21:

- Cold H100 ping: `delayTime=368.976s`, `executionTime=0.064s`.
- Warm H100 ping: `delayTime=1.658s`, `executionTime=0.053s`.
- First valid full-quality H100 run on the new volume:
  `delayTime=85.698s`, `executionTime=460.792s`, selected mode `reveal`.
- Warm full-quality H100 run after model load:
  `delayTime=458.891s`, `executionTime=113.325s`, selected mode `reveal`.
- Broker invalid-source smoke after Azure endpoint switch:
  `status=invalid_source` with expected source-quality issues returned through
  the public broker.

Historical H100 operational caveat:

- `EUR-IS-3` H100 fixed execution latency once a worker is running, but Runpod
  can still throttle or delay dispatch. In the measured warm full-quality run,
  execution was `113.325s` but provider delay was `458.891s`.

Blackwell benchmark evidence from 2026-04-22:

- Root cause for the earlier RTX PRO 6000 startup failure was the old worker
  image stack: CUDA `12.6` base image with `torch 2.6.0+cu124`. Blackwell GPUs
  required the CUDA `12.8` / PyTorch `2.7.1+cu128` image built from commit
  `329bb25079127997415789bb6d0fe9e5a539dc93`.
- RTX PRO 6000 endpoint: `3co74imgg53e3q`, template `foidwyis7u`, volume
  `0k9tnlryio`, data center `EUR-IS-1`.
- RTX PRO 6000 ping succeeded with `torch=2.7.1+cu128`,
  `delayTime=165.697s`, `executionTime=0.060s`.
- RTX PRO 6000 first valid full-quality run:
  `delayTime=0.784s`, `executionTime=92.059s`, selected mode `reveal`.
- RTX PRO 6000 warm full-quality repeat:
  `delayTime=0.764s`, `executionTime=14.735s`, selected mode `reveal`.
- RTX 5090 no-offload endpoint with network volume in `EUR-NO-1` did not reach
  the ping handler after more than `13m`; it stayed in worker initialization.
- RTX 5090 no-offload endpoint without network volume in high-stock `EUR-IS-2`
  also did not reach the ping handler after more than `13m`; it moved to
  provider `throttled` state. The job was cancelled and the endpoint deleted.
- Additional RTX 5090 serverless isolation showed the same failure with an exact
  `gpuTypeIds=["NVIDIA GeForce RTX 5090"]` endpoint and with Runpod's official
  hello-world serverless image. This points to current Runpod 5090 serverless
  provisioning/worker startup, not the Splash handler.
- Direct RTX 5090 pod benchmark succeeded on `NVIDIA GeForce RTX 5090`
  `32607 MiB`, driver `570.195.03`, `torch 2.7.1+cu128`, CUDA `12.8`.
- RTX 5090 full-quality no-offload failed with CUDA OOM after source
  localization and FLUX pipeline load. The process used about `31.15 GiB` of
  `31.37 GiB`; a further `288 MiB` allocation failed. This confirms 32GB VRAM
  is too tight for the full stack without offload.
- RTX 5090 full-quality offload preserved `top_k=5`, four generated candidates,
  source localization, generated localization, QA, and ranking. The selected
  candidate was `reveal`.
- RTX 5090 offload single request completed in `159.6007s`.
- RTX 5090 offload same-process pair completed in `270.0225s`: first request
  `143.6438s`, warm second request `125.3918s`.

Live Pro 6000 switch evidence from 2026-04-21 America/Vancouver
(`2026-04-22` UTC):

- Azure broker `PCP_RUNPOD_ENDPOINT_ID` was switched to `3co74imgg53e3q`.
- Pro 6000 endpoint was enabled with `workersMax=1`.
- Broker invalid-source smoke reached the Pro worker and returned the expected
  source-quality rejection:
  job `367d9fa6-ae3b-4c1f-a118-d7972decfa6b-u1`, `delayTime=83.590s`,
  `executionTime=31.952s`, status `invalid_source`.
- Broker valid-source smoke succeeded:
  job `a2f824ca-c7a0-4bc8-9f17-5c88a40a896d-u1`, `delayTime=0.800s`,
  `executionTime=104.624s`, selected mode `reveal`.
- Broker warm valid-source repeat succeeded:
  job `cd94fbd0-47fc-4d54-9692-a86e4205d61c-u2`, `delayTime=0.803s`,
  `executionTime=14.463s`, selected mode `reveal`.
- Returned smoke-test images were saved locally under
  `/tmp/splash-pro6000-live-smoke/`.
- Stale H100 endpoint `7lurkouf1lpzfk`, H100 volume `vb7l0nhag6`, and H100
  template `c95jm8srb3` were deleted after the Pro success smoke.

Execution-only cost estimate using observed worker rates:

| Backend | Mode | Time per image | Assumed rate | Cost per image | Images per dollar |
| --- | --- | ---: | ---: | ---: | ---: |
| H100 serverless | first measured full run | `460.792s` | `$2.99/hr` | `$0.3827` | `2.61` |
| H100 serverless | warm measured full run | `113.325s` | `$2.99/hr` | `$0.0941` | `10.63` |
| RTX PRO 6000 serverless | benchmark first full run | `92.059s` | `$1.89/hr` | `$0.0483` | `20.69` |
| RTX PRO 6000 serverless | benchmark warm full run | `14.735s` | `$1.89/hr` | `$0.0077` | `129.27` |
| RTX PRO 6000 serverless | live first valid smoke | `104.624s` | `$1.89/hr` | `$0.0549` | `18.21` |
| RTX PRO 6000 serverless | live warm valid smoke | `14.463s` | `$1.89/hr` | `$0.0076` | `131.70` |
| RTX 5090 pod | offload single request | `159.6007s` | `$0.99/hr` | `$0.0439` | `22.78` |
| RTX 5090 pod | offload warm request | `125.3918s` | `$0.99/hr` | `$0.0345` | `29.00` |

The 100GB Runpod network volume baseline is about `$7/month` at `$0.07/GB/month`.
These estimates exclude provider queue delay, startup, and exact billed worker
lifecycle time, which must be checked against Runpod billing records.

Current recommendation:

- Keep the public demo on RTX PRO 6000 serverless for the current low-traffic
  demo use case.
- Do not deploy on RTX 5090 serverless until Runpod fixes 5090 serverless worker
  startup. The direct pod path is useful for diagnosis but is not a low-idle-cost
  demo architecture.
- Keep H100 deleted unless a future workload shows a specific H100-only
  advantage. It was slower and less cost-efficient for this pipeline.

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
