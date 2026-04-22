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
- Azure `PCP_RUNPOD_ENDPOINT_ID`: `atl9cnzu9wzk53`
- Runpod endpoint: `splash-business-prior-demo-pro6000-euris1-volume`
- Runpod endpoint id: `atl9cnzu9wzk53`
- Runpod template: `splash-business-prior-serverless-pro6000-4d14fc6-slim-runtime`
- Runpod template id: `foidwyis7u`
- Runpod network volume: `splash-business-prior-cache-euris1-pro6000-100gb`
- Runpod network volume id: `0k9tnlryio`
- Runpod data center: `EUR-IS-1`
- Runpod GPU: `NVIDIA RTX PRO 6000`
- Runpod workers max: `1`
- Worker image: `ghcr.io/nyle-huang/splash-business-prior-serverless:4d14fc662f4dd21039073946ee210014beddfb90`
- Current health status: unhealthy as of 2026-04-22 UTC. The replacement
  endpoint is configured consistently but has not assigned a worker for
  `_internal_ping`.

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
- HF token was rotated after the old broad-permission token was deleted. The new
  token is recorded only by SHA-256 prefix `1a5ad8d5ec8b`; no token value is
  stored in this repository.
- Post-rotation broker smoke succeeded:
  job `e5fe3de5-7f16-4a6d-bc32-6a5a735e8208-u2`, `delayTime=0.804s`,
  `executionTime=14.584s`, selected mode `reveal`.
- After the post-rotation smoke, the Pro endpoint was recycled
  `workersMax=0 -> 1` so future workers start from the updated template env.
- The live Pro endpoint was then set to `idleTimeout=60` to avoid paying for a
  longer idle window while still allowing immediate retries to stay warm.
- Runpod cached-model investigation:
  - Official Runpod docs describe endpoint-level cached models configured
    through the Console `Model` field, with custom workers reading cached
    snapshots from `/runpod-volume/huggingface-cache/hub/`.
  - `runpodctl model list` returned
    `Model Repo feature is not enabled for this user`.
  - The documented REST endpoint schema returned no model/cache field for
    endpoint `3co74imgg53e3q`, and a temporary zero-worker REST create probe with
    a `model` field failed with HTTP `400` because `model` is not in the input
    schema. No endpoint was created by that probe.
  - GraphQL introspection from this machine failed with Cloudflare HTTP `403`
    `error code: 1010`, so GraphQL did not prove or disprove cached-model
    support.
  - Chrome/Console inspection was blocked by local Computer Use permission, so
    the UI path remains unverified here.
  - The worker now preserves the public model id but resolves
    `black-forest-labs/FLUX.2-klein-9B` to the Runpod cached snapshot path when
    that directory is mounted. This changes model load source only; it does not
    change candidate count, localizer, QA analysis, ranking, generation steps,
    image size, guidance, or device policy.
  - After the Console model field was saved and the worker image was updated,
    internal ping job `cb04bd4e-5059-4334-b85d-06cf4dd47a34-u2` reported
    `cached_model_root_exists=true` for
    `/runpod-volume/huggingface-cache/hub`, but
    `source_kind=model_id` and
    `resolved_model_load_source=black-forest-labs/FLUX.2-klein-9B`. That means
    the expected Runpod cached snapshot was not present on the mounted volume;
    the live worker is still using the normal Hugging Face model id path.
  - The same ping had `delayTime=94.780s` and `executionTime=0.085s`, which
    shows an image/container dispatch cold-start component can still dominate
    even before model loading.
  - Direct valid-source benchmark after the cached-snapshot worker change:
    cold job `06597cb7-e136-4d97-bcc6-faec7288eb5a-u1` completed with
    `delayTime=6.159s`, `executionTime=100.333s`, wall-clock `107.827s`,
    selected mode `hero`.
  - Immediate warm repeat
    `d278cf4d-efe0-43cd-96e6-6e08b6c3aa29-u1` completed with
    `delayTime=0.137s`, `executionTime=9.798s`, wall-clock `10.581s`, selected
    mode `hero`.
  - The Console screenshot showed Runpod persisted the cached model as a
    Hugging Face URL with the lowercased repo segment
    `flux.2-klein-9b:92196c8e11f7b6cf2b7493e037d8c5345c559216`. The first
    resolver only checked the canonical `FLUX.2-klein-9B` cache directory, so it
    missed the mounted lowercased cached snapshot.
  - Worker image
    `ghcr.io/nyle-huang/splash-business-prior-serverless:1c8c04c76abb4e8ec114c596193d426405716f8e`
    fixes that by matching Hugging Face cache directories case-insensitively
    and accepting Hugging Face URL strings with pinned revisions.
  - Internal ping job `258d0b2d-1152-4c7e-b47e-430f94fe320a-u1` verified
    `source_kind=runpod_cached_snapshot` and resolved the model load source to
    `/runpod-volume/huggingface-cache/hub/models--black-forest-labs--flux.2-klein-9b/snapshots/92196c8e11f7b6cf2b7493e037d8c5345c559216`.
    The same ping had `delayTime=865.351s`, `executionTime=0.209s`, showing a
    severe Runpod dispatch/provisioning stall independent of model loading.
  - First valid job after that ping on the already-started worker,
    `dd2df952-4617-470a-a6c8-4fccdc2e3da9-u2`, completed with
    `delayTime=0.133s`, `executionTime=56.760s`, wall-clock `61.528s`,
    selected mode `hero`. This isolates the cached-snapshot model/runtime
    initialization path from container dispatch.
  - Immediate warm repeat
    `fd75f696-c30d-442c-90d3-d62d9876d722-u2` completed with
    `delayTime=0.128s`, `executionTime=11.877s`, wall-clock `15.659s`, selected
    mode `reveal`.
  - Current conclusion: Runpod cached model now works through the lowercased
    mounted snapshot. It reduces in-process first valid execution from about
    `100s` to about `57s` on this smoke input, but it does not solve Runpod
    image/container dispatch stalls, which remain the largest unpredictable
    cold-start component.
- Slim runtime image update:
  - Worker image
    `ghcr.io/nyle-huang/splash-business-prior-serverless:4d14fc662f4dd21039073946ee210014beddfb90`
    moves build-only Python tooling into the Docker builder stage and runs the
    final worker from `/opt/venv`. It does not change candidate count,
    localizer use, generated-output localization, QA analysis, ranking,
    generation steps, guidance, image size, model revision, CUDA base image, or
    PyTorch package versions.
  - GitHub Actions run `24763186407` built and pushed the image successfully in
    `9m38s`.
  - Live template `foidwyis7u` was updated to the slim image. Internal ping job
    `0077e541-dfce-436a-afd2-9b9c24261abd-u1` reported build SHA
    `4d14fc662f4dd21039073946ee210014beddfb90`, Python executable
    `/opt/venv/bin/python`, `source_kind=runpod_cached_snapshot`, and resolved
    the model load source to the lowercased Runpod cached snapshot path. The
    same ping had `delayTime=613.095s`, `executionTime=0.102s`, proving the new
    image is live but also that Runpod dispatch/provisioning can still dominate
    cold starts before model loading.
  - Full valid smoke job `4699a90d-775f-4195-8ab4-6c258c88b401-u2` on the slim
    image completed with `delayTime=0.134s`, `executionTime=55.838s`,
    `status=succeeded`, `invalid_source=null`, selected mode `hero`, and a final
    output image saved locally under `/tmp/splash-slim-runtime-smoke/`.
- Cached-model/FlashBoot and endpoint replacement investigation:
  - Endpoint metadata with cached models reports `workersStandby=1`. Runpod
    billing and worker-state docs indicate idle/standby state is not billed as
    active GPU compute; active or initializing workers are billed. Account
    `currentSpendPerHr` was about `$0.01/hr`, matching the 100GB network-volume
    baseline rather than a PRO 6000 active worker.
  - With cached-model FlashBoot/standby enabled, six short-gap measurements
    reused worker `7bhq6lsqoaugml`. Ping median delay was `1.282s`; valid
    generation median wall time was `15.678s` and median execution time was
    `11.886s`. This measures warm retention, not independent cold starts.
  - Disabling `flashboot` did not clear `workersStandby=1` and caused valid
    generation job `2347238b-e899-4a16-a01e-ca2d1735dc32-u2` to remain
    `IN_QUEUE` for 30 minutes before manual cancellation. `flashboot=true` was
    restored afterward.
  - Old endpoint `3co74imgg53e3q` then stopped assigning workers even for
    `_internal_ping`. It was first set to `workersMax=0`, then deleted after
    replacement endpoint creation and explicit approval.
  - Replacement endpoint `atl9cnzu9wzk53` was created with template
    `foidwyis7u`, network volume `0k9tnlryio`, PRO 6000 GPU, `workersMin=0`,
    `workersMax=1`, `workersStandby=1`, `idleTimeout=60`, `flashboot=true`,
    and `QUEUE_DELAY=4`. Azure `PCP_RUNPOD_ENDPOINT_ID` now points to
    `atl9cnzu9wzk53`.
  - Replacement endpoint `_internal_ping` jobs
    `5caa9f62-b2d9-4153-910e-98c409d8eda2-u2`,
    `dde7846a-a880-4d55-b3ae-7dce51795f8a-u2`,
    `003744a4-c57d-45df-8353-9edb41401044-u1`, and
    `cfe720eb-03a2-42ae-a2f3-8d3803c59e87-u1` all stayed `IN_QUEUE` until
    manual timeout/cancellation. `REQUEST_COUNT=1`, old-endpoint capacity
    disablement, and old-endpoint deletion did not restore worker assignment.
  - Final observed replacement metadata: `worker_count=0`, `workersMin=0`,
    `workersMax=1`, `workersStandby=1`, `idleTimeout=60`, `flashboot=true`,
    `scalerType=QUEUE_DELAY`, `scalerValue=4`. Recent billing for replacement
    endpoint `atl9cnzu9wzk53` showed `0s` billed worker time and `$0` endpoint
    compute charge.
  - Current conclusion: cached-model and slim-runtime changes are not the
    failing path. The active failure boundary is Runpod serverless
    provisioning/worker assignment for the PRO 6000 + network-volume +
    cached-model endpoint configuration.
- A separate HF permission probe used a temporary diagnostic pod with the Pro
  network volume mounted at `/runpod-volume`; it did not target or modify
  `/runpod-volume/hf_home`.
- The probe set isolated `HF_HOME=/runpod-volume/hf_token_permission_probe_env`,
  created a temporary `/runpod-volume/hf_token_permission_probe_*` directory,
  verified `black-forest-labs/FLUX.2-klein-9B` metadata access with HTTP `200`,
  downloaded only `model_index.json` (`446` bytes) with HTTP `200`, then removed
  only the temporary probe directory. The probe reported `cleanup_ok=true`.
- The temporary diagnostic pod was deleted after the probe, and `runpodctl pod
  list` returned `[]`.

Execution-only cost estimate using observed worker rates:

| Backend | Mode | Time per image | Assumed rate | Cost per image | Images per dollar |
| --- | --- | ---: | ---: | ---: | ---: |
| H100 serverless | first measured full run | `460.792s` | `$2.99/hr` | `$0.3827` | `2.61` |
| H100 serverless | warm measured full run | `113.325s` | `$2.99/hr` | `$0.0941` | `10.63` |
| RTX PRO 6000 serverless | benchmark first full run | `92.059s` | `$1.89/hr` | `$0.0483` | `20.69` |
| RTX PRO 6000 serverless | benchmark warm full run | `14.735s` | `$1.89/hr` | `$0.0077` | `129.27` |
| RTX PRO 6000 serverless | live first valid smoke | `104.624s` | `$1.89/hr` | `$0.0549` | `18.21` |
| RTX PRO 6000 serverless | live warm valid smoke | `14.463s` | `$1.89/hr` | `$0.0076` | `131.70` |
| RTX PRO 6000 serverless | cached-model attempt cold valid run | `100.333s` | `$1.89/hr` | `$0.0527` | `18.95` |
| RTX PRO 6000 serverless | cached-model attempt warm valid run | `9.798s` | `$1.89/hr` | `$0.0051` | `194.66` |
| RTX PRO 6000 serverless | lowercased cached snapshot first valid run | `56.760s` | `$1.89/hr` | `$0.0298` | `33.57` |
| RTX PRO 6000 serverless | lowercased cached snapshot warm repeat | `11.877s` | `$1.89/hr` | `$0.0062` | `160.79` |
| RTX PRO 6000 serverless | slim runtime cached valid smoke | `55.838s` | `$1.89/hr` | `$0.0293` | `34.13` |
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

- Historical RTX PRO 6000 endpoint `3co74imgg53e3q` previously ran the
  full-quality path with current candidate count, source localizer,
  generated-output localizer, QA, ranking, and final image output enabled.
- Current Azure broker points to replacement endpoint `atl9cnzu9wzk53`, but this
  endpoint has not yet assigned a worker for `_internal_ping`.

Unverified here:

- billed duration including model load
- end-to-end public browser session after enabling Runpod cached-model routing
- whether the slim runtime image reduces dispatch latency across multiple
  independent cold starts; the first measured slim-image ping still had a
  `613.095s` provider delay
- root cause of Runpod failing to assign workers on replacement endpoint
  `atl9cnzu9wzk53`
