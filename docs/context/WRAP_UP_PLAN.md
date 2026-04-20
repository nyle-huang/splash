# Wrap-Up Plan

## Scope

This wrap-up closes the research and review phase around the accepted benchmark at
`outputs/final_benchmark_candidate_v5`.

The next operational target is narrower than the full research pipeline:

- deploy the `business_prior` line on Runpod
- expose it through a web interface that will be designed and built later
- keep the current benchmark and review stack as the regression and audit surface, not as the runtime product surface

## Current State

What is already ready:

- accepted final benchmark bundle at `outputs/final_benchmark_candidate_v5`
- upstream validity gating, localization, evidence extraction, retrieval planning, prompt composition, local FLUX generation, and candidate selection
- portable human review boards and merged benchmark assembly
- local runtime documented for the current vast.ai VM in `docs/context/ENVIRONMENT.md`

What is not deployment-ready yet:

- there is no actual deployed Runpod instance yet
- there is no final frontend/backend integration layer yet
- there is no auth, queueing, or rate-limiting layer
- there is no production traffic telemetry or monitoring yet

## Key Deployment Finding

The review scripts are not the production boundary.

The deployable business-prior runtime has now been extracted into a single-request contract and CLI surface, but the underlying generation path is still composed from:

- localization
- upstream validity / evidence extraction
- retrieval planning
- `build_business_prior(...)`
- prompt composition
- FLUX generation
- candidate selection

Current deployment-facing surfaces:

- `product_campaign_pipeline.production.run_business_prior_inference(...)`
- `pcp generate business-prior-photo`
- `docs/context/BUSINESS_PRIOR_RUNTIME_CONTRACT.md`

Those are the correct backend seam for future Runpod work. They still need to be wrapped in a real service/runtime package.

That means wrap-up must include extraction and documentation of a clean production inference boundary.

## Wrap-Up Deliverables

### 1. Freeze The Accepted Baseline

- treat `outputs/final_benchmark_candidate_v5` as the canonical acceptance checkpoint
- keep `benchmark_summary.json`, `generation_report.json`, and `excluded_or_held_sources.json` as the audit set
- stop further repair churn on the exhausted local source pool

### 2. Write The Final Project Summary

- summarize the end-to-end pipeline
- summarize the major upstream fixes that made the benchmark acceptable
- document benchmark coverage, exclusions, and accepted limitations
- document how to rerun benchmark assembly and targeted repair flows if needed later

### 3. Define The Production `business_prior` Contract

Completed in this phase:

- request/response schema defined in `docs/context/BUSINESS_PRIOR_RUNTIME_CONTRACT.md`
- production request models implemented in `src/product_campaign_pipeline/production.py`

Still required before Runpod deployment:

- define the single-request inputs
  - source image
  - optional title / product hint text
  - optional category hint
  - optional request config overrides
- define the outputs
  - final selected image
  - selected candidate mode
  - structured rejection for invalid-source inputs
  - optional debug payload for internal use only
- define the failure policy
  - invalid-source rejection
  - model-load failure
  - timeout / OOM handling
  - missing-model-access behavior

### 4. Extract A Production Inference Entry Point

Completed in this phase:

- single-request production inference entrypoint implemented in `src/product_campaign_pipeline/production.py`
- packaged CLI surface added at `pcp generate business-prior-photo`

Still required before Runpod deployment:

- create a production-oriented path that runs a single `business_prior` request end to end
- keep it separate from HTML board generation, benchmark merging, and review-only artifact assembly
- make the runtime return one production result instead of a review bundle
- keep the benchmark/review code as a separate regression and diagnostics layer

This production path should cover:

- source ingestion
- localization
- source validity checks
- evidence extraction
- retrieval planning
- business-prior prompt composition
- FLUX generation for the selected candidate set
- final candidate selection

### 5. Add Runpod Runtime Packaging

Completed in this phase:

- FastAPI service wrapper added in `src/product_campaign_pipeline/service.py`
- launch script added at `scripts/run_business_prior_service.py`
- Runpod container spec added at `Dockerfile.runpod`
- health endpoint and warmup endpoint added

Still required before actual deployment:

- build and run the container on the target Runpod environment
- verify HF auth and model access on the real deployment
- tune startup/warmup policy for the target GPU and cold-start budget

### 6. Add Runpod-Specific Runtime Controls

Partially completed in this phase:

- define request size limits
- define max image dimensions
- define default generation settings for production
  - width / height
  - inference steps
  - guidance scale
  - candidate modes
- define memory-protection behavior
  - CPU offload strategy
  - cleanup between requests
  - rejection or retry policy on OOM

Implemented:

- env-driven runtime settings in `src/product_campaign_pipeline/service.py`
- health and warmup surfaces for deployment orchestration
- retained lazy FLUX loading plus explicit warmup hook in `src/product_campaign_pipeline/flux/client.py`

### 7. Define The Future Web Interface Contract

Required before frontend work starts:

- specify the backend request and response schema that the future UI will call
- separate user-visible fields from internal debug fields
- define whether requests are synchronous or queued
- define artifact URLs / file locations returned to the frontend
- define user-visible error states
  - invalid input photo
  - unsupported source quality
  - runtime failure

This is needed now so the backend is wrapped correctly for Runpod instead of being retrofitted later around research scripts.

### 8. Add Production Smoke Tests

Completed in this phase:

- service-level tests in `tests/service/test_service.py`
- CLI/runtime regression in `tests/evaluation/test_cli.py`
- deployment smoke script in `scripts/run_business_prior_service_smoke.py`

Still required before actual deployment signoff:

- run the smoke path against a live Runpod container
- add one intentionally invalid-source smoke example to the deployment checklist

### 9. Add Operational Docs

Completed in this phase:

- operational runbook at `docs/context/RUNPOD_BACKEND_RUNBOOK.md`

Still required before actual deployment signoff:

- record the final chosen Runpod template / GPU / disk shape after first live deployment

### 10. Keep The Benchmark As The Regression Gate

Required after deployment prep:

- treat `final_benchmark_candidate_v5` as the regression baseline
- any future deployment-oriented refactor must not silently regress accepted rows
- use the benchmark and smoke set to validate changes before updating the Runpod deployment image

## Recommended Order

1. Freeze the accepted benchmark and write the final project summary.
2. Define the production `business_prior` request/response contract.
3. Extract the production single-request inference entrypoint.
4. Package that entrypoint for Runpod with startup, warmup, and healthcheck behavior.
5. Add smoke tests and operational docs.
6. Only then build the later web interface against the stabilized backend contract.

## Non-Goals For This Wrap-Up

- building the web UI now
- expanding the exhausted local review source pool further
- turning the full review-board pipeline into the product runtime
- treating invalid-source cases as model-quality debt

## Exit Criteria

Wrap-up is complete for the Runpod goal when all of the following are true:

- the accepted benchmark is frozen and documented
- the production `business_prior` runtime contract is documented
- a single-request production inference entrypoint exists
- Runpod packaging/startup requirements are documented
- smoke tests exist for valid and invalid requests
- operational docs are sufficient for deploying the backend before the future web UI is added
