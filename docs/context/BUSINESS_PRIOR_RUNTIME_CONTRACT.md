# Business-Prior Runtime Contract

## Purpose

This document defines the production-facing single-request contract for the
`business_prior` line.

It is the backend contract that a future Runpod handler or web interface should
target. It is intentionally narrower than the research review pipeline.

## Current Entry Point

Installed CLI surface:

```bash
pcp generate business-prior-photo \
  --image /path/to/upload.png \
  --product-title "Floral wallet" \
  --hint-phrase "wallet" \
  --retrieval-index /workspace/product_campaign_pipeline/data/creative_ranking/retrieval_index.train_top1024.json \
  --output-dir /workspace/runtime_outputs
```

Python surface:

- `product_campaign_pipeline.production.BusinessPriorInferenceRequest`
- `product_campaign_pipeline.production.BusinessPriorInferenceResult`
- `product_campaign_pipeline.production.run_business_prior_inference(...)`

## Request Schema

Defined by `BusinessPriorInferenceRequest`.

Required fields:

- `image_path`
- `product_title`
- `retrieval_index_path`
- `output_dir`

Optional fields:

- `hint_phrases`
- `request_id`
- `product_id`
- `source_page_url`
- `source_image_url`
- `model_id`
- `width`
- `height`
- `num_inference_steps`
- `guidance_scale`
- `device`
- `analysis_device`
- `localization_device`
- `candidate_modes`
- `skip_analysis`
- `top_k`
- `seed`

## Response Schema

Defined by `BusinessPriorInferenceResult`.

Possible statuses:

- `ok`
- `invalid_source`

On `ok`, the response includes:

- `output_path`
- `selected_candidate_mode`
- `candidate_count`
- `candidate_scores`
- `prompt`
- `prompt_readiness`
- `observed_evidence`
- `retrieval_metadata`
- `category_consistency`
- `semantic_plausibility`
- `evidence_consistency`
- `localization`

On `invalid_source`, the response includes:

- `source_validity`
- `source_validity_score`
- `source_validity_issues`
- `invalid_reason`
- `localization`
- `observed_evidence` when available

## Runtime Behavior

The single-request flow is:

1. Localize the product in the uploaded source image.
2. Export localization artifacts.
3. Build upstream identity and observed evidence.
4. Reject the request early if the source is invalid under current source-validity rules.
5. Build the `business_prior` retrieval plan.
6. Compose the candidate prompts for the selected candidate modes.
7. Generate candidate images with local FLUX.
8. Score and select the best candidate using the same validated selection policy used in benchmarked review runs.
9. Return one selected final output plus structured metadata.

## Request Directory Layout

Each request writes into:

- `<output_dir>/<request_id>/`

Current subdirectories:

- `localization/`
- `candidates/`
- `images/`

This layout is useful for debugging, smoke tests, and future service artifact management.

## Failure Policy

Current structured early failure:

- `localization_failed`
- `invalid_source_photo`

These should map directly to user-visible frontend errors later instead of surfacing raw tracebacks.

## Runpod Relevance

This contract is the correct backend seam for Runpod deployment because it:

- accepts one uploaded product photo at a time
- returns one selected `business_prior` result at a time
- preserves invalid-source rejection as a first-class outcome
- does not require the HTML review-board layer

What it still does not provide yet:

- an HTTP service
- a Runpod serverless handler
- container startup / warmup / healthcheck logic
- request queueing or async job tracking

Those remain the next deployment-prep steps after this contract.
