# Future Optimization And Experiment Guide

## Purpose

This guide explains how to use the project artifacts that already exist in this
repo for future optimization, regression checks, and new experiments.

The accepted system state is not defined by memory or chat history. It is
defined by the repo code, the maintained context docs, the retrieval assets, and
the accepted benchmark artifacts.

## The Current Baseline

The canonical accepted benchmark is:

- `outputs/final_benchmark_candidate_v5`

Use these files as the baseline reference:

- `outputs/final_benchmark_candidate_v5/human_review_board.html`
- `outputs/final_benchmark_candidate_v5/reports/benchmark_summary.json`
- `outputs/final_benchmark_candidate_v5/reports/generation_report.json`
- `outputs/final_benchmark_candidate_v5/reports/excluded_or_held_sources.json`

This is the accepted final major review checkpoint.

## The Most Important Artifacts

### Code

- `src/product_campaign_pipeline/`
- `scripts/`
- `tests/`
- `pyproject.toml`
- `README.md`

### Maintained Context

- `docs/context/ARCHITECTURE.md`
- `docs/context/ENVIRONMENT.md`
- `docs/context/DATA_CONTRACTS.md`
- `docs/context/EXPERIMENT_LOG.md`
- `docs/context/REVIEW_LOG.md`
- `docs/context/WRAP_UP_PLAN.md`
- `docs/context/BUSINESS_PRIOR_RUNTIME_CONTRACT.md`
- `docs/context/RUNPOD_BACKEND_RUNBOOK.md`
- `docs/context/RUNPOD_DEPLOYMENT_FINISH_GUIDE.md`

### Retrieval And Experiment Inputs

- `data/creative_ranking/retrieval_index.train_top1024.json`
- `data/generalization_diverse_*/`
- `data/final_benchmark_targeted_*/`
- `data/human_review_seed/`

These files define what was sampled, what was screened, and what targeted repair slices were used.

### Benchmark And Approved Review Outputs

Primary benchmark:

- `outputs/final_benchmark_candidate_v5`

Useful approved checkpoints for provenance and targeted regression:

- `outputs/generalization_diverse_v8_curated_v2`
- `outputs/generalization_diverse_v7_curated_v7`
- `outputs/generalization_diverse_v5_targeted_rootfix_v10_curated`
- `outputs/generalization_diverse_v4_approved_bundle`
- `outputs/generalization_diverse_v2_new_categories_final_bundle`
- `outputs/generalization_diverse_v6_approved_bundle`
- `outputs/human_review_batch_v33_final_bundle`
- `outputs/final_benchmark_targeted_v2_final_v1`
- `outputs/final_benchmark_targeted_v3_final_v1`
- `outputs/final_benchmark_targeted_v5_dressfix_v5`

## What To Use For Future Optimization

### If You Change The Runtime Or Backend

Use:

- `outputs/final_benchmark_candidate_v5`
- `tests/`
- `docs/context/BUSINESS_PRIOR_RUNTIME_CONTRACT.md`

Goal:

- confirm the backend contract still holds
- confirm the accepted benchmark does not regress

### If You Change Retrieval

Use:

- `data/creative_ranking/retrieval_index.train_top1024.json`
- `data/generalization_diverse_*`
- benchmark rows where business-prior quality depended on retrieval, especially bags, dress rows, and structurally specific products

Goal:

- improve retrieval quality without causing fallback drift or generic category stereotypes

### If You Change Upstream Validity Or Cropping

Use:

- `data/final_benchmark_targeted_v3/`
- `data/final_benchmark_targeted_v4/`
- `outputs/final_benchmark_targeted_v3_upstream_v2`
- `outputs/final_benchmark_candidate_v5`
- the notes in `EXPERIMENT_LOG.md` and `REVIEW_LOG.md` around the final benchmark repair passes

Goal:

- improve localization / evidence-export behavior without reintroducing prior crop regressions

### If You Change Prompting Or Candidate Selection

Use:

- `tests/composer/test_prompts.py`
- `tests/review/test_review_batch.py`
- `outputs/final_benchmark_targeted_v5_dressfix_v5`
- `outputs/final_benchmark_candidate_v5`

Goal:

- keep prior fixes around dress guardrails, compact accessories, soft goods, and gray-background collapse

## What Not To Do

- Do not mine the exhausted local source pool for another broad tranche.
- Do not treat invalid-source rows as model-quality failures.
- Do not wrap the review-board pipeline directly as the product backend.
- Do not evaluate deployment refactors only by smoke success; use the accepted benchmark as the regression gate.

## Recommended Experiment Workflow

1. Start from `outputs/final_benchmark_candidate_v5` as the regression baseline.
2. Decide which layer you are changing:
   - localization / crop validity
   - evidence extraction
   - retrieval
   - prompting
   - candidate selection
   - deployment/runtime only
3. Pick the smallest relevant targeted manifest or approved bundle first.
4. Run upstream-only checks before full generation when the change is upstream.
5. Only regenerate rows that are actually affected.
6. Compare against the accepted benchmark baseline before promoting the change.
7. Update `EXPERIMENT_LOG.md` and `REVIEW_LOG.md` with the new decision.

## Minimal Regression Gate

For most future work, the minimum gate should be:

```bash
pytest -q
python -m compileall src scripts
```

For backend/runtime work:

- service tests
- one valid smoke request
- one invalid-source request

For model-quality work:

- targeted rerender on the affected slice
- compare against the accepted baseline rows

## If You Need More Data

The current local review-photo pool is exhausted for broad generalization.

That means future model-quality work that needs broader proof should use:

- a new source pool
- a new manifest family
- the same upstream-first screening policy

Do not keep stretching the old pool past what the upstream validity screen already proved.
