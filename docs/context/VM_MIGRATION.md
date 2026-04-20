# VM Migration

## Goal

Move the project to a new VM without losing:
- code
- experiment artifacts
- raw source data
- local model cache
- Codex session context

## Current State

- Project root: `/workspace/product_campaign_pipeline`
- Raw CreativeRanking data root: `/workspace/data`
- Codex home/state: `/root/.codex`
- Hugging Face cache: `/workspace/.hf_home` or the active Hugging Face cache directory on the current VM
- Latest broader active tranche: `generalization_diverse_v5`
- Latest upstream-approved bundle:
  `outputs/generalization_diverse_v5_upstream_review_v3/human_review_board.html`
- Latest active generation target:
  `outputs/generalization_diverse_v5_final_bundle_v1`

## What Must Be Synced

### 1. Project workspace

Sync the full project directory:

`/workspace/product_campaign_pipeline`

This contains:
- source code in `src/`
- scripts in `scripts/`
- maintained context docs in `docs/context/`
- review manifests and small local data in `data/`
- generated review bundles and intermediate artifacts in `outputs/`

### 2. Raw CreativeRanking corpus

Sync the full external data tree:

`/workspace/data`

This is required if you want to:
- rebuild retrieval indexes
- regenerate from the original CreativeRanking source
- avoid being locked to only the already-exported lightweight retrieval indexes

### 3. Codex state

Sync:

`/root/.codex`

Important files inside it:
- `config.toml`
- `auth.json`
- `history.jsonl`
- `state_5.sqlite*`
- `logs_1.sqlite*`
- installed skills/plugins cache

This is the best available way to carry local Codex state and history onto the new VM.

### 4. Hugging Face cache

Sync:

`/workspace/.hf_home` or the active Hugging Face cache directory on the current VM

This avoids re-downloading model weights and usually preserves local token-based access state.

## What Should Usually Be Recreated Instead Of Synced

### Python venv

Optional to sync:

`/workspace/product_campaign_pipeline/.venv`

It is large and less portable across VM image differences.

Recommended:
- sync the project
- recreate the venv on the new VM
- reinstall packages there

Only sync `.venv` if the new VM is very similar and you want the fastest possible restart.

## What Contains The Real Working Context

If the goal is to restore not just files but the practical project context, these are the most important artifacts:

### Source of truth for project history

- `docs/context/EXPERIMENT_LOG.md`
- `docs/context/REVIEW_LOG.md`
- `docs/context/ARCHITECTURE.md`
- `docs/context/ENVIRONMENT.md`
- `docs/context/DATA_CONTRACTS.md`
- `docs/context/VM_MIGRATION.md`

### Latest approved and reference bundles

- `outputs/generalization_diverse_v6_approved_bundle`
- `outputs/generalization_diverse_v2_targeted_rootfix_v30_curated_final_bundle`
- `outputs/generalization_diverse_v3_targeted_repair_v5_final_bundle`
- `outputs/generalization_diverse_v4_targeted_repair_v8_curated_final_bundle`
- `outputs/generalization_diverse_v4_approved_bundle`

### Latest broader in-progress work

- `data/generalization_diverse_v5`
- `outputs/generalization_diverse_v5_localization_v1`
- `outputs/generalization_diverse_v5_upstream_review_v1`
- `outputs/generalization_diverse_v5_upstream_review_v2`
- `outputs/generalization_diverse_v5_upstream_review_v3`
- `outputs/generalization_diverse_v5_final_bundle_v1`
- `data/creative_ranking/retrieval_index.train_top1024.json`

## Practical Sync Sets

### Minimal but sufficient

If you want to restart work with full project context and current artifacts, sync:

- `/workspace/product_campaign_pipeline`
- `/workspace/data`
- `/root/.codex`
- `/workspace/.hf_home` or the active Hugging Face cache directory on the current VM

### Smaller fallback set

If transfer size is a problem, sync at minimum:

- `/workspace/product_campaign_pipeline/src`
- `/workspace/product_campaign_pipeline/scripts`
- `/workspace/product_campaign_pipeline/docs/context`
- `/workspace/product_campaign_pipeline/data`
- `/workspace/product_campaign_pipeline/outputs`
- `/workspace/data`
- `/root/.codex`

Then re-download models on the new VM instead of copying `.cache/huggingface`.

## Recommended rsync

Run after the active bundle has finished:

```bash
rsync -aH --info=progress2 /workspace/product_campaign_pipeline NEW_VM:/workspace/
rsync -aH --info=progress2 /workspace/data NEW_VM:/workspace/
rsync -aH --info=progress2 /root/.codex NEW_VM:/root/
rsync -aH --info=progress2 /workspace/.hf_home NEW_VM:/workspace/
```

If you decide not to copy the model cache, omit the last line.

## Restoring Context On The New VM

On the new VM, the best restore sequence is:

1. Open `docs/context/VM_MIGRATION.md`
2. Open `docs/context/EXPERIMENT_LOG.md`
3. Open `docs/context/REVIEW_LOG.md`
4. Open the latest `generalization_diverse_v5` upstream and final output directories
5. Continue from the latest finished checkpoint rather than re-deriving project history

## Important Limitation

Model conversation state is not literally portable as memory.

The practical replacement is:
- sync `.codex`
- keep the maintained context docs current
- keep the latest approved and in-progress review bundles
- start the next session by pointing the agent at this migration doc plus the two log files

That is the closest reliable equivalent to restoring working memory on a new VM.
