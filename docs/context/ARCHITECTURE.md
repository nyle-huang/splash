# Architecture

## Current Scope

- Input: one user-provided product photo, usually an everyday camera photo
- Shared preprocessing: product localization and identity extraction
- Baseline line: prompt composer only
- Business-prior line: CTR-aware planner plus prompt composer
- Shared generator: local `black-forest-labs/FLUX.2-klein-9B` running on this VM through `diffusers`

## Decisions

- All training runs on the local GCE L4 VM after host bootstrap
- Product identity is preserved from the input image, not from a separate catalog reference set
- CreativeRanking is the source of campaign-style priors and CTR supervision
- Input-photo robustness is bootstrapped synthetically from CreativeRanking items
- Generation no longer depends on the BFL API or any hosted FLUX service
- The Hugging Face token installed on this VM is the access path for gated model downloads
- The first human-review input set should be sampled from publicly viewable customer review photos in e-commerce comment sections, then stored locally with source URLs and capture notes

## Open Technical Notes

- Localization stack is expected to use a phrase-conditioned detector plus segmentation
- Local FLUX execution should stay behind a single client boundary so the rest of the pipeline does not depend on `diffusers` internals
- The FLUX.2 Klein 9B model card indicates a larger memory footprint than a single L4 can usually hold fully in VRAM, so the runtime plan should assume CPU offload and memory-saving inference unless profiling proves a better path
- Large model backbones should stay frozen on a single L4 unless profiling proves otherwise
