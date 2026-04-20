# Experiment Log

Record every meaningful run here.

## Template

- Date:
- Owner:
- Goal:
- Data snapshot:
- Code revision:
- Environment revision:
- Model config:
- Metrics:
- Qualitative notes:
- Follow-up:

## 2026-04-09 Human Review Batch V1

- Date: 2026-04-09T10:41:55Z
- Owner: Codex
- Goal: Run the first end-to-end review batch on messy customer-review photos without filtering bad localization or category cases, so failure intermediates and failure outputs stay visible.
- Data snapshot: 13 Walmart customer-review images in `data/human_review_seed/images` with localization artifacts in `outputs/human_review_seed_localization`
- Code revision: no git repository initialized in this workspace; use file timestamps plus output artifact paths as the run anchor
- Environment revision: local GCE `g2-standard-12` VM with `NVIDIA L4`, driver `550.163.01`, CUDA `12.4`, project venv at `.venv`
- Model config: local `black-forest-labs/FLUX.2-klein-9B` via `diffusers`, `512x512`, `num_inference_steps=4`, `guidance_scale=1.0`, CUDA device with CPU offload path available; baseline and business-prior lines both used localized crop plus source photo as references
- Metrics: 26 images generated successfully for 13 source photos; 13 baseline outputs and 13 business-prior outputs; mean elapsed time `34.44s` baseline and `28.18s` business-prior on this run
- Qualitative notes: bottles preserved shape and branding reasonably well but sometimes gained extra packaging details; pillows preserved texture and overall form well; patterned tote close-ups were brittle because the localized crop contained pattern information but weak category evidence, leading to one tote-to-shirt failure; business-prior retrieval was directionally useful for bags and room decor but still noisy for bottles and some tote cases
- Follow-up: tighten category inference before retrieval and prompt composition, increase category-conditioned retrieval filtering, and add prompt language that explicitly forbids changing product type when the source image is a close crop of surface pattern only

## 2026-04-09 Human Review Batch V2

- Date: 2026-04-09T11:16:00Z
- Owner: Codex
- Goal: Re-run the full 13-seed human review board after shipping three fixes from the first checkpoint: canonical product-type injection into prompts, post-generation category-consistency checks, and stricter retrieval filtering plus category fallback for weak drinkware retrieval.
- Data snapshot: same 13 Walmart customer-review images in `data/human_review_seed/images` with the same localization artifacts in `outputs/human_review_seed_localization`
- Code revision: no git repository initialized in this workspace; use the v2 output artifacts plus updated context logs as the run anchor
- Environment revision: local GCE `g2-standard-12` VM with `NVIDIA L4`, driver `550.163.01`, CUDA `12.4`, project venv at `.venv`
- Model config: local `black-forest-labs/FLUX.2-klein-9B` via `diffusers`, `512x512`, `num_inference_steps=4`, `guidance_scale=1.0`, CUDA device with memory-saving runtime; review batch now adds canonical product-type guardrails and CLIP-based category-consistency scoring
- Metrics: 26 images generated successfully for 13 source photos; 13 baseline outputs and 13 business-prior outputs; `0` rows flagged by the category-consistency check; all 4 CamelBak business-prior rows used `category_fallback` rather than noisy retrieval captions
- Qualitative notes: the previous `disney_tote_03` tote-to-shirt failure was removed and both lines now keep the product as a tote bag; CamelBak business-prior no longer inherits unrelated sauce or clutter captions and remains on-category; Dasein and pillow outputs remain broadly stable; the new board surfaces expected category and canonical type on every card so future drift is visible without filtering
- Follow-up: obtain the independent Claude review for the v2 board, then decide whether to expand evaluation or continue refining identity fidelity on the Dasein packaging-like source cases

## 2026-04-09 Human Review Batch V4 Semantics

- Date: 2026-04-09T19:36:41Z
- Owner: Codex
- Goal: Re-run the full 13-seed board after adding affordance-aware support modes, coherent scene/support planning, semantic plausibility scoring, and fixing the wallet-token affordance bug that falsely marked Dasein bags as wall-mounted.
- Data snapshot: same 13 Walmart customer-review images in `data/human_review_seed/images` with the same localization artifacts in `outputs/human_review_seed_localization`
- Code revision: no git repository initialized in this workspace; use the v4 semantic output artifacts plus updated context logs as the run anchor
- Environment revision: local GCE `g2-standard-12` VM with `NVIDIA L4`, driver `550.163.01`, CUDA `12.4`, project venv at `.venv`
- Model config: local `black-forest-labs/FLUX.2-klein-9B` via `diffusers`, `512x512`, `num_inference_steps=4`, `guidance_scale=1.0`; semantic planner now selects one support relation and one scene family per output and evaluates support plausibility post-generation with CLIP text-image similarity
- Metrics: 26 images generated successfully for 13 source photos; 13 baseline outputs and 13 business-prior outputs; `0` category-consistency flags; `0` semantic-plausibility flags in the final run
- Qualitative notes: pillow cases now consistently rest against seating or other support instead of standing implausibly on tables; tote and bag cases preserve product type while using either carried or surface-supported placements; CamelBak stays in coherent tabletop display scenes with category fallback when retrieval is weak; Dasein baseline no longer receives false mounted/hanging affordances from the `wallet` token
- Follow-up: obtain the independent Claude review for the v4 semantic board, then decide whether to promote this as the new benchmark baseline or continue with retrieval-quality improvements

## 2026-04-09 Dasein Evidence-Constrained Reinvention V3

- Date: 2026-04-09T23:59:00Z
- Owner: Codex
- Goal: Repair `dasein_handbag_02` by replacing viewpoint-preservation logic with evidence-constrained reinvention, then rerank multiple candidate completions by category, semantics, and observed-evidence consistency.
- Data snapshot: single Walmart customer-review image `data/human_review_seed/images/dasein_handbag_02.jpg` with localization artifacts in `outputs/human_review_seed_localization`
- Code revision: no git repository initialized in this workspace; use the v3 evidence-reinvention output artifacts plus updated context logs as the run anchor
- Environment revision: local GCE `g2-standard-12` VM with `NVIDIA L4`, driver `550.163.01`, CUDA `12.4`, project venv at `.venv`; generated-output evidence localization moved to CPU to avoid GPU contention with FLUX
- Model config: local `black-forest-labs/FLUX.2-klein-9B` via `diffusers`, `512x512`, requested `num_inference_steps=4`, `guidance_scale=1.0`; hard-case path generated 3 candidate prompts per line using `balanced`, `reveal`, and `hero` reinvention modes and selected the winner with a combined category/semantic/evidence score
- Metrics: baseline selected candidate `reveal` with combined score `0.7997` and evidence score `0.5572`; business-prior selected candidate `balanced` with combined score `0.8229` and evidence score `0.6220`; both selected outputs remained category-consistent and semantically plausible
- Qualitative notes: the observed-evidence extractor now prioritizes BLIP-derived evidence such as `blue and white floral print bag with a gold handle`; the selected outputs no longer collapse the bag into a generic dark tote and instead preserve the blue floral surface evidence while allowing a cleaner front/three-quarter campaign reinvention; the business-prior line produced the strongest final output in this repair pass
- Follow-up: run the same evidence-constrained reinvention path on the next held identity cases, especially future apparel inputs with partial or occluded evidence, and decide whether to promote this reranking path from targeted fallback to the default hard-case policy

## 2026-04-09 Human Review Batch V5 Evidence Reinvention

- Date: 2026-04-09T23:29:26Z
- Owner: Codex
- Goal: Promote evidence-constrained reinvention from a targeted `dasein_handbag_02` repair into the default hard-case policy, then rerun the full 13-seed human review board and verify that the previously held bag/tote cases improve without regressing the stronger bottle and pillow samples.
- Data snapshot: same 13 Walmart customer-review images in `data/human_review_seed/images` with the same localization artifacts in `outputs/human_review_seed_localization`
- Code revision: no git repository initialized in this workspace; use the v5 evidence-reinvention output artifacts plus updated context logs as the run anchor
- Environment revision: local GCE `g2-standard-12` VM with `NVIDIA L4`, driver `550.163.01`, CUDA `12.4`, project venv at `.venv`; generated-output evidence localization remains on CPU to avoid GPU contention with FLUX during candidate reranking
- Model config: local `black-forest-labs/FLUX.2-klein-9B` via `diffusers`, `512x512`, `num_inference_steps=4`, `guidance_scale=1.0`; hard partial-view or ambiguous samples generated `balanced`, `reveal`, and `hero` reinvention candidates per line and selected the winner with the combined category/semantic/evidence score
- Metrics: 26 images generated successfully for 13 source photos; `0` category-consistency failures; `0` semantic-plausibility failures; `0` evidence-consistency failures under the current advisory threshold; mean semantic score `0.6721`; mean evidence score `0.5881`; mean selected candidate score `0.8115`; selected modes were `balanced=11`, `hero=7`, `reveal=8`
- Qualitative notes: `dasein_handbag_02` retained the repaired evidence-constrained behavior and selected `hero` for both lines; `disney_tote_03` no longer required a hold and both lines remained compatible with the observed tote evidence while allowing cleaner campaign views; the weakest remaining rows shifted from outright contradiction into high-uncertainty partial-view cases such as `dasein_handbag_01` and `camelbak_bottle_03`; pillow cases remained strong after the semantic-support fixes; the review board is portable and uses only relative asset paths
- Follow-up: human-review the v5 board, decide whether the remaining weak evidence rows are acceptable as the new baseline, and then extend the same evidence-preserving reinvention policy to the first apparel-like hard cases

## 2026-04-10 Human Review Batch V6 Root Cause Fixes

- Date: 2026-04-10T01:24:00Z
- Owner: Codex
- Goal: Address the root preservation failures from the v5 review by upgrading the evidence model, adding evidence-compatible business-prior fallback, and replacing coarse evidence scoring with part-level contradiction checks, then rerun the full 13-seed board.
- Data snapshot: same 13 Walmart customer-review images in `data/human_review_seed/images` with the same localization artifacts in `outputs/human_review_seed_localization`
- Code revision: no git repository initialized in this workspace; use the v6 root-cause-fix output artifacts plus updated context logs as the run anchor
- Environment revision: local GCE `g2-standard-12` VM with `NVIDIA L4`, driver `550.163.01`, CUDA `12.4`, project venv at `.venv`; generated-output evidence localization and reranking remained CPU-heavy while FLUX generation stayed local on the same VM
- Model config: local `black-forest-labs/FLUX.2-klein-9B` via `diffusers`, `512x512`, `num_inference_steps=4`, `guidance_scale=1.0`; evidence-sensitive business-prior planning now falls back when retrieved creatives are scene-useful but evidence-incompatible
- Metrics: 26 images generated successfully for 13 source photos; `0` category-consistency failures; `0` semantic-plausibility failures; `0` evidence-consistency failures under the stricter advisory threshold; mean semantic score `0.6690`; mean evidence score `0.6953`; business-prior retrieval modes were `category_fallback=4`, `evidence_fallback=9`; key evidence-score improvements over v5 included `camelbak_bottle_04.business_prior +0.0955`, `rizzy_pillow_03.business_prior +0.1541`, `disney_tote_02.business_prior +0.1568`, and `dasein_handbag_02.business_prior +0.1105`
- Qualitative notes: bottles and pillows now preserve their surface treatments far more faithfully; `disney_tote_01` business-prior no longer collapses the print to partial coverage and uses evidence fallback instead of generic bag retrieval; `dasein_handbag_02` business-prior no longer invents the prior transparent side-panel contradiction; the main remaining residual is `disney_tote_02`, where the upper-component extractor still reads the handle/top-band evidence too coarsely and can preserve the full print while missing the dark braided-handle detail
- Follow-up: bring the v6 board to human review as the new root-cause-fix checkpoint, then decide whether the remaining upper-component extraction weakness warrants another targeted pass before expanding to apparel-like hard cases

## 2026-04-10 Human Review Batch V32 Final Upstream Fixed

- Date: 2026-04-10T09:26:00Z
- Owner: Codex
- Goal: Validate the upstream retrieval, evidence, persona, and label-polarity fixes on the focused 11-seed generated-image bundle, specifically checking whether the business-prior line stops drifting darker than baseline on bottles, Disney totes, and gray Dasein bags.
- Data snapshot: focused 11-seed review set in `data/human_review_seed/images` with localization artifacts in `outputs/human_review_seed_localization` and evidence-aware retrieval index `data/human_review_seed/retrieval_index.v2.json`
- Code revision: no git repository initialized; use output artifacts in `outputs/human_review_batch_v32_final_upstream_fixed` plus updated prompt and retrieval code as the run anchor
- Environment revision: local GCE `g2-standard-12` VM with `NVIDIA L4`, driver `550.163.01`, CUDA `12.4`, project venv at `.venv`; stable sequential per-seed FLUX runner on the same VM
- Model config: local `black-forest-labs/FLUX.2-klein-9B`, `512x512`, `num_inference_steps=6`, `guidance_scale=1.0`, CUDA generation with CPU analysis; baseline and business-prior lines both consumed the new evidence-aware prompt stack and retrieval index
- Metrics: 22 images generated successfully for 11 seeds; combined board and report emitted at `outputs/human_review_batch_v32_final_upstream_fixed`; bottle business-prior rows no longer showed the old dark-fallback look during spot checks
- Qualitative notes: the bottle set materially improved and business-prior no longer darkened labels or bodies relative to baseline; Disney tote cases moved to casual, product-compatible human styling and preserved body/handle evidence much better; the remaining residual was `dasein_handbag_03`, where both lines still invented a handle-like loop despite upstream evidence saying the wallet-sized bag had no visible handles
- Follow-up: fix the direct-grip interaction contract for handleless handheld items, strengthen refined-neutral anti-office wardrobe guidance, then selectively rerun the Dasein seeds into a final combined bundle

## 2026-04-10 Human Review Batch V33 Final Bundle

- Date: 2026-04-10T09:44:00Z
- Owner: Codex
- Goal: Roll the direct-grip interaction fix and stronger refined-neutral wardrobe guidance into the final focused 11-seed bundle without regenerating unaffected seeds.
- Data snapshot: carried forward `outputs/human_review_batch_v32_final_upstream_fixed` into `outputs/human_review_batch_v33_final_bundle`, then selectively regenerated `dasein_handbag_01` and `dasein_handbag_03`
- Code revision: no git repository initialized; use `outputs/human_review_batch_v33_final_bundle` plus the prompt-layer direct-grip and wardrobe updates as the run anchor
- Environment revision: unchanged local GCE `g2-standard-12` VM with `NVIDIA L4`, driver `550.163.01`, CUDA `12.4`, project venv at `.venv`
- Model config: local `black-forest-labs/FLUX.2-klein-9B`, `512x512`, `num_inference_steps=6`, `guidance_scale=1.0`; sequential rerun over the full 11-seed manifest with `--reuse-existing`, regenerating only the two Dasein seeds that depended on the prompt fix
- Metrics: final combined report contains `22` rows across `11` seeds and both lines; portable final board emitted at `outputs/human_review_batch_v33_final_bundle/human_review_board.html`; board HTML contains no absolute `/home/...` asset paths
- Qualitative notes: `dasein_handbag_03` now uses direct hand contact on the wallet body in both lines and no longer invents a loop or handle; `dasein_handbag_01` moved from suit-like styling toward casual knitwear/denim while preserving the bag identity; previously improved bottle, Disney tote, and pillow rows were retained unchanged from the v32 carry-forward bundle
- Follow-up: present the v33 board as the final focused generated-image checkpoint for the repaired cases

## 2026-04-10 Generalization Diverse V1 Final Bundle

- Date: 2026-04-10T17:35:00Z
- Owner: Codex
- Goal: Validate broader generalization on a deliberately diverse 15-seed set that mixes approved controls with new apparel, footwear, backpack, and home-lighting customer-review inputs, using the upstream-first prompt/evidence fixes before a final generated-image confirmation pass.
- Data snapshot: `data/generalization_diverse_v1/review_seed_manifest.json` with `15` Walmart customer-review seeds spanning bottles, handbags, Disney tote, pillow, `6` apparel rows, `2` footwear rows, `2` backpacks, and `1` table lamp; localization artifacts in `outputs/generalization_diverse_v1_localization`
- Code revision: no git repository initialized; use output artifacts in `outputs/generalization_diverse_v1_upstream_review_v4` and `outputs/generalization_diverse_v1_final_bundle` plus the updated type-structural evidence, persona damping, backpack wearability, lamp support, and minimum-reference-size normalization as the run anchor
- Environment revision: unchanged local GCE `g2-standard-12` VM with `NVIDIA L4`, driver `550.163.01`, CUDA `12.4`, project venv at `.venv`; sequential per-line runner with `--reuse-existing` was used after a lamp-row failure exposed a general FLUX conditioning requirement that every reference image must have both dimensions `>= 64px`
- Model config: local `black-forest-labs/FLUX.2-klein-9B`, `512x512`, `num_inference_steps=6`, `guidance_scale=1.0`; upstream board regenerated at `outputs/generalization_diverse_v1_upstream_review_v4/human_review_board.html`; final image bundle emitted at `outputs/generalization_diverse_v1_final_bundle/human_review_board.html`
- Metrics: final generated bundle contains `30` rows across `15` seeds and both lines; `0` category-consistency warnings; `0` semantic-plausibility warnings; `6` advisory evidence-consistency warnings concentrated in `apparel_longsleeve_top_02`, `bag_rawlings_backpack_02`, `control_camelbak_bottle_02.business_prior`, and `home_pineapple_lamp_01.baseline`; broad-tranche category means were apparel evidence `0.5767`, bag evidence `0.6131`, drinkware evidence `0.5622`, footwear evidence `0.5523`, home decor evidence `0.7778`, and home lighting evidence `0.4191`
- Qualitative notes: the new upstream contracts held across the wider set: close-crop apparel no longer collapsed into `cozy_home`, backpacks stayed backpacks and used worn/carry support instead of generic bag staging, and the lamp rows remained upright supported-display scenes after the structural prompt rewrite; sampled final images across all new categories looked materially stronger than the automated evidence scores suggest, indicating the current evidence scorer remains more conservative than human judgment on successful reinventions
- Follow-up: use `outputs/generalization_diverse_v1_final_bundle/human_review_board.html` as the next human checkpoint for broad generalization, and if the reviewer approves, promote this broader diverse tranche into the main benchmark/review track

## 2026-04-11 Generalization Diverse V4 Upstream Review

- Date: 2026-04-11T03:20:00Z
- Owner: Codex
- Goal: Validate the business-prior planner/composer refactor upstream before paying for another full diverse image bundle, specifically checking whether retrieval now drives typed creative choices instead of collapsing into near-baseline prompts.
- Data snapshot: `data/generalization_diverse_v1/review_seed_manifest.json` with localization reused from `outputs/generalization_diverse_v2_localization` and retrieval index `data/human_review_seed/retrieval_index.v2.json`
- Code revision: no git repository initialized; use `outputs/generalization_diverse_v4_upstream_review` plus the typed creative-hint planner/composer changes as the run anchor
- Environment revision: unchanged local GCE `g2-standard-12` VM with `NVIDIA L4`, driver `550.163.01`, CUDA `12.4`, project venv at `.venv`; upstream review ran CPU-only analysis
- Model config: upstream-only prompt/evidence review, no FLUX generation; business-prior prompts consumed retrieval-derived `creative_seed`, `lighting_hint`, `camera_hint`, `creative_direction`, and `cast_hint`
- Metrics: `15` seeds, `30` upstream prompt rows; all `15` business-prior rows remained in true `retrieval` mode; mean baseline/business-prior prompt-token overlap dropped to `0.6234`; `15/15` baseline and business-prior backgrounds were unique across the tranche; scene-family equality remained `13/15`
- Qualitative notes: the business-prior line now differs upstream through typed environment, camera, lighting, and casting choices instead of mostly through appended generic style prose; the remaining similarity is concentrated in shared support relations and a still-limited scene-family vote space rather than prompt text collapse
- Follow-up: run a full diverse `business_prior` image refresh only, keep the approved baseline bundle unchanged, and then inspect whether the stronger prior survives into outputs without reintroducing anatomy or evidence drift

## 2026-04-11 Generalization Diverse V4 Business-Prior Refresh

- Date: 2026-04-11T04:15:00Z
- Owner: Codex
- Goal: Re-render the broad diverse tranche for the `business_prior` line only using the new typed creative-hint planner/composer, then merge it back into the approved baseline board.
- Data snapshot: reused the `15`-seed diverse manifest and `outputs/generalization_diverse_v2_localization` artifacts; baseline carried forward from `outputs/generalization_diverse_v3_final_bundle`
- Code revision: no git repository initialized; use `outputs/generalization_diverse_v4_business_prior_only` and `outputs/generalization_diverse_v4_final_bundle` as the output anchor
- Environment revision: unchanged local GCE `g2-standard-12` VM with `NVIDIA L4`, driver `550.163.01`, CUDA `12.4`, project venv at `.venv`; one long-lived FLUX run refreshed all `15` business-prior rows
- Model config: local `black-forest-labs/FLUX.2-klein-9B`, `512x512`, `num_inference_steps=6`, `guidance_scale=1.0`, CUDA generation with CPU-side analysis and candidate reranking
- Metrics: the merged `outputs/generalization_diverse_v4_final_bundle` contained `30` rows with `1` semantic warning (`apparel_tshirt_01.business_prior`) and `2` evidence warnings (`apparel_longsleeve_top_02.business_prior`, `bag_rawlings_backpack_02.business_prior`)
- Qualitative notes: the stronger business prior materially improved prompt differentiation and many final rows, but it exposed two selector-level regressions: the T-shirt line selected an anatomy-weak candidate, and the backpack row accepted weak retrieval neighbors for a structurally specific subtype
- Follow-up: fix anatomy weakness as a stronger candidate-selection penalty, hard-reject incompatible retrieval neighbors for structured subtypes like backpacks, and rerun only the affected business-prior rows

## 2026-04-11 Generalization Diverse V5 Final Bundle

- Date: 2026-04-11T05:03:00Z
- Owner: Codex
- Goal: Promote the business-prior refresh after fixing the two regressions exposed in the v4 merged bundle, without re-running unaffected rows.
- Data snapshot: carried forward `outputs/generalization_diverse_v4_final_bundle` and selectively regenerated `apparel_tshirt_01.business_prior` plus `bag_rawlings_backpack_02.business_prior` into `outputs/generalization_diverse_v4_regression_patch`
- Code revision: no git repository initialized; use `outputs/generalization_diverse_v5_final_bundle` as the current broad review anchor
- Environment revision: unchanged local GCE `g2-standard-12` VM with `NVIDIA L4`, driver `550.163.01`, CUDA `12.4`, project venv at `.venv`
- Model config: local `black-forest-labs/FLUX.2-klein-9B`, `512x512`, `num_inference_steps=6`, `guidance_scale=1.0`; targeted rerun on `2` business-prior rows after selector/retrieval fixes
- Metrics: final merged bundle contains `30` rows; `0` semantic warnings; `1` evidence warning (`apparel_longsleeve_top_02.business_prior`); `1` baseline-only category warning remains on `control_dasein_handbag_03.baseline`, which earlier human review had already treated as acceptable
- Qualitative notes: the T-shirt regression was resolved by strengthening anatomy penalties in candidate selection, which switched the winning business-prior candidate away from the anatomy-weak reveal shot; the backpack regression was resolved by hard-rejecting structurally incompatible retrieval neighbors, which pushed `bag_rawlings_backpack_02.business_prior` to a category fallback with clean evidence alignment instead of forcing a weak retrieval prior
- Follow-up: use `outputs/generalization_diverse_v5_final_bundle/human_review_board.html` as the next broad human review checkpoint

## 2026-04-11 Generalization Diverse V6 Targeted Repair Bundle

- Date: 2026-04-11T09:10:00Z
- Owner: Codex
- Goal: Repair the four remaining broad-bundle human findings by fixing the upstream culprits first, then regenerating only the affected rows: `apparel_tshirt_01`, `footwear_easyspirit_01`, `footwear_easyspirit_02`, and `home_pineapple_lamp_01`.
- Data snapshot: reused `data/generalization_diverse_v1/review_seed_manifest.json`, localization artifacts from `outputs/generalization_diverse_v2_localization`, and retrieval index `data/human_review_seed/retrieval_index.v2.json`
- Code revision: no git repository initialized; use `outputs/generalization_diverse_v11_targeted_upstream_review` and `outputs/generalization_diverse_v6_targeted_final_bundle` as the run anchors
- Environment revision: unchanged local GCE `g2-standard-12` VM with `NVIDIA L4`, driver `550.163.01`, CUDA `12.4`, project venv at `.venv`
- Model config: upstream-only review on CPU, then local `black-forest-labs/FLUX.2-klein-9B` generation at `512x512`, `num_inference_steps=6`, `guidance_scale=1.0`, CUDA generation with CPU-side analysis and reranking
- Metrics: targeted upstream board emitted at `outputs/generalization_diverse_v11_targeted_upstream_review/human_review_board.html`; targeted image bundle emitted at `outputs/generalization_diverse_v6_targeted_final_bundle/human_review_board.html`; all four rows regenerated across both lines
- Qualitative notes: the upstream fix set split full-shape conditioning from surface-evidence inference for body-attached products; apparel references now use product-only evidence assets instead of person-bearing crops; low-profile footwear now uses a surface-focused evidence mask plus footwear-specific surface-treatment correction, which downgraded sock-driven false all-over patterns into localized or low-variation shoe evidence; lamp evidence now explicitly carries structured shade-ridge relief so business-prior no longer flattens the shade into a smooth surface
- Follow-up: bring the targeted v6 repair bundle to human review, and if it is accepted, merge these repaired rows back into the broad generalization board

## 2026-04-11 Generalization Diverse V6 Approved Bundle

- Date: 2026-04-11T09:35:00Z
- Owner: Codex
- Goal: Promote the human-approved targeted repair rows into the current broad diverse board without reopening the full tranche, and explicitly carry the remaining sneaker-baseline color issue as accepted open debt.
- Data snapshot: merged `outputs/generalization_diverse_v5_final_bundle` with the regenerated rows from `outputs/generalization_diverse_v6_targeted_final_bundle` for `apparel_tshirt_01`, `footwear_easyspirit_01`, `footwear_easyspirit_02`, and `home_pineapple_lamp_01`
- Code revision: no git repository initialized; use `outputs/generalization_diverse_v6_approved_bundle` as the current broad approved anchor
- Environment revision: unchanged local GCE `g2-standard-12` VM with `NVIDIA L4`, driver `550.163.01`, CUDA `12.4`, project venv at `.venv`
- Model config: no new generation in this promotion step; merged prior generated outputs and rebuilt the portable HTML review board
- Metrics: merged approved bundle contains `30` rows with `0` semantic warnings, `0` category warnings, and `0` evidence warnings in the generated report
- Qualitative notes: the user approved the targeted repair bundle and explicitly chose to move on despite a residual limitation in the baseline rows for `footwear_easyspirit_01` and `footwear_easyspirit_02`, where the white sneakers still render too gray; this issue remains tracked as open debt rather than a release blocker for the broader tranche
- Follow-up: treat `outputs/generalization_diverse_v6_approved_bundle/human_review_board.html` as the current broad approved board, then proceed to the next larger diverse evaluation tranche while keeping the sneaker-baseline color issue on the backlog

## 2026-04-11 Generalization Diverse V2 New Categories Bundle

- Date: 2026-04-11T10:45:00Z
- Owner: Codex
- Goal: Expand the broad evaluation beyond apparel, bags, footwear, drinkware, and lighting by adding new categories that stress broader generalization: kitchen appliances, bedding, furniture seating, and pet-home products.
- Data snapshot: new tranche at `data/generalization_diverse_v2/review_seed_manifest.json` with `14` seeds total: `4` approved controls plus `10` new-category rows across blenders, comforters, chairs, and pet beds
- Code revision: no git repository initialized; use `outputs/generalization_diverse_v2_upstream_review` for the upstream checkpoint and `outputs/generalization_diverse_v2_new_categories_final_bundle` for the clean generated-image bundle
- Environment revision: unchanged local GCE `g2-standard-12` VM with `NVIDIA L4`, driver `550.163.01`, CUDA `12.4`, project venv at `.venv`
- Model config: local `black-forest-labs/FLUX.2-klein-9B`, `512x512`, `num_inference_steps=6`, `guidance_scale=1.0`; CUDA generation with CPU-side analysis and candidate reranking
- Metrics: final generated bundle contains `28` rows with `0` semantic warnings, `0` category warnings, and `0` evidence warnings; upstream review confirmed coherent typing and support planning for the new categories before generation
- Qualitative notes: this tranche exposed three upstream tooling/modeling gaps that were fixed before generation: the manifest builder had versioned-path assumptions tied to `generalization_diverse_v1`; the Grounding DINO proposer was feeding text in an invalid shape; and the identity/support taxonomy only handled the earlier narrow set of categories, which initially caused blenders to collapse into apparel and broad soft/furniture products to collapse into generic `product`; after fixing those layers, the generated outputs preserved the new product classes well enough for a clean human checkpoint
- Follow-up: bring `outputs/generalization_diverse_v2_new_categories_final_bundle/human_review_board.html` to human review as the next broad final-image checkpoint

## 2026-04-12 Generalization Diverse V2 Targeted Rootfix V30 Curated Bundle

- Date: 2026-04-12T00:40:00Z
- Owner: Codex
- Goal: Rebuild the targeted new-category repair checkpoint under the latest upstream evaluator and evidence rules without regenerating images, so the bundle reflects the actual repaired state instead of stale warning surfaces.
- Data snapshot: reused the selected images from `outputs/generalization_diverse_v2_targeted_rootfix_v29_curated_final_bundle`, seed manifest `data/generalization_diverse_v2/review_seed_manifest.json`, and localization artifacts from `outputs/generalization_diverse_v2_targeted_rootfix_v25_localization`
- Code revision: updated `src/product_campaign_pipeline/review_batch.py` to use focus-crop category evaluation, compact hand-held semantic tolerance, and soft-goods-aware evidence gating; use `outputs/generalization_diverse_v2_targeted_rootfix_v30_curated_final_bundle` as the rebuilt checkpoint anchor
- Environment revision: unchanged local GCE `g2-standard-12` VM with `NVIDIA L4`, driver `550.163.01`, CUDA `12.4`, project venv at `.venv`
- Model config: no new FLUX generation; analysis-only rebuild on CPU with existing selected PNGs and regenerated focus artifacts for scoring
- Metrics: rebuilt curated bundle contains `14` rows with `0` category warnings, `0` semantic warnings, and `0` evidence warnings in the generated report
- Qualitative notes: this pass fixed three evaluator blind spots without reopening older output regressions: category scoring now uses the generated product focus crop when available instead of the full scene frame; compact hand-held carried-by-hand scenes tolerate a small negative support margin when anatomy and product evidence remain strong; and soft-goods evidence extraction/evaluation now stops misclassifying tonal comforters and low-profile pet beds as printed or structurally inflated due to shadow and fold contrast
- Follow-up: treat `outputs/generalization_diverse_v2_targeted_rootfix_v30_curated_final_bundle/human_review_board.html` as the current targeted repaired checkpoint and use it for independent review

## 2026-04-12 Review Board Relative-Path Repair

- Date: 2026-04-12T01:20:00Z
- Owner: Codex
- Goal: Fix the portable HTML review board so previously sanitized relative asset paths under `board_assets/...` still render correctly when the user downloads only the review directory.
- Data snapshot: reused `outputs/generalization_diverse_v2_targeted_rootfix_v30_curated_final_bundle/reports/generation_report.json` and the already staged `board_assets` tree in the same bundle
- Code revision: updated `src/product_campaign_pipeline/review_batch.py` so `_board_image_reference()` preserves already-staged relative asset paths when they exist under the board root, and added regression coverage in `tests/review/test_review_batch.py`
- Environment revision: unchanged local GCE `g2-standard-12` VM with `NVIDIA L4`, driver `550.163.01`, CUDA `12.4`, project venv at `.venv`
- Model config: no FLUX generation; board-render-only repair and verification
- Metrics: focused test suite now passes with `119` tests; rebuilt `v30` board has `0` empty image `src` entries, `28` relative image refs, and `0` absolute `/home/...` refs
- Qualitative notes: the bug was in the second-stage renderer, not missing assets; rendering a board from already-sanitized rows used to blank out image paths because relative `board_assets/...` values were checked from the repo cwd instead of the board root
- Follow-up: use the rebuilt `outputs/generalization_diverse_v2_targeted_rootfix_v30_curated_final_bundle/human_review_board.html` as the current approved portable checkpoint and keep the regression test in place for all future review bundles

## 2026-04-12 generalization_diverse_v3_targeted_repair_v5_final_bundle
- Merged the clean mug rerender into the targeted v3 repair bundle after disabling post-generation color repair for low-saturation glazed drinkware.
- Final bundle: outputs/generalization_diverse_v3_targeted_repair_v5_final_bundle
- Direct inspection: no remaining blocker-level issues across bedding_quilt_01, control_dasein_handbag_03, drinkware_mug_02, kitchen_coffee_maker_02, bag_backpack_cooler_01.
- Agent review: Euler found no blocker-level issues; Galileo found no blocker-level issues.
- Status: closed targeted repair loop; ready to fold forward into the broader evaluation track.

## 2026-04-13 Generalization Diverse V4 Targeted Repair V8 Curated Bundle

- Date: 2026-04-13T00:20:00Z
- Owner: Codex
- Goal: Close the remaining blocker-level issues in the four held `v4` rows without reopening earlier regressions, then clear the result through independent review before promotion.
- Data snapshot: reused the broader tranche at `outputs/generalization_diverse_v4_final_bundle_v1`, merged the repaired four-row bundle from `outputs/generalization_diverse_v4_targeted_repair_v8_curated_final_bundle`, and normalized the curated report so all output and candidate paths point back into the `v8` bundle itself
- Code revision: retrieval hard-conflict filtering, rigid-product color handling, no-human display-product guardrails, low-profile soft candidate-mode selection, and chromatic rigid post-generation color repair were all updated earlier in `src/product_campaign_pipeline/review_batch.py` and `src/product_campaign_pipeline/composer/prompts.py`; this step was the first blocker-level validation of those combined fixes on the curated four-row bundle
- Environment revision: unchanged local GCE `g2-standard-12` VM with `NVIDIA L4`, driver `550.163.01`, CUDA `12.4`, project venv at `.venv`
- Model config: no new broad rerender after the curated `v8` assembly; validation used the repaired final PNGs already staged in the curated bundle plus the paired generated report
- Metrics: curated bundle contains `8` rows with `0` category warnings, `0` semantic warnings, and `0` evidence warnings in the paired report; board is portable with `0` empty image refs, `0` missing image refs, and `0` absolute `/home/...` paths
- Qualitative notes: this checkpoint closes the original `v4` blockers without fallback churn; the backpack and office-chair rows preserve structure cleanly, the pet-bed row no longer invents an ottoman-like base, and the toaster pair stays product-only and chromatically aligned without the earlier torso bleed or mask-edge recolor artifact
- Follow-up: fold the repaired four rows into a promoted full-tranche artifact at `outputs/generalization_diverse_v4_approved_bundle` and continue to the next broader tranche rather than bringing `v4` back for another human checkpoint

## 2026-04-16 Environment Reconciliation On Migrated vast.ai VM

- Date: 2026-04-16T02:25:00Z
- Owner: Codex
- Goal: restore a runnable local development and generation environment after moving the project from the old GCE L4 VM to a new vast.ai VM, while treating the migrated filesystem as the source of truth
- Data snapshot: migrated project root at `/workspace/product_campaign_pipeline`, raw CreativeRanking corpus at `/workspace/data`, and restored Codex state at `/root/.codex`
- Code revision: patched `scripts/render_architecture_diagrams.py` to resolve its output directory from the repo root instead of the old absolute path; updated `README.md`, `docs/context/DATA_CONTRACTS.md`, `docs/context/ENVIRONMENT.md`, and `docs/context/VM_MIGRATION.md` so maintained docs reflect `/workspace/...` rather than `/home/nyle_j_huang/...`
- Environment revision: new host is Ubuntu 22.04.5 on vast.ai with `NVIDIA GeForce RTX 3090 Ti`, driver `550.163.01`, CUDA runtime `12.4`; recreated project venv at `/workspace/product_campaign_pipeline/.venv` with Python `3.12.13` because the migrated venv was broken and still pointed to the old machine
- Model config: installed project extras from `pyproject.toml`, then replaced the default `torch 2.11.0+cu130` wheel with `torch 2.6.0+cu124` and `torchvision 0.21.0+cu124` from the PyTorch `cu124` index so CUDA would initialize correctly against the current driver; restored `pillow` to `11.3.0` to satisfy the project pin after the torch reinstall had drifted it to `12.1.1`
- Metrics: `scripts/verify_environment.sh` now passes end to end; `torch.cuda.is_available()` is `True`; CUDA smoke test returns `[2.0, 4.0, 6.0]`; `pcp --help` succeeds; imports for `Flux2KleinPipeline`, `BlipProcessor`, `AutoModelForZeroShotObjectDetection`, `Sam2Processor`, and `Sam2Model` all succeed in the rebuilt venv
- Qualitative notes: the main migration blocker was not missing code or missing secrets but a wheel/driver mismatch introduced by reinstalling from unconstrained `torch>=2.4`; on this VM the default resolver chose a CUDA 13 build that could not initialize against the installed driver, so the environment needed an explicit driver-compatible torch stack rather than a host-driver change
- Follow-up: resume the post-`v5` review workflow from the migrated filesystem state, with `/workspace/...` treated as the live path root and historical `/home/nyle_j_huang/...` paths left intact only inside old reports

## 2026-04-16 Generalization Diverse V5 Curated Repair Pass On Migrated vast.ai VM

- Date: 2026-04-16T04:05:00Z
- Owner: Codex
- Goal: close the remaining real blocker-level issues in the migrated `v5` bundle without reopening older failures, and separate those from evaluator-only warning noise before the next human checkpoint
- Data snapshot: base board `outputs/generalization_diverse_v5_final_bundle_v1`, analysis rebuild `outputs/generalization_diverse_v5_rescored_v2`, curated bundle `outputs/generalization_diverse_v5_curated_v2`
- Code revision: updated `src/product_campaign_pipeline/review_batch.py` so generated-output relocalization uses product-only queries (`canonical_type + category`) instead of merchant-title tokens; compact direct-grip baselines now use the same candidate pool as business-prior; semantic review now detects multi-person scenes using the generated localizer; updated `src/product_campaign_pipeline/composer/prompts.py` to hard-ban background people in single-model scenes, avoid officewear for compact hand-held accessories, and bias human-scene backgrounds toward uncrowded settings; added regression coverage in `tests/review/test_review_batch.py` and `tests/composer/test_prompts.py`
- Environment revision: first local FLUX generation on the new vast.ai VM completed successfully after downloading the missing `black-forest-labs/FLUX.2-klein-9B` weights into `/workspace/.hf_home`
- Model config: local `black-forest-labs/FLUX.2-klein-9B`, `512x512`, current prompt/evaluator stack on CUDA generation with CPU-side review models
- Metrics: `pytest tests/composer/test_prompts.py tests/review/test_review_batch.py -q` passed with `155` tests; targeted repaired rows were regenerated at `outputs/generalization_diverse_v5_targeted_compact_repair_v3`, `outputs/generalization_diverse_v5_targeted_single_model_repair_v1`, and `outputs/generalization_diverse_v5_targeted_single_model_repair_v2`
- Qualitative notes: three real blocker classes were closed in this pass: `control_dasein_handbag_03 baseline` no longer collapses into a torso-dominant compact-accessory portrait; `bag_backpack_cooler_04 business_prior` no longer includes extra background people; and `footwear_easyspirit_04 business_prior` now satisfies the single-model requirement after tightening both prompt and review-time person-count handling. The remaining broad `v5` warnings are still dominated by evaluator and source-evidence calibration noise, not clearly bad final images.
- Follow-up: use `outputs/generalization_diverse_v5_curated_v2/human_review_board.html` as the next final-image checkpoint, and continue source-evidence calibration for low-saturation cool textiles and dark reflective rigid products separately from the blocker loop

## 2026-04-16 Generalization Diverse V5 Targeted Rootfix V10 Curated Bundle

- Date: 2026-04-16T07:10:00Z
- Owner: Codex
- Goal: close the remaining blocker-level failures in the six-row targeted `v5` set without reopening earlier artifact, casting, or color regressions, then validate the repaired board through independent review
- Data snapshot: base targeted repair rows from `outputs/generalization_diverse_v5_targeted_rootfix_v1_final`, refreshed `pet_dog_bed_04` rows from `outputs/generalization_diverse_v5_targeted_rootfix_v5_petbed`, refreshed `bedding_quilt_04 baseline` from `outputs/generalization_diverse_v5_targeted_rootfix_v10_quilt`, merged into `outputs/generalization_diverse_v5_targeted_rootfix_v10_curated`
- Code revision: updated `src/product_campaign_pipeline/composer/prompts.py` so the shared `clarity` reinvention clause no longer injects human-subject language into product-only rows, product-only soft-goods prompts explicitly forbid being worn/held/wrapped across a person, and bedding-specific furnished-interior planning now selects bed/daybed support language rather than generic standalone-furniture interiors; updated `src/product_campaign_pipeline/review_batch.py` so semantic evaluation detects any human presence when a product-only frame is required, penalizes it in candidate scoring, and filters those candidates out before final selection; added regression coverage in `tests/composer/test_prompts.py` and `tests/review/test_review_batch.py`
- Environment revision: unchanged vast.ai VM at `/workspace/product_campaign_pipeline` with local `FLUX.2-klein-9B` generation on the RTX 3090 Ti
- Model config: rerendered only `bedding_quilt_04 baseline` after the bedding support/prompt change; reused the already-accepted repaired pet-bed rerender rather than regenerating the whole targeted set
- Metrics: `pytest tests/composer/test_prompts.py tests/review/test_review_batch.py -q` passed with `174` tests; `python -m compileall src scripts` passed; final curated board contains `12` rows, `0` empty image refs, `0` absolute `/home/...` refs, and `0` absolute `/workspace/...` refs
- Qualitative notes: the earlier quilt blocker was real but upstream-causal: product-only bedding rows were still inheriting human wording from the generic `clarity` mode and could also drift into furniture-throw interiors because `furnished_interior` was not category-aware enough for bedding. The final rerender removes the human-presence path and reanchors bedding to bed/daybed support, while keeping the previously repaired pet-bed, watermark/logo, apparel color, and coffee-maker logo fixes intact.
- Follow-up: use `outputs/generalization_diverse_v5_targeted_rootfix_v10_curated/human_review_board.html` as the next targeted final-image checkpoint; remaining report flags in this bundle are evaluator residue, not blocker-level image defects

## 2026-04-16 Generalization Diverse V7 Curated V5 Final Repair Pass

- Date: 2026-04-16T11:10:00Z
- Owner: Codex
- Goal: close the last blocker-level issues in the broad `v7` tranche without reopening earlier failures, then rebuild a portable curated board from corrected rows only
- Data snapshot: base bundle `outputs/generalization_diverse_v7_final_bundle_v1`, prior curated bundle `outputs/generalization_diverse_v7_curated_v4`, targeted row rerenders `outputs/generalization_diverse_v7_targeted_bag_backpack_cooler_05_v5`, `outputs/generalization_diverse_v7_targeted_bag_backpack_cooler_05_v6_baseline_only`, and `outputs/generalization_diverse_v7_targeted_apparel_tshirt_05_v1`, merged into `outputs/generalization_diverse_v7_curated_v5`
- Code revision: updated `src/product_campaign_pipeline/review_batch.py` so uncertain laid-flat neutral apparel no longer locks in warm beige/gold color facts at high confidence, and updated `src/product_campaign_pipeline/composer/prompts.py` so uncertain apparel uses cutout-first reference ordering and refined-neutral apparel styling no longer suggests alternate hero tops like knitwear; added a shirt-specific anti-sweater guardrail and regression coverage in `tests/review/test_review_batch.py` and `tests/composer/test_prompts.py`
- Environment revision: unchanged vast.ai VM at `/workspace/product_campaign_pipeline`, local `FLUX.2-klein-9B` generation on the RTX 3090 Ti
- Model config: rerendered only the validated blocker rows instead of the full tranche, then rebuilt a curated portable bundle from report rows with sanitized relative assets
- Metrics: `pytest tests/composer/test_prompts.py tests/review/test_review_batch.py -q` passed with `184` tests; `python -m compileall src scripts` passed; final curated board at `outputs/generalization_diverse_v7_curated_v5` has `72` image refs, `0` empty refs, `0` absolute `/home/...` refs, and `0` absolute `/workspace/...` refs
- Qualitative notes: the two real blocker classes in `v7` were different. `bag_backpack_cooler_05` had an upstream subtype/context problem: the system preserved `backpack` but not the cooler-specific open insulated compartment. That was closed by adding functional-subtype evidence, source-context references, and subtype-aware semantic scoring, then selecting the corrected baseline and business-prior rerenders into the curated bundle. `apparel_tshirt_05 business_prior` had a different root cause: warm room-light contamination plus generic refined-neutral apparel styling could drift a laid-flat white T-shirt into a sweater-like beige top. That was closed by softening uncertain neutral apparel color evidence, prioritizing cutout references for uncertain apparel, and banning alternate-top reinterpretation for shirts.
- Follow-up: use `outputs/generalization_diverse_v7_curated_v5/human_review_board.html` as the next broad final-image checkpoint

## 2026-04-16 Generalization Diverse V7 Curated V7 Review-Feedback Closure

- Date: 2026-04-16T13:35:00Z
- Owner: Codex
- Goal: close the user-reviewed `v7_curated_v5` follow-up issues, exclude invalid-source rows instead of forcing churn on unrecoverable inputs, and rebuild a portable curated checkpoint that only contains still-valid review rows
- Data snapshot: prior curated board `outputs/generalization_diverse_v7_curated_v5`, targeted rerender bundles `outputs/generalization_diverse_v7_targeted_review_feedback_v5`, `outputs/generalization_diverse_v7_targeted_review_feedback_v6_quilt_bp`, and `outputs/generalization_diverse_v7_targeted_review_feedback_v7_quilt_validity`, merged into `outputs/generalization_diverse_v7_curated_v7`
- Code revision: updated `src/product_campaign_pipeline/review_batch.py` so localized contrast-panel evidence is category-aware, rigid drinkware masks repair small label-band notches before export, border-foreground intrusions on soft goods are detected and suppressed, low-variation soft-goods evidence no longer invents patterned/ink-like treatments, invalid-source scoring now rejects visually incompatible localized crops for soft nonhuman products, and semantic selection now includes single-model and dress-layering margins; updated `src/product_campaign_pipeline/composer/prompts.py` with apparel anti-gray-sleeve / anti-layering guardrails, drinkware anti-detached-patch guardrails, and soft-goods anti-ink / anti-panel guardrails
- Environment revision: unchanged vast.ai VM at `/workspace/product_campaign_pipeline` with local `FLUX.2-klein-9B` generation on the RTX 3090 Ti
- Model config: rerendered only the held rows that were still valid review targets (`control_camelbak_bottle_02`, `apparel_dress_05`) and moved four rows to invalid-source handling instead of forcing more generation (`furniture_folding_chair_05`, `kitchen_coffee_maker_05`, `drinkware_mug_05`, `bedding_quilt_05`)
- Metrics: `pytest tests/composer/test_prompts.py tests/review/test_review_batch.py -q` passed with `195` tests; `python -m compileall src scripts` passed; final curated bundle contains `28` valid review rows and `4` invalid-source exclusions; rebuilt board at `outputs/generalization_diverse_v7_curated_v7/human_review_board.html` has `56` image refs, `0` empty refs, `0` missing refs, and `0` absolute `/home/...` or `/workspace/...` paths
- Qualitative notes: the real closure in this pass was upstream. The bottle-band triangle artifact was caused by a notch in the exported rigid-body evidence mask, not by FLUX itself. The dress gray-sleeve failure came from over-generic contrast-panel evidence and weak apparel guardrails, which also risked reappearing on other mixed-surface garments. The folding-chair, coffee-maker, mug, and quilt rows turned out to be invalid-source cases rather than repairable generator misses, so they were moved out of the review set instead of being churned through more image fixes. The first `v7_curated_v6` merge had a portability defect because already-sanitized `board_assets/...` paths were copied as if they were raw paths; `v7_curated_v7` fixes that by rebuilding from original source reports and re-sanitizing in one pass.
- Follow-up: use `outputs/generalization_diverse_v7_curated_v7/human_review_board.html` as the new broad `v7` checkpoint; remaining report warnings on some good rows are evaluator residue, not blocker-level image debt

## 2026-04-16 Generalization Diverse V7 Curated V7 Human Approval

- Date: 2026-04-16T13:50:00Z
- Owner: Codex
- Goal: record human approval of the repaired broad `v7` checkpoint and promote it as the current approved bundle for the project
- Data snapshot: approved board `outputs/generalization_diverse_v7_curated_v7/human_review_board.html`, paired reports `outputs/generalization_diverse_v7_curated_v7/reports/generation_report.json` and `outputs/generalization_diverse_v7_curated_v7/reports/invalid_sources.json`
- Qualitative notes: the user reviewed the visible rows and approved the bundle as “good enough.” This promotes the repaired bottle and dress rows, preserves the invalid-source exclusions for the four unreliable inputs, and closes the `v7` repair loop without reopening earlier issues.
- Follow-up: use `outputs/generalization_diverse_v7_curated_v7` as the current approved broad checkpoint and continue to the next broader tranche under the same upstream-first review policy

## 2026-04-16 Generalization Diverse V8 Broad Tranche Build And Curation

- Date: 2026-04-16T16:10:00Z
- Owner: Codex
- Goal: build the next broader `v8` tranche from the remaining unseen local customer-review photos, run the upstream-first gate, fix any newly exposed root-cause issues, and prepare the next human checkpoint without reintroducing previously closed failure classes
- Data snapshot: source config `data/generalization_diverse_v8/source_selection.json`, final generation bundle `outputs/generalization_diverse_v8_final_bundle_v1`, upstream validity passes `outputs/generalization_diverse_v8_upstream_review_v1` through `v5`, targeted chair rerenders `outputs/generalization_diverse_v8_targeted_folding_chair_v1` and `v2`, curated checkpoint `outputs/generalization_diverse_v8_curated_v2`
- Code revision: updated `src/product_campaign_pipeline/review_batch.py` so structured kitchen-appliance crops are invalidated when the raw crop caption resolves to a competing standalone vessel/sub-object without sufficient canonical support, and so soft nonhuman products with border intrusion are invalidated when the raw caption is dominated by person/animal fragments instead of the product; updated `src/product_campaign_pipeline/composer/prompts.py` to add stronger open-frame folding-chair guardrails against ghosted / semi-transparent / overlapping chair planes and background furniture planes merging into the chair silhouette; updated `select_reinvention_candidate_modes_for_line()` so rigid placed folding chairs prefer `clarity` / `reveal` instead of the default `balanced` / `reveal`; added regression coverage in `tests/review/test_review_batch.py` and `tests/composer/test_prompts.py`
- Environment revision: unchanged vast.ai VM at `/workspace/product_campaign_pipeline` with local `FLUX.2-klein-9B` generation on the RTX 3090 Ti
- Model config: `v8` began as 18 seeds (4 controls + 14 unseen product rows), then source reselection and validity refinement reduced the reviewable final set to 12 valid ids; final generation ran only on the valid-source rows from `outputs/generalization_diverse_v8_upstream_review_v3`; later upstream passes `v4` and `v5` reclassified `kitchen_coffee_maker_06` and `pet_dog_bed_06` as invalid-source after the new validity rules landed
- Metrics: focused regression suite reached `198` passing tests; `python -m compileall src scripts` passed; curated checkpoint `outputs/generalization_diverse_v8_curated_v2` contains `24` valid review rows across `12` ids, `5` invalid-source exclusions (`bag_backpack_cooler_06`, `bedding_quilt_06`, `pet_dog_bed_06`, `kitchen_blender_05`, `kitchen_coffee_maker_06`), and `1` held row (`furniture_folding_chair_06`); the curated board has `48` image refs, `0` empty refs, `0` missing refs, and `0` absolute filesystem paths
- Qualitative notes: the most important `v8` work was upstream, not downstream. Several newly selected unseen source photos were simply not reviewable because the localized crop captured border text, animal fragments, or sub-objects instead of the intended product. The new validity rules now catch those cases earlier, especially for structured kitchen appliances and pet-home soft goods. `kitchen_coffee_maker_06` was initially allowed through, but both local captioning and independent review showed the crop and final outputs resolving to a mug/carafe-like sub-object, so it was moved to invalid-source. `pet_dog_bed_06` likewise turned out to be a hand/animal-fragment contamination case and is now excluded upstream. `furniture_folding_chair_06` remained contested after two narrow rerenders: one reviewer repeatedly saw a semi-transparent / ghosted chair artifact, the other cleared it. Rather than churn the broader tranche further on an unresolved row, the chair was held out of the broad curated checkpoint and left for its own narrow repair track.
- Follow-up: use `outputs/generalization_diverse_v8_curated_v2/human_review_board.html` as the next broad human checkpoint; keep `furniture_folding_chair_06` on a separate repair thread rather than reopening the full `v8` tranche around it

## 2026-04-17 Generalization Diverse V8 Human Approval

- Date: 2026-04-17T00:00:00Z
- Owner: Codex
- Goal: record human approval of the curated broad `v8` checkpoint and promote it as the current approved broad tranche
- Data snapshot: approved board `outputs/generalization_diverse_v8_curated_v2/human_review_board.html`, paired reports `outputs/generalization_diverse_v8_curated_v2/reports/generation_report.json`, `outputs/generalization_diverse_v8_curated_v2/reports/invalid_sources.json`, and `outputs/generalization_diverse_v8_curated_v2/reports/held_rows.json`
- Qualitative notes: the user reviewed the visible rows and judged them “good enough.” The five invalid-source exclusions remain intentionally out of scope for this checkpoint, and the held `furniture_folding_chair_06` row stays on its own repair track rather than blocking the broad tranche.
- Follow-up: use `outputs/generalization_diverse_v8_curated_v2` as the current approved broad checkpoint and continue to the next broader tranche under the same upstream-first review policy

## 2026-04-17 Generalization Diverse V9 Remaining-Source Screen

- Date: 2026-04-17T02:30:00Z
- Owner: Codex
- Goal: determine whether a real broad `v9` tranche still exists in the remaining unseen local customer-review pool, or whether the source pool is now too degraded to justify another broad generation pass
- Data snapshot: sampled broad manifest `data/generalization_diverse_v9/review_seed_manifest.json` with initial upstream pass `outputs/generalization_diverse_v9_upstream_review_v1`, plus exhaustive remaining-source screen `data/generalization_diverse_v9_source_screen/review_seed_manifest.json` and `outputs/generalization_diverse_v9_source_screen_upstream_v1`
- Code revision: no new modeling fix was shipped in this step; the purpose was source-pool measurement under the current upstream validity rules and retrieval stack
- Environment revision: unchanged vast.ai VM at `/workspace/product_campaign_pipeline` with local localization and upstream-review models on CUDA
- Metrics: the first sampled `v9` broad attempt contained `17` seeds (`4` controls + `13` products) but only `10` valid ids survived upstream review; the exhaustive remaining-source screen covered all `17` remaining unused customer photos and found only `6` valid product ids versus `11` invalid-source rows
- Qualitative notes: this is the first clear sign of source-pool exhaustion rather than model weakness. The surviving remainder is narrow (`shirt`, `dress`, `mug`, `folding chair`, `coffee maker`, `toaster`). The invalid remainder is dominated by the same upstream source-quality problems the pipeline has already learned to reject correctly: incomplete portable-product global shape on the remaining backpack rows; border-text or border-foreground contamination on the remaining bedding, pet-bed, blender, and several kitchen-appliance rows; and repeated localized-crop visual-type conflicts on the remaining slow-cooker images. Because these failures are in the source pool itself, forcing another broad generation tranche would mostly measure source degradation, not model generalization.
- Follow-up: stop broad-tranche expansion from the local remaining-source pool and pivot to the final benchmark / acceptance phase using the already approved broad checkpoints plus any still-useful targeted approved checkpoints

## 2026-04-17 Final Benchmark Candidate Assembly

- Date: 2026-04-17T02:45:00Z
- Owner: Codex
- Goal: assemble the final benchmark / acceptance candidate bundle from the already approved broad and targeted checkpoints, using explicit row-level precedence so newer approved repairs override older broad-board rows without hand-merging
- Data snapshot: consolidated benchmark bundle `outputs/final_benchmark_candidate_v1`, built from approved sources `human_review_batch_v33_final_bundle`, `generalization_diverse_v6_approved_bundle`, `generalization_diverse_v2_new_categories_final_bundle`, `generalization_diverse_v2_targeted_rootfix_v30_curated_final_bundle`, `generalization_diverse_v3_targeted_repair_v5_final_bundle`, `generalization_diverse_v4_approved_bundle`, `generalization_diverse_v5_approved_bundle`, `generalization_diverse_v5_targeted_rootfix_v10_curated`, `generalization_diverse_v7_curated_v7`, and `generalization_diverse_v8_curated_v2`
- Code revision: added `scripts/build_final_benchmark_bundle.py` to resolve row precedence, restage portable board assets from prior approved bundles, preserve targeted override rows over stale broad rows, and emit a consolidated benchmark board, row report, bundle-resolution summary, and excluded/held-source registry
- Environment revision: unchanged vast.ai VM at `/workspace/product_campaign_pipeline`
- Metrics: final candidate bundle contains `88` approved ids / `176` generated rows; the portable board at `outputs/final_benchmark_candidate_v1/human_review_board.html` has `352` image refs, `0` empty refs, `0` missing refs, and `0` absolute filesystem paths; benchmark summary and exclusion registry are emitted at `outputs/final_benchmark_candidate_v1/reports/benchmark_summary.json` and `outputs/final_benchmark_candidate_v1/reports/excluded_or_held_sources.json`
- Qualitative notes: the benchmark candidate intentionally includes approved non-blocking residual limitations from earlier broad checkpoints, because the purpose is to measure the accepted system as it stands rather than a selectively polished subset. It also carries the invalid-source and held-source registry forward explicitly so the final acceptance set is auditable: `21` rows remain excluded or held across the late broad-tranche and remaining-pool screens.
- Follow-up: treat `outputs/final_benchmark_candidate_v1/human_review_board.html` as the next and likely final major human review checkpoint for acceptance

## 2026-04-17 Final Benchmark Targeted Root-Cause Repair V2

- Date: 2026-04-17T05:55:00Z
- Owner: Codex
- Goal: fix the root causes behind the final benchmark review feedback instead of patching isolated rows, then rerender only the affected valid-source ids before rebuilding the benchmark board
- Data snapshot: targeted manifest `data/final_benchmark_targeted_v2/review_seed_manifest.json`; localization `outputs/final_benchmark_targeted_v2_localization_v1`; upstream passes `outputs/final_benchmark_targeted_v2_upstream_v1`, `v2`, and `v3`; repaired generation bundle `outputs/final_benchmark_targeted_v2_final_v1`
- Code revision: updated `src/product_campaign_pipeline/composer/prompts.py` to remove gray-prone scene variants, add explicit background-collapse guardrails, strengthen dress/shirt low-variation color-family guardrails, add mug identity guardrails, and add backpack harness/strap anti-duplication guardrails; updated `src/product_campaign_pipeline/review_batch.py` to detect low-detail gray background collapse, penalize and filter collapsed-background candidates during selection, disable synthetic color-anchor replacement for footwear, extend safe post-generation color repair to carried bags and footwear, tighten soft-product source-validity rejection under border-foreground contamination, and prefer `carried_by_hand` over `worn_on_body` when backpack evidence shows the harness/back-panel face; updated `scripts/build_final_benchmark_bundle.py` so bundle precedence is honored correctly and later bundles no longer overwrite earlier repaired rows; added regression coverage in `tests/composer/test_prompts.py`, `tests/review/test_review_batch.py`, and `tests/scripts/test_build_final_benchmark_bundle.py`
- Environment revision: unchanged vast.ai VM at `/workspace/product_campaign_pipeline` with local `FLUX.2-klein-9B` generation on the RTX 3090 Ti
- Metrics: targeted rerender generated `44` rows across `22` repaired ids; focused regression suite reached `206` passing tests; `python -m compileall src scripts` passed; the repaired benchmark candidate `outputs/final_benchmark_candidate_v2` now carries `22` ids from `final_benchmark_targeted_v2_final_v1` and excludes `27` invalid/held ids via the merged exclusion registry
- Qualitative notes: the key repair classes were gray-background collapse in apparel / footwear / bag rows, color-anchor contamination in footwear, unsafe carried-backpack support planning that produced duplicated strap systems, and a silent benchmark-merge bug that had reversed bundle precedence and could discard repaired rows without any visible failure in the merge script output. The targeted rerender also confirmed several previously flagged rows should stay excluded as invalid-source cases rather than being regenerated further (`bedding_quilt_01`, `kitchen_blender_04`, `bedding_comforter_02`, `bedding_comforter_03`, `pet_dog_bed_02`, `pet_dog_bed_03`).
- Follow-up: use `outputs/final_benchmark_candidate_v2/human_review_board.html` as the repaired final benchmark / acceptance checkpoint

## 2026-04-17 Final Benchmark Targeted Root-Cause Repair V3

- Date: 2026-04-17T09:30:00Z
- Owner: Codex
- Goal: fix the newly reviewed crop/export and stale benchmark issues at the root level, then rebuild the final benchmark candidate until both independent reviewers and local verification find no blocker remaining
- Data snapshot: targeted manifests `data/final_benchmark_targeted_v3/review_seed_manifest.json` and `data/final_benchmark_targeted_v4/review_seed_manifest.json`; rerun localization/upstream bundles `outputs/final_benchmark_targeted_v3_localization_v2`, `outputs/final_benchmark_targeted_v3_upstream_v2`, and `outputs/final_benchmark_targeted_v4_upstream_v1`; targeted generation bundles `outputs/final_benchmark_targeted_v3_final_v1`, `outputs/final_benchmark_targeted_v3_foldingchair_fix_v1`, and `outputs/final_benchmark_targeted_v5_dressfix_v5`; rebuilt merged benchmark bundle `outputs/final_benchmark_candidate_v5`
- Code revision:
  - `src/product_campaign_pipeline/review_batch.py`: passed the original localization crop through evidence extraction; when the system prefers the broader localization crop, it now drops mask-conditioned cutout/silhouette refs instead of mixing a trusted crop with an untrusted cutout; relaxed dark-olive bedding color recovery; suppressed dark reflective neutral overrides on warm wood / earthy rigid products; added a final dress-layering override so a clean dress candidate can beat a jeans-under-dress candidate; added support for extra exclusion reports in the benchmark builder flow
  - `src/product_campaign_pipeline/localization/artifacts.py`: added apparel-specific top-margin expansion so neckline-bearing crops are less likely to lose the top of the garment
  - `src/product_campaign_pipeline/composer/prompts.py`: removed denim / relaxed-separates persona-scene accents for dress hero rows so business-prior no longer contradicts the dress guardrail itself
  - `scripts/build_final_benchmark_bundle.py`: added `--exclusion-report` so newly invalidated rows from upstream-only screens can be removed from the merged benchmark without hand edits
  - tests: added and updated regression coverage in `tests/review/test_review_batch.py`, `tests/composer/test_prompts.py`, and `tests/scripts/test_build_final_benchmark_bundle.py`
- Environment revision: unchanged vast.ai VM at `/workspace/product_campaign_pipeline`, local `FLUX.2-klein-9B` generation on the RTX 3090 Ti
- Metrics: focused regression suite reached `214` passing tests; `python -m compileall src scripts` passed after each repair step; final repaired benchmark bundle `outputs/final_benchmark_candidate_v5` contains `80` ids / `160` rows, `0` automated category warnings, `0` automated semantic warnings, `0` automated evidence warnings, and excludes `23` invalid/held ids
- Qualitative notes:
  - The crop failures split into two real upstream causes: the localized box itself could be too tight, and the evidence-export layer could then over-trust a bad refined mask even after a good box existed. The fix was not “make the crop bigger everywhere.” It was: pass the original localization crop through the pipeline, prefer it when the refined evidence crop materially under-covers the product, and suppress cutout/silhouette conditioning when that refined mask is already suspect.
  - `kitchen_blender_02` and `bedding_quilt_03` were confirmed as invalid-source cases under the current upstream validity rules rather than generation bugs. `bedding_quilt_03` had survived the older merged benchmark as stale debt; it is now excluded via the new extra-exclusion-report path.
  - `control_dasein_handbag_03`, `dasein_handbag_03`, `disney_tote_02`, `bedding_comforter_01`, `furniture_office_chair_02`, and `furniture_folding_chair_03` were all rerun successfully under the repaired upstream path.
  - The last blocker was `apparel_dress_02 business_prior`: the business-prior prompt itself was reintroducing denim-styled wardrobe cues, which contradicted the dress-specific guardrail. Removing that contradiction and preferring non-layered dress candidates closed the loop.
- Follow-up: use `outputs/final_benchmark_candidate_v5/human_review_board.html` as the repaired final benchmark / acceptance checkpoint

## 2026-04-17 Final Wrap-Up Planning For Runpod Business-Prior Deployment

- Date: 2026-04-17T10:30:00Z
- Owner: Codex
- Goal: convert the completed benchmark / acceptance state into a concrete wrap-up plan that also prepares the `business_prior` line for later Runpod deployment behind a web interface
- Data snapshot: accepted benchmark bundle `outputs/final_benchmark_candidate_v5`; runtime docs `README.md` and `docs/context/ENVIRONMENT.md`; current entrypoints `src/product_campaign_pipeline/cli.py`, `scripts/run_human_review_generation.py`, and `scripts/run_human_review_generation_sequential.py`
- Code revision: added `docs/context/WRAP_UP_PLAN.md` and linked it from `README.md`
- Environment revision: unchanged vast.ai VM at `/workspace/product_campaign_pipeline`
- Qualitative notes: repo inspection confirmed that the project is benchmark-ready but not service-ready. The current runtime surface is a CLI plus offline review scripts; there is no API/service boundary, no Runpod container spec, no healthcheck/warmup contract, and no production single-request `business_prior` entrypoint. The wrap-up plan now explicitly includes extracting that production boundary, documenting the request/response contract, packaging a Runpod runtime, defining the future web-interface backend contract, and adding production smoke tests and operational docs.
- Follow-up: treat `docs/context/WRAP_UP_PLAN.md` as the current wrap-up checklist and include the Runpod deployment-prep items in the remaining closeout work instead of treating deployment as a separate undefined future phase

## 2026-04-17 Production Business-Prior Runtime Contract And Entry Point

- Date: 2026-04-17T11:00:00Z
- Owner: Codex
- Goal: execute the first concrete Runpod-prep step by extracting a production single-request `business_prior` inference surface and documenting its backend contract
- Data snapshot: accepted benchmark baseline `outputs/final_benchmark_candidate_v5`; runtime package surface in `src/product_campaign_pipeline`; wrap-up plan at `docs/context/WRAP_UP_PLAN.md`
- Code revision:
  - added `src/product_campaign_pipeline/production.py` with `BusinessPriorInferenceRequest`, `BusinessPriorInferenceResult`, and `run_business_prior_inference(...)`
  - extended `src/product_campaign_pipeline/cli.py` with `pcp generate business-prior-photo`
  - added backend contract doc `docs/context/BUSINESS_PRIOR_RUNTIME_CONTRACT.md`
  - updated `README.md` and `docs/context/WRAP_UP_PLAN.md` to reflect the new production boundary
  - added CLI regression coverage in `tests/evaluation/test_cli.py`
- Environment revision: unchanged vast.ai VM at `/workspace/product_campaign_pipeline`
- Metrics: targeted CLI regression suite passed with `3` tests; `python -m compileall src scripts` passed
- Qualitative notes: the deployment surface is now no longer just an idea in the wrap-up plan. The project has a concrete single-request inference boundary that localizes one uploaded image, applies upstream validity checks, runs the validated `business_prior` generation path, and returns a structured result suitable for a future Runpod handler or web backend. What still remains is service packaging: container image, warmup/healthcheck behavior, environment-driven config, and smoke tests at the deployment layer.
- Follow-up: next wrap-up step is Runpod packaging and operationalization of this new single-request runtime rather than further benchmark/model work

## 2026-04-17 Runpod Backend Packaging And Smoke-Test Prep

- Date: 2026-04-17T11:35:00Z
- Owner: Codex
- Goal: convert the new single-request `business_prior` runtime into a backend that can be hosted directly on Runpod, with service packaging, warmup/health behavior, smoke coverage, and an operational runbook
- Data snapshot: production runtime surface `src/product_campaign_pipeline/production.py`; accepted benchmark baseline `outputs/final_benchmark_candidate_v5`; wrap-up plan `docs/context/WRAP_UP_PLAN.md`
- Code revision:
  - added `src/product_campaign_pipeline/service.py` with FastAPI endpoints `GET /healthz`, `POST /warmup`, and `POST /generate/business-prior`
  - added `scripts/run_business_prior_service.py` as the stable backend startup command
  - added `scripts/run_business_prior_service_smoke.py` for live-service smoke validation
  - added `Dockerfile.runpod` and `.dockerignore`
  - added `docs/context/RUNPOD_BACKEND_RUNBOOK.md`
  - added `service` optional dependency group in `pyproject.toml`
  - added explicit FLUX client warmup support in `src/product_campaign_pipeline/flux/client.py`
  - added service regression coverage in `tests/service/test_service.py`
  - updated `README.md` and `docs/context/WRAP_UP_PLAN.md` to reflect the new deployment state
- Environment revision: installed the new `service` extra into the project venv on the vast.ai VM for validation
- Metrics: `pytest tests/evaluation/test_cli.py tests/service/test_service.py -q` passed with `5` tests; `python -m compileall src scripts` passed
- Qualitative notes: the backend is now packageable instead of just callable. The service wrapper preserves the validated single-request `business_prior` runtime, exposes health and warmup control for orchestration, accepts direct image uploads, and returns the structured runtime result defined by the production contract. What still remains is not repo prep but live deployment execution and tuning on the eventual Runpod target.
- Follow-up: the next wrap-up step is to deploy and validate this backend on the actual Runpod environment rather than adding more local packaging layers

## 2026-04-17 Final Handoff Docs And Save-Set Planning

- Date: 2026-04-17T12:05:00Z
- Owner: Codex
- Goal: write explicit handoff docs for finishing Runpod deployment and for future experiments, then define the minimal and recommended save set before retreating from this VM
- Data snapshot: accepted benchmark baseline `outputs/final_benchmark_candidate_v5`; deployment assets `Dockerfile.runpod`, `src/product_campaign_pipeline/service.py`, and `scripts/run_business_prior_service.py`; maintained context docs in `docs/context`
- Code revision:
  - added `docs/context/RUNPOD_DEPLOYMENT_FINISH_GUIDE.md`
  - added `docs/context/FUTURE_OPTIMIZATION_AND_EXPERIMENT_GUIDE.md`
  - updated `README.md` to link the new handoff docs
- Environment revision: measured current artifact sizes on the vast.ai VM to support the retreat plan; repo is ~`19G`, raw dataset `/workspace/data` is ~`112G`, Hugging Face cache `/workspace/.hf_home` is ~`37G`, and Codex global state `/root/.codex` is ~`2.1G`
- Qualitative notes: the repo now contains explicit instructions for both near-term deployment completion and later research continuation. The remaining task is operational: preserve the right artifacts when leaving this VM so the next environment can either finish deployment quickly or continue experiments without reconstructing the accepted baseline from scratch.
- Follow-up: preserve the code, context docs, retrieval assets, accepted benchmark artifacts, and any required caches according to the save set defined at handoff time
