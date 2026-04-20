# Review Log

Every deliverable should be reviewed twice: once by Codex and once independently by Claude.

## Template

- Date:
- Deliverable:
- Reviewer:
- Findings:
- Decision:
- Follow-up owner:

## 2026-04-09 Review Batch V1

- Date: 2026-04-09T10:41:55Z
- Deliverable: first side-by-side human review board at `outputs/human_review_batch_v1/human_review_board.html`
- Reviewer: Codex
- Findings: localization quality was sufficient to proceed without filtering bad cases; bottle outputs generally preserved silhouette and major branding but occasionally introduced non-source accessories; pillow outputs were the most stable; business-prior retrieval still leaks cross-category cues, especially for bottles and close-crop tote images; one clear failure mode is category drift from patterned tote crop to patterned shirt output
- Decision: proceed to human checkpoint with full artifact set and failure notes preserved
- Follow-up owner: Codex

## 2026-04-09 Review Batch V1

- Date: 2026-04-09T10:41:55Z
- Deliverable: first side-by-side human review board at `outputs/human_review_batch_v1/human_review_board.html`
- Reviewer: Claude
- Findings: flagged `disney_tote_03` as a critical product-type drift case where the business-prior line rendered a shirt and the baseline line drifted toward a pillow-like soft good; flagged `dasein_handbag_01` as a logo and identity distortion case; flagged CamelBak retrieval neighbors as off-category and therefore untrustworthy despite visually acceptable bottle renders; noted Rizzy pillow and the clearer tote views as strong
- Decision: do not proceed as-is; fix the blocking drift cases and regenerate the affected seeds before the next review
- Follow-up owner: Codex

## 2026-04-09 Review Batch V2

- Date: 2026-04-09T11:16:00Z
- Deliverable: updated full human review board at `outputs/human_review_batch_v2/human_review_board.html`
- Reviewer: Codex
- Findings: the tote close-up drift case is fixed and `disney_tote_03` now remains a tote bag in both lines; CamelBak business-prior now uses category fallback instead of polluted neighbors and remains on-category; the new board includes category-consistency metadata and no rows were flagged in this run; Dasein `01` still looks packaging-like, but that behavior is consistent with the source reference rather than a product-type mutation
- Decision: proceed to the next human checkpoint
- Follow-up owner: Codex

## 2026-04-09 Review Batch V2

- Date: 2026-04-09T11:16:00Z
- Deliverable: updated full human review board at `outputs/human_review_batch_v2/human_review_board.html`
- Reviewer: Claude
- Findings: confirmed that the prior tote drift is fixed and the weak-shape tote guard is working; flagged the current category-consistency classifier as non-discriminative and therefore advisory rather than reliable as a gate; noted that all CamelBak business-prior rows now use category fallback because the retrieval index lacks drinkware support; noted that bag retrieval is still generic and repetitive across Disney and Dasein
- Decision: conditional proceed; continue with the batch but do not rely on the category-consistency layer as a blocking automation gate yet
- Follow-up owner: Codex

## 2026-04-09 Review Batch V4 Semantics

- Date: 2026-04-09T19:36:41Z
- Deliverable: final semantic review board at `outputs/human_review_batch_v4_semantics/human_review_board.html`
- Reviewer: Codex
- Findings: coherent scene/support planning removed the prior upright-pillow semantic failure; the new semantic plausibility layer reports `0` flagged rows in the final run; Dasein false mounted/hanging flags were resolved by fixing wallet-token parsing in affordance inference; tote, pillow, bottle, and bag cases now all remain category-consistent and semantically plausible under the current heuristic scorer
- Decision: proceed to the next human checkpoint
- Follow-up owner: Codex

## 2026-04-09 Review Batch V4 Semantics

- Date: 2026-04-09T19:36:41Z
- Deliverable: final semantic review board at `outputs/human_review_batch_v4_semantics/human_review_board.html`
- Reviewer: Claude
- Findings: confirmed that the prior pillow-support issue is fixed and the batch is semantically coherent overall; flagged `dasein_handbag_01` as an identity hallucination case with fabricated brand lockup; flagged `disney_tote_03` as the weakest Disney seed due to weak shape evidence and borderline support margin; noted that the current semantic and category scorers have limited margin separation and should still be treated as advisory rather than decisive
- Decision: conditional proceed with holds on `dasein_handbag_01` and `disney_tote_03`
- Follow-up owner: Codex

## 2026-04-09 Dasein Evidence-Constrained Reinvention V3

- Date: 2026-04-09T23:59:00Z
- Deliverable: targeted repair board at `outputs/human_review_dasein02_evidence_reinvent_v3/human_review_board.html`
- Reviewer: Codex
- Findings: the old side-view-preservation bias was removed and replaced with observed-evidence modeling, multi-candidate reinvention, and contradiction-aware reranking; the selected baseline output now reveals a cleaner front/three-quarter campaign presentation while keeping the blue floral surface evidence compatible with the input; the selected business-prior output preserves the same core evidence and produces the strongest final composition of the repair pass; early evidence extraction was initially skewed toward dark neutrals, but the final extractor now prioritizes the BLIP evidence caption and blue/floral cues correctly
- Decision: targeted fix accepted for `dasein_handbag_02`; use this evidence-constrained reinvention path for future hard partial-view cases rather than reverting to source-view preservation
- Follow-up owner: Codex

## 2026-04-09 Review Batch V5 Evidence Reinvention

- Date: 2026-04-09T23:29:26Z
- Deliverable: full portable board at `outputs/human_review_batch_v5_evidence_reinvention/human_review_board.html`
- Reviewer: Codex
- Findings: the evidence-constrained reinvention policy now applies across the full batch and removes the earlier need to hold `dasein_handbag_02`; both lines for `disney_tote_03` remain tote-compatible while allowing cleaner hero views; all 26 rows remain category-consistent and semantically plausible under the current advisory scorers; no rows are flagged by the current evidence-consistency threshold, but the weakest evidence rows are still concentrated in high-uncertainty partial-view cases, especially `dasein_handbag_01`, `dasein_handbag_03`, and `camelbak_bottle_03`; pillow rows remain the strongest combination of semantic and evidence scores; the board is now download-safe because all image references are relative
- Decision: proceed to a human checkpoint on the v5 board and treat `dasein_handbag_01` as the main remaining identity-risk case rather than a broad policy failure
- Follow-up owner: Codex

## 2026-04-10 Review Batch V6 Root Cause Fixes

- Date: 2026-04-10T01:24:00Z
- Deliverable: full portable board at `outputs/human_review_batch_v6_root_cause_fixes/human_review_board.html`
- Reviewer: Codex
- Findings: the new evidence graph, evidence-aware business-prior fallback, and stricter contradiction-aware selector materially improved the previously reported preservation failures across bottles, pillows, and Disney totes; all 26 rows remained category-consistent and semantically plausible, and the full-batch mean evidence score rose from `0.5881` in v5 to `0.6953` in v6; `disney_tote_01` business-prior now preserves full print coverage and falls back away from weak generic bag retrieval; `dasein_handbag_02` business-prior remains visually plausible without the prior transparent side-panel contradiction; the main remaining residual is `disney_tote_02`, where the print and body evidence are improved but the dark braided handle detail is still not extracted or enforced strongly enough
- Decision: proceed to a human checkpoint on the v6 board with one explicit residual hold on `disney_tote_02` handle fidelity if the reviewer still considers it blocking
- Follow-up owner: Codex

## 2026-04-10 Review Batch V32 Final Upstream Fixed

- Date: 2026-04-10T09:30:00Z
- Deliverable: focused 11-seed generated-image bundle at `outputs/human_review_batch_v32_final_upstream_fixed/human_review_board.html`
- Reviewer: Codex
- Findings: the bottle set no longer shows the earlier business-prior darkening or black-label drift; Disney tote rows preserve the product evidence while using casual, product-compatible human styling; pillow rows remain strong and no longer show the earlier false warm edging; the main residual is `dasein_handbag_03`, where both lines still invent a handle-like loop even though the upstream evidence and prompt say the wallet-sized bag has no visible handles
- Decision: do not send to final human review yet; fix the direct-grip interaction contract first
- Follow-up owner: Codex

## 2026-04-10 Review Batch V33 Final Bundle

- Date: 2026-04-10T09:44:00Z
- Deliverable: focused final bundle at `outputs/human_review_batch_v33_final_bundle/human_review_board.html`
- Reviewer: Codex
- Findings: the direct-grip prompt fix removed the remaining invented-loop failure on `dasein_handbag_03`; `dasein_handbag_01` also improved from suit-like styling to casual knitwear/denim without regressing bag identity; previously repaired bottle, Disney tote, and pillow rows stayed stable because the bundle carried them forward unchanged from v32; the combined final board is portable and contains all 22 line-level rows across the 11 focused seeds
- Decision: ready for final human review
- Follow-up owner: Codex

## 2026-04-10 Generalization Diverse V1 Final Bundle

- Date: 2026-04-10T17:35:00Z
- Deliverable: broad 15-seed diverse generated-image bundle at `outputs/generalization_diverse_v1_final_bundle/human_review_board.html`
- Reviewer: Codex
- Findings: the broader set confirms the upstream fixes generalize materially beyond the original bottle/bag/pillow tranche; close-crop shirts and dresses remain apparel instead of drifting into `cozy_home` or officewear; both backpack rows stay as worn utility backpacks with visible strap support instead of reverting to generic bag handling; the lamp rows stay upright and physically supported after the table-lamp structural rewrite; sampled outputs across bottles, apparel, footwear, backpacks, and the lamp look human-acceptable even where the current evidence scorer remains conservative; the one runtime failure in this run exposed a real infrastructure issue, and the new minimum-reference-size normalization in the FLUX client resolved it without requiring sample-specific handling
- Decision: ready for the next human review checkpoint on broad generalization
- Follow-up owner: Codex

## 2026-04-11 Generalization Diverse V4 Upstream Review

- Date: 2026-04-11T03:20:00Z
- Deliverable: upstream-only prompt/evidence board at `outputs/generalization_diverse_v4_upstream_review/human_review_board.html`
- Reviewer: Codex
- Findings: the business-prior line is no longer mostly a prose-level appendage to baseline; all `15` business-prior rows remained true `retrieval` rather than fallback; typed creative hints now drive environment, lighting, camera, and cast differences upstream; prompt overlap between baseline and business-prior dropped materially, and every selected background description remained unique across the diverse tranche; shared support relations still keep some rows in the same scene family, but the remaining similarity is now substantially narrower and structurally understandable
- Decision: proceed to a generated-image refresh for the `business_prior` line only, keeping the approved baseline bundle unchanged
- Follow-up owner: Codex

## 2026-04-11 Generalization Diverse V4 Final Bundle

- Date: 2026-04-11T04:15:00Z
- Deliverable: merged broad bundle at `outputs/generalization_diverse_v4_final_bundle/human_review_board.html`
- Reviewer: Codex
- Findings: the stronger business-prior planner/composer materially improved line differentiation, but the first merged image bundle was not yet review-ready because it exposed two selector-level regressions: `apparel_tshirt_01.business_prior` selected an anatomy-weak candidate, and `bag_rawlings_backpack_02.business_prior` accepted weak retrieval neighbors for a structurally specific backpack input; these were not planner-quality reasons to revert the broader refactor, but they were blocking issues for a human checkpoint
- Decision: hold the broad board; fix selector anatomy penalties and structured-subtype retrieval conflicts, then rerun only the affected business-prior rows
- Follow-up owner: Codex

## 2026-04-11 Generalization Diverse V5 Final Bundle

- Date: 2026-04-11T05:03:00Z
- Deliverable: broad merged bundle at `outputs/generalization_diverse_v5_final_bundle/human_review_board.html`
- Reviewer: Codex
- Findings: the anatomy selector fix removed the remaining `apparel_tshirt_01.business_prior` warning and switched the winner to a cleaner balanced candidate; the structured-subtype retrieval fix forced `bag_rawlings_backpack_02.business_prior` away from bad wallet/glasses neighbors and into a clean category fallback with no evidence warning; the final bundle now has `0` semantic warnings and only `1` remaining business-prior evidence warning on `apparel_longsleeve_top_02`, which matches a row the earlier human review had already considered good enough despite conservative automated scoring; the only remaining category warning is a baseline-only false positive on `control_dasein_handbag_03`
- Decision: ready for the next broad human review checkpoint
- Follow-up owner: Codex

## 2026-04-11 Generalization Diverse V6 Targeted Repair Bundle

- Date: 2026-04-11T09:12:00Z
- Deliverable: targeted repair bundle at `outputs/generalization_diverse_v6_targeted_final_bundle/human_review_board.html`
- Reviewer: Codex
- Findings: the apparel conditioning fix removed the source-face leakage path for `apparel_tshirt_01`, and the regenerated business-prior row now shows a clearly different model instead of reproducing the input woman; the footwear repair separated full-shape conditioning from surface-evidence inference, so `footwear_easyspirit_02` now reads as a neutral shoe surface rather than inheriting the sock print, and `footwear_easyspirit_01` was downgraded from an all-over printed reading to a mostly neutral shoe with limited accent zones; the lamp repair added structured surface-relief evidence, and the regenerated business-prior lamp now preserves the visible shade ridges instead of smoothing them away
- Decision: ready for human review on the targeted repair bundle
- Follow-up owner: Codex

## 2026-04-11 Generalization Diverse V6 Approved Bundle

- Date: 2026-04-11T09:36:00Z
- Deliverable: promoted broad approved bundle at `outputs/generalization_diverse_v6_approved_bundle/human_review_board.html`
- Reviewer: Human + Codex
- Findings: the user approved the targeted repair bundle overall and accepted the remaining sneaker-baseline issue as non-blocking for now; specifically, `footwear_easyspirit_01` and `footwear_easyspirit_02` still render the baseline white sneakers too gray, but all other repaired rows were accepted; the merged broad approved bundle is portable and its generated report contains `0` semantic warnings, `0` category warnings, and `0` evidence warnings
- Decision: approved for promotion; move on to the next broader evaluation tranche and keep the sneaker-baseline color drift as accepted open debt
- Follow-up owner: Codex

## 2026-04-11 Generalization Diverse V2 New Categories Bundle

- Date: 2026-04-11T10:46:00Z
- Deliverable: broader new-category generated-image bundle at `outputs/generalization_diverse_v2_new_categories_final_bundle/human_review_board.html`
- Reviewer: Codex
- Findings: the upstream-first gate paid off on this tranche because the first pass exposed real taxonomy and support-planning gaps rather than generator-specific defects; after those upstream fixes, the final images for the new categories are visually coherent and preserve the right object class and support behavior: blenders remain countertop appliances, comforters remain broad bed-spread textiles, chairs remain self-standing furniture, and pet beds remain low soft products rather than generic pillows or apparel; the final bundle is portable and its report contains `0` semantic warnings, `0` category warnings, and `0` evidence warnings
- Decision: ready for the next human review checkpoint
- Follow-up owner: Codex

## 2026-04-12 Generalization Diverse V2 Targeted Rootfix V30 Curated Bundle

- Date: 2026-04-12T00:55:00Z
- Deliverable: rebuilt targeted repair checkpoint at `outputs/generalization_diverse_v2_targeted_rootfix_v30_curated_final_bundle/human_review_board.html`
- Reviewer: Codex
- Findings: the stale warning surface from `v29` was real metric debt, not image debt; after rebuilding under the current evaluator, the accepted wallet row no longer trips compact-handheld category or semantic false positives, the comforter rows no longer get penalized for false localized-pattern readings, and the low-profile pet-bed rows no longer get penalized for shadow-driven color/value drift; the rebuilt bundle is portable and its report contains `0` category warnings, `0` semantic warnings, and `0` evidence warnings
- Decision: ready for independent review
- Follow-up owner: Codex

## 2026-04-12 Generalization Diverse V2 Targeted Rootfix V30 Independent Review

- Date: 2026-04-12T01:05:00Z
- Deliverable: same `v30` targeted repair checkpoint and the current `src/product_campaign_pipeline/review_batch.py` implementation
- Reviewer: Euler + Galileo
- Findings: Galileo returned `None`, meaning no materially valid remaining blocker was found after inspecting the repaired bundle and implementation; Euler raised one medium-risk note that the compact wallet baseline is still less product-dominant than the stronger business-prior variant, but after artifact verification this was judged non-blocking for the current checkpoint because the wallet remains clearly visible, anatomically coherent, and product-led enough to satisfy the compact-handheld requirements without reopening the earlier false-positive gating problem
- Decision: no remaining validated blocker in the rebuilt `v30` targeted repair checkpoint
- Follow-up owner: Codex

## 2026-04-12 Generalization Diverse V2 Targeted Rootfix V30 Approval And Board Repair

- Date: 2026-04-12T01:22:00Z
- Deliverable: approved targeted repair checkpoint at `outputs/generalization_diverse_v2_targeted_rootfix_v30_curated_final_bundle/human_review_board.html`
- Reviewer: Human + Codex
- Findings: the user approved the rendered outputs and reported that the only remaining issue was the portable HTML board failing to load images after download; verification confirmed the assets were present and the defect was limited to the renderer blanking already-sanitized relative `board_assets/...` paths; after the renderer fix and rebuild, the board now loads relative image refs correctly with no empty `src` entries and no absolute filesystem paths
- Decision: approved; board portability issue closed without reopening output quality issues
- Follow-up owner: Codex

## 2026-04-12 generalization_diverse_v3_targeted_repair_v5_final_bundle
- Reviewer: Codex
  Verdict: no blocker-level issues after direct inspection of the merged final bundle.
- Reviewer: Euler
  Verdict: no blocker-level issues.
- Reviewer: Galileo
  Verdict: no blocker-level issues.

## 2026-04-13 Generalization Diverse V4 Targeted Repair V8 Curated Bundle

- Date: 2026-04-13T00:25:00Z
- Deliverable: curated four-row repair bundle at `outputs/generalization_diverse_v4_targeted_repair_v8_curated_final_bundle/human_review_board.html`
- Reviewer: Codex
- Findings: direct bundle verification confirmed the curated board is loadable and self-contained, the paired report is normalized to the curated bundle paths, and the repaired rows no longer show the earlier blocker-level failures in toaster color/body-fragment leakage, office-chair lower-body artifacting, backpack product drift, or pet-bed ottoman-like base invention
- Decision: clean at blocker level; send to independent review before promotion
- Follow-up owner: Codex

## 2026-04-13 Generalization Diverse V4 Targeted Repair V8 Independent Review

- Date: 2026-04-13T00:40:00Z
- Deliverable: same `v8` targeted repair bundle and its paired `generation_report.json`
- Reviewer: Euler + Galileo
- Findings: Euler found no blocker-level issues after direct artifact inspection and stated the report only carried weak alignment notes that did not survive human-level review in the final images; Galileo found no blocker-level issues after checking the final images and paired report; both findings were verified locally against the bundle structure, the final PNGs, and the normalized report
- Decision: no remaining validated blocker in the `v8` targeted repair bundle
- Follow-up owner: Codex

## 2026-04-16 Generalization Diverse V5 Curated V2 Independent Review

- Date: 2026-04-16T04:08:00Z
- Deliverable: curated migrated-VM bundle at `outputs/generalization_diverse_v5_curated_v2/human_review_board.html`
- Reviewer: Codex + Lagrange + Hilbert
- Findings: Hilbert initially flagged three rows in the earlier migrated `v5` repair loop, but after direct verification only two findings held: the single-model failures in `bag_backpack_cooler_04 business_prior` and `footwear_easyspirit_04 business_prior`; both were fixed and replaced in the curated `v2` bundle. Hilbert also raised `bag_rawlings_backpack_04 baseline` as a possible remaining blocker, but direct source-vs-output verification did not support that as a real contradiction: the generated backpack remains consistent with the observed open baseball-backpack / harness-face evidence, so that finding was rejected as non-blocking. Lagrange's final short review on `v2` reported no blocker-level real visual issues remaining. Codex agreed after direct image inspection of the repaired rows and the previously suspect controls.
- Decision: no remaining validated blocker-level visual issue in `outputs/generalization_diverse_v5_curated_v2`
- Follow-up owner: Codex

## 2026-04-16 Generalization Diverse V5 Targeted Rootfix V10 Independent Review

- Date: 2026-04-16T07:18:00Z
- Deliverable: curated six-row targeted repair bundle at `outputs/generalization_diverse_v5_targeted_rootfix_v10_curated/human_review_board.html`
- Reviewer: Codex + Hilbert + Lagrange
- Findings: Hilbert initially flagged `bedding_quilt_04 baseline` in the earlier `v9` bundle as still reading like a decor throw on furniture rather than bedding. Codex verified that this was a valid blocker path because the prompt/runtime still allowed furniture-like furnished-interior locations for bedding and the generic `clarity` clause still mentioned a human subject for product-only rows. After the prompt and semantic-gating fixes, Codex rerendered only `bedding_quilt_04 baseline` and rechecked it with the local vision backbone: the refreshed image now scores closer to “quilt on a bed” than “throw on a bench,” the generated caption reads as a bed scene, and semantic evaluation reports `people_out_of_frame_required = true` with `person_presence_flag = false`. Hilbert then cleared the refreshed `v10` bundle with no blocker-level remaining issue. Lagrange also reported no blocker-level issue in the same `v10` bundle. Codex verified both findings against the rebuilt board, the repaired row reports, and the final staged images.
- Decision: no remaining validated blocker-level issue in `outputs/generalization_diverse_v5_targeted_rootfix_v10_curated`
- Follow-up owner: Codex

## 2026-04-16 Generalization Diverse V7 Curated V5 Independent Review

- Date: 2026-04-16T11:16:00Z
- Deliverable: curated broad tranche bundle at `outputs/generalization_diverse_v7_curated_v5/human_review_board.html`
- Reviewer: Codex + Hilbert + Lagrange
- Findings: Hilbert initially found the earlier `bag_backpack_cooler_05` blocker in `v7_final_bundle_v1`, which Codex verified as real because the source clearly showed an open cooler backpack with visible cans while both lines had collapsed toward generic daypacks. Lagrange later found a second blocker in the first curated board: `apparel_tshirt_05 business_prior` had drifted from a white short-sleeve T-shirt into a sweater-like beige long-sleeve top. Codex verified that finding against the source, crop, upstream evidence, and final image. After the subtype/context repair for the cooler backpack and the uncertain-apparel prompt/evidence repair for the T-shirt, Codex rebuilt `outputs/generalization_diverse_v7_curated_v5` from sanitized raw report rows and directly verified the corrected cooler-backpack and T-shirt outputs. Hilbert then reviewed the final `v7_curated_v5` bundle and found no true blocker-level issues remaining, classifying the residual report warnings as conservative evaluator residue. Lagrange independently re-reviewed the same final bundle and also found no true blocker-level issue after the T-shirt repair.
- Decision: no remaining validated blocker-level issue in `outputs/generalization_diverse_v7_curated_v5`
- Follow-up owner: Codex

## 2026-04-16 Generalization Diverse V7 Curated V7 Independent Review

- Date: 2026-04-16T13:42:00Z
- Deliverable: repaired broad tranche bundle at `outputs/generalization_diverse_v7_curated_v7/human_review_board.html`
- Reviewer: Codex + Hilbert + Lagrange
- Findings: this review pass started from direct user feedback on `v7_curated_v5`. Codex verified the bottle-band triangle artifact on `control_camelbak_bottle_02 baseline` and the gray-sleeve drift on `apparel_dress_05 business_prior` as real upstream-conditioned failures, fixed them at the evidence/prompt level, and rerendered those two valid rows. Codex also verified that `furniture_folding_chair_05`, `kitchen_coffee_maker_05`, `drinkware_mug_05`, and `bedding_quilt_05` should be treated as invalid-source cases rather than further generation targets because the localized crops do not preserve enough reliable product evidence for fair review. After the row repairs, the first `v7_curated_v6` board merge was found to have a portability defect: most carried-forward rows lost their staged source/crop refs and rendered with empty image `src` values. Codex rejected that bundle, rebuilt `v7_curated_v7` from original source reports with normalized asset paths, and directly verified that the repaired board has no empty refs, no missing refs, and no absolute filesystem paths. Hilbert then reviewed the rebuilt `v7_curated_v7` bundle and returned `no blocker`. Lagrange independently reviewed the same rebuilt bundle and also returned `no blocker`.
- Decision: no remaining validated blocker-level issue in `outputs/generalization_diverse_v7_curated_v7`; invalid-source exclusions are intentional and not treated as open blockers
- Follow-up owner: Codex

## 2026-04-16 Generalization Diverse V7 Curated V7 Human Review

- Date: 2026-04-16T13:50:00Z
- Deliverable: approved broad tranche bundle at `outputs/generalization_diverse_v7_curated_v7/human_review_board.html`
- Reviewer: Human
- Findings: the user reviewed the rebuilt portable board and judged the visible rows “good enough.” No additional visible blocker was called out, and the excluded invalid-source rows remained intentionally out of scope for this checkpoint.
- Decision: approved
- Follow-up owner: Codex

## 2026-04-16 Generalization Diverse V8 Final Bundle Review And Curation

- Date: 2026-04-16T16:18:00Z
- Deliverable: broad tranche generation bundle at `outputs/generalization_diverse_v8_final_bundle_v1`, refreshed upstream validity passes `outputs/generalization_diverse_v8_upstream_review_v4` and `v5`, targeted chair rerenders `outputs/generalization_diverse_v8_targeted_folding_chair_v1` and `v2`, and curated broad checkpoint `outputs/generalization_diverse_v8_curated_v2/human_review_board.html`
- Reviewer: Codex + Hilbert + Lagrange
- Findings: the first independent pass on `v8_final_bundle_v1` surfaced a real upstream validity miss on `kitchen_coffee_maker_06`: Hilbert flagged both final outputs as mug/carafe-like rather than coffee-maker-like, and Codex verified that the source crop caption itself was `a cup of coffee` while the validity gate still marked the row valid. That finding held up against the code and was fixed at the source-validity layer, not by rerendering. Hilbert also flagged `pet_dog_bed_06` for identity drift into a boxed/bolstered bed with unsupported text/side-panel treatment; Codex verified that the source crop caption still saw an animal fragment rather than the bed, so this also became an upstream invalid-source exclusion after the new soft-goods validity rule. Lagrange independently flagged `furniture_folding_chair_06` as a blocker-level ghosted / semi-transparent chair artifact. Codex attempted two narrow rerender passes under stronger chair-specific guardrails and a clarity-preferring mode policy. Hilbert cleared the rerendered chair row both times, but Lagrange continued to flag it. A local CLIP-style comparison also leaned slightly toward the “ghosted / duplicated chair” prompts over the “normal opaque chair” prompts, so Codex did not promote the chair into the broad checkpoint. Instead it was held out in `reports/held_rows.json` while the rest of the broad tranche was curated into `outputs/generalization_diverse_v8_curated_v2`. The final curated board excludes the five invalid-source rows and the held folding-chair row.
- Decision: broad `v8` checkpoint is ready for human review as `outputs/generalization_diverse_v8_curated_v2`; `furniture_folding_chair_06` remains an unresolved held row and is intentionally not part of the checkpoint
- Follow-up owner: Codex

## 2026-04-17 Generalization Diverse V8 Human Review

- Date: 2026-04-17T00:00:00Z
- Deliverable: approved broad tranche bundle at `outputs/generalization_diverse_v8_curated_v2/human_review_board.html`
- Reviewer: Human
- Findings: the user reviewed the visible rows and judged them “good enough.” No additional visible blocker was called out. The invalid-source exclusions remained intentionally out of scope for the checkpoint, and `furniture_folding_chair_06` remained on the separate held-row track rather than blocking approval of the broad tranche.
- Decision: approved
- Follow-up owner: Codex

## 2026-04-17 Generalization Diverse V9 Remaining-Source Screen Review

- Date: 2026-04-17T02:30:00Z
- Deliverable: sampled upstream checkpoint `outputs/generalization_diverse_v9_upstream_review_v1/human_review_board.html` and exhaustive remaining-source upstream screen `outputs/generalization_diverse_v9_source_screen_upstream_v1/human_review_board.html`
- Reviewer: Codex
- Findings: the sampled `v9` tranche already showed a weak source-validity rate, so Codex escalated to a full remaining-source screen instead of pushing more generation. The exhaustive screen covered every unused customer-review photo still cached locally. Only `6` product ids remained valid upstream: `screen_apparel_tshirt_idx3`, `screen_apparel_dress_idx3`, `screen_drinkware_mug_idx3`, `screen_furniture_folding_chair_idx4`, `screen_kitchen_coffee_maker_idx2`, and `screen_kitchen_toaster_idx2`. The other `11` remaining unused photos were correctly rejected as invalid-source cases due to border-text overlays, border-foreground intrusions, incomplete portable-product global shape, multipart-appliance structure loss, or localized crop visual-type conflicts. This means another broad tranche from the remaining local source pool would be dominated by source failures rather than new model generalization signal.
- Decision: do not build another broad generated-image tranche from the remaining local source pool; pivot to the final benchmark / acceptance phase instead
- Follow-up owner: Codex

## 2026-04-17 Final Benchmark Candidate Merge Review

- Date: 2026-04-17T02:45:00Z
- Deliverable: consolidated benchmark bundle at `outputs/final_benchmark_candidate_v1/human_review_board.html`, with `reports/benchmark_summary.json`, `reports/generation_report.json`, and `reports/excluded_or_held_sources.json`
- Reviewer: Codex + Hilbert
- Findings: Codex verified the merge mechanically rather than by hand curation: later approved bundles override earlier ones at the row-id level, and targeted approved bundles override stale broad rows where applicable. The assembled bundle contains `88` approved ids / `176` rows. Structural verification passed: `352` image refs, `0` empty refs, `0` missing refs, `0` absolute `/home/...` paths, and `0` absolute `/workspace/...` paths. Hilbert independently reviewed the consolidated board and summary for merge-level issues only and returned `no blocker`. Lagrange was requested to do the same merge-level review but did not return a concrete finding within the bounded wait window, so no additional blocker was validated from that path.
- Decision: benchmark candidate bundle is ready for the final human review checkpoint
- Follow-up owner: Codex

## 2026-04-17 Final Benchmark Targeted Root-Cause Repair V2 Review

- Date: 2026-04-17T06:05:00Z
- Deliverable: repaired targeted rerender `outputs/final_benchmark_targeted_v2_final_v1/human_review_board.html` and repaired consolidated benchmark board `outputs/final_benchmark_candidate_v2/human_review_board.html`
- Reviewer: Codex + Hilbert + Lagrange
- Findings: this pass began from direct human review feedback on `final_benchmark_candidate_v1`. Codex grouped the visible failures into four root causes: gray-background collapse, contaminated color anchors / color-family drift, unsafe backpack back-view support planning, and stale invalid-source rows surviving into the benchmark board. Codex implemented those fixes upstream, rerendered the affected valid-source ids, and then discovered a separate benchmark assembly bug: `scripts/build_final_benchmark_bundle.py` was overwriting early-precedence repaired rows with later stale rows. Codex fixed that precedence bug, added a regression test, and rebuilt `outputs/final_benchmark_candidate_v2` so the targeted rerender actually takes effect. Structural verification on the rebuilt board passed: `164` generated image refs, `0` empty refs, `0` absolute filesystem paths, and `22` ids now resolved from `final_benchmark_targeted_v2_final_v1`. Hilbert independently reviewed the repaired benchmark board for blocker-level remaining visual issues and returned `no blocker`. Lagrange independently reviewed the same repaired board and also returned `no blocker` after an interrupted bounded wait. No validated blocker-level issue remained after direct local verification.
- Decision: no remaining validated blocker-level issue in `outputs/final_benchmark_candidate_v2`; this becomes the repaired final benchmark / acceptance checkpoint
- Follow-up owner: Codex

## 2026-04-17 Final Benchmark Targeted Root-Cause Repair V3 Review

- Date: 2026-04-17T09:45:00Z
- Deliverable: targeted rerenders `outputs/final_benchmark_targeted_v3_final_v1/human_review_board.html`, `outputs/final_benchmark_targeted_v3_foldingchair_fix_v1/human_review_board.html`, `outputs/final_benchmark_targeted_v5_dressfix_v5/human_review_board.html`, and rebuilt merged benchmark board `outputs/final_benchmark_candidate_v5/human_review_board.html`
- Reviewer: Codex + Hilbert + Lagrange
- Findings:
  - This pass began from direct human review feedback on the prior benchmark merge. Codex traced the crop complaints to two different layers: localization-box sizing and evidence-export over-trimming. The fix was to let the pipeline fall back to the broader localization crop when the refined evidence crop materially under-covered the product, while also dropping cutout/silhouette conditioning in that case so the generator would not see a contradictory bad cutout alongside a good crop.
  - After the first targeted rerender, Hilbert surfaced one real blocker outside the explicitly flagged slice: `bedding_quilt_03 baseline` had drifted into throw-blanket sofa use. Codex reran current upstream on that row and verified it is now invalid-source (`localized_crop_visual_type_conflict` plus border-foreground intrusion). That issue was fixed at the benchmark-builder layer by teaching `scripts/build_final_benchmark_bundle.py` to accept extra exclusion reports so stale invalidated rows can be removed from the merged benchmark instead of surviving forever through older approved bundles.
  - Hilbert then surfaced the final real blocker: `apparel_dress_02 business_prior` still showed jeans under the dress. Codex verified that the business-prior prompt itself was contradicting the dress guardrail by injecting denim / relaxed-separates styling language through the refined-neutral fashion-lifestyle persona accents. Codex removed those cues for dress hero rows, strengthened the dress-layering evaluation prompts, and added a final dress-selection override so a clean dress candidate can beat a layered-dress candidate when one exists.
  - Final independent review on `outputs/final_benchmark_candidate_v5` returned `no blocker remains` from both Hilbert and Lagrange after direct local verification of the repaired rows and the merged board. `bedding_quilt_03` is no longer on the visible review surface.
- Decision: no remaining validated blocker-level issue in `outputs/final_benchmark_candidate_v5`; this becomes the current repaired final benchmark / acceptance checkpoint
- Follow-up owner: Codex
