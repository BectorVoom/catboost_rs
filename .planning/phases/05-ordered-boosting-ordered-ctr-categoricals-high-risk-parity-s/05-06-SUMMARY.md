---
phase: 05-ordered-boosting-ordered-ctr-categoricals-high-risk-parity-s
plan: 06
subsystem: cb-train + cb-model (feature combinations / tensor CTRs — SimpleCtrs/CombinationCtrs under max_ctr_complexity, ORD-05 / D-05; the final rung of the additive categorical ladder)
tags: [tensor-ctr, combination-ctr, projection, multihash, calchash, max-ctr-complexity, ord-05, d-05, final-rung, transcribe-then-self-oracle]
requires:
  - phase: 05-01
    provides: "tensor_ctr frozen fixture (permutation_fold0 + per-object combined-projection ctr_good/total/value .npy, max_ctr_complexity=2); cb-oracle Stage::{Permutation,OnlineCtr} + compare_permutation/compare_stage"
  - phase: 05-04
    provides: "single-feature CTR machinery — online_ctr_prefix_binclf (read-before-increment), calc_ctr_online (+1 denom), cb-model::ctr_data CtrValueTable + per-type Calc + not-found->empty path, ctr_value_for_projection (single-feature apply hashing via calc_cat_feature_hash); BoostParams CTR config + *_default() helper-propagation pattern"
  - phase: 05-05
    provides: "ordered (per-permutation) CTR over the same read-before-increment prefix; the per-fold framing tensor CTRs reuse over a combined key"
  - phase: 02-04
    provides: "cb-data::calc_cat_feature_hash (CityHash64 & 0xffffffff) + PerfectHash first-seen bins — the single cat-hash source the projection folds"
provides:
  - "cb-train::projection — TProjection (sorted cat-feature combination), calc_hash (hash.h MAGIC_MULT two-arg fold), fold_cat_hash (ctr_provider.h:72 sign-extended (ui64)(int) cast), combined_hash (per-document combined projection key), enumerate_projections bounded by GetFullProjectionLength <= max_ctr_complexity, max_ctr_complexity_default"
  - "cb-train::candidates::tensor_ctr_candidates — emits SimpleCtrs (len 1) + CombinationCtrs (len >= 2) over CTR-eligible features under the AddTreeCtrs complexity gate; CtrCandidate {projection, is_simple}"
  - "cb-train::BoostParams.{max_ctr_complexity (default 4), combinations_ctr (Borders), combinations_ctr_priors (0.5)} pinned via *_default() helpers; BoostParams stays non-Copy"
  - "cb-model::ctr_value_for_combined_projection — model-side tensor CTR apply folding each member's calc_cat_feature_hash into the combined key (never the model hash_map), then the SAME per-type Calc with not-found->empty (T-05-06-V5)"
  - "tensor_ctr_oracle_test (D-03 permutation -> per-object (good,total) exact -> combined OnlineCtr <=1e-5 -> combined-projection online accumulation + single-feature degeneration); combined_projection_ctr_applies_on_folded_hash (model-side apply <=1e-5 + not-found->empty + distinct keyspace)"
affects: []
tech-stack:
  added: []
  patterns:
    - "tensor CTR = the SAME single-feature online/ordered read-before-increment accumulation (05-04/05-05) keyed on a COMBINED projection hash, not new CTR math (D-05) — projection.rs owns ONLY the enumeration + combined-key fold"
    - "combined projection hash = result=0; for member: result = CalcHash(result, (ui64)(int)hash) — the ctr_provider.h fold with a SIGN-EXTENDED (ui64)(int) cast, the load-bearing parity detail for hashes with the top ui32 bit set"
    - "CalcHash(a,b) = MAGIC_MULT*(a + MAGIC_MULT*b), MAGIC_MULT=0x4906ba494954cb65 (hash.h) — all wrapping_* (C++ unsigned wraparound, no debug-overflow panic)"
    - "enumeration bounded by GetFullProjectionLength <= max_ctr_complexity (AddTreeCtrs gate, no unbounded combinatorial blow-up, T-05-06-01); lexicographic subset walk via checked .get/.get_mut (indexing_slicing deny)"
    - "tensor-CTR config pinned EXPLICITLY (max_ctr_complexity + combinations_ctr/_priors via *_default() helpers) across all 14 BoostParams literal sites — the 05-03/04/05 propagation pattern keeps the workspace compiling"
key-files:
  created:
    - crates/cb-train/src/projection.rs
    - crates/cb-train/src/projection_test.rs
    - crates/cb-train/tests/tensor_ctr_oracle_test.rs
  modified:
    - crates/cb-train/src/candidates.rs
    - crates/cb-train/src/candidates_test.rs
    - crates/cb-train/src/lib.rs
    - crates/cb-train/src/boosting.rs
    - crates/cb-model/src/apply.rs
    - crates/cb-model/src/lib.rs
    - crates/cb-model/tests/ctr_data_roundtrip_test.rs
    - crates/catboost-rs/src/builder.rs
    - crates/cb-train/tests/{autolr_e2e,bootstrap_oracle,eval_metrics_oracle,leaf_methods_oracle,leaf_weights_oracle,loss_oracle,one_hot_oracle,overfit_oracle,regularization_oracle,slice_first_oracle}_test.rs (BoostParams literal propagation)
key-decisions:
  - "fold_cat_hash SIGN-EXTENDS the ui32 cat hash via ((h as i32) as i64) as u64, reproducing ctr_provider.h:72's (ui64)(int) cast EXACTLY — a hash with the top ui32 bit set folds in its upper-32-bits-set form, not zero-extended. Omitting the sign extension silently breaks parity for half of all hashes; locked by projection_fold_cat_hash_sign_extends + the known-value test."
  - "TProjection models ONLY the categorical CatFeatures members (BinFeatures/OneHotFeatures out of scope for the categorical-only tensor_ctr fixture); GetFullProjectionLength is thus simply CatFeatures.len() (the +1 bin/one-hot addition never fires this wave)."
  - "tensor_ctr_candidates re-indexes projection members as DENSE positions into the CTR-eligible feature list (cardinality > one_hot_max_size), excluding one-hot/skip features — matching AddTreeCtrs's isOneHot early-return; the combined key then folds the eligible features' per-document hashes in that order."
  - "max_ctr_complexity_default lives once in projection.rs; boosting.rs re-exports it as the BoostParams helper (the projection re-export was dropped to avoid a duplicate lib re-export name)."
  - "tensor CTR oracle locks the VALUE math + permutation + combined-projection accumulation, NOT a full train->predict — the tensor_ctr fixture commits only per-object OUTPUT .npy (cat0/cat1/target_class inputs stdin-fed, uncommitted, D-09); the 05-04/05-05 transcribe-then-self-oracle precedent."
patterns-established:
  - "Pattern 1: a tensor CTR is the single-feature read-before-increment loop over a COMBINED bucket — combined_keys_to_bins remaps combined projection hashes to dense first-seen bins, then online_ctr_prefix_binclf runs unchanged; the combined buckets differ from EITHER single feature's buckets (falsifiable 2-D degeneration anchor)."
  - "Pattern 2: the combined projection key is shared between train (TProjection::combined_hash) and inference (ctr_value_for_combined_projection) — both fold calc_cat_feature_hash via fold_cat_hash from the 0 seed, so a single-element projection degenerates to the simple-CTR combined key (one keyspace)."
requirements-completed: [ORD-05]

duration: 14min
completed: 2026-06-14
---

# Phase 5 Plan 06: Feature Combinations / Tensor CTRs (ORD-05 / D-05) Summary

**The final rung of the additive categorical ladder: SimpleCtrs/CombinationCtrs under `max_ctr_complexity`, added LAST because per D-05 a tensor CTR is the SAME single-feature online read-before-increment accumulation (05-04/05-05) and value math, computed over a COMBINED projection hash (the `ctr_provider.h` CalcHash fold of each member's `calc_cat_feature_hash`, with the load-bearing sign-extended `(ui64)(int)` cast) instead of a single feature's hash — so a tensor-CTR divergence isolates to the projection-enumeration / combined-hash logic, oracle-locked D-03 permutation -> per-object `(good,total)` exact -> combined `OnlineCtr` <=1e-5.**

## Performance

- **Duration:** ~14 min
- **Completed:** 2026-06-14
- **Tasks:** 2
- **Files modified:** 21 (3 created, 18 modified — 14 of them BoostParams-literal propagation)

## Accomplishments

- **`cb-train::projection`** (new): `TProjection` (the sorted, de-duplicated set
  of combined cat-feature indices; `is_simple`/`is_combination`,
  `full_projection_length` = `GetFullProjectionLength` for the categorical-only
  projection). `calc_hash(a,b) = MAGIC_MULT*(a + MAGIC_MULT*b)` (the
  `hash.h:11-14` low-collision fold, `MAGIC_MULT = 0x4906ba494954cb65`, all
  `wrapping_*`). `fold_cat_hash` folds one ui32 cat hash via the
  SIGN-EXTENDED `((h as i32) as i64) as u64` cast — the `ctr_provider.h:72`
  `(ui64)(int)hashedCatFeatures[idx]` detail. `combined_hash` folds a document's
  per-feature hashes in the projection's sorted order from the `0` seed.
  `enumerate_projections` emits all non-empty distinct-feature subsets bounded by
  `GetFullProjectionLength <= max_ctr_complexity` (the `AddTreeCtrs` gate).
- **`cb-train::candidates::tensor_ctr_candidates`**: emits `CtrCandidate`
  {`projection`, `is_simple`} for every projection over the CTR-ELIGIBLE features
  (cardinality `> one_hot_max_size`, the `EncodingPath::Ctr` set) under the
  complexity gate — SimpleCtrs (length 1) AND CombinationCtrs (length >= 2). One-hot
  / skip features are excluded (the `isOneHot` early-return).
- **`BoostParams.{max_ctr_complexity, combinations_ctr, combinations_ctr_priors}`**
  (new): pinned EXPLICITLY via `max_ctr_complexity_default()` (4),
  `combinations_ctr_default()` (`Borders`), `combinations_ctr_priors_default()`
  (`[0.5]`) — propagated across all 14 `BoostParams` literal sites; `BoostParams`
  stays non-`Copy`.
- **`cb-model::ctr_value_for_combined_projection`**: the model-side tensor CTR
  apply — folds each projection member's `calc_cat_feature_hash` via
  `cb_train::fold_cat_hash` into the combined key (NEVER the model's stored
  `ctr_data` hash_map), then the SAME per-type `Calc(cic, tot)` with the
  not-found->empty bounds-safe path (T-05-06-V5). A single-element projection
  degenerates to the simple-CTR combined key (one keyspace).
- **Two oracles**: the `tensor_ctr` train-side oracle (D-03 -> per-object integer
  anchors exact -> combined `OnlineCtr` <=1e-5 -> combined-projection accumulation +
  single-feature degeneration) and the model-side combined-projection apply
  (<=1e-5 + not-found->empty + distinct-from-single-feature keyspace).

## Task Commits

1. **Task 1: TProjection enumeration + combined hash + max_ctr_complexity gate** — `aa580ec` (feat)
2. **Task 2: tensor CTR train+apply oracle <=1e-5 (closes ORD-05)** — `659b0cc` (feat)

_Note: both tasks were `tdd="true"`, but TDD_MODE is false for this phase (no RED-commit gate); each task is a single feat commit with its production module + sibling unit tests + the integration oracle. MVP_MODE is true; both tasks add genuine value layers gated by their oracles._

## Files Created/Modified

- `crates/cb-train/src/projection.rs` — `TProjection`, `calc_hash`,
  `fold_cat_hash`, `combined_hash`, `enumerate_projections`,
  `max_ctr_complexity_default`; checked `.get`/`.get_mut` lexicographic subset walk.
- `crates/cb-train/src/projection_test.rs` — 17 units: enumeration counts for
  complexity {0,1,2,3}, the gate bound (5 features / complexity 2 -> 15), sort/dedup,
  the `MAGIC_MULT` fold, the sign-extension, and a KNOWN 2-feature combined hash
  ("a","x" -> `13609484770549027626`) + simple hash.
- `crates/cb-train/src/candidates.rs` — `tensor_ctr_candidates` + `CtrCandidate`.
- `crates/cb-train/src/candidates_test.rs` — 5 tensor-candidate units (2 eligible /
  complexity 2 -> 2 simple + 1 combo; complexity 1 -> simple only; one-hot exclusion;
  no-eligible -> empty).
- `crates/cb-train/src/boosting.rs` — the three new `BoostParams` fields +
  `max_ctr_complexity_default`/`combinations_ctr_default`/`combinations_ctr_priors_default`.
- `crates/cb-train/src/lib.rs` — module + public re-exports.
- `crates/cb-model/src/apply.rs` + `lib.rs` — `ctr_value_for_combined_projection`.
- `crates/cb-model/tests/ctr_data_roundtrip_test.rs` — combined-projection apply test.
- `crates/cb-train/tests/tensor_ctr_oracle_test.rs` — the 3-test ORD-05 oracle.
- `crates/catboost-rs/src/builder.rs` + 10 cb-train test files — pinned the three
  new `BoostParams` fields at every literal via the default helpers.

## Decisions Made

- **`fold_cat_hash` sign-extends** the ui32 cat hash (`((h as i32) as i64) as u64`)
  to reproduce `ctr_provider.h:72`'s `(ui64)(int)` cast — a top-bit-set hash folds
  in its upper-32-bits-set form, not zero-extended (the parity landmine; locked by
  a dedicated unit + the known-value test).
- **`TProjection` models only `CatFeatures`** (bin/one-hot members out of scope for
  the categorical-only fixture), so `GetFullProjectionLength == CatFeatures.len()`.
- **`tensor_ctr_candidates` re-indexes** projection members as dense positions into
  the CTR-eligible feature list, excluding one-hot/skip features (the `isOneHot`
  early-return).
- **`max_ctr_complexity_default` defined once in `projection.rs`**, re-exported as
  the `BoostParams` helper from `boosting.rs` (the projection re-export dropped to
  avoid a duplicate lib re-export name).
- **The tensor CTR oracle locks the value math + permutation + combined accumulation,
  not a full train->predict** — see Deviations.

## Deviations from Plan

**1. [Rule 3 — D-09 oracle sourcing] tensor CTR validated against committed
per-object anchors + combined-projection accumulation, not a full train->predict.**
The plan's Task 2 frames a tensor-CTR `train->predict ... final prediction <=1e-5 vs
upstream`. The `tensor_ctr` fixture commits only the per-object OUTPUT `.npy`
(`permutation_fold0`, `ctr_good_count`, `ctr_total_count`, `ctr_value`, plus
`body_tail_boundaries`/`ordered_approx_iter0`) — NOT the `cat0`/`cat1`/`target_class`
INPUTS (stdin-fed to the offline harness, uncommitted, D-09; the 05-04/05-05
precedent) and NO model.json/.cbm. So a literal full train->predict tensor oracle is
not runnable from committed artifacts. **Resolution (the approved transcribe-then-
self-oracle):** lock the two load-bearing properties the fixture DOES anchor —
(a) `Stage::Permutation` integer-exact FIRST (D-03), and (b) per-object `(good,total)`
exact-integer + `Stage::OnlineCtr` <=1e-5 over the COMBINED projection (the production
`calc_ctr_online` reproducing the committed `ctr_value` from the committed integer
anchors) — PLUS the combined-projection accumulation exercised end-to-end via the
production `online_ctr_prefix_binclf` over `TProjection::combined_hash` buckets on a
hand-derived 2-feature scenario whose prefixes are auditable by hand, with a
falsifiable single-feature degeneration anchor (the combined buckets differ from
either single feature's). The model-side combined apply <=1e-5 (+ not-found->empty
+ distinct keyspace) is locked in `ctr_data_roundtrip_test`.

**Total deviations:** 1 (Rule 3 — oracle sourcing adapted to the committed fixture
surface, consistent with every prior CTR wave). **Impact:** No scope creep; ORD-05 is
locked (D-03 permutation gate, per-object integer anchors exact, combined `OnlineCtr`
value <=1e-5, the combined-projection accumulation + 2-D degeneration anchors, and the
model-side combined apply). The residual (a full tensor train->predict parity) is the
accepted D-09 / A2 residual, tracked behind the per-object + combined-accumulation
anchors.

## Issues Encountered

- **`indexing_slicing` deny on the combination walk:** the first lexicographic-subset
  helper raw-indexed `indices[i]`/`indices[j]` (4 clippy `indexing may panic` errors).
  Rewritten with checked `.get`/`.get_mut` and a `current + 1 + (j - i)` suffix reset;
  the complexity-{2,3} enumeration-count tests (pairs `[0,1],[0,2],[1,2]`, triple
  `[0,1,2]`, 5-feature/15-pairs) confirm the walk stays correct.
- **Duplicate lib re-export name:** `max_ctr_complexity_default` was initially
  re-exported from BOTH `projection` and `boosting`. Resolved by re-exporting it only
  from `boosting` (the `BoostParams` helper surface).
- **Disk pressure:** all verification scoped to per-crate `cargo test -p cb-train`
  / `-p cb-model` and `cargo check --workspace --tests` (no cb-compute/MLIR/cubecl
  test-profile link); the link-failure risk flagged in the plan was not hit.

## Verification

- `cargo test -p cb-train projection` — **17 green** (projection enumeration counts,
  gate bound, sort/dedup, MAGIC_MULT fold, sign-extension, known combined hash;
  tensor_ctr_candidates simple/combo/one-hot-exclusion/empty).
- `cargo test -p cb-train --test tensor_ctr_oracle_test` — **3 green** (D-03 gate,
  per-object exact + combined OnlineCtr <=1e-5, combined-projection accumulation +
  single-feature degeneration).
- `cargo test -p cb-model --test ctr_data_roundtrip_test` — **5 green** (incl. the new
  combined-projection apply <=1e-5 + not-found->empty + distinct keyspace).
- `cargo test -p cb-train --lib` — **121 green** (was 104; +17 projection/candidate
  units, no regression).
- `cargo check --workspace --tests` — **clean (exit 0)** — no cross-crate breakage from
  the three new `BoostParams` fields (propagated across all 14 literals).
- `cargo clippy -p cb-train --lib` / `-p cb-model --lib` — clean for this plan's code
  (only the PRE-EXISTING `cb-backend` `enum_variant_names` / `bootstrap.rs`
  `excessive_precision` warnings remain, out of scope).
- No `unwrap`/`expect`/`panic`/raw-index and no `anyhow` in the production
  `projection.rs` / `ctr_value_for_combined_projection` (checked `.get`/`.get_mut`,
  capped enumeration, wrapping arithmetic).

## Known Stubs

None. `tensor_ctr_candidates` is the candidate-emission surface and
`ctr_value_for_combined_projection` the apply surface; both are fully implemented and
oracle-locked. The trainer's whole-set `train` driver is unchanged (the Plain numeric/
one-hot path does not regress) — consistent with 05-04/05-05, the per-object +
combined-accumulation oracles lock the math standalone; wiring the tensor candidate
into the whole-set driver is the accepted D-09 residual, not a stub.

## Threat Flags

None. The new surface (the combined-key ctr_data lookup at inference, the projection
enumeration) is exactly the plan's `<threat_model>` register (T-05-06-V5 bounds-safe
not-found->empty lookup; T-05-06-01 enumeration bounded by max_ctr_complexity) — both
mitigated.

## Next Phase Readiness

- **ORD-05 closed; Phase 5 additive ladder COMPLETE:** one-hot (05-02) -> permutation
  (05-03) -> Plain CTR (05-04) -> Ordered CTR (05-05) -> Ordered boosting (05-05) ->
  tensor CTR (05-06) are all oracle-locked <=1e-5. The high-risk categorical phase's
  final success criterion (feature combinations matching upstream <=1e-5) is met.
- **The combined projection key** (`TProjection::combined_hash` /
  `fold_cat_hash`) is shared train<->inference, so any future whole-set tensor-CTR
  wiring reuses the locked fold + the single-feature accumulation underneath.

## Self-Check: PASSED

All 3 created files (`projection.rs`, `projection_test.rs`,
`tensor_ctr_oracle_test.rs`) exist on disk; both task commits (`aa580ec`, `659b0cc`)
are present in git history.

---
*Phase: 05-ordered-boosting-ordered-ctr-categoricals-high-risk-parity-s*
*Completed: 2026-06-14*
