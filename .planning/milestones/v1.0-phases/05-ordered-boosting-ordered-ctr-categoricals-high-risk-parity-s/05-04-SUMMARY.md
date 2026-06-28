---
phase: 05-ordered-boosting-ordered-ctr-categoricals-high-risk-parity-s
plan: 04
subsystem: cb-train + cb-model (Plain-mode CTR math + model-side ctr_data serde/apply, ORD-03 / D-06)
tags: [ctr, plain-ctr, target-statistics, ctr_data, serde, online-ctr, read-before-increment, six-ctr-types, d-06, ord-03]
requires:
  - phase: 05-01
    provides: "plain_ctr frozen fixture (permutation_fold0 + per-object online ctr good/total/value .npy); cb-oracle Stage::{Permutation,OnlineCtr} + compare_permutation; model.json ctr_data parser (CtrTableJson)"
  - phase: 05-03
    provides: "fisher_yates_permutation (D-03 integer-exact, fold-0); BoostParams permutation_count/fold_len_multiplier + *_default() helper-propagation pattern"
  - phase: 04-01
    provides: "cb-model::generated::ctr_data_generated (ECtrType bindings); canonical cb-model::Model"
  - phase: 04-03
    provides: ".cbm CBM1 framing + VERIFYING bounds-before-slice decode discipline (cbm.rs); model.json export/import; ModelError typed/panic-free serde"
  - phase: 02-04
    provides: "cb-data::calc_cat_feature_hash (CityHash64 & 0xffffffff) + PerfectHash first-seen bins — the single cat-hash source"
provides:
  - "cb-train::ctr module — online (whole-set, Plain) accumulation, online/inference CalcCTR (separate fns), final-CTR table build for all six ECtrType"
  - "cb-train::online_ctr_prefix_binclf — per-object online (read-before-increment) binclf CTR over the permutation (the no-leakage prefix)"
  - "cb-train::ECtrType + CounterCalcMethod + BoostParams CTR config (simple_ctr/simple_ctr_priors/counter_calc_method) pinned via *_default() helpers"
  - "cb-model::ctr_data — CtrValueTable/CtrData serde over both model.json (hash_map/hash_stride/counter_denominator) and a self-describing .cbm binary blob; per-type inference Calc + not-found→empty path"
  - "cb-model::ctr_value_for_projection — model-side CTR apply hashing the projection via calc_cat_feature_hash"
  - "plain_ctr_oracle_test (D-03 permutation → D-06 OnlineCtr value lock); ctr_data_roundtrip_test (both wire forms + per-type apply ≤1e-5 + malformed-blob Err)"
affects:
  - 05-05 (ordered CTR — a focused delta on accumulation ORDER over this locked Plain CTR math)
  - 05-06 (tensor/combination CTRs build on the ctr_data serde + final-CTR table)
tech-stack:
  added: []
  patterns:
    - "online (+1 denom) vs inference (+PriorDenom) CalcCTR as SEPARATE functions (Pitfall 1; coincide only at PriorDenom==1)"
    - "Counter denom = MAX bucket total vs FeatureFreq denom = total sample count (Pitfall 4; same numerator, distinct denominator)"
    - "FloatTargetMeanValue final-CTR-path-only, never in the online dispatch (Pitfall 5)"
    - "local ECtrType duplicated in cb-train (below cb-model in the dep graph) mirroring the upstream i8 discriminants; cb-model maps losslessly"
    - "bounds-before-slice BlobReader cursor for the untrusted .cbm ctr_data blob (declared-length cap + per-byte bound; Security V5)"
    - "read-before-increment online CTR prefix = the no-leakage property (a doc's CTR never sees its own label)"
key-files:
  created:
    - crates/cb-train/src/ctr/online.rs
    - crates/cb-train/src/ctr/online_test.rs
    - crates/cb-train/src/ctr/calc_ctr.rs
    - crates/cb-train/src/ctr/calc_ctr_test.rs
    - crates/cb-train/src/ctr/final_ctr.rs
    - crates/cb-train/src/ctr/final_ctr_test.rs
    - crates/cb-train/src/ctr/mod.rs
    - crates/cb-model/src/ctr_data.rs
    - crates/cb-model/src/ctr_data_test.rs
    - crates/cb-train/tests/plain_ctr_oracle_test.rs
    - crates/cb-model/tests/ctr_data_roundtrip_test.rs
  modified:
    - crates/cb-train/src/lib.rs
    - crates/cb-train/src/boosting.rs
    - crates/cb-model/src/apply.rs
    - crates/cb-model/src/lib.rs
    - crates/cb-model/Cargo.toml
    - crates/catboost-rs/src/builder.rs
    - crates/cb-train/tests/{overfit,autolr_e2e,loss,regularization,eval_metrics,one_hot,bootstrap,leaf_methods,slice_first,leaf_weights}_*_test.rs (BoostParams literal propagation)
key-decisions:
  - "ECtrType is DUPLICATED in cb-train (mirroring the upstream i8 discriminants) rather than imported from cb-model's generated bindings — cb-model depends on cb-train, so cb-train cannot depend on cb-model (circular). The two map losslessly via as_i8/from_i8."
  - "BoostParams DROPPED Copy: the CTR config carries an owned Vec<f64> of explicit priors (simple_ctr_priors). All call sites already pass &BoostParams; the three new fields propagate via simple_ctr_default()/simple_ctr_priors_default()/counter_calc_method_default() helpers (the 05-03 one_hot_max_size pattern) across all 12 literal sites — workspace stays compiling."
  - "ctr_data .cbm wire form is a self-describing little-endian blob (the model-parts region after the FlatBuffers core), NOT a FlatBuffers table — the committed ctr_data_generated bindings define ECtrType + feature/split structs but no TCtrValueTable table, so the bucket blob is serialized directly with a bounds-before-slice BlobReader mirroring cbm.rs:240-270."
  - "Plain CTR oracle locks the VALUE math (calc_ctr_online over the committed integer good/total anchors) + the permutation (D-03), NOT a full train→predict, because the plain_ctr fixture commits only per-object OUTPUT .npy (the cat_bin/target_class INPUTS were stdin-fed to the offline harness and are uncommitted, D-09) — the 05-02/D-04 transcribe-then-self-oracle precedent."
patterns-established:
  - "Pattern 1: six-type CTR math in one whole-set accumulation pass (OnlineCtrAccumulator holds class/mean/total histograms; each ECtrType reads whichever it needs in build_final_ctr)."
  - "Pattern 2: untrusted ctr_data blob decode via a bounds-checked BlobReader — every declared length bounded before slice + a 16M declared-length cap; malformed/oversized/truncated → typed ModelError, never panic (Security V5, T-05-04-V5/01/02)."
  - "Pattern 3: read-before-increment online prefix as the no-leakage signature — locked falsifiably (reversed permutation changes the prefixes)."
requirements-completed: [ORD-03]

duration: 19min
completed: 2026-06-14
---

# Phase 5 Plan 04: Plain CTR (ORD-03 / D-06 lock) Summary

**All six CTR types computed as whole-set target statistics (online/inference CalcCTR as separate `+1`/`+PriorDenom` functions, Counter-max vs FeatureFreq-total denominators, FloatTargetMeanValue final-path-only) plus the model-side `ctr_data` (de)serialize over both `.cbm` (bounds-checked binary blob) and `model.json` (hash_map shape) and the per-type inference apply — Plain-mode CTR locked BEFORE ordered CTR (D-06 key isolation), with the plain_ctr permutation (D-03) and per-object online CTR value (≤1e-5) oracle-locked.**

## Performance

- **Duration:** ~19 min
- **Completed:** 2026-06-14
- **Tasks:** 3
- **Files modified:** 21 (11 created, 10 modified)

## Accomplishments

- **Six-type CTR math** (`cb-train::ctr`): `online.rs` whole-set per-bucket
  class-count / mean / total accumulation (one pass, cat-hash via
  `calc_cat_feature_hash` + `PerfectHash`, never a model `ctr_data` hash_map);
  `calc_ctr.rs` the online (`(cic+prior)/(total+1)`) and inference
  (`(cic+PriorNum)/(total+PriorDenom)`) CalcCTR as SEPARATE functions (Pitfall 1);
  `final_ctr.rs` the `CalcFinalCtrsImpl` table build with Counter (max bucket)
  vs FeatureFreq (total sample) denominators (Pitfall 4) and FloatTargetMeanValue
  final-path-only (Pitfall 5). Integer class counts exact-int (exempt from
  `sum_f64`); the Buckets Σ-classes routes through `cb_core::sum_f64`.
- **Online prefix loop** (`online_ctr_prefix_binclf`): the per-object
  read-before-increment binclf CTR over the fold permutation — `good=N[1]`,
  `total=N[0]+N[1]` READ before `++N[targetClass]` (the no-leakage property).
- **`cb-model::ctr_data`**: `CtrValueTable`/`CtrData` over a local `ECtrType`
  (i8 discriminants matching cb-train + the generated bindings); round-trips
  through BOTH the upstream `model.json` `hash_map`/`hash_stride`/
  `counter_denominator` flat-array shape AND a self-describing `.cbm` binary blob
  decoded by a bounds-before-slice `BlobReader` (Security V5). Per-type inference
  `Calc(cic,tot)` with the not-found→empty path (`Calc(0,denom)`/`Calc(0,0)`),
  never an OOB index.
- **Model-side apply** (`ctr_value_for_projection`): hashes the projection via
  `cb_data::calc_cat_feature_hash` (single cat-hash source) and applies the
  table's per-type `Calc`.
- **BoostParams CTR config**: `simple_ctr` (type) + `simple_ctr_priors`
  (explicit per-prior numerators) + `counter_calc_method` (default `SkipTest`),
  pinned EXPLICITLY via `*_default()` helpers; `BoostParams` dropped `Copy`
  (owned priors `Vec`) and propagated across all 12 literal sites.

## Task Commits

1. **Task 1: Six-type CTR math — online accumulation + CalcCTR + final-CTR table** — `944684f` (feat)
2. **Task 2: ctr_data .cbm/model.json serde + model-side CTR apply (bounds-checked)** — `5dc1dfb` (feat)
3. **Task 3: Plain CTR online-prefix loop + plain_ctr oracle (D-06 lock)** — `02ed8a2` (feat)

_Note: all three tasks were `tdd="true"`, but TDD_MODE is false for this phase (no RED-commit gate); each task is a single feat commit with its production module + sibling unit tests + the integration oracle._

## Files Created/Modified

- `crates/cb-train/src/ctr/online.rs` — whole-set `accumulate_online` + `online_ctr_prefix_binclf` (read-before-increment); `TCtrHistory`/`TCtrMeanHistory`.
- `crates/cb-train/src/ctr/calc_ctr.rs` — `calc_ctr_online` (+1), `calc_ctr_inference` (+PriorDenom), `calc_normalization`, `Prior`.
- `crates/cb-train/src/ctr/final_ctr.rs` — `build_final_ctr` per `ECtrType`; `FinalCtrTable`.
- `crates/cb-train/src/ctr/mod.rs` — local `ECtrType` (i8 discriminants, default_priors), `CounterCalcMethod`, re-exports.
- `crates/cb-train/src/ctr/{online,calc_ctr,final_ctr}_test.rs` — 27 per-type unit tests.
- `crates/cb-train/src/boosting.rs` — `BoostParams.{simple_ctr,simple_ctr_priors,counter_calc_method}` + `*_default()` helpers; dropped `Copy`.
- `crates/cb-train/src/lib.rs` — `mod ctr;` + public re-exports.
- `crates/cb-model/src/ctr_data.rs` — `CtrValueTable`/`CtrData`, model.json `CtrTableJson` serde, `.cbm` `encode/decode_ctr_data` (BlobReader), per-type `Calc`.
- `crates/cb-model/src/ctr_data_test.rs` — 15 unit tests (bounds rejection, malformed-blob Err, per-type Calc, round-trips).
- `crates/cb-model/src/apply.rs` — `ctr_value_for_projection` (projection hash via `calc_cat_feature_hash`).
- `crates/cb-model/src/lib.rs` + `Cargo.toml` — module + re-exports; `cb-data` dependency added.
- `crates/catboost-rs/src/builder.rs` + 10 cb-train test files — pinned the three new `BoostParams` fields at every literal via the default helpers.
- `crates/cb-train/tests/plain_ctr_oracle_test.rs` — D-03 permutation → D-06 OnlineCtr value lock + no-leakage ordering.
- `crates/cb-model/tests/ctr_data_roundtrip_test.rs` — both-wire-form round-trip + per-type apply ≤1e-5 + malformed-blob Err.

## Decisions Made

- **ECtrType duplicated in cb-train** (mirroring the upstream i8 discriminants),
  not imported from cb-model's generated bindings — cb-model depends ON cb-train,
  so the reverse import would be circular. `as_i8`/`from_i8` map losslessly.
- **BoostParams dropped `Copy`** to carry the owned `simple_ctr_priors: Vec<f64>`;
  all call sites pass `&BoostParams` already, and the three new fields propagate
  via `*_default()` helpers across all 12 literal sites (the 05-03
  `one_hot_max_size` pattern) — `cargo check --workspace --tests` stays green.
- **ctr_data `.cbm` is a self-describing binary blob**, not a FlatBuffers table:
  the committed `ctr_data_generated` bindings define `ECtrType` + feature/split
  structs but no `TCtrValueTable` table, so the bucket blob is serialized directly
  with a bounds-before-slice `BlobReader` mirroring `cbm.rs:240-270`.
- **Plain CTR oracle locks the value math + permutation, not full train→predict**
  — see Deviations.

## Deviations from Plan

The plan's Task 3 framed a full end-to-end `train→predict` plain-CTR oracle
(splits/leaves/staged/predict ≤1e-5) wired through `boosting.rs` candidate
generation. The committed `plain_ctr` fixture, however, carries ONLY the
per-object OUTPUT `.npy` (`permutation_fold0`, `ctr_good_count`,
`ctr_total_count`, `ctr_value`) — NO input features/target, NO `cat_bin`/
`target_class` (those were stdin-fed to the offline harness and are uncommitted,
D-09), and NO model.json/.cbm. So a literal full train→predict oracle is not
runnable from committed artifacts.

**Resolution (Rule 3 — blocking-issue handling, the 05-02/D-04
transcribe-then-self-oracle precedent):** Task 3 locks the two load-bearing
properties the fixture DOES anchor: (a) `Stage::Permutation` integer-exact FIRST
(D-03), and (b) `Stage::OnlineCtr` ≤1e-5 — the production `calc_ctr_online`
reproducing the committed `ctr_value` from the committed integer `(good, total)`
anchors — plus a hand-derived no-leakage read-before-increment ordering lock
(reversed permutation changes the prefixes). The model-side per-type apply
≤1e-5 is locked in `ctr_data_roundtrip_test` against the independently-computed
trainer-side whole-set table. This preserves the D-06 isolation intent (Plain
CTR math locked before any ordered math) without fabricating an uncommitted
input corpus. The `boosting.rs` change in this plan is the CTR config surface
(Task 1); the candidate-generation CTR-split emission is deferred to the ordered
CTR wave (05-05), which needs the permutation-ordered prefix anyway.

**Total deviations:** 1 (Rule 3 — oracle sourcing adapted to the committed
fixture surface). **Impact:** No scope creep; the D-06 lock (Plain CTR value +
permutation) holds. The residual (full train→predict CTR oracle) is tracked for
the ordered-CTR wave, where the per-object prefix is the primary subject.

## Issues Encountered

- **Circular-dependency on ECtrType:** the plan's interface listed `ECtrType`
  "from cb-model's generated bindings," but cb-train is BELOW cb-model in the dep
  graph. Resolved by a local cb-train `ECtrType` mirroring the same i8
  discriminants (lossless `as_i8`/`from_i8` bridge).
- **Fragile fixture reconstruction:** a first attempt reconstructed the
  `cat_bin`/`target_class` inputs from the committed prefix counts to drive the
  full read-before-increment loop end-to-end; the running-state bucket match was
  ambiguous across buckets with identical prefix states and failed. Replaced with
  a hand-derived no-leakage ordering lock (deterministic, falsifiable) — the value
  math is already locked against the committed anchors directly.
- **Disk pressure (~91% full, 22G free):** all verification scoped to per-crate
  `cargo test -p cb-train` / `-p cb-model` (no MLIR/cubecl test-profile link); the
  cb-compute link-failure risk noted in the plan was never hit.

## Next Phase Readiness

- **D-06 isolation achieved:** Plain-mode CTR math (whole-set + the online
  read-before-increment prefix) is locked. Wave 5's ordered CTR is now a focused
  delta on accumulation ORDER alone — the `online_ctr_prefix_binclf` loop is the
  substrate; the only change is computing the prefix UNDER the ordered permutation
  with the body/tail boundaries (05-03) rather than the whole set.
- **`ctr_data` serde + apply are ready** for the ordered/tensor CTR waves to bake
  and load real CTR tables; the bounds-checked decode is the panic-free model-load
  path for untrusted CTR blobs.

## Self-Check: PASSED

All 11 created files and the modified files exist on disk; all three task commits
(`944684f`, `5dc1dfb`, `02ed8a2`) are present in git history.

---
*Phase: 05-ordered-boosting-ordered-ctr-categoricals-high-risk-parity-s*
*Completed: 2026-06-14*
