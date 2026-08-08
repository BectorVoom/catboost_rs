//! T09 / DCTR-07 — the FIRST covering test for [`super::ctr_covered`]'s per-type admission
//! list, and for the C-7 invariant that admitting a second CTR type changes **nothing else**
//! about the coverage predicate.
//!
//! # Why this module is mounted inside `session.rs`, not `gpu_runtime/mod.rs`
//!
//! [`super::ctr_covered`] is a **private** free function inside `mod session`. A module
//! mounted in `gpu_runtime/mod.rs` (which is how `session_residency` and
//! `session_depth_gt1_test` are mounted) is a *sibling* of `session` and cannot name it —
//! `E0603`. Those two files work from `mod.rs` only because they exercise the **public**
//! `GpuTrainSession` surface. The pattern copied here is instead the in-module `#[path]`
//! sibling used by `cb-train/src/boosting.rs:7469-7474`. `ctr_covered` deliberately stays
//! private: a test that does not compile is not a reason to widen production visibility.
//!
//! # Source/test separation (CLAUDE.md / AGENTS.md)
//!
//! `session.rs` is production code; every `#[test]` for it lives here.
//!
//! # Backend
//!
//! `ctr_covered` is a pure host classifier — it touches no `ComputeClient` — so this module
//! runs on **every** non-wgpu backend, exactly like `session_depth_gt1_test`'s gate arm.
//! wgpu is excluded at module scope because `ctr_covered` force-declines there (WR-02: the
//! f64 CTR seam does not exist), which would make every positive assertion below vacuous.

#![cfg(not(feature = "wgpu"))]

use cb_compute::{
    DeviceCtrAveraging, DeviceCtrColumn, DeviceCtrConfig, DeviceTrainConfig,
};

use super::ctr_covered;

const N: usize = 32;
const N_BINS: usize = 16;

/// `cb_train::ctr::ECtrType`'s i8 discriminants, transcribed BY VALUE (Pattern B / C-3 —
/// `cb-backend` must never gain a `cb-train` dep). Upstream `restrictions.h:20-32` limits the
/// CPU task type to `{Borders, Buckets, BinarizedTargetMeanValue, Counter}`; the other two are
/// GPU-only upstream and have no parity surface here.
const BORDERS: i8 = 0;
const BUCKETS: i8 = 1;
const BINARIZED_TARGET_MEAN_VALUE: i8 = 2;
const FLOAT_TARGET_MEAN_VALUE: i8 = 3;
const COUNTER: i8 = 4;
const FEATURE_FREQ: i8 = 5;
/// NOT an `ECtrType` discriminant — the first value past the enum. Used by the mixed
/// structure/averaging pair below, which needs an operand that `ctr_covered` declines; since
/// DCTR-14 admitted the last CPU-legal type, a stray is the only choice no future task can
/// invalidate.
const UNKNOWN_DISCRIMINANT: i8 = 6;

/// `n_bins - 1` borders spanning `(0, 1)` ⇒ `borders.len() + 1 == n_bins`, the uniform-histogram
/// shape invariant `ctr_covered` enforces.
fn covered_borders() -> Vec<f64> {
    (1..N_BINS).map(|b| b as f64 / N_BINS as f64).collect()
}

/// One otherwise-covered CTR column carrying the given type/numerator selector: a single
/// small-cardinality member and a border table that binarizes to exactly `N_BINS` buckets.
/// `projection_members` is populated explicitly (rather than left to `Default`) so this fixture
/// never joins the `projection_members: vec![]` re-review set T17/T19 inherit.
fn column(ctr_type: i8, target_border_idx: u32) -> DeviceCtrColumn {
    DeviceCtrColumn {
        member_bins: vec![(0..N).map(|k| (k % 5) as u32).collect()],
        prior: 0.5,
        borders: covered_borders(),
        bucket_count: 5,
        weight_group: 0,
        ctr_type,
        target_border_idx,
        projection_members: vec![0],
    }
}

/// A two-permutation CTR config (GDC-09: the averaging half is mandatory) whose structure and
/// averaging columns may carry DIFFERENT types, so a conjunct added to only one of
/// `ctr_covered`'s two `.all(..)` closures is detectable.
fn config_with(structure: DeviceCtrColumn, averaging: DeviceCtrColumn) -> DeviceTrainConfig {
    let permutation: Vec<u32> = (0..N as u32).collect();
    let target_class: Vec<u32> = (0..N).map(|k| (k % 2) as u32).collect();
    DeviceTrainConfig {
        ctr: Some(DeviceCtrConfig {
            permutation: permutation.clone(),
            target_class: target_class.clone(),
            columns: vec![structure],
            averaging: Some(DeviceCtrAveraging {
                // A DIFFERENT (reversed) object order — the two materializations genuinely
                // diverge, mirroring `session_depth_gt1_test`'s covered fixture.
                permutation: permutation.into_iter().rev().collect(),
                target_class,
                columns: vec![averaging],
            }),
            ..DeviceCtrConfig::default()
        }),
        ..DeviceTrainConfig::default()
    }
}

/// Both halves carry the same type — the ordinary shape a real fit produces.
fn config(ctr_type: i8) -> DeviceTrainConfig {
    config_with(column(ctr_type, 0), column(ctr_type, 0))
}

/// DCTR-07 / DCTR-10 / DCTR-14: `ctr_covered` admits EXACTLY
/// `{Borders, Buckets, BinarizedTargetMeanValue, Counter}` — the four CPU-legal types — and
/// declines every other `ECtrType` discriminant.
///
/// # The admitted set moves with the device statistics, one task at a time
///
/// T09 seeded this loop with `{Borders, Buckets}` and told T12/T16 that widening it is their
/// **expected Red**, not a regression: a task that adds a device statistic must, in the same
/// hunk, add its discriminant to both `ctr_covered` closures, add its arm to
/// [`super::build_ctr_cindex_columns`]'s dispatch, and move the discriminant from the decline
/// loop below to the admit loop above. T12 did that for `Counter` (`4`, DCTR-10) and T16 for
/// `BinarizedTargetMeanValue` (`2`, DCTR-14). **The admitted set is now CLOSED**: what stays in
/// the decline loop **forever** is `{FloatTargetMeanValue (3), FeatureFreq (5)}` — GPU-only
/// upstream (`restrictions.h:20-32`), so no CPU oracle could ever prove parity — plus every
/// out-of-enum stray.
///
/// # Why declining loudly is safe (C-14 — the property this task records)
///
/// `ctr_covered` has TWO callers (`session.rs`'s `None` grow-policy arm and its CTR
/// augmentation arm). The first feeds the coverage disjunction whose failure path is
/// `return Ok(None)`, declining the **whole fit** to the byte-unchanged CPU path. So a
/// mismatch between the `cb-train` gate's admitted type list and THIS list can never silently
/// drop CTR columns onto a device fit — it degrades loudly to `grown == 0`. That is why the
/// two lists may be widened in separate hunks with no torn intermediate state.
#[test]
fn ctr_covered_declines_unimplemented_ctr_types() {
    for (ctr_type, name) in [
        (BORDERS, "Borders"),
        (BUCKETS, "Buckets"),
        // DCTR-14 (T16): BTMV is device-implemented via `launch_btmv_ctr_resident` (a
        // permutation-dependent prefix over a running FLOAT sum, NOT a class count — so it
        // cannot use the class-prefix kernel either).
        (BINARIZED_TARGET_MEAN_VALUE, "BinarizedTargetMeanValue"),
        // DCTR-10 (T12): Counter is device-implemented via `launch_counter_ctr_resident`
        // (a whole-set tally, NOT the class-prefix kernel).
        (COUNTER, "Counter"),
    ] {
        assert!(
            ctr_covered(&config(ctr_type), N, N_BINS),
            "{name} ({ctr_type}) is device-implemented and must be ADMITTED by ctr_covered"
        );
    }

    for (ctr_type, name) in [
        (FLOAT_TARGET_MEAN_VALUE, "FloatTargetMeanValue"),
        (FEATURE_FREQ, "FeatureFreq"),
    ] {
        assert!(
            !ctr_covered(&config(ctr_type), N, N_BINS),
            "{name} ({ctr_type}) has no device numerator yet and must be DECLINED by \
             ctr_covered — admitting it would route its columns through the class-prefix \
             kernel and silently return the Borders numerator"
        );
    }

    // Out-of-enum discriminants are declined too (an admission LIST, never a range check).
    for stray in [-1i8, 6, i8::MIN, i8::MAX] {
        assert!(
            !ctr_covered(&config(stray), N, N_BINS),
            "ctr_type {stray} is not an ECtrType discriminant and must be DECLINED"
        );
    }

    // BOTH `.all(..)` closures must carry the conjunct: a config whose structure half is
    // covered but whose averaging half is not (or vice versa) must decline. A one-sided edit
    // would leave the leaf-value gather materializing an unimplemented type.
    //
    // The mixed pair uses the OUT-OF-ENUM stray [`UNKNOWN_DISCRIMINANT`]. (T09 used `Counter`
    // here; T12 implemented it and moved the pair to `BinarizedTargetMeanValue`; T16
    // implemented that too. With the CPU-legal set now fully admitted, the discriminants that
    // still decline are the GPU-only pair and the strays — and a stray is the DURABLE choice,
    // because no future task can implement one. **Swap the operand if that ever stops being
    // true; never delete these two assertions**, which are the only pins that BOTH closures
    // carry the type conjunct.)
    assert!(
        !ctr_covered(
            &config_with(column(BORDERS, 0), column(UNKNOWN_DISCRIMINANT, 0)),
            N,
            N_BINS
        ),
        "an unimplemented AVERAGING column must decline (the averaging `.all(..)` closure \
         needs the same conjunct as the structure one)"
    );
    assert!(
        !ctr_covered(
            &config_with(column(UNKNOWN_DISCRIMINANT, 0), column(BORDERS, 0)),
            N,
            N_BINS
        ),
        "an unimplemented STRUCTURE column must decline"
    );
}

/// C-7: admitting a type adds an admission LIST entry and nothing else — the border-table
/// shape check (`borders.len() + 1 == n_bins`) and the non-empty-member check are UNCHANGED
/// for every type. Buckets (T09) and Counter (T12) both quantize through the same
/// `calc_ctr_online_bin` border table as Borders on the CPU (`quantize_in_f32 == false`;
/// Counter's `total` is the CONSTANT max-bucket denominator repeated per object, so the
/// resident triple has the ordered path's shape exactly), and BTMV (T16) — the one admitted
/// type whose CPU quantizer IS the `quantize_in_f32` arm — was measured bit-identical to that
/// same f64 border table for every prior in `[0, 1]` (research spike Q2, 4,504,501
/// pairs/prior), which is why it needs no per-type border handling either. So a per-type shape
/// special case would be a divergence, not a fix. Every task that widens the admission list
/// adds its type to the loop below.
#[test]
fn buckets_keeps_the_borders_shape_checks() {
    let wrong_borders = |ctr_type: i8| {
        let mut col = column(ctr_type, 0);
        col.borders = vec![0.5_f64]; // 2 buckets != N_BINS
        col
    };
    let no_members = |ctr_type: i8| {
        let mut col = column(ctr_type, 0);
        col.member_bins = vec![];
        col
    };

    for (ctr_type, name) in [
        (BORDERS, "Borders"),
        (BUCKETS, "Buckets"),
        (BINARIZED_TARGET_MEAN_VALUE, "BinarizedTargetMeanValue"),
        (COUNTER, "Counter"),
    ] {
        assert!(
            !ctr_covered(
                &config_with(wrong_borders(ctr_type), column(ctr_type, 0)),
                N,
                N_BINS
            ),
            "{name}: a column not binarizing to n_bins buckets must still decline \
             (uniform-histogram invariant)"
        );
        assert!(
            !ctr_covered(
                &config_with(column(ctr_type, 0), wrong_borders(ctr_type)),
                N,
                N_BINS
            ),
            "{name}: the AVERAGING half's border-table shape check is unchanged too"
        );
        assert!(
            !ctr_covered(
                &config_with(no_members(ctr_type), column(ctr_type, 0)),
                N,
                N_BINS
            ),
            "{name}: a column with no member bins must still decline"
        );
    }

    // The `target_border_idx = 1` Buckets column — the second of the two columns per
    // `(projection, prior)` — is admitted on shape exactly like its `b = 0` sibling.
    assert!(
        ctr_covered(&config_with(column(BUCKETS, 1), column(BUCKETS, 1)), N, N_BINS),
        "the Buckets b=1 numerator column must be admitted; it shares every shape invariant \
         with its b=0 sibling"
    );
}
