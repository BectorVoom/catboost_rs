//! T27b / SPEC-OH-24 + SPEC-OH-25 — the one-hot channel reaches the resident scorer with
//! the RIGHT VALUES.
//!
//! # Why this lives here and not in `crates/cb-train/tests/` or `crates/cb-backend/tests/`
//!
//! `GpuTrainSession` lives in the PRIVATE `mod session;` of `gpu_runtime`, so neither
//! integration-test directory can see it. A `#[cfg(test)] mod` declared in
//! `gpu_runtime/mod.rs` is a descendant of `gpu_runtime` and reaches
//! `super::GpuTrainSession` (re-exported by `pub use session::*;`) — exactly how
//! `session_residency` and `session_depth_gt1_test` already do it.
//!
//! # The load-bearing assertion
//!
//! `real_folds[c] == 2` for a cardinality-2 one-hot column, NOT `32`.
//!
//! `32` is precisely the value `TCFeature.folds` would supply: the production path packs
//! the cindex with `vec![n_bins_line; eff_n_features]`, and `pack_cindex` copies that
//! argument straight into `folds`. Using it as a one-hot candidate bound would make
//! `border < folds[feature]` the loop bound itself, admitting 30 phantom
//! "all-objects-right" candidates on a 32-wide padded line. Asserting the value HERE, at
//! the seam, localizes such a regression instead of letting it surface as an
//! unattributable ≤1e-5 device-vs-CPU gap in the parity gate.

use cb_compute::{DeviceTrainConfig, EScoreFunction, Loss};

use super::GpuTrainSession;

/// A pool with 1 float column (31 borders → 32 bins, so `n_bins_line == 32`) and 2
/// cardinality-2 one-hot columns. The padded line width and the real cardinality differ
/// maximally, which is what makes a wrong data source visible.
#[test]
fn one_hot_flags_real_folds_and_n_float_reach_the_resident_scorer() {
    if !cfg!(any(feature = "rocm", feature = "cuda")) {
        println!(
            "[T27b] SKIP one_hot_flags_real_folds_and_n_float_reach_the_resident_scorer: the \
             resident session needs Atomic<u64> (rocm/cuda); cpu/wgpu lack it"
        );
        return;
    }

    let n = 64usize;
    let n_float = 1usize;
    let n_features = 3usize; // 1 float + 2 one-hot
    let n_bins = 32usize;
    let depth = 2usize;

    // Feature-major bins: the float column cycles the full 32-bin line; the two cat
    // columns only ever take 0 or 1.
    let mut bins = Vec::with_capacity(n_features * n);
    for i in 0..n {
        bins.push((i % n_bins) as u32);
    }
    for i in 0..n {
        bins.push((i % 2) as u32);
    }
    for i in 0..n {
        bins.push(((i / 2) % 2) as u32);
    }
    let weight = vec![1.0_f64; n];
    let target: Vec<f64> = (0..n).map(|i| (i as f64) - (n as f64) / 2.0).collect();
    let scaled_l2 = cb_compute::scale_l2_reg(3.0, n as f64, n);

    // The config the production host builds for this pool.
    let config = DeviceTrainConfig {
        one_hot_flags: vec![false, true, true],
        real_folds: vec![32, 2, 2],
        n_float,
        ..DeviceTrainConfig::default()
    };
    assert!(
        config.is_covered_regime(),
        "a one-hot pool with no CTR must be a COVERED regime (SPEC-OH-25); \
         `ctr.is_none()` is deliberately NOT relaxed"
    );

    let session = GpuTrainSession::begin(
        &Loss::Rmse,
        depth,
        true, // Plain
        1,    // fold_count
        EScoreFunction::Cosine,
        &bins,
        &weight,
        n,
        n_features,
        n_bins,
        0.3,
        scaled_l2,
        &config,
    )
    .expect("begin must not error on a covered one-hot config")
    .expect("a one-hot pool with no CTR must open a session (SPEC-OH-25)");

    let (flags, real_folds, stored_n_float, feature_lo, feature_hi) = session.one_hot_channel();

    assert_eq!(
        flags,
        vec![false, true, true],
        "the per-feature one-hot flags must reach the session as the host built them"
    );
    assert_eq!(
        real_folds,
        vec![32u32, 2, 2],
        "real_folds must carry the TRUE per-feature cardinality. `real_folds[1] == 32` \
         would mean the PADDED line width (`TCFeature.folds`) was wired in, which bounds \
         nothing and lets 30 phantom candidates per cat column into the argmax"
    );
    assert_eq!(stored_n_float, n_float, "the float boundary must be preserved");
    assert_eq!(
        (feature_lo, feature_hi),
        (1, 3),
        "pass B must sweep exactly the one-hot suffix [n_float, n_features)"
    );

    drop(session);
}

/// A float-only pool must be byte-unchanged: pass B is EMPTY (`feature_lo == feature_hi`),
/// so the scorer makes exactly one launch with today's arguments (SPEC-OH-31).
#[test]
fn float_only_pool_leaves_pass_b_empty() {
    if !cfg!(any(feature = "rocm", feature = "cuda")) {
        println!("[T27b] SKIP float_only_pool_leaves_pass_b_empty: needs rocm/cuda");
        return;
    }

    let n = 64usize;
    let n_features = 2usize;
    let n_bins = 32usize;
    let mut bins = Vec::with_capacity(n_features * n);
    for f in 0..n_features {
        for i in 0..n {
            bins.push(((i + f) % n_bins) as u32);
        }
    }
    let weight = vec![1.0_f64; n];
    let scaled_l2 = cb_compute::scale_l2_reg(3.0, n as f64, n);

    // Exactly what the trainer emits for a float-only device-eligible pool: an all-`false`
    // flag set and a fully populated `real_folds`. `real_folds` is NEVER empty on a real
    // fit — the trainer always routes through the one-hot quantizer.
    let config = DeviceTrainConfig {
        one_hot_flags: vec![false; n_features],
        real_folds: vec![32u32; n_features],
        n_float: n_features,
        ..DeviceTrainConfig::default()
    };

    let session = GpuTrainSession::begin(
        &Loss::Rmse,
        2,
        true,
        1,
        EScoreFunction::Cosine,
        &bins,
        &weight,
        n,
        n_features,
        n_bins,
        0.3,
        scaled_l2,
        &config,
    )
    .expect("begin must not error")
    .expect("a float-only covered config must open a session");

    let (flags, _real_folds, stored_n_float, feature_lo, feature_hi) = session.one_hot_channel();
    assert!(
        flags.iter().all(|&f| !f),
        "a float-only pool must carry no one-hot flag"
    );
    assert_eq!(stored_n_float, n_features);
    assert_eq!(
        feature_lo, feature_hi,
        "pass B must be EMPTY on a float-only pool, so the scorer launches once with \
         today's arguments (SPEC-OH-31)"
    );

    drop(session);
}
