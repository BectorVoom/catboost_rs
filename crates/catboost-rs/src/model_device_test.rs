//! GINF-01-S5 facade tests for [`crate::Model::predict_raw_on_device`].
//!
//! Mounted at the crate root via `#[cfg(test)] mod model_device_test;`, mirroring
//! `model_sum_test.rs` / `error_test.rs`'s in-crate `#[cfg(test)]`-module
//! precedent (`crates/catboost-rs/src/lib.rs`). Uses
//! [`crate::Model::from_canonical`] (`pub(crate)`), so this MUST live inside the
//! crate rather than under `tests/` (same rationale as `model_sum_test.rs`).
//!
//! The numeric oracle is the shipped CPU apply `cb_model::predict_raw` (D-04,
//! read-only). Parity runs under the DEFAULT `cpu` backend (CubeCL `CpuRuntime`,
//! f64) in ordinary `cargo test`; the bound is `SCORE_BOUND` (`1e-9` for f64
//! backends). Device accumulation is documented as within-`SCORE_BOUND` of the
//! order-locked `sum_f64`, NOT bit-exact (SPEC §9 R1 / D-08).

use crate::Model;

/// `SCORE_BOUND` under the default `cpu` (f64) backend — the project report
/// convention (`score_split.rs:70-73`; `1e-3` only under wgpu-f32).
const SCORE_BOUND: f64 = 1e-9;

/// A 2-tree, depth-2, float-only oblivious SCALAR model over 2 float features,
/// with distinct borders, distinct leaf values, and a nonzero bias — a fixture
/// exercising both trees and multiple leaves per tree.
fn sample_model() -> cb_model::Model {
    cb_model::Model::new(
        vec![
                cb_model::ObliviousTree {
                    splits: vec![
                        cb_model::ModelSplit::Float(cb_model::Split { feature: 0, border: 0.5 }),
                        cb_model::ModelSplit::Float(cb_model::Split { feature: 1, border: 0.3 }),
                    ],
                    leaf_values: vec![0.1, -0.2, 0.7, 1.3],
                    leaf_weights: vec![1.0, 1.0, 1.0, 1.0],
                },
                cb_model::ObliviousTree {
                    splits: vec![
                        cb_model::ModelSplit::Float(cb_model::Split { feature: 1, border: 0.7 }),
                        cb_model::ModelSplit::Float(cb_model::Split { feature: 0, border: 0.2 }),
                    ],
                    leaf_values: vec![-0.5, 0.4, 0.9, -1.1],
                    leaf_weights: vec![1.0, 1.0, 1.0, 1.0],
                },
            ],
        0.375,
        vec![vec![0.2, 0.5], vec![0.3, 0.7]],
    )
}

/// The same structure but multi-dimensional (`approx_dimension = 2`) — an
/// unsupported model the device-apply guard rejects (`MultiDimensional`).
fn multiclass_model() -> cb_model::Model {
    let mut model = sample_model();
    model.approx_dimension = 2;
    model
}

/// A float-only oblivious scalar model whose SECOND split references float
/// feature index 3 — a HIGH index. Applied via `predict_raw_on_device` with FEWER
/// than 4 feature columns, feature 3 is absent, so BOTH the CPU `predict_raw`
/// (checked `.get` → `false`) and the device kernel (`f >= n_features` guard →
/// bit 0) treat that split as bit 0. Exercises the Finding #1 kernel guard.
fn high_feature_model() -> cb_model::Model {
    cb_model::Model::new(
        vec![cb_model::ObliviousTree {
                splits: vec![
                    cb_model::ModelSplit::Float(cb_model::Split { feature: 0, border: 0.5 }),
                    // Feature index 3 — deliberately beyond the 2 columns supplied below.
                    cb_model::ModelSplit::Float(cb_model::Split { feature: 3, border: -1.0 }),
                ],
                leaf_values: vec![0.1, -0.2, 0.7, 1.3],
                leaf_weights: vec![1.0, 1.0, 1.0, 1.0],
            }],
        0.375,
        // Borders for features 0..=3; the inner vecs' contents are irrelevant to
        // apply (splits carry their own borders), only the index range matters.
        vec![vec![0.5], vec![], vec![], vec![-1.0]],
    )
}

/// Element-wise max absolute difference between two equal-length prediction
/// vectors (also asserts equal length).
fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "prediction vectors differ in length: {} vs {}", a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// AT-S5a: `predict_raw_on_device` matches the shipped CPU `predict_raw`
/// element-wise within `SCORE_BOUND` on a well-formed float batch.
#[test]
fn predict_on_device_matches_cpu() {
    let model = Model::from_canonical(sample_model());
    // 2 feature columns of 4 objects; values straddle every split border.
    let features: Vec<Vec<f32>> =
        vec![vec![0.1, 0.6, 0.9, 0.05], vec![0.2, 0.8, 0.1, 0.5]];

    let cpu = cb_model::predict_raw(model.as_canonical(), &features);
    let device = model
        .predict_raw_on_device(&features)
        .expect("device apply succeeds for a supported model");

    let diff = max_abs_diff(&cpu, &device);
    println!("predict_on_device_matches_cpu: max|diff| = {diff:e}");
    assert!(diff <= SCORE_BOUND, "device vs CPU max|diff| {diff:e} exceeds {SCORE_BOUND:e}");
}

/// GINF-01 CR (Finding #1): a float-only oblivious model that splits on a feature
/// index BEYOND the supplied columns matches the CPU `predict_raw` element-wise —
/// both treat the missing feature as absent (bit 0), and the device kernel must NOT
/// read out of bounds. `high_feature_model` splits on feature 3 (border `-1.0`); we
/// supply only 2 columns, so an UNGUARDED kernel would OOB-read (~0.0) and evaluate
/// `~0.0 > -1.0` → bit 1, diverging from the CPU's bit 0.
#[test]
fn predict_on_device_matches_cpu_missing_feature_column() {
    let model = Model::from_canonical(high_feature_model());
    // Only 2 feature columns of 3 objects; the model splits on feature index 3.
    let features: Vec<Vec<f32>> = vec![vec![0.1, 0.6, 0.9], vec![0.2, 0.8, 0.1]];

    let cpu = cb_model::predict_raw(model.as_canonical(), &features);
    let device = model
        .predict_raw_on_device(&features)
        .expect("device apply succeeds for a model referencing an absent feature");

    let diff = max_abs_diff(&cpu, &device);
    println!("missing-feature-column: n={} max|diff| = {diff:e}", cpu.len());
    assert_eq!(cpu.len(), 3, "n_objects is the first column length (3)");
    assert!(
        diff <= SCORE_BOUND,
        "missing-feature-column device vs CPU max|diff| {diff:e} exceeds {SCORE_BOUND:e}"
    );
}

/// AT-S5b: an unsupported (multi-dimensional) model is rejected with a typed
/// [`crate::CatBoostError`], never a panic and never a silent wrong result.
#[test]
fn predict_on_device_rejects_unsupported() {
    let model = Model::from_canonical(multiclass_model());
    let features: Vec<Vec<f32>> = vec![vec![0.1, 0.6], vec![0.2, 0.8]];

    let result = model.predict_raw_on_device(&features);
    assert!(result.is_err(), "multi-dimensional model must be rejected");
}

/// AT-S5c: ragged (unequal-length) feature columns match the CPU
/// first-column-governs-`n_objects` behavior on BOTH a shorter-later-column
/// (NaN-pad) config and a longer-later-column (truncation) config.
#[test]
fn predict_on_device_matches_cpu_ragged_columns() {
    let model = Model::from_canonical(sample_model());

    // (a) A LATER column SHORTER than the first: n_objects = 3, column 1 is
    // NaN-padded for objects 1..3 on BOTH the CPU and device paths. Exercises
    // the device kernel's `v > b` against a NaN operand (IEEE unordered → false).
    let shorter_later: Vec<Vec<f32>> = vec![vec![0.1, 0.6, 0.9], vec![0.2]];
    let cpu_a = cb_model::predict_raw(model.as_canonical(), &shorter_later);
    let device_a = model
        .predict_raw_on_device(&shorter_later)
        .expect("device apply (shorter later column)");
    let diff_a = max_abs_diff(&cpu_a, &device_a);
    println!("ragged shorter-later: n={} max|diff| = {diff_a:e}", cpu_a.len());
    assert_eq!(cpu_a.len(), 3, "n_objects is the FIRST column length (3)");
    assert!(diff_a <= SCORE_BOUND, "shorter-later max|diff| {diff_a:e} exceeds {SCORE_BOUND:e}");

    // (b) A LATER column LONGER than the first: n_objects = 2 (first column),
    // column 1's tail is truncated (never read past object 2) — the ONLY config
    // that distinguishes first-column-governs from a wrong max-over-columns rule.
    let longer_later: Vec<Vec<f32>> = vec![vec![0.1, 0.6], vec![0.2, 0.8, 0.9, 0.5]];
    let cpu_b = cb_model::predict_raw(model.as_canonical(), &longer_later);
    let device_b = model
        .predict_raw_on_device(&longer_later)
        .expect("device apply (longer later column)");
    let diff_b = max_abs_diff(&cpu_b, &device_b);
    println!("ragged longer-later: n={} max|diff| = {diff_b:e}", cpu_b.len());
    assert_eq!(cpu_b.len(), 2, "n_objects is the FIRST column length (2), NOT max-over-columns");
    assert!(diff_b <= SCORE_BOUND, "longer-later max|diff| {diff_b:e} exceeds {SCORE_BOUND:e}");
}
