//! GINF-01-S6 throughput benchmark (NON-GATING): time device vs CPU apply over a
//! large synthetic float-only oblivious model + batch, print wall times (and, when
//! `CB_GPU_PROF` is set, a per-stage attribution), and assert element-wise parity
//! within `SCORE_BOUND` as the correctness guard.
//!
//! This is `#[ignore]` by design (SPEC §6 AT-S6): it does NOT run in the default
//! `cargo test` set. Run it on demand with
//! `cargo test -p catboost-rs --test predict_on_device_bench_test -- --ignored`.
//! The numeric oracle is the shipped CPU `cb_model::predict_raw` (D-04, read-only);
//! device accumulation is within-`SCORE_BOUND` of the order-locked `sum_f64`, NOT
//! bit-exact (D-08).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::time::Instant;

use catboost_rs::{CatBoostBuilder, IngestSource, Loss, Model, OwnedColumns, Pool};

/// `SCORE_BOUND` under the default `cpu` (f64) backend (score_split.rs:70-73).
const SCORE_BOUND: f64 = 1e-9;

const N_FEATURES: usize = 6;

/// Deterministic pseudo-random f64 in `[0, 1)` from a 64-bit SplitMix state — no
/// `rand` dependency, reproducible across runs.
fn next_unit(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    // Top 53 bits → [0, 1).
    ((z >> 11) as f64) / ((1u64 << 53) as f64)
}

/// Generate `n_objects` per-feature `f64` columns and an RMSE target that is a
/// noisy linear combination of the features (so trees actually split).
fn synth_columns(n_objects: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut state = seed;
    let mut columns: Vec<Vec<f64>> = vec![Vec::with_capacity(n_objects); N_FEATURES];
    let mut label = Vec::with_capacity(n_objects);
    for _ in 0..n_objects {
        let mut y = 0.0_f64;
        for (f, column) in columns.iter_mut().enumerate() {
            let v = next_unit(&mut state);
            column.push(v);
            y += ((f + 1) as f64) * v;
        }
        // A little noise so the target is not perfectly separable.
        y += 0.1 * (next_unit(&mut state) - 0.5);
        label.push(y);
    }
    (columns, label)
}

/// Train a large synthetic float-only oblivious RMSE regressor through the public
/// facade (no `from_canonical`, which is `pub(crate)`).
fn train_large_model() -> Model {
    let (columns, label) = synth_columns(4_000, 0x1234_5678);
    let pool: Pool = OwnedColumns::new(columns, label)
        .into_pool()
        .expect("synthetic training pool builds");
    CatBoostBuilder::new()
        .loss(Loss::Rmse)
        .iterations(200)
        .depth(6)
        .learning_rate(0.1)
        .random_seed(0)
        .fit(&pool)
        .expect("synthetic RMSE model trains")
}

/// AT-S6: large-batch device==CPU parity guard + printed timings.
#[test]
#[ignore = "throughput benchmark; run explicitly with -- --ignored"]
fn bench_predict_on_device() {
    let profile = std::env::var("CB_GPU_PROF").is_ok();

    let t_train = Instant::now();
    let model = train_large_model();
    let train_ms = t_train.elapsed().as_secs_f64() * 1e3;

    // Large apply batch: many objects, N_FEATURES columns of f32.
    let n_apply = 200_000usize;
    let (apply_cols_f64, _) = synth_columns(n_apply, 0xABCD_EF01);
    let features: Vec<Vec<f32>> = apply_cols_f64
        .iter()
        .map(|col| col.iter().map(|&v| v as f32).collect())
        .collect();

    // CPU oracle (shipped predict_raw, D-04).
    let t_cpu = Instant::now();
    let cpu = cb_model::predict_raw(model.as_canonical(), &features);
    let cpu_ms = t_cpu.elapsed().as_secs_f64() * 1e3;

    // Device apply.
    let t_dev = Instant::now();
    let device = model
        .predict_raw_on_device(&features)
        .expect("device apply succeeds on the large synthetic model");
    let dev_ms = t_dev.elapsed().as_secs_f64() * 1e3;

    // Parity guard (the actual gate; timing is informational).
    assert_eq!(cpu.len(), device.len(), "device / CPU length mismatch");
    let max_diff = cpu
        .iter()
        .zip(device.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    println!("--- GINF-01-S6 predict_on_device bench ---");
    println!("objects            : {n_apply}");
    println!("features           : {N_FEATURES}");
    println!("trees              : {}", model.as_canonical().oblivious_trees.len());
    println!("train wall (once)  : {train_ms:.2} ms");
    println!("CPU predict_raw    : {cpu_ms:.2} ms");
    println!("device predict     : {dev_ms:.2} ms");
    println!("device/CPU ratio   : {:.3}x", dev_ms / cpu_ms.max(f64::MIN_POSITIVE));
    println!("parity max|diff|   : {max_diff:e} (bound {SCORE_BOUND:e})");
    if profile {
        // Per-stage attribution (CB_GPU_PROF style): re-time the CPU/device legs
        // in isolation so upload+launch+read-back dominate the device figure.
        let t = Instant::now();
        let _ = Model::predict_raw_on_device(&model, &features);
        println!(
            "[CB_GPU_PROF] device apply (guard+flatten+marshal+launch+readback): {:.2} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    assert!(
        max_diff <= SCORE_BOUND,
        "device vs CPU parity max|diff| {max_diff:e} exceeds SCORE_BOUND {SCORE_BOUND:e}"
    );
}
