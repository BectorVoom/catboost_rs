//! The wire-up params: `model_size_reg` and the snapshot trio
//! (`save_snapshot` / `snapshot_file` / `snapshot_interval`).
//!
//! Both were IMPLEMENTED in the engine but unreachable from the public API:
//! `model_size_reg`'s weight function existed and the boosting loop pinned it to
//! the default, and `cb_train::train_with_snapshot` existed with no caller.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;
use std::time::Duration;

use catboost_rs::{CatBoostBuilder, IngestSource, OwnedColumns, Pool};
use cb_compute::Loss;

/// A deterministic numeric corpus — no fixture needed, these tests assert
/// reachability and self-consistency rather than upstream numerics.
fn corpus(n: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut cols = vec![Vec::with_capacity(n); 3];
    let mut y = Vec::with_capacity(n);
    let mut state = 12_345_u64;
    for _ in 0..n {
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let a = next();
        let b = next();
        let c = next();
        cols[0].push(a);
        cols[1].push(b);
        cols[2].push(c);
        y.push(2.0 * a - b + 0.5 * c);
    }
    (cols, y)
}

fn pool_of(cols: Vec<Vec<f64>>, target: Vec<f64>) -> Pool {
    OwnedColumns::new(cols, target)
        .into_pool()
        .expect("pool must build")
}

fn builder() -> CatBoostBuilder {
    CatBoostBuilder::new()
        .loss(Loss::Rmse)
        .iterations(5)
        .depth(3)
        .learning_rate(0.3)
        .l2_leaf_reg(3.0)
        .random_strength(0.0)
        .random_seed(0)
        .border_count(32)
        .score_function(cb_compute::EScoreFunction::L2)
        .leaf_method(cb_compute::LeafMethod::Gradient)
}

fn unique_tmp(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("catboost_rs_{tag}_{nanos}.snapshot"))
}

/// `model_size_reg` only penalizes CTR PROJECTIONS, so on a numeric pool it must
/// be inert — changing it there would mean it had leaked into the float path.
#[test]
fn model_size_reg_is_inert_on_a_numeric_pool() {
    let (cols, y) = corpus(200);
    let baseline = builder()
        .fit(&pool_of(cols.clone(), y.clone()))
        .expect("fit")
        .predict(&pool_of(cols.clone(), y.clone()))
        .expect("predict");
    for reg in [0.0, 0.5, 2.0] {
        let preds = builder()
            .model_size_reg(reg)
            .fit(&pool_of(cols.clone(), y.clone()))
            .expect("fit")
            .predict(&pool_of(cols.clone(), y.clone()))
            .expect("predict");
        assert_eq!(
            preds, baseline,
            "model_size_reg={reg} must be inert without categorical features"
        );
    }
}

/// Snapshotting round-trips: a snapshotted fit produces the SAME model as an
/// unsnapshotted one, and leaves a checkpoint file behind.
#[test]
fn a_snapshotted_fit_matches_an_unsnapshotted_one_and_writes_the_file() {
    let (cols, y) = corpus(200);
    let plain = builder()
        .fit(&pool_of(cols.clone(), y.clone()))
        .expect("fit")
        .predict(&pool_of(cols.clone(), y.clone()))
        .expect("predict");

    let path = unique_tmp("wireup_roundtrip");
    let snapped = builder()
        .save_snapshot(true)
        .snapshot_file(path.clone())
        // Zero interval => write on every completed iteration.
        .snapshot_interval(Duration::from_secs(0))
        .fit(&pool_of(cols.clone(), y.clone()))
        .expect("a snapshotted fit must succeed")
        .predict(&pool_of(cols.clone(), y.clone()))
        .expect("predict");

    assert_eq!(
        plain, snapped,
        "checkpointing must not change the trained model"
    );
    assert!(path.is_file(), "the checkpoint file must exist after the fit");
    let _ = std::fs::remove_file(&path);
}

/// Resuming from a completed checkpoint reproduces the same model rather than
/// re-training from scratch into something different.
#[test]
fn resuming_from_a_checkpoint_reproduces_the_model() {
    let (cols, y) = corpus(200);
    let path = unique_tmp("wireup_resume");

    let first = builder()
        .save_snapshot(true)
        .snapshot_file(path.clone())
        .snapshot_interval(Duration::from_secs(0))
        .fit(&pool_of(cols.clone(), y.clone()))
        .expect("first fit")
        .predict(&pool_of(cols.clone(), y.clone()))
        .expect("predict");

    // Second fit sees the existing checkpoint and resumes from it.
    let second = builder()
        .save_snapshot(true)
        .snapshot_file(path.clone())
        .snapshot_interval(Duration::from_secs(0))
        .fit(&pool_of(cols.clone(), y.clone()))
        .expect("resumed fit")
        .predict(&pool_of(cols.clone(), y.clone()))
        .expect("predict");

    assert_eq!(first, second, "a resumed fit must reproduce the model");
    let _ = std::fs::remove_file(&path);
}

/// `save_snapshot` without a path is a configuration error, not a silent no-op.
#[test]
fn save_snapshot_without_a_file_is_rejected() {
    let (cols, y) = corpus(50);
    let err = builder()
        .save_snapshot(true)
        .fit(&pool_of(cols, y))
        .expect_err("save_snapshot without snapshot_file must be rejected");
    assert!(err.to_string().contains("snapshot_file"));
}

/// Snapshotting is refused on a CATEGORICAL fit rather than silently training
/// without checkpoints — that would drop the durability the caller asked for.
#[test]
fn save_snapshot_is_refused_on_a_categorical_pool() {
    let (cols, y) = corpus(100);
    let cats: Vec<String> = (0..100).map(|i| format!("c{}", i % 7)).collect();
    let pool = OwnedColumns::new(cols, y)
        .with_cat_features(vec![cats])
        .into_pool()
        .expect("pool");
    let err = builder()
        .save_snapshot(true)
        .snapshot_file(unique_tmp("wireup_cat"))
        .fit(&pool)
        .expect_err("snapshotting a categorical fit must be refused");
    let msg = err.to_string();
    assert!(
        msg.contains("save_snapshot") && msg.contains("categorical"),
        "the refusal must name the parameter and the cause; got: {msg}"
    );
}

/// Not enabling snapshotting leaves the fit path untouched.
#[test]
fn snapshot_params_are_inert_when_disabled() {
    let (cols, y) = corpus(150);
    let baseline = builder()
        .fit(&pool_of(cols.clone(), y.clone()))
        .expect("fit")
        .predict(&pool_of(cols.clone(), y.clone()))
        .expect("predict");
    // A path and interval set, but `save_snapshot` left off.
    let path = unique_tmp("wireup_disabled");
    let preds = builder()
        .snapshot_file(path.clone())
        .snapshot_interval(Duration::from_secs(0))
        .fit(&pool_of(cols.clone(), y.clone()))
        .expect("fit")
        .predict(&pool_of(cols, y))
        .expect("predict");
    assert_eq!(preds, baseline, "snapshot params must be inert when disabled");
    assert!(
        !path.exists(),
        "no checkpoint may be written when save_snapshot is off"
    );
}
