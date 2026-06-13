//! Unit tests for the [`crate::fixture`] loaders. Loads the committed skeleton
//! fixture from disk. Dedicated `*_test.rs` file per D-17.

use std::path::Path;

use crate::error::OracleError;
use crate::fixture::{load_config, load_f64_vec};

/// Canonical skeleton values written by `src/bin/write_skeleton.rs`.
const SKELETON_VALUES: [f64; 5] = [0.0, 0.25, -1.5, 3.14159, 2.71828];

fn fixtures_dir() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn load_f64_vec_reads_committed_skeleton_bit_exactly() {
    let path = fixtures_dir().join("fixtures/skeleton/predictions.npy");
    let v = load_f64_vec(&path).expect("read skeleton predictions.npy");
    assert_eq!(v, SKELETON_VALUES.to_vec());
}

#[test]
fn load_f64_vec_errors_on_missing_file() {
    let path = fixtures_dir().join("fixtures/skeleton/does_not_exist.npy");
    match load_f64_vec(&path) {
        Err(OracleError::Npy(_)) => {}
        other => panic!("expected Npy error for missing file, got {other:?}"),
    }
}

#[test]
fn load_config_parses_skeleton_metadata() {
    let path = fixtures_dir().join("fixtures/skeleton/config.json");
    let cfg = load_config(&path).expect("parse skeleton config.json");
    assert_eq!(cfg.seed, 0);
    assert_eq!(cfg.catboost_version, "1.2.10");
    assert_eq!(cfg.thread_count, 1);
}

#[test]
fn load_config_errors_on_missing_file() {
    let path = fixtures_dir().join("fixtures/skeleton/does_not_exist.json");
    match load_config(&path) {
        Err(OracleError::Io(_)) => {}
        other => panic!("expected Io error for missing config, got {other:?}"),
    }
}
