//! Fixture loaders: the `.npy` (binary f64 arrays) + `config.json` (metadata)
//! halves of the hybrid fixture format (D-09).
//!
//! `ndarray-npy`'s `read_npy` returns an error (not a panic) on a dtype/shape
//! mismatch, which is the desired guard against malformed fixtures (Pitfall 3,
//! T-01-01).

use std::path::Path;

use ndarray::Array1;
use ndarray_npy::read_npy;
use serde::Deserialize;

use crate::error::OracleError;

/// Reads a 1-D f64 `.npy` fixture into a `Vec<f64>`.
///
/// # Errors
/// [`OracleError::Npy`] if the file is missing, has the wrong dtype/shape, or is
/// otherwise unreadable.
pub fn load_f64_vec(path: &Path) -> Result<Vec<f64>, OracleError> {
    let arr: Array1<f64> = read_npy(path)?;
    Ok(arr.to_vec())
}

/// Metadata half of a fixture (`config.json`): pinned seed, oracle version, and
/// thread count (always 1 for determinism, D-12).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct FixtureConfig {
    /// Seed used to generate the fixture.
    pub seed: u64,
    /// Pinned CatBoost oracle version (e.g. `"1.2.10"`).
    pub catboost_version: String,
    /// Thread count used during generation (pinned to 1 for determinism).
    pub thread_count: u32,
}

/// Parses a fixture's `config.json` into a [`FixtureConfig`].
///
/// # Errors
/// [`OracleError::Io`] if the file cannot be read; [`OracleError::Json`] if it
/// cannot be parsed.
pub fn load_config(path: &Path) -> Result<FixtureConfig, OracleError> {
    let contents = std::fs::read_to_string(path)?;
    let config = serde_json::from_str(&contents)?;
    Ok(config)
}
