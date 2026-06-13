//! Owned-`Vec` ingestion primitive (D-04): the trivial [`IngestSource`] used by
//! the Builder API and by `.npy` fixture loading.
//!
//! [`OwnedColumns`] gathers owned column buffers, then [`OwnedColumns::into_pool`]
//! validates that every supplied column has the same length (`n_rows`) before a
//! [`Pool`] exists. Any mismatch is a typed [`cb_core::CbError`] — never a panic
//! and never an out-of-bounds index (threats T-02-04 / T-02-05). No `unwrap` /
//! `expect` / `panic` / `[]`-indexing appears in this module (Shared Pattern C).

use cb_core::{CbError, CbResult};

use crate::ingest::IngestSource;
use crate::pool::{Pair, Pool};

/// Owned column inputs for building a [`Pool`].
///
/// Construct with [`OwnedColumns::new`] (float features + label only — the
/// minimal supervised case) and attach optional columns with the `with_*`
/// builder methods. Feature columns are Structure-of-Arrays: one inner `Vec`
/// per feature, each of length `n_rows`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OwnedColumns {
    float_features: Vec<Vec<f64>>,
    cat_features: Vec<Vec<String>>,
    text_features: Vec<Vec<String>>,
    embedding_features: Vec<Vec<Vec<f32>>>,
    label: Vec<f64>,
    weights: Vec<f64>,
    group_id: Vec<u64>,
    subgroup_id: Vec<u64>,
    pairs: Vec<Pair>,
    baseline: Vec<f64>,
}

impl OwnedColumns {
    /// Start from the minimal supervised case: float feature columns (SoA) plus
    /// a label. Optional columns default to empty and are attached via the
    /// `with_*` methods.
    #[must_use]
    pub fn new(float_features: Vec<Vec<f64>>, label: Vec<f64>) -> Self {
        Self {
            float_features,
            label,
            ..Self::default()
        }
    }

    /// Attach categorical feature columns (SoA, raw owned strings).
    #[must_use]
    pub fn with_cat_features(mut self, cat_features: Vec<Vec<String>>) -> Self {
        self.cat_features = cat_features;
        self
    }

    /// Attach text feature columns (SoA, raw owned strings).
    #[must_use]
    pub fn with_text_features(mut self, text_features: Vec<Vec<String>>) -> Self {
        self.text_features = text_features;
        self
    }

    /// Attach embedding feature columns (`[feature][row][dim]`).
    #[must_use]
    pub fn with_embedding_features(mut self, embedding_features: Vec<Vec<Vec<f32>>>) -> Self {
        self.embedding_features = embedding_features;
        self
    }

    /// Attach per-object weights.
    #[must_use]
    pub fn with_weights(mut self, weights: Vec<f64>) -> Self {
        self.weights = weights;
        self
    }

    /// Attach per-object group ids (grouped / ranking data).
    #[must_use]
    pub fn with_group_id(mut self, group_id: Vec<u64>) -> Self {
        self.group_id = group_id;
        self
    }

    /// Attach per-object subgroup ids.
    #[must_use]
    pub fn with_subgroup_id(mut self, subgroup_id: Vec<u64>) -> Self {
        self.subgroup_id = subgroup_id;
        self
    }

    /// Attach ranking pairs.
    #[must_use]
    pub fn with_pairs(mut self, pairs: Vec<Pair>) -> Self {
        self.pairs = pairs;
        self
    }

    /// Attach per-object baseline values.
    #[must_use]
    pub fn with_baseline(mut self, baseline: Vec<f64>) -> Self {
        self.baseline = baseline;
        self
    }

    /// The reference object count (`n_rows`) every column must match.
    ///
    /// Derived from the first non-empty signal in a fixed precedence order
    /// (float feature 0, then label) so that an all-empty `Pool` (no features,
    /// no label) is a legitimate zero-row dataset rather than an error.
    fn reference_n_rows(&self) -> usize {
        if let Some(first) = self.float_features.first() {
            return first.len();
        }
        self.label.len()
    }
}

/// Returns a [`CbError::LengthMismatch`] if `actual != expected`, naming the
/// offending column. Centralizes the length check so no call site indexes or
/// panics. Uses the dedicated `LengthMismatch` variant (matching the Arrow path)
/// so callers can `match` on a shape mismatch distinctly from any other range
/// violation (WR-04).
fn check_len(name: &str, expected: usize, actual: usize) -> CbResult<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(CbError::LengthMismatch {
            column: name.to_string(),
            expected,
            actual,
        })
    }
}

/// Returns an `OutOfRange` error if a ranking pair's row id is not a valid
/// object index (`id >= n_rows`), naming the offending pair and field. An
/// `n_rows == 0` dataset rejects any pair (no valid object id exists).
fn check_pair_id(field: &str, pair_index: usize, id: u32, n_rows: usize) -> CbResult<()> {
    if (id as usize) < n_rows {
        Ok(())
    } else {
        Err(CbError::OutOfRange(format!(
            "pair[{pair_index}].{field} = {id} is out of range for n_rows {n_rows}"
        )))
    }
}

impl IngestSource for OwnedColumns {
    fn into_pool(self) -> CbResult<Pool> {
        let n_rows = self.reference_n_rows();

        // Every feature column (all kinds) and every metadata column that is
        // present must agree with `n_rows`. Empty metadata columns are the
        // "not supplied" sentinel and are skipped.
        for (index, col) in self.float_features.iter().enumerate() {
            check_len(&format!("float_feature[{index}]"), n_rows, col.len())?;
        }
        for (index, col) in self.cat_features.iter().enumerate() {
            check_len(&format!("cat_feature[{index}]"), n_rows, col.len())?;
        }
        for (index, col) in self.text_features.iter().enumerate() {
            check_len(&format!("text_feature[{index}]"), n_rows, col.len())?;
        }
        for (index, col) in self.embedding_features.iter().enumerate() {
            check_len(&format!("embedding_feature[{index}]"), n_rows, col.len())?;
        }
        if !self.label.is_empty() {
            check_len("label", n_rows, self.label.len())?;
        }
        if !self.weights.is_empty() {
            check_len("weights", n_rows, self.weights.len())?;
        }
        if !self.group_id.is_empty() {
            check_len("group_id", n_rows, self.group_id.len())?;
        }
        if !self.subgroup_id.is_empty() {
            check_len("subgroup_id", n_rows, self.subgroup_id.len())?;
        }
        if !self.baseline.is_empty() {
            check_len("baseline", n_rows, self.baseline.len())?;
        }

        // Ranking pairs reference object (row) indices; an id beyond `n_rows`
        // is a latent out-of-bounds for any downstream code that indexes
        // objects by pair id (threats T-02-04 / T-02-05). Validate every pair so
        // the `Pool` length-consistency guarantee actually holds for pairs (WR-02).
        for (index, pair) in self.pairs.iter().enumerate() {
            check_pair_id("winner_id", index, pair.winner_id, n_rows)?;
            check_pair_id("loser_id", index, pair.loser_id, n_rows)?;
        }

        Ok(Pool::from_validated_columns(
            n_rows,
            self.float_features,
            self.cat_features,
            self.text_features,
            self.embedding_features,
            self.label,
            self.weights,
            self.group_id,
            self.subgroup_id,
            self.pairs,
            self.baseline,
        ))
    }
}
