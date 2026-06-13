//! [`Pool`] — the owned in-memory dataset, mirroring upstream CatBoost's
//! `TRawObjectsDataProvider` shape (DATA-01): float / categorical / text /
//! embedding feature columns plus the full target-side metadata (`label`,
//! `weights`, `group_id`, `subgroup_id`, `pairs`, `baseline`).
//!
//! # Owned now, zero-copy seam later (D-02)
//!
//! Every column is an owned `Vec` — there is **no** lifetime generic and **no**
//! `Cow`. A borrowed / zero-copy view is introduced at Phase 8 by adding a new
//! [`crate::ingest::IngestSource`] implementation, not by reshaping `Pool`
//! itself. Keeping `Pool` lifetime-free here is the deliberate D-02 decision.
//!
//! # SoA float layout (D-12-consistent)
//!
//! Float features are stored Structure-of-Arrays: one `Vec<f64>` per feature,
//! each of length `n_rows`. This is the layout the quantizer
//! ([`crate::select_borders_greedy_logsum`]) consumes column-by-column.

/// A pair `(winner, loser)` for ranking/pairwise objectives, identified by
/// object (row) index. Mirrors upstream `TPair` (winner ranked above loser).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pair {
    /// Row index of the object that should rank higher.
    pub winner_id: u32,
    /// Row index of the object that should rank lower.
    pub loser_id: u32,
}

/// The owned dataset: feature columns (by kind) + target-side metadata.
///
/// Construct through the ingestion seam — see
/// [`crate::ingest::OwnedColumns::into_pool`] — which validates that every
/// supplied column has the same length before a `Pool` exists. A `Pool`
/// obtained that way is guaranteed internally length-consistent.
#[derive(Debug, Clone, PartialEq)]
pub struct Pool {
    /// Number of objects (rows).
    n_rows: usize,
    /// Float features, Structure-of-Arrays: `float_features[f][row]`. Each inner
    /// `Vec` has length `n_rows`.
    float_features: Vec<Vec<f64>>,
    /// Categorical features as raw owned strings (hashing happens in the
    /// cat-hash plan): `cat_features[f][row]`. Each inner `Vec` has length
    /// `n_rows`.
    cat_features: Vec<Vec<String>>,
    /// Text features as raw owned strings: `text_features[f][row]`. Each inner
    /// `Vec` has length `n_rows`.
    text_features: Vec<Vec<String>>,
    /// Embedding features: one dense `Vec<f32>` per object per feature
    /// (`embedding_features[f][row]`). Each inner `Vec` has length `n_rows`.
    embedding_features: Vec<Vec<Vec<f32>>>,
    /// Target / label, one value per object (empty if unsupervised).
    label: Vec<f64>,
    /// Per-object weight (empty when unweighted; callers treat empty as
    /// all-ones).
    weights: Vec<f64>,
    /// Group id per object for grouped (ranking) data (empty when ungrouped).
    group_id: Vec<u64>,
    /// Subgroup id per object (empty when absent).
    subgroup_id: Vec<u64>,
    /// Ranking pairs (empty for non-pairwise data).
    pairs: Vec<Pair>,
    /// Baseline (prior prediction) per object (empty when absent).
    baseline: Vec<f64>,
}

impl Pool {
    /// Construct a `Pool` from already-validated owned columns.
    ///
    /// This is the single private constructor the ingestion seam funnels
    /// through; it performs no validation itself (the caller —
    /// [`crate::ingest::OwnedColumns::into_pool`] — has already checked every
    /// length), it merely moves the owned buffers into place.
    // This is a purely internal funnel that moves the already-validated owned
    // buffers into place; the column kinds are intrinsically many, so the arg
    // count is inherent rather than a design smell (the public surface is the
    // builder on `OwnedColumns`).
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub(crate) fn from_validated_columns(
        n_rows: usize,
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
    ) -> Self {
        Self {
            n_rows,
            float_features,
            cat_features,
            text_features,
            embedding_features,
            label,
            weights,
            group_id,
            subgroup_id,
            pairs,
            baseline,
        }
    }

    /// Number of objects (rows) in the dataset.
    #[must_use]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of float feature columns.
    #[must_use]
    pub fn n_float_features(&self) -> usize {
        self.float_features.len()
    }

    /// Number of categorical feature columns.
    #[must_use]
    pub fn n_cat_features(&self) -> usize {
        self.cat_features.len()
    }

    /// Number of text feature columns.
    #[must_use]
    pub fn n_text_features(&self) -> usize {
        self.text_features.len()
    }

    /// Number of embedding feature columns.
    #[must_use]
    pub fn n_embedding_features(&self) -> usize {
        self.embedding_features.len()
    }

    /// All float feature columns (SoA): `[feature][row]`.
    #[must_use]
    pub fn float_features(&self) -> &[Vec<f64>] {
        &self.float_features
    }

    /// The `index`-th float feature column, or `None` if out of range.
    #[must_use]
    pub fn float_feature(&self, index: usize) -> Option<&[f64]> {
        self.float_features.get(index).map(Vec::as_slice)
    }

    /// All categorical feature columns: `[feature][row]`.
    #[must_use]
    pub fn cat_features(&self) -> &[Vec<String>] {
        &self.cat_features
    }

    /// All text feature columns: `[feature][row]`.
    #[must_use]
    pub fn text_features(&self) -> &[Vec<String>] {
        &self.text_features
    }

    /// All embedding feature columns: `[feature][row][dim]`.
    #[must_use]
    pub fn embedding_features(&self) -> &[Vec<Vec<f32>>] {
        &self.embedding_features
    }

    /// Per-object label (empty when unsupervised).
    #[must_use]
    pub fn label(&self) -> &[f64] {
        &self.label
    }

    /// Per-object weight (empty when unweighted).
    #[must_use]
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Per-object group id (empty when ungrouped).
    #[must_use]
    pub fn group_id(&self) -> &[u64] {
        &self.group_id
    }

    /// Per-object subgroup id (empty when absent).
    #[must_use]
    pub fn subgroup_id(&self) -> &[u64] {
        &self.subgroup_id
    }

    /// Ranking pairs (empty for non-pairwise data).
    #[must_use]
    pub fn pairs(&self) -> &[Pair] {
        &self.pairs
    }

    /// Per-object baseline (empty when absent).
    #[must_use]
    pub fn baseline(&self) -> &[f64] {
        &self.baseline
    }
}
