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
/// supplied column has the same length, and that every ranking [`Pair`]'s
/// `winner_id`/`loser_id` is a valid object index (`< n_rows`), before a `Pool`
/// exists. A `Pool` obtained that way is guaranteed internally length-consistent
/// and pair-index-consistent.
#[derive(Debug, Clone, PartialEq)]
pub struct Pool {
    /// Number of objects (rows).
    n_rows: usize,
    /// Float features, Structure-of-Arrays: `float_features[f][row]`. Each inner
    /// `Vec` has length `n_rows`.
    float_features: Vec<Vec<f64>>,
    /// OPTIONAL f32 narrowing cache of `float_features` (empty = absent), SoA with
    /// the same shape. Attached by ingestion sources whose input was ALREADY f32
    /// (the Python NumPy path), where `f64::from(v) as f32 == v` bit-exactly — so
    /// a consumer that needs the f32 storage view (fit-prep) can skip the full
    /// re-narrowing pass (SPD-03 wave 3: that pass was a top host term at 1M×50).
    /// Never load-bearing: every consumer must fall back to narrowing
    /// `float_features` itself when this is empty or its shape disagrees.
    float_features_f32: Vec<Vec<f32>>,
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
            float_features_f32: Vec::new(),
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

    /// Attach the f32 narrowing cache (see the field doc). Called by the
    /// ingestion seam AFTER shape validation; a wrong-shape cache would be
    /// silently ignored by consumers (they re-narrow), never wrong data.
    pub(crate) fn set_float_f32_cache(&mut self, cache: Vec<Vec<f32>>) {
        self.float_features_f32 = cache;
    }

    /// The f32 narrowing cache of the float columns, or an EMPTY slice when no
    /// ingestion source attached one. Consumers must treat empty (or any shape
    /// disagreement with [`Self::float_features`]) as "narrow it yourself".
    #[must_use]
    pub fn float_features_f32(&self) -> &[Vec<f32>] {
        &self.float_features_f32
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

    /// Build a new [`Pool`] containing only `indices` rows, in the given order,
    /// across every populated column (float/cat/text/embedding features + label,
    /// weights, group_id, subgroup_id, baseline). An empty source column stays
    /// empty (the gather never fabricates values for an absent column).
    ///
    /// Ranking `pairs` are DROPPED: row re-indexing would invalidate their
    /// object-index ids, and the first-slice cross-validation path (ORCH-01) is
    /// numeric / grouped, not pairwise.
    ///
    /// An out-of-range index is skipped via checked access (never panics), so
    /// the result has `<= indices.len()` rows and `n_rows()` equals the number
    /// of in-range indices supplied.
    #[must_use]
    pub fn select_rows(&self, indices: &[usize]) -> Pool {
        fn gather<T: Clone>(col: &[T], indices: &[usize]) -> Vec<T> {
            indices
                .iter()
                .filter_map(|&i| col.get(i).cloned())
                .collect()
        }

        let float_features = self
            .float_features
            .iter()
            .map(|col| gather(col, indices))
            .collect();
        let float_features_f32: Vec<Vec<f32>> = self
            .float_features_f32
            .iter()
            .map(|col| gather(col, indices))
            .collect();
        let cat_features = self
            .cat_features
            .iter()
            .map(|col| gather(col, indices))
            .collect();
        let text_features = self
            .text_features
            .iter()
            .map(|col| gather(col, indices))
            .collect();
        let embedding_features = self
            .embedding_features
            .iter()
            .map(|col| gather(col, indices))
            .collect();

        // The actual gathered valid-index count. Every populated per-row column
        // (float SoA columns and `label` when present) has length `n_rows`, so
        // its gathered length equals this count — the Pool stays internally
        // length-consistent. Empty columns gather to empty and are unaffected.
        let n_rows = indices.iter().filter(|&&i| i < self.n_rows).count();

        let mut out = Pool::from_validated_columns(
            n_rows,
            float_features,
            cat_features,
            text_features,
            embedding_features,
            gather(&self.label, indices),
            gather(&self.weights, indices),
            gather(&self.group_id, indices),
            gather(&self.subgroup_id, indices),
            Vec::new(),
            gather(&self.baseline, indices),
        );
        // The f32 cache gathers with the same indices, so it stays a bit-exact
        // narrowing of the gathered f64 columns (empty stays empty).
        out.set_float_f32_cache(float_features_f32);
        out
    }
}
