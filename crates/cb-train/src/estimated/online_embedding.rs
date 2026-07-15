//! Online (ordered) embedding-feature estimation — the per-fold read-before-update
//! prefix loop for the target-AWARE LDA embedding calcer (D-03 leakage control).
//!
//! # The read-before-update prefix (D-03, mirrors `ctr/online.rs` + `online_text.rs`)
//!
//! LDA is an `IOnlineFeatureEstimator`: a document's projected feature is computed
//! from the per-class scatter accumulated from EARLIER documents in the learn
//! permutation ONLY, THEN the current document's `(class, embedding)` updates that
//! scatter. A document's encoding therefore never sees its own label — the
//! no-leakage property. This is the same loop upstream runs in the embedding base
//! estimator (`base_embedding_feature_estimator.h`), with the LDA visitor
//! (`TLinearDACalcerVisitor::Update`/`Flush`, `lda.cpp:185-227`) re-fitting the
//! projection on a doubling schedule (`if (2*LastFlush <= Size) Flush`).
//!
//! # Plain vs ordered
//!
//! For `boosting_type=Plain` (the FEAT-02 fixtures) the estimated features fed to
//! the TREE SPLITS are the OFFLINE whole-set estimate ([`offline_lda_features`]):
//! accumulate EVERY learn document's `(class, embedding)`, fit the projection
//! once, and project every document against it. The online read-before-update
//! prefix ([`online_lda_prefix`]) feeds the ordered-boosting leaf-approximation
//! path. Both are target-AWARE (they accumulate labels); only the online one is
//! leakage-controlled.
//!
//! # Output is OBJECT-indexed
//!
//! Like the text seam, the returned columns are object-indexed
//! (`columns[f][doc]`), ready to append to the float-feature layout exactly like
//! the offline calcer columns. Width = the LDA `projection_dimension`.
//!
//! # Parity discipline
//!
//! The calcer compute math (`cb_compute::LdaCalcer` + `IncrementalCloud`) owns the
//! documented-upstream-order f32 BLAS scatter/projection cells (D-04). This module
//! only sequences the cloud accumulation, the doubling Flush re-fit, and the
//! Compute-then-Update ordering. Checked `.get(..)` only; no
//! `unwrap`/`expect`/panic/raw index; no `anyhow`.
//!
//! # Eigensolver parity (06.5-05)
//!
//! The LDA scatter is bit-faithful to upstream; the hand-rolled eigensolve is
//! reference-faithful but diverges 4.9e-2 from upstream's vendored-CLAPACK
//! `ssyev_` iterate (a ~3.9e-2 raw-projection divergence that does NOT cross any
//! split border, so the model per-stage oracle is byte-identical). See
//! `cb-oracle::lda_oracle_test` and `06.5-05-SUMMARY.md`.

use cb_compute::{IncrementalCloud, KnnCalcer, LdaCalcer};
use cb_core::{CbError, CbResult};

/// LDA projection-dimension default (`embedding_feature_estimators.cpp`:
/// `min(nClasses-1, dim-1)`). For binary classification over a `dim`-dim
/// embedding this is `min(1, dim-1)`.
#[must_use]
pub fn lda_projection_dim(num_classes: usize, dim: usize) -> usize {
    num_classes.saturating_sub(1).min(dim.saturating_sub(1))
}

/// The object-indexed online-LDA estimated feature columns plus the
/// permutation-order per-prefix projection trace.
#[derive(Debug, Clone, PartialEq)]
pub struct OnlineLdaPrefix {
    /// One estimated feature column per projection dimension, OBJECT-indexed
    /// (`columns[f][doc]`), `f32`-valued (estimated features are `f32`).
    pub columns: Vec<Vec<f32>>,
    /// The per-document projection in PERMUTATION order (`projection_in_order[p]`
    /// is the projected feature vector computed for the document at learn-order
    /// position `p`, BEFORE its own update). The per-prefix oracle anchor that
    /// localizes any leakage-order bug.
    pub projection_in_order: Vec<Vec<f32>>,
}

/// Fit the OFFLINE (whole-set) LDA projection and project every document
/// (the Plain-boosting estimate).
///
/// - `embeddings[doc]` is object `doc`'s embedding vector (length `dim`).
/// - `classes[doc]` is object `doc`'s class in `[0, num_classes)`.
/// - `num_classes` is the number of target classes (2 for binclf).
/// - `reg` is the LDA regularization (fixtures: 0.05).
///
/// Returns OBJECT-indexed columns (`columns[f][doc]`), width =
/// [`lda_projection_dim`].
///
/// # Errors
/// [`CbError::Degenerate`] on length mismatch / empty input;
/// [`CbError::OutOfRange`] (propagated) on a dimension mismatch or a non-SPD
/// betweenMatrix in the eigensolve.
pub fn offline_lda_features(
    embeddings: &[Vec<f32>],
    classes: &[usize],
    num_classes: usize,
    reg: f32,
) -> CbResult<Vec<Vec<f32>>> {
    let n = embeddings.len();
    if classes.len() != n {
        return Err(CbError::Degenerate(
            "offline_lda_features: embeddings / classes length mismatch".to_owned(),
        ));
    }
    let Some(first) = embeddings.first() else {
        return Err(CbError::Degenerate(
            "offline_lda_features: empty embedding set".to_owned(),
        ));
    };
    let dim = first.len();
    let proj_dim = lda_projection_dim(num_classes, dim);

    // Accumulate the WHOLE learn set into per-class clouds, then fit once.
    let mut clouds: Vec<IncrementalCloud> = (0..num_classes.max(1))
        .map(|_| IncrementalCloud::new(dim))
        .collect();
    let mut size = 0.0f32;
    let class_cap = clouds.len().saturating_sub(1);
    for (doc, embed) in embeddings.iter().enumerate() {
        let Some(&class) = classes.get(doc) else {
            continue;
        };
        let Some(cloud) = clouds.get_mut(class.min(class_cap)) else {
            continue;
        };
        cloud.add_vector(embed)?;
        size += 1.0;
    }
    let calcer = LdaCalcer::fit(&clouds, size, dim, proj_dim, reg)?;

    let mut columns: Vec<Vec<f32>> = vec![vec![0.0_f32; n]; proj_dim];
    for (doc, embed) in embeddings.iter().enumerate() {
        let proj = calcer.compute(embed)?;
        for (f, &v) in proj.iter().enumerate() {
            if let Some(col) = columns.get_mut(f) {
                if let Some(slot) = col.get_mut(doc) {
                    *slot = v;
                }
            }
        }
    }
    Ok(columns)
}

/// Compute the online (ordered) LDA projections over the learn `permutation` with
/// the read-before-update prefix (D-03) and the doubling Flush re-fit schedule.
///
/// - `permutation[p]` is the object index at learn-order position `p` (the fold's
///   `Fold::permutation`, NOT a fresh one).
/// - `embeddings[doc]` is object `doc`'s embedding vector (length `dim`).
/// - `classes[doc]` is object `doc`'s class in `[0, num_classes)`.
/// - `num_classes` is the number of target classes.
/// - `reg` is the LDA regularization.
///
/// For each `p`: read `doc = permutation[p]`, project it against the LAST-flushed
/// projection (the prefix-fitted clouds), store it object-indexed, THEN update the
/// cloud with `(class, embedding)` and re-fit on the doubling schedule
/// (`2*LastFlush <= Size`).
///
/// Until the first Flush (Size < 1) there is no projection yet, so the leading
/// document(s) project to zero — the same all-zero prefix the offline columns
/// start from and upstream's pre-first-Flush state produces.
///
/// # Errors
/// [`CbError::Degenerate`] on length mismatch / out-of-range permutation index;
/// [`CbError::OutOfRange`] (propagated) from the eigensolve.
pub fn online_lda_prefix(
    permutation: &[i32],
    embeddings: &[Vec<f32>],
    classes: &[usize],
    num_classes: usize,
    reg: f32,
) -> CbResult<OnlineLdaPrefix> {
    let n = permutation.len();
    if embeddings.len() != n || classes.len() != n {
        return Err(CbError::Degenerate(
            "online_lda_prefix: permutation / embeddings / classes length mismatch".to_owned(),
        ));
    }
    let Some(first) = embeddings.first() else {
        return Err(CbError::Degenerate(
            "online_lda_prefix: empty embedding set".to_owned(),
        ));
    };
    let dim = first.len();
    let proj_dim = lda_projection_dim(num_classes, dim);

    let mut clouds: Vec<IncrementalCloud> = (0..num_classes.max(1))
        .map(|_| IncrementalCloud::new(dim))
        .collect();
    let mut size = 0.0f32;
    let mut last_flush = 0.0f32;
    // The current (last-flushed) projection calcer; None until the first Flush.
    let mut calcer: Option<LdaCalcer> = None;

    let mut columns: Vec<Vec<f32>> = vec![vec![0.0_f32; n]; proj_dim];
    let mut projection_in_order: Vec<Vec<f32>> = Vec::with_capacity(n);

    for &doc_i in permutation {
        let doc = doc_i as usize;
        let Some(embed) = embeddings.get(doc) else {
            return Err(CbError::Degenerate(
                "online_lda_prefix: permutation index out of range for embeddings".to_owned(),
            ));
        };
        let Some(&class) = classes.get(doc) else {
            return Err(CbError::Degenerate(
                "online_lda_prefix: permutation index out of range for classes".to_owned(),
            ));
        };

        // COMPUTE from the prefix-fitted projection (read-before-update, D-03).
        let proj: Vec<f32> = match calcer.as_ref() {
            Some(c) => c.compute(embed)?,
            None => vec![0.0_f32; proj_dim],
        };
        for (f, &v) in proj.iter().enumerate() {
            if let Some(col) = columns.get_mut(f) {
                if let Some(slot) = col.get_mut(doc) {
                    *slot = v;
                }
            }
        }
        projection_in_order.push(proj);

        // THEN UPDATE the cloud and re-fit on the doubling schedule. Scope the
        // mutable borrow so the immutable `&clouds` fit below is allowed.
        {
            let class_cap = clouds.len().saturating_sub(1);
            let class_idx = class.min(class_cap);
            let Some(cloud) = clouds.get_mut(class_idx) else {
                continue;
            };
            cloud.add_vector(embed)?;
        }
        size += 1.0;
        if 2.0 * last_flush <= size {
            // Flush all pending batches, then re-fit the projection.
            for c in clouds.iter_mut() {
                c.update();
            }
            calcer = Some(LdaCalcer::fit(&clouds, size, dim, proj_dim, reg)?);
            last_flush = size;
        }
    }

    Ok(OnlineLdaPrefix {
        columns,
        projection_in_order,
    })
}

// ===========================================================================
// KNN online/offline embedding seam (06.5-06, brute-force-exact, D-03/D-04).
// ===========================================================================
//
// KNN is an `IOnlineFeatureEstimator` exactly like LDA: a document's neighbor-vote
// feature is computed from the prefix of EARLIER documents in the learn
// permutation ONLY (read-before-update, D-03), THEN the current `(target,
// embedding)` is inserted. The neighbor set is the spike-validated brute-force-
// exact k-NN (`cb_compute::KnnCalcer`); NO third-party HNSW crate (A2/D-05).
//
// Width: classification -> `num_classes` per-class vote counts; regression -> 1
// (neighbor target mean). For `boosting_type=Plain` the TREE SPLITS see the
// OFFLINE whole-set estimate ([`offline_knn_features`]); the online prefix
// ([`online_knn_prefix`]) feeds the ordered-boosting leaf path.

/// The object-indexed online-KNN estimated feature columns plus the
/// permutation-order per-prefix neighbor-id trace (the spike's oracle anchor).
#[derive(Debug, Clone, PartialEq)]
pub struct OnlineKnnPrefix {
    /// One estimated feature column per output dimension, OBJECT-indexed
    /// (`columns[f][doc]`), `f32`-valued.
    pub columns: Vec<Vec<f32>>,
    /// The per-document neighbor-id list in PERMUTATION order
    /// (`neighbors_in_order[p]` is the k-NN id list for the document at learn-order
    /// position `p`, computed over the prefix BEFORE its own insertion). This is the
    /// exact surface the instrumented `knn_neighbors` dump captures.
    pub neighbors_in_order: Vec<Vec<usize>>,
}

/// The KNN classification output width (`num_classes`) or regression width (`1`).
#[must_use]
pub fn knn_feature_count(num_classes: usize, is_classification: bool) -> usize {
    if is_classification {
        num_classes.max(1)
    } else {
        1
    }
}

/// Fit the OFFLINE (whole-set) KNN feature and compute every document (the
/// Plain-boosting estimate): insert EVERY learn document `(target, embedding)`,
/// then compute each document's `k`-NN vote over the full inserted set (the
/// document is its own nearest neighbor at distance 0, matching upstream's
/// whole-set `Compute`).
///
/// - `embeddings[doc]` is object `doc`'s embedding vector (length `dim`).
/// - `targets[doc]` is object `doc`'s target (class label for classification,
///   regression target otherwise).
/// - `num_classes` is the number of target classes (classification width).
/// - `close_num` is the query `k` (`KNN:k=...`).
/// - `is_classification` selects the vote-count vs mean arm.
///
/// Returns OBJECT-indexed columns (`columns[f][doc]`), width =
/// [`knn_feature_count`].
///
/// # Errors
/// [`CbError::Degenerate`] on length mismatch / empty input;
/// [`CbError::OutOfRange`] (propagated) on a dimension mismatch.
pub fn offline_knn_features(
    embeddings: &[Vec<f32>],
    targets: &[f32],
    num_classes: usize,
    close_num: usize,
    is_classification: bool,
) -> CbResult<Vec<Vec<f32>>> {
    let n = embeddings.len();
    if targets.len() != n {
        return Err(CbError::Degenerate(
            "offline_knn_features: embeddings / targets length mismatch".to_owned(),
        ));
    }
    let Some(first) = embeddings.first() else {
        return Err(CbError::Degenerate(
            "offline_knn_features: empty embedding set".to_owned(),
        ));
    };
    let dim = first.len();
    let width = knn_feature_count(num_classes, is_classification);

    let mut calcer = KnnCalcer::new(dim, close_num, is_classification, num_classes)?;
    for (doc, embed) in embeddings.iter().enumerate() {
        let Some(&target) = targets.get(doc) else {
            continue;
        };
        calcer.update(target, embed)?;
    }

    let mut columns: Vec<Vec<f32>> = vec![vec![0.0_f32; n]; width];
    for (doc, embed) in embeddings.iter().enumerate() {
        let feat = calcer.compute(embed)?;
        for (f, &v) in feat.iter().enumerate() {
            if let Some(col) = columns.get_mut(f) {
                if let Some(slot) = col.get_mut(doc) {
                    *slot = v;
                }
            }
        }
    }
    Ok(columns)
}

/// Compute the online (ordered) KNN features over the learn `permutation` with the
/// read-before-update prefix (D-03).
///
/// - `permutation[p]` is the object index at learn-order position `p` (the fold's
///   `Fold::permutation`, NOT a fresh one).
/// - `embeddings[doc]` / `targets[doc]` as in [`offline_knn_features`].
///
/// For each `p`: read `doc = permutation[p]`, compute its `k`-NN vote over the
/// PREFIX of already-inserted documents (NOT including itself — leakage control),
/// store it object-indexed AND record the neighbor-id list, THEN insert
/// `(target, embedding)`.
///
/// The first document(s) have an empty prefix and so produce an all-zero feature
/// with an empty neighbor list — exactly the leading `"neighbors":[]` rows the
/// instrumented dump shows.
///
/// # Errors
/// [`CbError::Degenerate`] on length mismatch / out-of-range permutation index;
/// [`CbError::OutOfRange`] (propagated) on a dimension mismatch.
pub fn online_knn_prefix(
    permutation: &[i32],
    embeddings: &[Vec<f32>],
    targets: &[f32],
    num_classes: usize,
    close_num: usize,
    is_classification: bool,
) -> CbResult<OnlineKnnPrefix> {
    let n = permutation.len();
    if embeddings.len() != n || targets.len() != n {
        return Err(CbError::Degenerate(
            "online_knn_prefix: permutation / embeddings / targets length mismatch".to_owned(),
        ));
    }
    let Some(first) = embeddings.first() else {
        return Err(CbError::Degenerate(
            "online_knn_prefix: empty embedding set".to_owned(),
        ));
    };
    let dim = first.len();
    let width = knn_feature_count(num_classes, is_classification);

    let mut calcer = KnnCalcer::new(dim, close_num, is_classification, num_classes)?;
    let mut columns: Vec<Vec<f32>> = vec![vec![0.0_f32; n]; width];
    let mut neighbors_in_order: Vec<Vec<usize>> = Vec::with_capacity(n);

    for &doc_i in permutation {
        let doc = doc_i as usize;
        let Some(embed) = embeddings.get(doc) else {
            return Err(CbError::Degenerate(
                "online_knn_prefix: permutation index out of range for embeddings".to_owned(),
            ));
        };
        let Some(&target) = targets.get(doc) else {
            return Err(CbError::Degenerate(
                "online_knn_prefix: permutation index out of range for targets".to_owned(),
            ));
        };

        // COMPUTE from the prefix-inserted vectors (read-before-update, D-03). The
        // neighbor ids here are PREFIX-LOCAL insertion ids (0..prefix_len), exactly
        // as the instrumented `knn_neighbors` dump records them.
        let neighbors = calcer.neighbors(embed)?;
        let feat = calcer.compute(embed)?;
        for (f, &v) in feat.iter().enumerate() {
            if let Some(col) = columns.get_mut(f) {
                if let Some(slot) = col.get_mut(doc) {
                    *slot = v;
                }
            }
        }
        neighbors_in_order.push(neighbors);

        // THEN INSERT the current document (its own vector + target).
        calcer.update(target, embed)?;
    }

    Ok(OnlineKnnPrefix {
        columns,
        neighbors_in_order,
    })
}
