//! Embedding feature calcers — the pure numeric primitives that turn a per-object
//! embedding vector into estimated float feature columns.
//!
//! # Source of truth (D-04)
//!
//! This module transcribes the upstream CatBoost 1.2.10 embedding-feature calcer
//! math VERBATIM. The first calcer is LDA (`TLinearDACalcer`), transcribed from
//! `catboost-master/catboost/private/libs/embedding_features/lda.cpp` and
//! `lda.h`. The generalized symmetric eigensolve itself lives in
//! [`crate::lda_linalg`] (the `ssygst_`/`ssyev_` analog); this module owns the
//! [`IncrementalCloud`] per-class scatter accumulation, [`TotalScatter`]
//! computation, the regularized between-class scatter assembly, and the
//! [`LdaCalcer::compute`] `cblas_sgemv` projection.
//!
//! LDA is a target-AWARE ONLINE estimator (D-03): the per-class scatter is
//! accumulated over the TFold learn permutation with a read-before-update prefix.
//! This module owns only the PURE accumulation + projection math; the ordered
//! per-fold prefix loop that drives it lives behind the `cb-train`
//! online-embedding seam (`cb-train::estimated::online_embedding`).
//!
//! # Summation routing (D-04 / D-08)
//!
//! The scatter-matrix rank updates ([`IncrementalCloud::update`]) and the
//! projection GEMV ([`crate::lda_linalg::sgemv_rowmajor`]) are the **documented
//! upstream-order f32 BLAS cells** (`cblas_sgemm` / `cblas_sgemv`) and so use raw
//! f32 `+=` to bit-match the BLAS accumulation order — the D-04 carve-out for
//! "documented upstream-order scatter cells". Object-order reductions that are NOT
//! BLAS cells (none in the current LDA path) would route through
//! [`cb_core::sum_f64`].
//!
//! # Eigensolver parity status (06.5-05)
//!
//! The hand-rolled f32 eigensolve in [`crate::lda_linalg`] is reference-faithful
//! (matches f64 scipy AND f32 LAPACK `ssygv`). The upstream vendored-CLAPACK
//! `ssyev_` iterate differs by 4.9e-2 on the dominant eigenvector, yielding a
//! ~3.9e-2 raw-projection divergence — BUT this does NOT cross any split border
//! (the class clouds project to ≈±2 with a ≥1.34 margin to the 0.59 border), so
//! the model's per-stage oracle (Splits/LeafValues/StagedApprox/Predictions) is
//! byte-identical. See `06.5-05-SUMMARY.md` and `lda_linalg.rs`.
//!
//! # Robustness (V5 / INFRA-02)
//!
//! Every embedding-vector length is checked against the calcer dimension; a
//! mismatch yields a typed [`CbError::OutOfRange`] (the `CB_ENSURE` analog,
//! `helpers.h:21`). No `unwrap`/`expect`/`panic`/raw index appears in this module
//! (CLAUDE.md library discipline).

use cb_core::{CbError, CbResult};

use crate::lda_linalg::{calculate_projection, sgemv_rowmajor};

/// LDA default regularization (`embedding_feature_estimators.cpp`:reg default
/// `0.00005`). The fixtures pin `reg=0.05`; the value is a constructor argument.
pub const LDA_DEFAULT_REG: f32 = 0.000_05;

/// Per-class incremental scatter accumulator — the [`IncrementalCloud`] analog
/// (`lda.h:13-35`, `lda.cpp:93-129`).
///
/// Accumulates, in object order, a running shifted mean ([`Self::base_center`])
/// and an f32 scatter matrix ([`Self::scatter`], row-major `dim × dim`) via the
/// same `cblas_sgemm` rank-update schedule upstream uses (flush when the buffered
/// batch reaches the threshold). The arithmetic is f32 with raw `+=` on the BLAS
/// cells to bit-match the upstream accumulation order (D-04).
#[derive(Debug, Clone)]
pub struct IncrementalCloud {
    dimension: usize,
    base_size: i64,
    additional_size: i64,
    base_center: Vec<f32>,
    new_shift: Vec<f32>,
    scatter: Vec<f32>,
    buffer: Vec<f32>,
}

impl IncrementalCloud {
    /// Create an empty cloud for `dimension`-dimensional embeddings
    /// (`IncrementalCloud(int dim)`, `lda.h:15-20`).
    #[must_use]
    pub fn new(dimension: usize) -> Self {
        IncrementalCloud {
            dimension,
            base_size: 0,
            additional_size: 0,
            base_center: vec![0.0; dimension],
            new_shift: vec![0.0; dimension],
            scatter: vec![0.0; dimension.saturating_mul(dimension)],
            buffer: Vec::new(),
        }
    }

    /// `BaseSize + AdditionalSize` as f32 (`IncrementalCloud::TotalSize`,
    /// `lda.h:24-26`).
    #[must_use]
    pub fn total_size(&self) -> f32 {
        (self.base_size + self.additional_size) as f32
    }

    /// The running shifted class mean (`BaseCenter`). Length = `dimension`.
    #[must_use]
    pub fn base_center(&self) -> &[f32] {
        &self.base_center
    }

    /// The accumulated (un-regularized) class scatter, row-major `dim × dim`.
    #[must_use]
    pub fn scatter(&self) -> &[f32] {
        &self.scatter
    }

    /// Add one embedding vector (`IncrementalCloud::AddVector`, `lda.cpp:93-102`).
    ///
    /// Pushes the shifted vector into the batch buffer and updates `NewShift`,
    /// then flushes via [`Self::update`] on the same `BaseSize < 128 ||
    /// AdditionalSize >= 32` schedule upstream uses.
    ///
    /// # Errors
    /// [`CbError::OutOfRange`] if `embed.len()` != the cloud dimension.
    pub fn add_vector(&mut self, embed: &[f32]) -> CbResult<()> {
        if embed.len() != self.dimension {
            return Err(CbError::OutOfRange(format!(
                "IncrementalCloud::add_vector: embedding dim {} != cloud dim {}",
                embed.len(),
                self.dimension
            )));
        }
        self.additional_size += 1;
        for idx in 0..self.dimension {
            // checked access; lengths verified above / by construction.
            let e = *embed.get(idx).unwrap_or(&0.0);
            let bc = *self.base_center.get(idx).unwrap_or(&0.0);
            let v = e - bc;
            self.buffer.push(v);
            if let Some(slot) = self.new_shift.get_mut(idx) {
                *slot += v;
            }
        }
        if self.base_size < 128 || self.additional_size >= 32 {
            self.update();
        }
        Ok(())
    }

    /// Flush the buffered batch into the scatter matrix
    /// (`IncrementalCloud::Update`, `lda.cpp:104-129`).
    ///
    /// Mirrors the two `cblas_sgemm` rank updates exactly:
    /// `scatter = (1/ts)·Bufferᵀ·Buffer + (BaseSize/ts)·scatter`, then
    /// `scatter += (-1)·NewShift·NewShiftᵀ`. Raw f32 `+=` on the GEMM cells (D-04).
    pub fn update(&mut self) {
        if self.additional_size == 0 {
            return;
        }
        let ts = self.total_size();
        for idx in 0..self.dimension {
            if let Some(ns) = self.new_shift.get_mut(idx) {
                *ns /= ts;
            }
            let ns = *self.new_shift.get(idx).unwrap_or(&0.0);
            if let Some(bc) = self.base_center.get_mut(idx) {
                *bc += ns;
            }
        }
        let alpha = 1.0f32 / ts;
        let beta = self.base_size as f32 / ts;
        let k = self.additional_size as usize;
        let dim = self.dimension;
        // scatter = alpha · Bufferᵀ·Buffer + beta · scatter  (CblasTrans, NoTrans)
        let mut new_scatter = vec![0.0f32; dim.saturating_mul(dim)];
        for i in 0..dim {
            for j in 0..dim {
                let mut acc = 0.0f32; // documented upstream-order GEMM rank cell (raw f32 +=)
                for r in 0..k {
                    let bi = *self.buffer.get(r * dim + i).unwrap_or(&0.0);
                    let bj = *self.buffer.get(r * dim + j).unwrap_or(&0.0);
                    acc += bi * bj;
                }
                let prev = *self.scatter.get(i * dim + j).unwrap_or(&0.0);
                if let Some(slot) = new_scatter.get_mut(i * dim + j) {
                    *slot = alpha * acc + beta * prev;
                }
            }
        }
        self.scatter = new_scatter;
        self.buffer.clear();
        // scatter += (-1) · NewShift·NewShiftᵀ  (cblas_sgemm alpha=-1.0 rank-1
        // update, lda.cpp:120-125). f32 negation is exact, so `-(ni*nj)` is
        // bit-identical to the upstream `-1.0 *` GEMM cell.
        for i in 0..dim {
            for j in 0..dim {
                let ni = *self.new_shift.get(i).unwrap_or(&0.0);
                let nj = *self.new_shift.get(j).unwrap_or(&0.0);
                if let Some(slot) = self.scatter.get_mut(i * dim + j) {
                    *slot += -(ni * nj);
                }
            }
        }
        self.base_size += self.additional_size;
        self.additional_size = 0;
        for ns in self.new_shift.iter_mut() {
            *ns = 0.0;
        }
    }
}

/// Total-scatter computation (`TLinearDACalcer::TotalScatterCalculation`,
/// `lda.cpp:162-183`): the weighted outer product of class means minus the
/// weighted-total-mean outer product. `clouds` are the per-class
/// [`IncrementalCloud`]s; `size` is the total object count seen so far.
///
/// Returns the `dim × dim` row-major total scatter. Raw f32 `+=` on the GEMM cells.
///
/// # Errors
/// [`CbError::OutOfRange`] if any cloud's dimension disagrees with `dim`.
pub fn total_scatter(clouds: &[IncrementalCloud], size: f32, dim: usize) -> CbResult<Vec<f32>> {
    let mut result = vec![0.0f32; dim.saturating_mul(dim)];
    let mut total_mean = vec![0.0f32; dim];
    for cloud in clouds {
        if cloud.dimension != dim {
            return Err(CbError::OutOfRange(format!(
                "total_scatter: cloud dim {} != requested dim {}",
                cloud.dimension, dim
            )));
        }
        let weight = cloud.total_size() / size;
        for i in 0..dim {
            let ci = *cloud.base_center.get(i).unwrap_or(&0.0);
            for j in 0..dim {
                let cj = *cloud.base_center.get(j).unwrap_or(&0.0);
                if let Some(slot) = result.get_mut(i * dim + j) {
                    *slot += weight * ci * cj;
                }
            }
            if let Some(tm) = total_mean.get_mut(i) {
                *tm += weight * ci;
            }
        }
    }
    for i in 0..dim {
        let ti = *total_mean.get(i).unwrap_or(&0.0);
        for j in 0..dim {
            let tj = *total_mean.get(j).unwrap_or(&0.0);
            if let Some(slot) = result.get_mut(i * dim + j) {
                // cblas_sgemm alpha=-1.0 (lda.cpp:177-182); f32 negation is exact.
                *slot += -(ti * tj);
            }
        }
    }
    Ok(result)
}

/// Assemble the regularized between-class scatter (`TLinearDACalcerVisitor::Flush`,
/// `lda.cpp:203-218`): for classification, the class-weighted sum of per-class
/// scatters; then `+ RegParam` on the diagonal.
///
/// `clouds` are the per-class clouds; `size` is the total object count; `dim` the
/// embedding dimension; `reg` the regularization. Returns the `dim × dim`
/// row-major regularized betweenMatrix.
///
/// # Errors
/// [`CbError::OutOfRange`] if any cloud's dimension disagrees with `dim`.
pub fn between_matrix(
    clouds: &[IncrementalCloud],
    size: f32,
    dim: usize,
    reg: f32,
) -> CbResult<Vec<f32>> {
    let n2 = dim.saturating_mul(dim);
    let mut between = vec![0.0f32; n2];
    if clouds.len() == 1 {
        // Regression path (lda.cpp:214): BetweenMatrix = ClassesDist[0].ScatterMatrix.
        if let Some(c) = clouds.first() {
            if c.dimension != dim {
                return Err(CbError::OutOfRange(format!(
                    "between_matrix: cloud dim {} != requested dim {}",
                    c.dimension, dim
                )));
            }
            between.copy_from_slice(&c.scatter);
        }
    } else {
        // Classification path (lda.cpp:204-212): Σ_c (TotalSize_c / Size) · Scatter_c.
        for cloud in clouds {
            if cloud.dimension != dim {
                return Err(CbError::OutOfRange(format!(
                    "between_matrix: cloud dim {} != requested dim {}",
                    cloud.dimension, dim
                )));
            }
            let weight = cloud.total_size() / size;
            for i in 0..n2 {
                let s = *cloud.scatter.get(i).unwrap_or(&0.0);
                if let Some(slot) = between.get_mut(i) {
                    *slot += weight * s;
                }
            }
        }
    }
    // Diagonal += RegParam (lda.cpp:216-218).
    for d in 0..dim {
        if let Some(slot) = between.get_mut(d * dim + d) {
            *slot += reg;
        }
    }
    Ok(between)
}

/// The trained LDA calcer projection state — the subset of [`TLinearDACalcer`]
/// needed for [`LdaCalcer::compute`] (`lda.cpp:131-160`).
///
/// `projection_matrix` is `projection_dim × total_dimension` row-major (the
/// trailing eigenvector rows from [`calculate_projection`]). For the fixtures
/// `projection_dim = 1`, `total_dimension = 4`.
#[derive(Debug, Clone)]
pub struct LdaCalcer {
    total_dimension: usize,
    projection_dimension: usize,
    projection_matrix: Vec<f32>,
}

impl LdaCalcer {
    /// Build a calcer from a pre-computed projection (`projection_dim × total_dim`,
    /// row-major).
    ///
    /// # Errors
    /// [`CbError::OutOfRange`] if `projection_matrix.len()` != `projection_dim *
    /// total_dim`.
    pub fn new(
        total_dimension: usize,
        projection_dimension: usize,
        projection_matrix: Vec<f32>,
    ) -> CbResult<Self> {
        if projection_matrix.len() != projection_dimension.saturating_mul(total_dimension) {
            return Err(CbError::OutOfRange(format!(
                "LdaCalcer::new: projection len {} != proj_dim*total_dim {}*{}",
                projection_matrix.len(),
                projection_dimension,
                total_dimension
            )));
        }
        Ok(LdaCalcer {
            total_dimension,
            projection_dimension,
            projection_matrix,
        })
    }

    /// Fit a calcer end-to-end from the per-class clouds (the `Flush` +
    /// `CalculateProjection` path, `lda.cpp:197-223`): assemble the regularized
    /// betweenMatrix B and totalScatter A, solve the generalized symmetric
    /// eigenproblem, and keep the trailing `projection_dim` eigenvector rows.
    ///
    /// `clouds` are the per-class clouds; `size` the total object count; `dim` the
    /// embedding dimension; `projection_dim` the output width; `reg` the
    /// regularization.
    ///
    /// # Errors
    /// [`CbError::OutOfRange`] on dimension mismatch, non-SPD betweenMatrix, or
    /// `projection_dim > dim`.
    pub fn fit(
        clouds: &[IncrementalCloud],
        size: f32,
        dim: usize,
        projection_dim: usize,
        reg: f32,
    ) -> CbResult<Self> {
        let total = total_scatter(clouds, size, dim)?;
        let between = between_matrix(clouds, size, dim, reg)?;
        // Generalized problem: scatterTotal · x = λ · scatterInner · x
        // (ssygst_ factors scatterInner = betweenMatrix = B; reduces scatterTotal = A).
        let (projection, _eigenvalues) = calculate_projection(&between, &total, dim, projection_dim)?;
        LdaCalcer::new(dim, projection_dim, projection)
    }

    /// Project one embedding vector (`TLinearDACalcer::Compute`, `lda.cpp:131-140`):
    /// `proj = ProjectionMatrix · embed` (`cblas_sgemv`, row-major `proj_dim ×
    /// total_dim`). Likelihood probabilities (`ComputeProbabilities`) are NOT
    /// included here — the fixtures' default-fallback probs are not the parity
    /// surface (see SUMMARY); the projection scalar(s) are.
    ///
    /// Returns the `projection_dim`-length projected feature row.
    ///
    /// # Errors
    /// [`CbError::OutOfRange`] if `embed.len()` != `total_dimension`.
    pub fn compute(&self, embed: &[f32]) -> CbResult<Vec<f32>> {
        sgemv_rowmajor(
            &self.projection_matrix,
            self.projection_dimension,
            self.total_dimension,
            embed,
        )
    }

    /// The projection matrix (`projection_dim × total_dim`, row-major).
    #[must_use]
    pub fn projection_matrix(&self) -> &[f32] {
        &self.projection_matrix
    }

    /// Output feature width (`FeatureCount`, `lda.h:64-69`, without probabilities).
    #[must_use]
    pub fn feature_count(&self) -> usize {
        self.projection_dimension
    }
}

// ===========================================================================
// KNN (k-nearest-neighbor vote) embedding calcer — `knn.cpp` / `knn.h`.
// ===========================================================================
//
// Upstream `TKNNCalcer` (`knn.h:94-142`) stores the inserted embedding vectors in
// an ONLINE HNSW approximate index (`TOnlineHnswDenseVectorIndex<float,
// TL2SqrDistance>`) and, for each query, votes over the `CloseNum` nearest
// neighbors: classification accumulates per-class counts
// (`++result[TargetClasses[id]]`, `knn.cpp:56-59`), regression averages the
// neighbor targets (`result[0] = mean(Targets[id])`, `knn.cpp:60-66`).
//
// # Why brute-force-exact L2 (06.5-06 spike, A2/A5 — NOT a third-party HNSW crate)
//
// The 06.5-06 neighbor-id SPIKE compared a brute-force-exact L2-squared k-NN
// against the INSTRUMENTED upstream HNSW neighbor-id dump
// (`fixtures/text_tokenizer/knn_neighbors.json`, the Plan-01 D-07 `knn_neighbors`
// hook) on the frozen 16-row / 4-dim fixture. Result: **0 / 64 neighbor-id
// mismatches** (ordered AND set), across all online prefixes. At fixture scale the
// HNSW index degenerates to exact (A5), so brute-force-exact reproduces the
// upstream neighbor set BIT-FOR-BIT. The class-vote encoding is an integer count
// over that neighbor set, so the per-stage model oracle is byte-identical (a
// strictly stronger result than the LDA binarization-stability argument). A
// third-party HNSW/ANN crate is FORBIDDEN (any non-identical graph -> different
// neighbors -> parity fail, A2 / D-05).
//
// # Summation routing (D-04)
//
// The L2-squared distance and the regression neighbor mean route through
// [`cb_core::sum_f64`] (object-order reductions, NOT BLAS cells). The distance
// per-component squares are summed in an f64 accumulator over the (widened) f32
// component differences, matching upstream's `TL2SqrDistance<float>` summed in the
// HNSW comparator. Ties break by ascending neighbor id (the stable secondary key
// the dump exhibits), reproduced here by a stable sort on `(distance, id)`.

/// Brute-force-exact L2-squared nearest-neighbor cloud (the spike-validated
/// `IKNNCloud` analog, `knn.h:22-46`). Stores inserted embedding vectors in
/// INSERTION order (= the learn permutation when driven by the online seam) and
/// answers a `k`-NN query by an exact distance scan.
#[derive(Debug, Clone, Default)]
pub struct KnnCloud {
    /// Dimension of each stored vector.
    dimension: usize,
    /// Inserted vectors, flattened row-major (`points[i*dim .. (i+1)*dim]`), in
    /// insertion order. Insertion id `i` is the neighbor id the vote indexes.
    points: Vec<f32>,
    /// Number of inserted vectors (`points.len() / dimension`).
    size: usize,
}

impl KnnCloud {
    /// A new empty cloud over `dimension`-dim vectors.
    #[must_use]
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            points: Vec::new(),
            size: 0,
        }
    }

    /// Number of inserted vectors so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.size
    }

    /// Whether no vectors have been inserted yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Insert `embed` at the next insertion id (`AddItem` analog, `knn.h:35-37`).
    ///
    /// # Errors
    /// [`CbError::OutOfRange`] if `embed.len()` differs from the cloud dimension
    /// (the `CB_ENSURE` analog; never indexes past the buffer).
    pub fn add_vector(&mut self, embed: &[f32]) -> CbResult<()> {
        if embed.len() != self.dimension {
            return Err(CbError::OutOfRange(format!(
                "KnnCloud::add_vector: embedding length {} != cloud dimension {}",
                embed.len(),
                self.dimension
            )));
        }
        self.points.extend_from_slice(embed);
        self.size += 1;
        Ok(())
    }

    /// L2-squared distance from `query` to inserted vector `id`, summed in f64
    /// over the f32 component squares (`TL2SqrDistance<float>` analog, D-04).
    ///
    /// # Errors
    /// [`CbError::OutOfRange`] on a dimension mismatch or an out-of-range id.
    fn l2_sqr(&self, query: &[f32], id: usize) -> CbResult<f64> {
        if query.len() != self.dimension {
            return Err(CbError::OutOfRange(format!(
                "KnnCloud::l2_sqr: query length {} != cloud dimension {}",
                query.len(),
                self.dimension
            )));
        }
        let start = id.checked_mul(self.dimension).ok_or_else(|| {
            CbError::OutOfRange("KnnCloud::l2_sqr: id*dim overflow".to_owned())
        })?;
        let end = start.checked_add(self.dimension).ok_or_else(|| {
            CbError::OutOfRange("KnnCloud::l2_sqr: id range overflow".to_owned())
        })?;
        let row = self
            .points
            .get(start..end)
            .ok_or_else(|| CbError::OutOfRange("KnnCloud::l2_sqr: id out of range".to_owned()))?;
        let squares: Vec<f64> = row
            .iter()
            .zip(query.iter())
            .map(|(&p, &q)| {
                let d = p - q;
                f64::from(d * d)
            })
            .collect();
        Ok(cb_core::sum_f64(&squares))
    }

    /// The `k` nearest insertion ids to `query`, sorted by ascending L2-squared
    /// distance with an ascending-id tie-break (`GetNearestNeighbors` analog,
    /// `knn.cpp:31-49`). Returns at most `min(k, len)` ids.
    ///
    /// # Errors
    /// [`CbError::OutOfRange`] on a dimension mismatch.
    pub fn nearest_neighbors(&self, query: &[f32], k: usize) -> CbResult<Vec<usize>> {
        if query.len() != self.dimension {
            return Err(CbError::OutOfRange(format!(
                "KnnCloud::nearest_neighbors: query length {} != cloud dimension {}",
                query.len(),
                self.dimension
            )));
        }
        let mut scored: Vec<(f64, usize)> = Vec::with_capacity(self.size);
        for id in 0..self.size {
            scored.push((self.l2_sqr(query, id)?, id));
        }
        // Ascending distance, ascending id on ties (the dump's stable secondary key).
        scored.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(core::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
        Ok(scored.into_iter().take(k).map(|(_, id)| id).collect())
    }
}

/// KNN neighbor-vote calcer (`TKNNCalcer`, `knn.h:94-142` / `knn.cpp:51-90`).
///
/// Classification emits per-class neighbor-vote COUNTS (`width = num_classes`);
/// regression emits the neighbor target MEAN (`width = 1`, Pitfall 5). The
/// neighbor set is the brute-force-exact spike-validated `k`-NN (see module note).
#[derive(Debug, Clone)]
pub struct KnnCalcer {
    cloud: KnnCloud,
    /// `CloseNum` — the query `k` (`knn.h:107`; fixtures `KNN:k=3`).
    close_num: usize,
    /// `true` -> classification (per-class vote counts); `false` -> regression mean.
    is_classification: bool,
    /// `FeatureCount_` — `num_classes` (clf) or `1` (reg) (`knn.cpp:110`).
    feature_count: usize,
    /// Per-insertion-id target class (classification) — `TargetClasses`, parallel
    /// to the cloud insertion ids.
    target_classes: Vec<usize>,
    /// Per-insertion-id target value (regression) — `Targets`.
    targets: Vec<f32>,
}

impl KnnCalcer {
    /// A new empty KNN calcer over `dimension`-dim embeddings.
    ///
    /// - `close_num` is the query `k` (`KNN:k=...`).
    /// - `is_classification` selects the vote-count vs mean arm.
    /// - `num_classes` sets the classification output width (ignored for reg).
    #[must_use]
    pub fn new(
        dimension: usize,
        close_num: usize,
        is_classification: bool,
        num_classes: usize,
    ) -> Self {
        let feature_count = if is_classification {
            num_classes.max(1)
        } else {
            1
        };
        Self {
            cloud: KnnCloud::new(dimension),
            close_num,
            is_classification,
            feature_count,
            target_classes: Vec::new(),
            targets: Vec::new(),
        }
    }

    /// Output feature width (`FeatureCount`, `knn.h:115-117`).
    #[must_use]
    pub fn feature_count(&self) -> usize {
        self.feature_count
    }

    /// Number of inserted neighbors so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cloud.len()
    }

    /// Whether no neighbors have been inserted yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cloud.is_empty()
    }

    /// The k nearest insertion ids to `embed` over the currently-inserted prefix
    /// (the spike's neighbor-id surface — used by the per-query neighbor oracle).
    ///
    /// # Errors
    /// [`CbError::OutOfRange`] on a dimension mismatch.
    pub fn neighbors(&self, embed: &[f32]) -> CbResult<Vec<usize>> {
        self.cloud.nearest_neighbors(embed, self.close_num)
    }

    /// Insert `embed` with its `target` (`TKNNCalcerVisitor::Update`,
    /// `knn.cpp:76-90`): add to the cloud, then push the parallel class/target.
    ///
    /// # Errors
    /// [`CbError::OutOfRange`] on a dimension mismatch.
    pub fn update(&mut self, target: f32, embed: &[f32]) -> CbResult<()> {
        self.cloud.add_vector(embed)?;
        if self.is_classification {
            // `(ui32)target` — the class label as a non-negative integer.
            let class = if target.is_finite() && target >= 0.0 {
                target as usize
            } else {
                0
            };
            self.target_classes.push(class);
        } else {
            self.targets.push(target);
        }
        Ok(())
    }

    /// Compute the KNN vote feature row for `embed` over the inserted prefix
    /// (`TKNNCalcer::Compute`, `knn.cpp:51-74`).
    ///
    /// Classification: `result[class(neighbor)] += 1` over the k neighbors
    /// (width = `num_classes`). Regression: `result[0] = mean(target(neighbor))`
    /// via [`cb_core::sum_f64`] (width = 1); an empty neighbor set yields all-zero.
    ///
    /// # Errors
    /// [`CbError::OutOfRange`] on a dimension mismatch or a neighbor id with no
    /// recorded class/target (a parallel-array desync — never silently indexed).
    pub fn compute(&self, embed: &[f32]) -> CbResult<Vec<f32>> {
        let mut result = vec![0.0_f32; self.feature_count];
        let neighbors = self.cloud.nearest_neighbors(embed, self.close_num)?;
        if self.is_classification {
            for &id in &neighbors {
                let class = *self.target_classes.get(id).ok_or_else(|| {
                    CbError::OutOfRange(format!(
                        "KnnCalcer::compute: neighbor id {id} has no recorded class"
                    ))
                })?;
                let slot = result.get_mut(class.min(self.feature_count.saturating_sub(1)));
                if let Some(s) = slot {
                    *s += 1.0;
                }
            }
        } else if !neighbors.is_empty() {
            let vals: Vec<f64> = neighbors
                .iter()
                .map(|&id| {
                    self.targets
                        .get(id)
                        .copied()
                        .map(f64::from)
                        .ok_or_else(|| {
                            CbError::OutOfRange(format!(
                                "KnnCalcer::compute: neighbor id {id} has no recorded target"
                            ))
                        })
                })
                .collect::<CbResult<Vec<f64>>>()?;
            #[allow(clippy::cast_possible_truncation)]
            let mean = (cb_core::sum_f64(&vals) / neighbors.len() as f64) as f32;
            if let Some(s) = result.get_mut(0) {
                *s = mean;
            }
        }
        Ok(result)
    }
}
