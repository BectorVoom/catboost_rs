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
