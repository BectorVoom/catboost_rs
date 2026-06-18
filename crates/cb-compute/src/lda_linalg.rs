//! LDA linear-algebra primitives — the hand-rolled f32 GEMM/GEMV and the
//! generalized symmetric eigensolver that back the LinearDA embedding calcer
//! (FEAT-02). This is the **spike candidate** produced by Plan 06.5-05 Task 1.
//!
//! # Re-measurement status (06.5-05, instrumented re-measure)
//!
//! This eigensolver is **reference-faithful**: on the bit-matched upstream scatter
//! inputs (`lda_scatter` dump) it reproduces **both** f64 scipy `eigh` **and** f32
//! reference LAPACK `ssygv` to ~6 digits (dominant eigenvalue 38.45, dominant
//! eigenvector `[0.597,0.563,-0.439,-0.365]`). The hand-rolled scatter construction
//! is also bit-faithful (`scatter_inner` ≤4.66e-10, `scatter_total` 0 vs upstream).
//!
//! However, the upstream **vendored-CLAPACK `ssyev_`** dump
//! (`fixtures/embedding_calcers/LDA/scatter_projection_gt.json`) reports a dominant
//! eigenvalue (376.67) inconsistent with its own eigenvector (Rayleigh quotient
//! 38.28; `||A_reduced·v − 376.67·v|| = 338`) and an eigenvector 4.9e-2 away from
//! the reference dominant eigenvector, yielding a per-document projected-feature
//! divergence of 3.9e-2 — above the ≤1e-5 bar. This divergence is therefore a
//! **documented-tolerance escalation candidate**, NOT a transcription bug in this
//! module. See `06.5-05-SUMMARY.md` and `instrument_text_pipeline_README.md`.
//!
//! # Source of truth (D-04)
//!
//! Transcribed from
//! `catboost-master/catboost/private/libs/embedding_features/lda.cpp:31-160`
//! (`CalculateProjection` — `ssygst_` reduce + `ssyev_` symmetric eigen, then the
//! trailing-`ProjectionDim`-rows copy; the `cblas_sgemv` project in
//! `TLinearDACalcer::Compute`). Upstream computes the generalized symmetric
//! eigenproblem in **single precision (`float`)** via CLAPACK; this module
//! reproduces that arithmetic with a hand-rolled f32 path so the parity decision
//! can be driven by measured divergence rather than a new LAPACK dependency.
//!
//! The eigensolver pair mirrors LAPACK exactly:
//!
//! * [`reduce_generalized`] is the `ssygst_(itype=1,'L')` analog: given the SPD
//!   matrix `b = scatterInner` (the regularized between-class scatter) factored
//!   as `b = L Lᵀ`, it overwrites `a = scatterTotal` with `inv(L) a inv(Lᵀ)`,
//!   turning `a x = λ b x` into the standard problem `(inv(L) a inv(Lᵀ)) y = λ y`.
//! * [`jacobi_symmetric_eig`] is the `ssyev_('V','L')` analog: a cyclic-Jacobi
//!   symmetric eigensolver (chosen over tridiagonal-QR because the LDA matrices
//!   are small — `dim ≤ embeddingDim`, e.g. 4×4 — so Jacobi is exact, branch-free
//!   and order-deterministic).
//!
//! The projection is the eigenvector column of the **largest** eigenvalue (the
//! trailing `ProjectionDim` rows of the column-major eigenvector matrix that
//! `lda.cpp:57` copies; `ProjectionDim = min(nClasses-1, dim-1) = 1` for the
//! binary fixtures).
//!
//! # Summation routing (D-04 / D-08)
//!
//! The GEMM/GEMV reductions here are the **documented upstream-order f32 scatter
//! / projection cells** (`cblas_sgemm` / `cblas_sgemv` rank updates) and so use
//! raw f32 `+=` to bit-match the BLAS accumulation order — exactly the D-04
//! carve-out for "documented upstream-order scatter cells". This is NOT a general
//! float reduction; callers that reduce in object order still route through
//! `cb_core::sum_f64`.

use cb_core::{CbError, CbResult};

/// Row-major dense f32 GEMV `y := alpha * A·x` (the `cblas_sgemv` project step,
/// `lda.cpp:134`). `a` is `rows × cols` row-major; `x` has length `cols`; the
/// returned vector has length `rows`. Raw f32 `+=` matches the BLAS dot order.
///
/// # Errors
/// [`CbError::OutOfRange`] if the dimensions are inconsistent with `a.len()`.
pub fn sgemv_rowmajor(a: &[f32], rows: usize, cols: usize, x: &[f32]) -> CbResult<Vec<f32>> {
    if a.len() != rows.saturating_mul(cols) {
        return Err(CbError::OutOfRange(format!(
            "sgemv: matrix len {} != rows*cols {}*{}",
            a.len(),
            rows,
            cols
        )));
    }
    if x.len() != cols {
        return Err(CbError::OutOfRange(format!(
            "sgemv: x len {} != cols {}",
            x.len(),
            cols
        )));
    }
    let mut y = vec![0.0f32; rows];
    for (i, yi) in y.iter_mut().enumerate() {
        let mut acc = 0.0f32; // documented upstream-order BLAS dot (raw f32 +=)
        for (j, &xj) in x.iter().enumerate() {
            // checked access; bounds guaranteed by the length checks above.
            if let Some(&aij) = a.get(i.saturating_mul(cols).saturating_add(j)) {
                acc += aij * xj;
            }
        }
        *yi = acc;
    }
    Ok(y)
}

/// Lower-triangular f32 Cholesky factor `L` of an SPD `dim × dim` row-major matrix
/// `b` (`b = L Lᵀ`). Returns `OutOfRange` if `b` is not positive-definite (a
/// non-positive pivot), which the caller surfaces rather than panicking.
fn cholesky_lower(b: &[f32], dim: usize) -> CbResult<Vec<f32>> {
    let mut l = vec![0.0f32; dim.saturating_mul(dim)];
    for i in 0..dim {
        for j in 0..=i {
            let mut sum = *b.get(i * dim + j).unwrap_or(&0.0);
            for k in 0..j {
                let lik = *l.get(i * dim + k).unwrap_or(&0.0);
                let ljk = *l.get(j * dim + k).unwrap_or(&0.0);
                sum -= lik * ljk;
            }
            if i == j {
                // `!(sum > 0.0)` also rejects non-finite (NaN/inf) pivots, which
                // `sum <= 0.0` silently lets through (NaN <= 0.0 is false) (CR-01).
                if !(sum > 0.0) {
                    return Err(CbError::OutOfRange(format!(
                        "cholesky: non-SPD pivot {sum} at diagonal {i}"
                    )));
                }
                if let Some(slot) = l.get_mut(i * dim + j) {
                    *slot = sum.sqrt();
                }
            } else {
                let ljj = *l.get(j * dim + j).unwrap_or(&1.0);
                if let Some(slot) = l.get_mut(i * dim + j) {
                    *slot = sum / ljj;
                }
            }
        }
    }
    Ok(l)
}

/// `ssygst_(itype=1,'L')` analog: reduce the generalized symmetric problem
/// `a x = λ b x` to the standard problem by overwriting (a copy of) `a` with
/// `inv(L) a inv(Lᵀ)`, where `b = L Lᵀ`. `a` and `b` are `dim × dim` row-major.
///
/// # Errors
/// [`CbError::OutOfRange`] on dimension mismatch or non-SPD `b`.
pub fn reduce_generalized(a: &[f32], b: &[f32], dim: usize) -> CbResult<Vec<f32>> {
    let n2 = dim.saturating_mul(dim);
    if a.len() != n2 || b.len() != n2 {
        return Err(CbError::OutOfRange(format!(
            "reduce_generalized: expected {n2}-len matrices, got a={} b={}",
            a.len(),
            b.len()
        )));
    }
    let l = cholesky_lower(b, dim)?;
    // Y = inv(L) * A : forward substitution per column.
    let mut y = vec![0.0f32; n2];
    for col in 0..dim {
        for i in 0..dim {
            let mut s = *a.get(i * dim + col).unwrap_or(&0.0);
            for k in 0..i {
                let lik = *l.get(i * dim + k).unwrap_or(&0.0);
                let ykc = *y.get(k * dim + col).unwrap_or(&0.0);
                s -= lik * ykc;
            }
            let lii = *l.get(i * dim + i).unwrap_or(&1.0);
            if let Some(slot) = y.get_mut(i * dim + col) {
                *slot = s / lii;
            }
        }
    }
    // C = Y * inv(Lᵀ) : solve C Lᵀ = Y row by row (forward over columns).
    let mut c = vec![0.0f32; n2];
    for i in 0..dim {
        for j in 0..dim {
            let mut s = *y.get(i * dim + j).unwrap_or(&0.0);
            for k in 0..j {
                let cik = *c.get(i * dim + k).unwrap_or(&0.0);
                let ljk = *l.get(j * dim + k).unwrap_or(&0.0);
                s -= cik * ljk;
            }
            let ljj = *l.get(j * dim + j).unwrap_or(&1.0);
            if let Some(slot) = c.get_mut(i * dim + j) {
                *slot = s / ljj;
            }
        }
    }
    // Symmetrize to damp f32 round-off before the eigensolver.
    let mut out = vec![0.0f32; n2];
    for i in 0..dim {
        for j in 0..dim {
            let cij = *c.get(i * dim + j).unwrap_or(&0.0);
            let cji = *c.get(j * dim + i).unwrap_or(&0.0);
            if let Some(slot) = out.get_mut(i * dim + j) {
                *slot = 0.5 * (cij + cji);
            }
        }
    }
    Ok(out)
}

/// Result of [`jacobi_symmetric_eig`]: ascending eigenvalues and the matching
/// orthonormal eigenvector **columns** (column-major `dim × dim`, mirroring how
/// `ssyev_` overwrites its input).
#[derive(Debug, Clone, PartialEq)]
pub struct SymmetricEig {
    /// Eigenvalues sorted ascending (the `ssyev_` convention).
    pub eigenvalues: Vec<f32>,
    /// Eigenvector columns aligned with `eigenvalues`, column-major.
    pub eigenvectors: Vec<f32>,
}

/// `ssyev_('V','L')` analog: cyclic-Jacobi symmetric eigensolver for a small
/// `dim × dim` row-major symmetric f32 matrix. Returns ascending eigenvalues and
/// the matching orthonormal eigenvector columns.
///
/// # Errors
/// [`CbError::OutOfRange`] on dimension mismatch.
pub fn jacobi_symmetric_eig(a_in: &[f32], dim: usize) -> CbResult<SymmetricEig> {
    let n2 = dim.saturating_mul(dim);
    if a_in.len() != n2 {
        return Err(CbError::OutOfRange(format!(
            "jacobi: matrix len {} != dim*dim {}",
            a_in.len(),
            n2
        )));
    }
    let mut a = a_in.to_vec();
    let mut v = vec![0.0f32; n2];
    for i in 0..dim {
        if let Some(slot) = v.get_mut(i * dim + i) {
            *slot = 1.0;
        }
    }
    for _sweep in 0..100 {
        let mut off = 0.0f32;
        for p in 0..dim {
            for q in (p + 1)..dim {
                let apq = *a.get(p * dim + q).unwrap_or(&0.0);
                off += apq * apq;
            }
        }
        if off < 1e-20 {
            break;
        }
        for p in 0..dim {
            for q in (p + 1)..dim {
                let apq = *a.get(p * dim + q).unwrap_or(&0.0);
                if apq.abs() < 1e-30 {
                    continue;
                }
                let app = *a.get(p * dim + p).unwrap_or(&0.0);
                let aqq = *a.get(q * dim + q).unwrap_or(&0.0);
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let cval = 1.0 / (t * t + 1.0).sqrt();
                let s = t * cval;
                jacobi_rotate(&mut a, dim, p, q, cval, s);
                rotate_vectors(&mut v, dim, p, q, cval, s);
            }
        }
    }
    let mut pairs: Vec<(f32, usize)> = (0..dim)
        .map(|i| (*a.get(i * dim + i).unwrap_or(&0.0), i))
        .collect();
    pairs.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut eigenvalues = vec![0.0f32; dim];
    let mut eigenvectors = vec![0.0f32; n2];
    for (new_col, (lambda, old_col)) in pairs.into_iter().enumerate() {
        if let Some(slot) = eigenvalues.get_mut(new_col) {
            *slot = lambda;
        }
        for row in 0..dim {
            let val = *v.get(row * dim + old_col).unwrap_or(&0.0);
            if let Some(slot) = eigenvectors.get_mut(row * dim + new_col) {
                *slot = val;
            }
        }
    }
    Ok(SymmetricEig {
        eigenvalues,
        eigenvectors,
    })
}

/// Apply a Jacobi rotation `(p,q,c,s)` to the symmetric matrix `a` (both sides).
fn jacobi_rotate(a: &mut [f32], dim: usize, p: usize, q: usize, c: f32, s: f32) {
    for k in 0..dim {
        let akp = *a.get(k * dim + p).unwrap_or(&0.0);
        let akq = *a.get(k * dim + q).unwrap_or(&0.0);
        if let Some(slot) = a.get_mut(k * dim + p) {
            *slot = c * akp - s * akq;
        }
        if let Some(slot) = a.get_mut(k * dim + q) {
            *slot = s * akp + c * akq;
        }
    }
    for k in 0..dim {
        let apk = *a.get(p * dim + k).unwrap_or(&0.0);
        let aqk = *a.get(q * dim + k).unwrap_or(&0.0);
        if let Some(slot) = a.get_mut(p * dim + k) {
            *slot = c * apk - s * aqk;
        }
        if let Some(slot) = a.get_mut(q * dim + k) {
            *slot = s * apk + c * aqk;
        }
    }
}

/// Accumulate the Jacobi rotation into the eigenvector matrix `v`.
fn rotate_vectors(v: &mut [f32], dim: usize, p: usize, q: usize, c: f32, s: f32) {
    for k in 0..dim {
        let vkp = *v.get(k * dim + p).unwrap_or(&0.0);
        let vkq = *v.get(k * dim + q).unwrap_or(&0.0);
        if let Some(slot) = v.get_mut(k * dim + p) {
            *slot = c * vkp - s * vkq;
        }
        if let Some(slot) = v.get_mut(k * dim + q) {
            *slot = s * vkp + c * vkq;
        }
    }
}

/// Full `CalculateProjection` analog (`lda.cpp:31-59`): given the regularized
/// between-class scatter `scatter_inner` and the total scatter `scatter_total`
/// (`dim × dim` row-major), reduce the generalized problem and take the
/// eigenvector of the **largest** eigenvalue as the `proj_dim`-row projection
/// (here `proj_dim` rows are the trailing eigenvector columns; for the binary
/// LDA fixtures `proj_dim = 1`).
///
/// Returns `(projection, eigenvalues)` where `projection` is `proj_dim × dim`
/// row-major (each row a projection direction) and `eigenvalues` is ascending.
///
/// # Errors
/// [`CbError::OutOfRange`] on dimension mismatch, non-SPD `scatter_inner`, or
/// `proj_dim > dim`.
pub fn calculate_projection(
    scatter_inner: &[f32],
    scatter_total: &[f32],
    dim: usize,
    proj_dim: usize,
) -> CbResult<(Vec<f32>, Vec<f32>)> {
    if proj_dim > dim {
        return Err(CbError::OutOfRange(format!(
            "calculate_projection: proj_dim {proj_dim} > dim {dim}"
        )));
    }
    let reduced = reduce_generalized(scatter_total, scatter_inner, dim)?;
    let eig = jacobi_symmetric_eig(&reduced, dim)?;
    // Trailing proj_dim eigenvector columns = the largest eigenvalues' vectors.
    let mut projection = vec![0.0f32; proj_dim.saturating_mul(dim)];
    for r in 0..proj_dim {
        let col = dim.saturating_sub(proj_dim).saturating_add(r);
        for k in 0..dim {
            let val = *eig.eigenvectors.get(k * dim + col).unwrap_or(&0.0);
            if let Some(slot) = projection.get_mut(r * dim + k) {
                *slot = val;
            }
        }
    }
    Ok((projection, eig.eigenvalues))
}
