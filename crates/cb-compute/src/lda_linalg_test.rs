//! Tests for the LDA linear-algebra spike candidate ([`crate::lda_linalg`]).
//!
//! These cover the eigensolver's internal correctness (round-trip on a known SPD
//! generalized pair) and the GEMV project step. The *parity vs the instrumented
//! dump* divergence measurement lives in the Plan 06.5-05 Task 1 checkpoint
//! report, not as an assertion here, because the eigensolver decision
//! (hand-roll-f32 vs LAPACK crate vs documented tolerance) is the open checkpoint.

use crate::lda_linalg::{
    calculate_projection, jacobi_symmetric_eig, reduce_generalized, sgemv_rowmajor,
};

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn sgemv_matches_manual_dot() {
    // 2x3 row-major A, x len 3.
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = [1.0f32, 0.0, -1.0];
    let y = sgemv_rowmajor(&a, 2, 3, &x).expect("dims ok");
    // row0: 1*1 + 2*0 + 3*-1 = -2 ; row1: 4 - 6 = -2
    assert!((y[0] - (-2.0)).abs() < 1e-6);
    assert!((y[1] - (-2.0)).abs() < 1e-6);
}

#[test]
fn sgemv_rejects_bad_dims() {
    let a = [1.0f32, 2.0, 3.0];
    assert!(sgemv_rowmajor(&a, 2, 2, &[1.0, 2.0]).is_err());
    assert!(sgemv_rowmajor(&[1.0, 2.0, 3.0, 4.0], 2, 2, &[1.0]).is_err());
}

#[test]
fn jacobi_diagonal_is_identity_eig() {
    // Diagonal matrix -> eigenvalues == diagonal (ascending), eigenvectors == basis.
    let a = [3.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0];
    let eig = jacobi_symmetric_eig(&a, 3).expect("dims ok");
    assert!((eig.eigenvalues[0] - 1.0).abs() < 1e-5);
    assert!((eig.eigenvalues[1] - 2.0).abs() < 1e-5);
    assert!((eig.eigenvalues[2] - 3.0).abs() < 1e-5);
}

#[test]
fn jacobi_reconstructs_symmetric_matrix() {
    // Symmetric 2x2 [[2,1],[1,2]] -> eigenvalues 1 and 3.
    let a = [2.0f32, 1.0, 1.0, 2.0];
    let eig = jacobi_symmetric_eig(&a, 2).expect("dims ok");
    assert!((eig.eigenvalues[0] - 1.0).abs() < 1e-5);
    assert!((eig.eigenvalues[1] - 3.0).abs() < 1e-5);
    // Eigenvectors orthonormal: column dot products.
    let v = &eig.eigenvectors;
    let dot00 = v[0] * v[0] + v[2] * v[2];
    let dot01 = v[0] * v[1] + v[2] * v[3];
    assert!((dot00 - 1.0).abs() < 1e-5);
    assert!(dot01.abs() < 1e-5);
}

#[test]
fn reduce_generalized_recovers_known_pair() {
    // a = 2*I, b = I -> generalized eigenvalues all 2.0 ; reduced = 2*I.
    let a = [2.0f32, 0.0, 0.0, 2.0];
    let b = [1.0f32, 0.0, 0.0, 1.0];
    let reduced = reduce_generalized(&a, &b, 2).expect("spd ok");
    assert!((reduced[0] - 2.0).abs() < 1e-5);
    assert!((reduced[3] - 2.0).abs() < 1e-5);
    let eig = jacobi_symmetric_eig(&reduced, 2).expect("dims ok");
    assert!((eig.eigenvalues[1] - 2.0).abs() < 1e-5);
}

#[test]
fn reduce_generalized_rejects_non_spd() {
    // b not positive-definite (zero diagonal) -> error, never panic.
    let a = [1.0f32, 0.0, 0.0, 1.0];
    let b = [0.0f32, 0.0, 0.0, 0.0];
    assert!(reduce_generalized(&a, &b, 2).is_err());
}

#[test]
fn calculate_projection_is_largest_eigenvector() {
    // a = diag(10,1,1,1), b = I -> largest generalized eigenvalue is 10 along axis 0.
    // proj_dim=1 -> projection row ~ e0 (up to sign).
    let mut a = vec![0.0f32; 16];
    a[0] = 10.0;
    a[5] = 1.0;
    a[10] = 1.0;
    a[15] = 1.0;
    let mut b = vec![0.0f32; 16];
    for d in 0..4 {
        b[d * 4 + d] = 1.0;
    }
    let (proj, eig) = calculate_projection(&b, &a, 4, 1).expect("spd ok");
    // a is scatter_total here (passed 2nd); b is scatter_inner (=I). Largest eig = 10.
    assert!((eig[3] - 10.0).abs() < 1e-4);
    // projection is e0 up to sign.
    let aligned = if proj[0] < 0.0 {
        proj.iter().map(|v| -v).collect::<Vec<_>>()
    } else {
        proj.clone()
    };
    let e0 = [1.0f32, 0.0, 0.0, 0.0];
    assert!(max_abs_err(&aligned, &e0) < 1e-4);
}
