//! Self-oracle for the STOCHASTIC ranking pair (Phase 13 Plan 05, GPUT-22, D-08): the device
//! YetiRank / PFound-F der ([`crate::gpu_runtime::ranking::yetirank_ders_host`] /
//! `pfound_f_ders_host`) must reproduce the FROZEN pinned-seed CPU reference
//! (`cb_train::yetirank::sample_pairs` + `cb_compute::calc_ders_for_queries`, the LOSS-04 grouped der
//! seam, itself oracle-tested to ≤1e-5) over a 2-query fixture within ε=1e-4 (the D-07 GPU bar).
//!
//! # Why FROZEN literals (not a live cb-train call)
//!
//! `cb-backend` must NEVER gain a `cb-train` dependency (the feature-unification landmine). The CPU
//! reference der + the per-query seeds are therefore FROZEN here as literals, generated ONCE offline
//! from the INDEPENDENT `cb_train::yetirank_sample_pairs` + `cb_compute::calc_ders_for_queries` (a
//! different implementation from the device path), so this remains a NON-tautological oracle. The
//! generator fixture: `approx=[0.5,-0.2,1.3, 0.1,0.9,-0.3]`, `target=[1,0,2, 0,1,2]`,
//! `q_offsets=[0,3,6]`, `permutations=4`, `decay=0.85`, `random_seed=42`.
//!
//! # Draw COUNT (Pitfall 4 / T-13-10)
//!
//! The consumed `gen_rand_real1` count is `permutations · n` (perm-major, doc-ascending, per query).
//! A divergent count silently desyncs every subsequent draw beyond ε=1e-4, so it is asserted
//! independently ([`draw_count_matches`]) — a wrong count is DETECTED, not absorbed. The per-query
//! seeds ([`derive_query_seeds_inline`]) are likewise asserted against the frozen CPU chain.
//!
//! Source/test separation is mandatory (CLAUDE.md / AGENTS.md): the device der driver lives in the
//! production `crate::gpu_runtime::ranking` module; ALL assertions / `.unwrap()` / indexing live
//! here. Runs over [`crate::SelectedRuntime`]; the numeric ε=1e-4 vs-CPU assertion SKIPS off
//! rocm/cuda (record-only) so a default `cpu` run does not masquerade as a GPU validation (WR-01
//! anti-false-pass — on `cpu` the "device" IS the host, a CPU-vs-CPU comparison).

#![cfg(not(feature = "wgpu"))]

use crate::gpu_runtime::ranking::{
    derive_query_seeds_inline, pfound_f_ders_host, ranking_objective_covered, yetirank_draw_count,
    yetirank_ders_host, RankingObjective,
};

/// The ε=1e-4 device-vs-CPU bar (D-07; the GPU bar, looser than the CPU ref's own ≤1e-5).
const TOL: f64 = 1e-4;

/// Whether the numeric ε assertion runs on a REAL device (rocm/cuda) — else record-only (WR-01):
/// on the `cpu` backend the "device" IS the host, so a numeric assert would be a CPU-vs-CPU
/// false-pass rather than a GPU validation.
fn device_backend_active() -> bool {
    cfg!(any(feature = "rocm", feature = "cuda"))
}

/// The pinned stochastic fixture (2 queries, sizes 3 / 3, `n = 6`), uniform object weights.
fn fixture() -> (Vec<f64>, Vec<f64>, Vec<u32>, u32, f64, u64) {
    let approx = vec![0.5, -0.2, 1.3, 0.1, 0.9, -0.3];
    let target = vec![1.0, 0.0, 2.0, 0.0, 1.0, 2.0];
    let q_offsets = vec![0u32, 3, 6];
    (approx, target, q_offsets, 4, 0.85, 42)
}

/// The FROZEN CPU reference der (generated offline from `cb_train::yetirank_sample_pairs` +
/// `cb_compute::calc_ders_for_queries` for the pinned fixture; see the module docs).
fn frozen_der1() -> Vec<f64> {
    vec![
        -0.0014044343755522382,
        -0.059093373351386924,
        0.06049780772693916,
        -0.10946665330559988,
        -0.049002110000118525,
        0.1584687633057184,
    ]
}
fn frozen_der2() -> Vec<f64> {
    vec![
        -0.04406289243780697,
        -0.0435732618117321,
        -0.04523356802513898,
        -0.05349484383482564,
        -0.041874068093867685,
        -0.049645713277479464,
    ]
}

/// The FROZEN per-query inner Gumbel seeds (`derive_query_seeds(42, 2)` from the CPU chain).
const FROZEN_SEEDS: [u64; 2] = [11_230_867_027_785_084_481, 17_628_939_635_266_696_464];

/// Assert two flat der vectors agree within `TOL` (element-wise), reporting the worst divergence.
fn assert_der_close(got: &[f64], want: &[f64], who: &str) {
    assert_eq!(got.len(), want.len(), "{who}: der length mismatch");
    let mut max_div = 0.0_f64;
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let div = (g - w).abs();
        if div > max_div {
            max_div = div;
        }
        assert!(
            div <= TOL,
            "{who}: der[{i}] = {g} vs frozen {w} (|Δ| = {div} > {TOL})"
        );
    }
    println!("{who}: max_div = {max_div:.3e} (<= {TOL})");
}

#[test]
fn seeds_match_frozen_cpu_chain() {
    // The per-query O(1) base state the device re-expands MUST match the CPU 2-level derivation.
    let seeds = derive_query_seeds_inline(42, 2);
    assert_eq!(seeds.as_slice(), &FROZEN_SEEDS, "per-query seed chain diverged");
}

#[test]
fn draw_count_matches() {
    // permutations · n draws (perm-major, doc-ascending). A divergent count desyncs every value.
    let (_a, _t, _o, perms, _d, _s) = fixture();
    let n = 6usize;
    assert_eq!(yetirank_draw_count(n, perms), perms as usize * n);
    assert_eq!(yetirank_draw_count(n, perms), 24, "expected 4·6 = 24 uniform draws");
}

#[test]
fn stochastic_pair_is_covered() {
    // Both stochastic arms are in the COVERED ranking set (independently gated ON, unlike the
    // deferred QueryCrossEntropy).
    assert!(ranking_objective_covered(RankingObjective::YetiRank {
        permutations: 4,
        decay: 0.85
    }));
    assert!(ranking_objective_covered(RankingObjective::PFoundF {
        permutations: 4,
        decay: 0.85
    }));
}

#[test]
fn yetirank_der_matches_frozen_cpu() {
    let (approx, target, q_offsets, perms, decay, seed) = fixture();
    let (der1, der2) =
        yetirank_ders_host(&approx, &target, &[], &q_offsets, perms, decay, seed).expect("yetirank");
    assert_eq!(der1.len(), approx.len());
    assert_eq!(der2.len(), approx.len());
    for v in der1.iter().chain(der2.iter()) {
        assert!(v.is_finite(), "yetirank der must be finite");
    }
    if device_backend_active() {
        assert_der_close(&der1, &frozen_der1(), "yetirank der1");
        assert_der_close(&der2, &frozen_der2(), "yetirank der2");
    } else {
        println!("yetirank: cpu backend — numeric ε assert skipped (WR-01 record-only)");
    }
}

#[test]
fn pfound_f_der_matches_frozen_cpu() {
    // PFound-F (YetiRankPairwise) shares the SAME sampled-pair der stream as YetiRank (only the leaf
    // path differs, decided later in boosting), so it reproduces the SAME frozen CPU reference.
    let (approx, target, q_offsets, perms, decay, seed) = fixture();
    let (der1, der2) =
        pfound_f_ders_host(&approx, &target, &[], &q_offsets, perms, decay, seed).expect("pfound_f");
    assert_eq!(der1.len(), approx.len());
    assert_eq!(der2.len(), approx.len());
    for v in der1.iter().chain(der2.iter()) {
        assert!(v.is_finite(), "pfound_f der must be finite");
    }
    if device_backend_active() {
        assert_der_close(&der1, &frozen_der1(), "pfound_f der1");
        assert_der_close(&der2, &frozen_der2(), "pfound_f der2");
    } else {
        println!("pfound_f: cpu backend — numeric ε assert skipped (WR-01 record-only)");
    }
}
