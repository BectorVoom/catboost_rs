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
    compute_group_max_weighted_host, derive_query_seeds_inline, descending_order_per_query,
    pfound_f_ders_host, query_softmax_ders_host, ranking_objective_covered, yetirank_draw_count,
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

/// The INDEPENDENT reference: a CPU stable DESCENDING sort per query, ties broken by ASCENDING
/// original index. This is deliberately NOT the complemented-key radix path under test — it is a
/// plain `sort_by` over `(value desc, index asc)` — so the oracle is non-tautological (a different
/// algorithm reaching the same contract). `perturbed` values are all non-negative here (they model
/// `exp(approx)` ratios), matching the production precondition.
fn cpu_stable_descending_order(perturbed: &[f64], q_offsets: &[u32]) -> Vec<u32> {
    let mut out: Vec<u32> = Vec::with_capacity(perturbed.len());
    for w in q_offsets.windows(2) {
        let begin = w[0] as usize;
        let end = w[1] as usize;
        let mut idx: Vec<u32> = (begin as u32..end as u32).collect();
        // Stable sort by value DESCENDING; `sort_by` is stable, so equal-value docs retain their
        // pre-sort (ascending original-index) order — exactly the upstream stable descending sort.
        idx.sort_by(|&a, &b| {
            let va = perturbed[a as usize];
            let vb = perturbed[b as usize];
            vb.partial_cmp(&va).expect("perturbed values are finite in the fixture")
        });
        out.extend(idx);
    }
    out
}

#[test]
fn tie_order_matches_cpu_stable_descending() {
    // RV-13-01 (HARD-03, D-02): prove `descending_order_per_query` preserves ORIGINAL-index order
    // for TIED values, matching a CPU stable descending sort. This is the divergent path that the
    // frozen-fixture "no ties" assertion never exercised.
    //
    // Fixture: two queries, sizes 5 / 4. Deliberate exact ties WITHIN each query (repeated f64 bit
    // patterns model `exp(approx)` collisions + f32-Gumbel-induced equal values). Ties are placed at
    // non-adjacent original indices so a tie-flip would be observable, and one tie value is repeated
    // three times (index 1,2,4 in query 0) to catch a partial reversal.
    let perturbed = vec![
        // query 0 (indices 0..5): 2.0, 1.5, 1.5, 3.0, 1.5  → desc: idx3(3.0), idx0(2.0), then the
        // three 1.5 ties MUST come out ascending-index: idx1, idx2, idx4.
        2.0, 1.5, 1.5, 3.0, 1.5, //
        // query 1 (indices 5..9): 0.5, 0.5, 0.9, 0.5 → desc: idx7(0.9), then 0.5 ties ascending:
        // idx5, idx6, idx8.
        0.5, 0.5, 0.9, 0.5,
    ];
    let q_offsets = vec![0u32, 5, 9];

    // Sanity: the fixture actually contains ties (else the oracle is vacuous — Pitfall 1 guard).
    assert_eq!(perturbed[1], perturbed[2], "fixture must contain an exact tie");
    assert_eq!(perturbed[2], perturbed[4], "fixture must contain a 3-way tie");

    let got = descending_order_per_query(&perturbed, &q_offsets).expect("descending order");
    let want = cpu_stable_descending_order(&perturbed, &q_offsets);
    assert_eq!(got.len(), perturbed.len(), "order must be a full permutation");

    // The order is produced by the device `segmented_radix_sort` (`plane_inclusive_sum`), which is
    // UNSUPPORTED on the cubecl `cpu` backend — so the exact-permutation assertion is device-gated
    // (rocm/cuda), matching the WR-01 record-only discipline used for the ε der asserts above. On
    // `cpu` the "device" sort is non-functional, so an order assert there would be meaningless, not a
    // validation.
    if device_backend_active() {
        assert_eq!(
            got, want,
            "RV-13-01: device descending order diverged from CPU stable descending sort\n \
             got  = {got:?}\n want = {want:?}"
        );
        // Explicit expected order documents the tie contract for the reader / 15-EVIDENCE.
        assert_eq!(
            got,
            vec![3u32, 0, 1, 2, 4, 7, 5, 6, 8],
            "RV-13-01: tied values must stay in ascending original-index order per query"
        );
        println!(
            "RV-13-01 tie_order: device order == CPU stable descending sort (ties index-ascending)"
        );
    } else {
        // Still exercise the code path (build + launch) and expose the reference for the record.
        println!(
            "RV-13-01 tie_order: cpu backend — order assert skipped (radix sort is device-only, \
             WR-01 record-only). cpu_reference want = {want:?}"
        );
    }
}

/// The RV-13-02 fixture (2 queries, sizes 3 / 3, `n = 6`). Query 0 is WEIGHTED and its global-max
/// -approx document (`approx[0] = 3.0`) has `weight = 0.0` — so the weight-blind (pre-fix) seed 3.0
/// diverges from the CPU weight>0 seed 1.0. Query 1 is uniform-weight (regression guard: weight>0
/// max == global max == 0.8).
fn softmax_fixture() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<u32>, f64, f64) {
    let approx = vec![3.0, 1.0, 0.5, 0.2, 0.8, -0.1];
    let target = vec![1.0, 1.0, 1.0, 1.0, 0.0, 1.0];
    let weights = vec![0.0, 2.0, 1.0, 1.0, 1.0, 1.0];
    let q_offsets = vec![0u32, 3, 6];
    let beta = 1.0;
    let lambda = 0.0;
    (approx, target, weights, q_offsets, beta, lambda)
}

/// FROZEN CPU QuerySoftMax der for [`softmax_fixture`], generated ONCE offline from the INDEPENDENT
/// `cb_compute` QuerySoftMax math (`ranking_der.rs:251-322` + `loss.rs::querysoftmax_der`,
/// `:608-619`) seeded with the WEIGHT>0 per-query max. Generation recipe (transcribe the CPU
/// formula, do NOT call the device path — non-tautology):
/// ```text
/// per query g with begin/end:
///   max_a  = max{ approx[i] : weight[i] > 0 }               # weight>0 seed (the parity target)
///   sum_wt = Σ_{w>0, t>0} weight[i]·target[i]
///   if sum_wt > 0:
///     sum_exp = Σ_i exp(beta·(approx[i] − max_a))·weight[i]
///     for i in begin..end:
///       if weight[i] > 0 and sum_exp > 0:
///         p    = exp(beta·(approx[i] − max_a))·weight[i] / sum_exp
///         der2 = beta·sum_wt·(beta·p·(p−1) − lambda)
///         der1 = beta·(−sum_wt·p + weight[i]·target[i])
///       else: der1 = der2 = 0
///   else: der1 = der2 = 0
/// ```
/// (beta = 1.0, lambda = 0.0). A weight-blind seed (max over ALL objects) yields the SAME der to
/// ~1e-16 here because softmax is shift-invariant in exact arithmetic; the oracle's TEETH are in
/// [`softmax_weight_max_seed`]'s direct assertion that the SEED is the weight>0 max (1.0), not the
/// global max (3.0) — the pre-fix behaviour that this fix corrects.
fn frozen_softmax_der1() -> Vec<f64> {
    vec![
        0.0,
        -0.3019103871433044,
        0.3019103871433041,
        0.4386653515985748,
        -1.022818416162833,
        0.5841530645642584,
    ]
}
fn frozen_softmax_der2() -> Vec<f64> {
    vec![
        0.0,
        -0.5356465769972252,
        -0.5356465769972254,
        -0.40378635465344936,
        -0.4997396599419099,
        -0.32938259858009267,
    ]
}

#[test]
fn softmax_weight_max_seed() {
    // RV-13-02 (HARD-03, D-02): the QuerySoftMax exp-shift must be seeded from the WEIGHT>0 max,
    // matching CPU `ranking_der.rs:257-266`.
    let (approx, target, weights, q_offsets, beta, lambda) = softmax_fixture();

    // (a) TEETH — direct, backend-agnostic assertion on the SEED SELECTION. The weighted max helper
    // is pure host code (no device kernel), so it runs everywhere. The pre-fix weight-blind seed
    // would return 3.0 for query 0 (the global max on the weight-0 doc); the fix returns 1.0.
    let weight_col = weights.clone();
    let seeds = compute_group_max_weighted_host(&approx, &weight_col, &q_offsets);
    assert_eq!(
        seeds,
        vec![1.0, 0.8],
        "RV-13-02: per-query seed must be the WEIGHT>0 max (q0=1.0, not the global 3.0 on the \
         weight-0 doc; q1=0.8)"
    );

    // (b) der parity — device-gated ε (WR-01 record-only on cpu; still exercised + finiteness-checked
    // there). The der driver runs on `SelectedRuntime`; on a real device it must match the frozen CPU
    // reference within ε=1e-4.
    let (der1, der2) =
        query_softmax_ders_host(&approx, &target, &weights, &q_offsets, beta, lambda)
            .expect("query_softmax der");
    assert_eq!(der1.len(), approx.len());
    assert_eq!(der2.len(), approx.len());
    for v in der1.iter().chain(der2.iter()) {
        assert!(v.is_finite(), "RV-13-02 softmax der must be finite (no inf/NaN)");
    }
    if device_backend_active() {
        assert_der_close(&der1, &frozen_softmax_der1(), "softmax der1");
        assert_der_close(&der2, &frozen_softmax_der2(), "softmax der2");
    } else {
        println!(
            "RV-13-02 softmax_weight_max_seed: cpu backend — numeric ε assert skipped (WR-01 \
             record-only); seed-selection assert (weight>0 max) ran and passed"
        );
    }
}
