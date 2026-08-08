//! Self-oracle for the device CTR accumulation (Phase 12 Plan 08, GPUT-10, Pattern F): the device
//! ordered / one-hot / tensor CTR column ([`crate::kernels::ctr_device`]) must reproduce a FROZEN
//! CPU reference — an inline serial transcription of `cb_train::ctr::online::online_ctr_prefix_binclf`
//! (read-before-increment, object-order output) + `calc_ctr_online` — within the ε=1e-4 device bar,
//! and the CTR→cindex binarize JOIN must reproduce the host binarization bit-exact.
//!
//! Source/test separation is mandatory (CLAUDE.md / AGENTS.md): the device kernels live in the
//! production `kernels::ctr_device` module; ALL assertions / `.unwrap()` / indexing live here. The
//! CPU reference is TRANSCRIBED inline (NOT `use cb_train` — the landmine is cb-TRAIN, the whole
//! reason cb-backend must not depend on it), so this is a genuine device-`#[cube]`-vs-independent-
//! host-scan oracle, not a tautology.
//!
//! Runs over [`crate::SelectedRuntime`]. Unlike the resident-grow oracles, the ordered CTR is a
//! SERIAL SCAN with EXACT INTEGER prefix counting (no `Atomic<u64>`, no resident histogram), so it
//! executes IN-ENV on the default cpu backend as well as rocm/cuda. Only the wgpu backend (no f64)
//! cannot run the f64 CTR value seam — the assertions SKIP there (WR-01 anti-false-pass).

use crate::kernels::ctr_device::{
    binarize_counter_column_host, binarize_ctr_column_host, combine_projection_bins,
    compute_counter_ctr_host, compute_ordered_ctr_host, compute_ordered_ctr_host_mode,
};

/// `cb_train::ctr::ECtrType::Borders` discriminant (`cb-train/src/ctr/mod.rs:96-108`),
/// transcribed by value — cb-backend must never gain a `cb-train` dep (Pattern B / C-3).
const CTR_TYPE_BORDERS: i8 = 0;
/// `cb_train::ctr::ECtrType::Buckets` discriminant (`cb-train/src/ctr/mod.rs:96-108`).
const CTR_TYPE_BUCKETS: i8 = 1;

/// The ε=1e-4 device-vs-CPU bar (D-07; the GPU bar, looser than the CPU ref's own ≤1e-5).
const TOL: f64 = 1e-4;

/// Whether the device f64 CTR seam runs on this backend (cpu/rocm/cuda have f64; wgpu does not).
fn device_ctr_active() -> bool {
    !cfg!(feature = "wgpu")
}

/// Deterministic pseudo-random `u32` stream (LCG — no `rand` dep), matching the cindex oracle's
/// generator so the fixtures are reproducible.
fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state >> 8
}

/// A deterministic categorical bin column (`obj -> bucket in 0..cardinality`) + a binclf class
/// column + a learn permutation, all in object order, from one seed.
fn synth_fixture(n: usize, cardinality: u32, seed: u32) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut s = seed;
    let bins: Vec<u32> = (0..n).map(|_| lcg(&mut s) % cardinality).collect();
    let class: Vec<u32> = (0..n).map(|_| lcg(&mut s) % 2).collect();
    // A non-identity learn permutation (rotate + swap) so read-order matters.
    let mut perm: Vec<u32> = (0..n as u32).collect();
    for i in 0..n {
        let j = (lcg(&mut s) as usize) % n;
        perm.swap(i, j);
    }
    (bins, class, perm)
}

/// The FROZEN CPU ordered CTR reference: an inline serial transcription of
/// `online_ctr_prefix_binclf` (read-before-increment, object order) + `calc_ctr_online`
/// (`(good + prior) / (total + 1)`). Returns `(good, total, value)` in OBJECT order.
///
/// DCTR-06: the per-bucket NUMERATOR is selected by `(is_buckets, target_border_idx)` through an
/// inline transcription of `cb_train::ctr::online::online_class_prefix` (`online.rs:552-569`,
/// upstream `UpdateGoodCount`, `online_ctr.cpp:115-121`) kept in its GENERIC (loop-over-classes)
/// form deliberately: the device kernel implements the 2-class COLLAPSE, so comparing the two is a
/// genuine cross-check rather than a copy of the same arithmetic. `(false, 0)` — Borders at target
/// border 0 — is the historical behaviour and reduces to `good = N1` exactly.
fn cpu_ordered_ctr(
    perm: &[u32],
    bins: &[u32],
    class: &[u32],
    prior: f64,
    is_buckets: bool,
    target_border_idx: usize,
) -> (Vec<i64>, Vec<i64>, Vec<f64>) {
    let n = perm.len();
    let bucket_count = bins.iter().copied().max().map_or(0, |m| m as usize + 1);
    let mut counts = vec![[0i64; 2]; bucket_count];
    let mut good = vec![0i64; n];
    let mut total = vec![0i64; n];
    let mut value = vec![0f64; n];
    for &doc_i in perm {
        let doc = doc_i as usize;
        let bucket = bins[doc] as usize;
        // READ before increment (the no-leakage invariant).
        let prefix = counts[bucket];
        // `online_class_prefix`: total is ALWAYS the bucket total, whatever the CTR type.
        let t: i64 = prefix.iter().sum();
        let g = if is_buckets {
            prefix.get(target_border_idx).copied().unwrap_or(0)
        } else {
            let head: i64 = (0..=target_border_idx)
                .map(|c| prefix.get(c).copied().unwrap_or(0))
                .sum();
            t.saturating_sub(head)
        };
        good[doc] = g;
        total[doc] = t;
        value[doc] = (g as f64 + prior) / (t as f64 + 1.0);
        counts[bucket][class[doc] as usize] += 1;
    }
    (good, total, value)
}

/// The FROZEN CPU Counter reference: an inline transcription of
/// `cb_train::ctr::online::online_counter_column` (`cb-train/src/ctr/online.rs:493-521`) followed
/// by `calc_ctr_online` (`cb-train/src/ctr/calc_ctr.rs:77-80`), i.e. exactly what
/// `ctr_feature.rs:296-309` composes for [`cb_train::ctr::ECtrType::Counter`].
///
/// Counter is a **whole-set tally with a CONSTANT denominator**: `totals[b] = #{obj : bins[obj] == b}`
/// over the entire learn set, `denominator = max_b totals[b]`, and per object
/// `value = (totals[bins[obj]] + prior) / (denominator + 1)`. It takes **no permutation parameter
/// at all** — `IsPermutationDependentCtrType(Counter) == false` (`ctr_type.cpp:43-56`).
///
/// `extra_bins` (the `counter_calc_method = Full` eval widening, `online_ctr.cpp:713-729`) is
/// deliberately absent: P1 ships no eval-widening device code and the boundary is pinned by T13.
///
/// Returns `(good, denominator, value)` in OBJECT order.
fn cpu_counter_ctr(bins: &[u32], prior: f64, bucket_count: usize) -> (Vec<i64>, i64, Vec<f64>) {
    let mut totals = vec![0i64; bucket_count];
    for &bin in bins {
        if let Some(slot) = totals.get_mut(bin as usize) {
            *slot += 1;
        }
    }
    let denominator = totals.iter().copied().max().unwrap_or(0);
    let good: Vec<i64> = bins
        .iter()
        .map(|&bin| totals.get(bin as usize).copied().unwrap_or(0))
        .collect();
    let value: Vec<f64> = good
        .iter()
        .map(|&g| (g as f64 + prior) / (denominator as f64 + 1.0))
        .collect();
    (good, denominator, value)
}

/// `max |a - b|` over two equal-length f64 vectors; `INFINITY` on a length mismatch so a truncated
/// device buffer fails loudly (WR-06, mirroring `grow_loop::max_divergence`).
fn max_divergence(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// The bucket count `max(bins) + 1` a device launch must size its scratch to.
fn bucket_count(bins: &[u32]) -> usize {
    bins.iter().copied().max().map_or(0, |m| m as usize + 1)
}

#[test]
fn ordered_ts_matches_cpu_reference() {
    // Ordered target statistic (the headline): the device read-before-increment prefix column
    // must reproduce the frozen CPU reference — good/total EXACT integers, value ≤1e-4.
    if !device_ctr_active() {
        eprintln!("SKIP ordered_ts_matches_cpu_reference: wgpu has no f64 CTR seam (WR-01)");
        return;
    }
    let n = 400usize;
    let prior = 0.5;
    for &seed in &[1u32, 42, 12345] {
        let (bins, class, perm) = synth_fixture(n, 7, seed);
        let (cg, ct, cv) = cpu_ordered_ctr(&perm, &bins, &class, prior, false, 0);
        let (dg, dt, dv) =
            compute_ordered_ctr_host(&perm, &bins, &class, prior, bucket_count(&bins)).unwrap();
        assert_eq!(dg, cg, "device good count != CPU (seed {seed})");
        assert_eq!(dt, ct, "device total count != CPU (seed {seed})");
        let div = max_divergence(&dv, &cv);
        assert!(div <= TOL, "ordered CTR value divergence {div} > {TOL} (seed {seed})");
    }
}

#[test]
fn first_doc_in_segment_reads_prior() {
    // The FIRST document to land in each bucket (in permutation order) reads the PRIOR alone —
    // good = total = 0, value = prior / 1 — never its own label (the no-leakage invariant).
    if !device_ctr_active() {
        eprintln!("SKIP first_doc_in_segment_reads_prior: wgpu has no f64 CTR seam (WR-01)");
        return;
    }
    let n = 200usize;
    let prior = 0.5;
    let (bins, class, perm) = synth_fixture(n, 5, 777);
    let (dg, dt, dv) =
        compute_ordered_ctr_host(&perm, &bins, &class, prior, bucket_count(&bins)).unwrap();

    // Walk the permutation, tracking each bucket's first appearance.
    let mut seen = vec![false; bucket_count(&bins)];
    for &doc_i in &perm {
        let doc = doc_i as usize;
        let bucket = bins[doc] as usize;
        if !seen[bucket] {
            seen[bucket] = true;
            assert_eq!(dg[doc], 0, "first doc in bucket {bucket} must read good=0");
            assert_eq!(dt[doc], 0, "first doc in bucket {bucket} must read total=0");
            assert!(
                (dv[doc] - prior).abs() <= TOL,
                "first doc in bucket {bucket} must read value=prior ({prior}), got {}",
                dv[doc]
            );
        }
    }
}

#[test]
fn one_hot_ctr_matches_cpu_reference() {
    // One-hot regime: a SMALL-cardinality categorical (each category its own bucket) rides the
    // SAME device prefix kernel as the ordered TS (A5 — only the bucket source differs).
    if !device_ctr_active() {
        eprintln!("SKIP one_hot_ctr_matches_cpu_reference: wgpu has no f64 CTR seam (WR-01)");
        return;
    }
    let n = 256usize;
    let prior = 1.0;
    let (bins, class, perm) = synth_fixture(n, 3, 9091); // cardinality 3 => one-hot regime
    let (cg, ct, cv) = cpu_ordered_ctr(&perm, &bins, &class, prior, false, 0);
    let (dg, dt, dv) =
        compute_ordered_ctr_host(&perm, &bins, &class, prior, bucket_count(&bins)).unwrap();
    assert_eq!(dg, cg, "one-hot device good != CPU");
    assert_eq!(dt, ct, "one-hot device total != CPU");
    assert!(max_divergence(&dv, &cv) <= TOL, "one-hot CTR value divergence > {TOL}");
}

#[test]
fn tensor_combination_ctr_matches_cpu_reference() {
    // Tensor / feature-combination CTR (A5): TWO cat columns fold into one combined-projection
    // bin column (device-host `combine_projection_bins`), then the SAME ordered prefix runs on the
    // combined bins. Both the device path and the CPU reference consume the SAME combined bins, so
    // the oracle validates the ordered-CTR-over-combined-bins MATH (the exact upstream combined_hash
    // parity is the Kaggle CUDA sign-off, Plan 09). Identical (categoryA, categoryB) pairs must
    // share the SAME combined bucket (asserted below).
    if !device_ctr_active() {
        eprintln!("SKIP tensor_combination_ctr_matches_cpu_reference: wgpu has no f64 seam (WR-01)");
        return;
    }
    let n = 300usize;
    let prior = 0.5;
    let (col_a, class, perm) = synth_fixture(n, 4, 202);
    let (col_b, _c2, _p2) = synth_fixture(n, 3, 606);
    let (combined, buckets) = combine_projection_bins(&[col_a.clone(), col_b.clone()], n).unwrap();
    assert_eq!(combined.len(), n);
    // Same (a, b) pair => same combined bin (canonical CTR per category-within-combination).
    for i in 0..n {
        for j in 0..n {
            if col_a[i] == col_a[j] && col_b[i] == col_b[j] {
                assert_eq!(combined[i], combined[j], "same (a,b) pair must share a combined bucket");
            }
        }
    }
    assert!(buckets <= 4 * 3, "combined buckets bounded by the cartesian product");

    let (cg, ct, cv) = cpu_ordered_ctr(&perm, &combined, &class, prior, false, 0);
    let (dg, dt, dv) = compute_ordered_ctr_host(&perm, &combined, &class, prior, buckets).unwrap();
    assert_eq!(dg, cg, "tensor device good != CPU");
    assert_eq!(dt, ct, "tensor device total != CPU");
    assert!(max_divergence(&dv, &cv) <= TOL, "tensor CTR value divergence > {TOL}");
}

#[test]
fn ctr_binarized_cindex_column_bit_exact() {
    // The CTR→cindex binarize JOIN: the device binarizes the accumulated CTR VALUES into an extra
    // cindex bin column (`bin = #{borders < value}`, the `> bin` convention the histogram loop
    // reads). It must reproduce the host binarization of the CPU CTR column BIT-EXACT (integer
    // equality — tighter than the ε=1e-4 float bar, mirroring the cindex D-07 discipline).
    if !device_ctr_active() {
        eprintln!("SKIP ctr_binarized_cindex_column_bit_exact: wgpu has no f64 seam (WR-01)");
        return;
    }
    let n = 300usize;
    let prior = 0.5;
    let (bins, class, perm) = synth_fixture(n, 6, 314);
    // CTR values lie in (0, 1] for these priors; borders split that range into cindex buckets.
    let borders = vec![0.2_f64, 0.4, 0.5, 0.6, 0.8];

    // Host reference: CPU CTR column, then binarize each value against the borders.
    let (_cg, _ct, cv) = cpu_ordered_ctr(&perm, &bins, &class, prior, false, 0);
    let host_cindex: Vec<u32> = cv
        .iter()
        .map(|&v| borders.iter().filter(|&&b| v > b).count() as u32)
        .collect();

    // Device: accumulate CTR on device, binarize on device, read the extra cindex column back.
    let dev_cindex =
        binarize_ctr_column_host(&perm, &bins, &class, prior, bucket_count(&bins), &borders).unwrap();

    assert_eq!(dev_cindex.len(), n, "device CTR cindex column truncated");
    assert_eq!(
        dev_cindex, host_cindex,
        "device CTR->cindex binarization must be bit-exact vs the host CTR column"
    );
    // Every emitted bin is a valid index into the borders+1 buckets (bounds sanity).
    assert!(
        dev_cindex.iter().all(|&b| (b as usize) <= borders.len()),
        "a CTR cindex bin exceeds the borders+1 bucket count"
    );
}

#[test]
fn ctr_averaging_permutation_column_bit_exact() {
    // GDC-09 (T11): the SAME online-CTR + binarize kernels called with the AVERAGING
    // permutation must reproduce the CPU averaging materialization bit-exact — i.e.
    // the second-permutation pass is the same math over a different object order,
    // exactly like the CPU `averaging_ctr_features` reference (a second
    // `materialize_ctr_columns_for_perm` call). The structure-permutation column at
    // the same inputs must DIFFER (the discriminating precondition for GDC-10's
    // leaf-value gather — a fixture where the two orders agree cannot detect a
    // structure-only bug).
    if !device_ctr_active() {
        eprintln!("SKIP ctr_averaging_permutation_column_bit_exact: wgpu has no f64 seam (WR-01)");
        return;
    }
    let n = 300usize;
    let prior = 0.5;
    let (bins, class, structure_perm) = synth_fixture(n, 6, 314);
    // The averaging order: a DIFFERENT deterministic permutation over the same objects
    // (rotate + reverse of the structure order, guaranteed ≠ structure for n > 2).
    let averaging_perm: Vec<u32> = {
        let mut p: Vec<u32> = structure_perm.iter().rev().copied().collect();
        p.rotate_left(7);
        p
    };
    assert_ne!(structure_perm, averaging_perm);
    let borders = vec![0.2_f64, 0.4, 0.5, 0.6, 0.8];

    // Host reference: the CPU ordered CTR under the AVERAGING order, binarized.
    let (_ag, _at, av) = cpu_ordered_ctr(&averaging_perm, &bins, &class, prior, false, 0);
    let host_avg_cindex: Vec<u32> = av
        .iter()
        .map(|&v| borders.iter().filter(|&&b| v > b).count() as u32)
        .collect();

    // Device: the SAME kernels, averaging permutation input.
    let dev_avg_cindex = binarize_ctr_column_host(
        &averaging_perm, &bins, &class, prior, bucket_count(&bins), &borders,
    )
    .unwrap();
    assert_eq!(
        dev_avg_cindex, host_avg_cindex,
        "device averaging-permutation CTR->cindex must be bit-exact vs the CPU \
         averaging materialization"
    );

    // Discriminator: the structure-permutation column differs in ≥1 object.
    let dev_structure_cindex = binarize_ctr_column_host(
        &structure_perm, &bins, &class, prior, bucket_count(&bins), &borders,
    )
    .unwrap();
    assert_ne!(
        dev_avg_cindex, dev_structure_cindex,
        "the averaging and structure materializations coincide on this fixture — it \
         cannot discriminate a structure-only leaf-gather bug (pick another order)"
    );
}

#[test]
fn buckets_numerator_matches_cpu_reference() {
    // DCTR-06 (T08): the ordered prefix kernel's per-bucket NUMERATOR is now selected by
    // `(ctr_type, target_border_idx)` instead of being hard-coded to `N1`. For every
    // (ctr_type, target_border_idx) pair reachable at binclf — `(Borders, 0)`, `(Buckets, 0)`,
    // `(Buckets, 1)` — the device `good`/`total` must be INTEGER-EQUAL to the
    // `online_class_prefix` reference (`cb-train/src/ctr/online.rs:552-569`), transcribed inline
    // above in its generic loop-over-classes form. `total` is the bucket total in every mode.
    if !device_ctr_active() {
        eprintln!("SKIP buckets_numerator_matches_cpu_reference: wgpu has no f64 CTR seam (WR-01)");
        return;
    }
    let n = 128usize;
    let prior = 0.5;
    let (bins, class, perm) = synth_fixture(n, 5, 7);
    let buckets = bucket_count(&bins);

    let modes: [(i8, u32, &str); 3] = [
        (CTR_TYPE_BORDERS, 0, "Borders@0"),
        (CTR_TYPE_BUCKETS, 0, "Buckets@0"),
        (CTR_TYPE_BUCKETS, 1, "Buckets@1"),
    ];
    let mut goods: Vec<Vec<i64>> = Vec::new();
    for &(ctr_type, target_border_idx, label) in &modes {
        let (cg, ct, cv) = cpu_ordered_ctr(
            &perm,
            &bins,
            &class,
            prior,
            ctr_type == CTR_TYPE_BUCKETS,
            target_border_idx as usize,
        );
        let (dg, dt, dv) = compute_ordered_ctr_host_mode(
            &perm,
            &bins,
            &class,
            prior,
            buckets,
            ctr_type,
            target_border_idx,
        )
        .unwrap();
        assert_eq!(dg, cg, "device good != online_class_prefix ({label})");
        assert_eq!(dt, ct, "device total != bucket total ({label})");
        let div = max_divergence(&dv, &cv);
        assert!(div <= TOL, "CTR value divergence {div} > {TOL} ({label})");
        goods.push(dg);
    }

    // NON-VACUITY GUARD (mandatory): on a fixture where every bucket were singleton the three
    // modes would all read `good = 0` and the assertions above would pass trivially. Buckets@0
    // reads N0 where Borders@0 reads N1, so the two vectors MUST differ somewhere.
    assert_ne!(
        goods.first(),
        goods.get(1),
        "Buckets@0 and Borders@0 coincide on this fixture — it cannot discriminate the \
         numerator selection (pick another fixture)"
    );
    // Collapse identity at SIMPLE_CLASSES_COUNT == 2 (`cb-train/src/ctr/online.rs:52`):
    // Buckets@1 selects N[1] and Borders@0 selects Total - N[0] == N[1]. Documented, not
    // load-bearing — the per-mode equalities above are what pin the behaviour.
    assert_eq!(
        goods.first(),
        goods.get(2),
        "at binclf Buckets@1 and Borders@0 must both select N1"
    );
}

#[test]
fn out_of_range_ctr_mode_is_rejected() {
    // DCTR-06 (T08): the numerator selector is host-validated BEFORE dispatch, exactly like the
    // pre-existing bin/class range guards. A `target_border_idx > 1` at binclf, or a CTR type this
    // class-prefix kernel does not implement, must surface a typed `CbError::OutOfRange` — never a
    // silent fallback onto the `b = 0` numerator (which would be a WRONG answer, not a worse one).
    if !device_ctr_active() {
        eprintln!("SKIP out_of_range_ctr_mode_is_rejected: wgpu has no f64 CTR seam (WR-01)");
        return;
    }
    let n = 32usize;
    let (bins, class, perm) = synth_fixture(n, 4, 11);
    let buckets = bucket_count(&bins);

    let too_big = compute_ordered_ctr_host_mode(
        &perm, &bins, &class, 0.5, buckets, CTR_TYPE_BUCKETS, 2,
    );
    assert!(
        matches!(too_big, Err(cb_core::CbError::OutOfRange(_))),
        "target_border_idx = 2 must be rejected at binclf, got {too_big:?}"
    );

    // `ECtrType::Counter` (4) is not a class-prefix type — `online_counter_column` owns it.
    let wrong_type =
        compute_ordered_ctr_host_mode(&perm, &bins, &class, 0.5, buckets, 4, 0);
    assert!(
        matches!(wrong_type, Err(cb_core::CbError::OutOfRange(_))),
        "a non-class-prefix ctr_type must be rejected, got {wrong_type:?}"
    );
}

/// The Counter CTR border table used by the two DCTR-09 oracles. The Counter values on
/// `synth_fixture(96, 5, 11)` are `(t + 0.5) / 23` for `t in {13, 20, 21, 22}` — i.e.
/// `{0.5870, 0.8913, 0.9348, 0.9783}` — so these five borders separate all four of them and the
/// emitted cindex column is non-degenerate (asserted below, not assumed).
const COUNTER_BORDERS: [f64; 5] = [0.2, 0.7, 0.91, 0.95, 0.99];

#[test]
fn counter_ctr_matches_cpu_reference() {
    // DCTR-09 (T11): the device Counter statistic must reproduce `online_counter_column` +
    // `calc_ctr_online` exactly — per-object `good = totals[bin]` (whole-set tally, EXACT
    // integers), a CONSTANT `max_b totals[b]` denominator for EVERY object (mirroring
    // `ctr_feature.rs:304`'s `denoms = vec![denominator; n]`), and the f64 value within the
    // ε=1e-4 device bar. The binarized `u32` cindex column must be BIT-EXACT against the host
    // binarization of the CPU value column (the `ctr_binarized_cindex_column_bit_exact`
    // template) — Counter quantizes through the SAME f64 border table as Borders/Buckets
    // (`quantize_in_f32 == false`, `ctr_feature.rs:309`), so no per-type border handling exists.
    if !device_ctr_active() {
        eprintln!("SKIP counter_ctr_matches_cpu_reference: wgpu has no f64 CTR seam (WR-01)");
        return;
    }
    let n = 96usize;
    let prior = 0.5;
    let (bins, _class, _perm) = synth_fixture(n, 5, 11);
    let buckets = bucket_count(&bins);
    let (cg, cdenom, cv) = cpu_counter_ctr(&bins, prior, buckets);

    let (dg, dt, dv) = compute_counter_ctr_host(&bins, prior, buckets).unwrap();
    assert_eq!(dg, cg, "device Counter good != online_counter_column totals[bin]");
    assert_eq!(
        dt,
        vec![cdenom; n],
        "device Counter denominator must be the CONSTANT max bucket total for every object"
    );
    let div = max_divergence(&dv, &cv);
    assert!(div <= TOL, "Counter CTR value divergence {div} > {TOL}");

    // NON-VACUITY GUARD (mandatory): on a fixture where every bucket held the same number of
    // documents, `good == total` for every object and the constant-denominator assertion would
    // pass trivially. The tally must actually vary across buckets.
    let distinct_totals: std::collections::BTreeSet<i64> = cg.iter().copied().collect();
    assert!(
        distinct_totals.len() >= 2,
        "every bucket has the same tally on this fixture — it cannot discriminate a per-object \
         denominator from the constant max (pick another fixture); totals {distinct_totals:?}"
    );
    assert!(
        cg.iter().any(|&g| g != cdenom),
        "every bucket is the max bucket on this fixture — good and the denominator coincide and \
         the oracle cannot tell them apart (pick another fixture)"
    );

    // The CTR->cindex binarize JOIN over the Counter values, bit-exact vs the host reference.
    let borders = COUNTER_BORDERS.to_vec();
    let host_cindex: Vec<u32> = cv
        .iter()
        .map(|&v| borders.iter().filter(|&&b| v > b).count() as u32)
        .collect();
    let dev_cindex = binarize_counter_column_host(&bins, prior, buckets, &borders).unwrap();
    assert_eq!(dev_cindex.len(), n, "device Counter cindex column truncated");
    assert_eq!(
        dev_cindex, host_cindex,
        "device Counter->cindex binarization must be bit-exact vs the host Counter column"
    );
    let distinct_bins: std::collections::BTreeSet<u32> = dev_cindex.iter().copied().collect();
    assert!(
        distinct_bins.len() >= 3,
        "the Counter cindex column is near-constant on these borders — a bit-exact comparison \
         against it proves little (pick other borders); bins {distinct_bins:?}"
    );
}

#[test]
fn counter_ctr_is_permutation_independent() {
    // DCTR-09 (T11): `IsPermutationDependentCtrType(Counter) == false` (`ctr_type.cpp:43-56`).
    // `launch_counter_ctr_resident` takes NO permutation argument, so the only route an order
    // dependence could take is the order the documents are presented in. Present the SAME
    // objects in two different orders (identity and a full reversal), undo the reordering on
    // the output, and require the emitted cindex columns to be BIT-IDENTICAL.
    //
    // The contrast half is what makes this discriminating: the ORDERED (Borders) statistic on
    // the SAME objects and the SAME two orders DOES change, so an "everything on this fixture
    // is invariant" explanation is ruled out.
    if !device_ctr_active() {
        eprintln!("SKIP counter_ctr_is_permutation_independent: wgpu has no f64 CTR seam (WR-01)");
        return;
    }
    let n = 96usize;
    let prior = 0.5;
    let (bins, class, _perm) = synth_fixture(n, 5, 11);
    let buckets = bucket_count(&bins);
    let counter_borders = COUNTER_BORDERS.to_vec();

    let identity: Vec<u32> = (0..n as u32).collect();
    let reversed: Vec<u32> = (0..n as u32).rev().collect();
    assert_ne!(identity, reversed);

    // Counter under the identity order, and under the reversal with the reordering undone.
    let bins_reversed: Vec<u32> = reversed.iter().map(|&i| bins[i as usize]).collect();
    let col_identity = binarize_counter_column_host(&bins, prior, buckets, &counter_borders).unwrap();
    let col_reversed_raw =
        binarize_counter_column_host(&bins_reversed, prior, buckets, &counter_borders).unwrap();
    let mut col_reversed = vec![u32::MAX; n];
    for (pos, &obj) in reversed.iter().enumerate() {
        col_reversed[obj as usize] = col_reversed_raw[pos];
    }
    assert_eq!(
        col_identity, col_reversed,
        "the Counter statistic must be permutation independent (ctr_type.cpp:43-56)"
    );

    // Contrast: the ordered class-prefix statistic over the SAME objects and the SAME two orders
    // is order DEPENDENT. Its values live in (0, 1] around the prior, so it needs its own
    // border table (the Counter borders sit far above most ordered values).
    let ordered_borders = vec![0.2_f64, 0.4, 0.5, 0.6, 0.8];
    let ordered_identity =
        binarize_ctr_column_host(&identity, &bins, &class, prior, buckets, &ordered_borders).unwrap();
    let ordered_reversed =
        binarize_ctr_column_host(&reversed, &bins, &class, prior, buckets, &ordered_borders).unwrap();
    assert_ne!(
        ordered_identity, ordered_reversed,
        "the ORDERED statistic coincides under both orders on this fixture — it cannot show that \
         the Counter invariance above is a property of Counter rather than of the fixture"
    );
}
