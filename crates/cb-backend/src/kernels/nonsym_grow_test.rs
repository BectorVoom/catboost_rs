//! Serial CPU self-oracle for the Phase 12 Plan 03 (GPUT-18) device Depthwise / Lossguide
//! non-symmetric grow (`kernels::nonsym_grow::grow_nonsym_tree`). The device driver SELECTS
//! each node's split via the device argmin (`launch_find_optimal_split_pointwise`); this
//! oracle grows the SAME tree with an inline HOST leaf-wise reference (host score argmax, the
//! SAME `cb_train::tree::leaf_wise_grower` bookkeeping — TRANSCRIBED, never `use cb_train`,
//! the feature-unification landmine) and asserts:
//!
//! - STRUCTURE is INTEGER-exact: `step_nodes`, `node_id_to_leaf_id`, per-node `(feature, bin)`
//!   splits, and per-object `leaf_of` all match (the STRICT bar — on the clear-gain-margin
//!   fixture the device argmin picks the SAME split as the CPU reference, `score_split` oracle).
//! - LEAF VALUES match within ε=1e-4 (`max_divergence`, which returns `(INF, INF)` on a length
//!   mismatch so a truncated device buffer fails loudly). Kaggle CUDA ε=1e-4 sign-off is
//!   deferred to Plan 09; the in-env self-oracle + ROCm smoke is the local gate.
//!
//! Runs over `SelectedRuntime`: `launch_find_optimal_split_pointwise` uses the whole-subset
//! `pointwise_hist2` (`Atomic<f64>`) path, so this builds AND RUNS under every backend
//! (cpu/wgpu host + rocm gfx1100 in-env + cuda compile) — UNLIKE the resident oblivious grow
//! (which needs `Atomic<u64>` and skips on cpu/wgpu).

use cb_compute::{calc_average, cosine_split_score, l2_split_score, LeafStats};
use cb_core::sum_f64;

use crate::kernels::nonsym_grow::{grow_nonsym_tree, NonsymPolicy};
use crate::kernels::{SCORE_FN_COSINE, SCORE_FN_L2};

/// Max abs / rel divergence over two equal-length buffers (the `grow_loop::max_divergence`
/// reporter shape). A length mismatch yields a sentinel infinite divergence (WR-06).
fn max_divergence(device: &[f64], baseline: &[f64]) -> (f64, f64) {
    if device.len() != baseline.len() {
        return (f64::INFINITY, f64::INFINITY);
    }
    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;
    for (&d, &b) in device.iter().zip(baseline) {
        let abs = (d - b).abs();
        let rel = if b.abs() > 0.0 { abs / b.abs() } else { abs };
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
    }
    (max_abs, max_rel)
}

/// Host split-score dispatch (L2 / Cosine — the two the oracle exercises).
fn host_score(leaves: &[LeafStats], scaled_l2: f64, score_fn: u32) -> f64 {
    if score_fn == SCORE_FN_COSINE {
        cosine_split_score(leaves, scaled_l2)
    } else {
        l2_split_score(leaves, scaled_l2)
    }
}

/// The CPU leaf-wise reference tree (mirrors `grow_nonsym_tree` but with HOST split selection).
struct CpuTree {
    splits: Vec<(u32, u32, bool)>,
    step_nodes: Vec<(u16, u16)>,
    node_id_to_leaf_id: Vec<u32>,
    leaf_values: Vec<f64>,
    leaf_of: Vec<u32>,
}

struct HostBest {
    feature: u32,
    bin: u32,
    gain: f64,
    left: Vec<usize>,
    right: Vec<usize>,
}

/// Host best split for one node's doc subset (the reference SELECTION: strict first-wins
/// argmax over ascending `(feature, bin)`, gate on `gain >= 1e-9`).
#[allow(clippy::too_many_arguments)]
fn host_best(
    docs: &[usize],
    der1: &[f64],
    weight: &[f64],
    cindex: &[u32],
    n: usize,
    n_bins: usize,
    n_features: usize,
    min_data_in_leaf: usize,
    scaled_l2: f64,
    score_fn: u32,
) -> Option<HostBest> {
    if docs.len() < min_data_in_leaf || docs.len() < 2 {
        return None;
    }
    let der_all: Vec<f64> = docs.iter().map(|&i| der1[i]).collect();
    let w_all: Vec<f64> = docs.iter().map(|&i| weight[i]).collect();
    let baseline = host_score(
        &[LeafStats { sum_weighted_delta: sum_f64(&der_all), sum_weight: sum_f64(&w_all) }],
        scaled_l2,
        score_fn,
    );

    let mut best_score = f64::NEG_INFINITY;
    let mut best: Option<(u32, u32)> = None;
    for feature in 0..n_features {
        for bin in 0..n_bins.saturating_sub(1) {
            let mut ld: Vec<f64> = Vec::new();
            let mut lw: Vec<f64> = Vec::new();
            let mut rd: Vec<f64> = Vec::new();
            let mut rw: Vec<f64> = Vec::new();
            for &obj in docs {
                if (cindex[feature * n + obj] as usize) > bin {
                    rd.push(der1[obj]);
                    rw.push(weight[obj]);
                } else {
                    ld.push(der1[obj]);
                    lw.push(weight[obj]);
                }
            }
            let leaves = [
                LeafStats { sum_weighted_delta: sum_f64(&ld), sum_weight: sum_f64(&lw) },
                LeafStats { sum_weighted_delta: sum_f64(&rd), sum_weight: sum_f64(&rw) },
            ];
            let score = host_score(&leaves, scaled_l2, score_fn);
            if score > best_score {
                best_score = score;
                best = Some((feature as u32, bin as u32));
            }
        }
    }
    let (feature, bin) = best?;
    let gain = best_score - baseline;
    if gain < 1e-9 {
        return None;
    }
    let mut left = Vec::new();
    let mut right = Vec::new();
    for &obj in docs {
        if (cindex[feature as usize * n + obj] as usize) > bin as usize {
            right.push(obj);
        } else {
            left.push(obj);
        }
    }
    Some(HostBest { feature, bin, gain, left, right })
}

enum RefNode {
    Interior { feature: u32, bin: u32, left: usize, right: usize },
    Leaf,
}

/// Grow the CPU reference tree (host selection + the identical leaf_wise_grower bookkeeping).
#[allow(clippy::too_many_arguments)]
fn cpu_leaf_wise(
    policy: NonsymPolicy,
    der1: &[f64],
    weight: &[f64],
    cindex: &[u32],
    n: usize,
    n_bins: usize,
    n_features: usize,
    max_depth: usize,
    max_leaves: usize,
    min_data_in_leaf: usize,
    scaled_l2: f64,
    score_fn: u32,
) -> CpuTree {
    let mut nodes: Vec<RefNode> = Vec::new();
    let mut node_docs: Vec<Vec<usize>> = Vec::new();
    let mut node_depth: Vec<usize> = Vec::new();

    let mut new_node = |nodes: &mut Vec<RefNode>,
                        node_docs: &mut Vec<Vec<usize>>,
                        node_depth: &mut Vec<usize>,
                        docs: Vec<usize>,
                        depth: usize|
     -> usize {
        let id = nodes.len();
        nodes.push(RefNode::Leaf);
        node_docs.push(docs);
        node_depth.push(depth);
        id
    };

    let root = new_node(&mut nodes, &mut node_docs, &mut node_depth, (0..n).collect(), 0);
    let mut leaf_owner: Vec<usize> = vec![root; n];

    let mut do_split = |nodes: &mut Vec<RefNode>,
                        node_docs: &mut Vec<Vec<usize>>,
                        node_depth: &mut Vec<usize>,
                        leaf_owner: &mut [usize],
                        id: usize,
                        bs: &HostBest|
     -> (usize, usize) {
        let depth = node_depth[id] + 1;
        let left = new_node(nodes, node_docs, node_depth, bs.left.clone(), depth);
        let right = new_node(nodes, node_docs, node_depth, bs.right.clone(), depth);
        nodes[id] = RefNode::Interior { feature: bs.feature, bin: bs.bin, left, right };
        for &obj in &bs.left {
            leaf_owner[obj] = left;
        }
        for &obj in &bs.right {
            leaf_owner[obj] = right;
        }
        (left, right)
    };

    match policy {
        NonsymPolicy::Depthwise => {
            let mut current_level = vec![root];
            for _ in 0..max_depth {
                let mut next_level = Vec::new();
                for &leaf in &current_level {
                    let docs = node_docs[leaf].clone();
                    if let Some(bs) = host_best(
                        &docs, der1, weight, cindex, n, n_bins, n_features, min_data_in_leaf,
                        scaled_l2, score_fn,
                    ) {
                        let (l, r) = do_split(
                            &mut nodes, &mut node_docs, &mut node_depth, &mut leaf_owner, leaf, &bs,
                        );
                        next_level.push(l);
                        next_level.push(r);
                    }
                }
                if next_level.is_empty() {
                    break;
                }
                current_level = next_level;
            }
        }
        NonsymPolicy::Lossguide => {
            use std::cmp::Ordering;
            use std::collections::BinaryHeap;
            struct QItem {
                gain: f64,
                seq: u64,
                node: usize,
                best: HostBest,
            }
            impl PartialEq for QItem {
                fn eq(&self, other: &Self) -> bool {
                    self.gain == other.gain && self.seq == other.seq
                }
            }
            impl Eq for QItem {}
            impl Ord for QItem {
                fn cmp(&self, other: &Self) -> Ordering {
                    match self.gain.partial_cmp(&other.gain).unwrap_or(Ordering::Equal) {
                        Ordering::Equal => other.seq.cmp(&self.seq),
                        ord => ord,
                    }
                }
            }
            impl PartialOrd for QItem {
                fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                    Some(self.cmp(other))
                }
            }
            let mut heap: BinaryHeap<QItem> = BinaryHeap::new();
            let mut seq = 0u64;
            let mut leaf_count = 1usize;
            macro_rules! enqueue {
                ($node:expr) => {{
                    let node = $node;
                    if node_depth[node] < max_depth {
                        let docs = node_docs[node].clone();
                        if let Some(bs) = host_best(
                            &docs, der1, weight, cindex, n, n_bins, n_features, min_data_in_leaf,
                            scaled_l2, score_fn,
                        ) {
                            heap.push(QItem { gain: bs.gain, seq, node, best: bs });
                            seq += 1;
                        }
                    }
                }};
            }
            enqueue!(root);
            while let Some(item) = heap.pop() {
                if leaf_count >= max_leaves {
                    break;
                }
                let (l, r) = do_split(
                    &mut nodes, &mut node_docs, &mut node_depth, &mut leaf_owner, item.node,
                    &item.best,
                );
                leaf_count += 1;
                enqueue!(l);
                enqueue!(r);
            }
        }
    }

    // Finalize (identical to grow_nonsym_tree).
    let node_count = nodes.len();
    let mut step_nodes = Vec::with_capacity(node_count);
    let mut node_id_to_leaf_id = vec![u32::MAX; node_count];
    let mut splits = Vec::with_capacity(node_count);
    let mut leaf_values = Vec::new();
    let mut node_to_leaf = vec![None; node_count];
    let mut next_leaf_id = 0u32;
    for (id, node) in nodes.iter().enumerate() {
        match node {
            RefNode::Interior { feature, bin, left, right } => {
                splits.push((*feature, *bin, false));
                node_id_to_leaf_id[id] = u32::MAX;
                step_nodes.push((
                    u16::try_from(left - id).unwrap(),
                    u16::try_from(right - id).unwrap(),
                ));
            }
            RefNode::Leaf => {
                splits.push((0, 0, false));
                step_nodes.push((0, 0));
                node_to_leaf[id] = Some(next_leaf_id);
                node_id_to_leaf_id[id] = next_leaf_id;
                let docs = &node_docs[id];
                let ds: Vec<f64> = docs.iter().map(|&i| der1[i]).collect();
                let ws: Vec<f64> = docs.iter().map(|&i| weight[i]).collect();
                leaf_values.push(calc_average(sum_f64(&ds), sum_f64(&ws), scaled_l2));
                next_leaf_id += 1;
            }
        }
    }
    let leaf_of: Vec<u32> =
        leaf_owner.iter().map(|&node| node_to_leaf[node].unwrap_or(0)).collect();

    CpuTree { splits, step_nodes, node_id_to_leaf_id, leaf_values, leaf_of }
}

/// Build a clear-gain-margin fixture (feature 0 aligned with the der1 ramp) — the SAME
/// primitives the `grow_loop` / `score_split` oracles use, so the device argmin and the host
/// argmax agree on every node's split.
fn fixture(n: usize, n_features: usize, n_bins: usize) -> (Vec<f64>, Vec<f64>, Vec<u32>) {
    let der1 = crate::kernels::test_fixtures::ramp_centred(n);
    let weight = crate::kernels::test_fixtures::weight_mod5(n);
    let cindex = crate::kernels::test_fixtures::cindex_feature_major(n, n_features, n_bins);
    (der1, weight, cindex)
}

fn scaled_l2_for(weight: &[f64], n: usize, l2: f64) -> f64 {
    cb_compute::scale_l2_reg(l2, sum_f64(weight), n)
}

/// Assert the device non-sym grow matches the host reference for one policy + score fn.
fn assert_matches(policy: NonsymPolicy, score_fn: u32, label: &str) {
    // The device split scorer runs real GPU kernels; the cubecl-cpu backend cannot JIT the
    // per-node score/argmin over these subset shapes (an `elem.rs` visitor panic), so SKIP on
    // cpu/wgpu and validate on the real device in-env (rocm gfx1100) — the WR-01 anti-false-pass
    // convention shared with the resident grow oracles. Kaggle CUDA ε sign-off is Plan 09's.
    if !cfg!(any(feature = "rocm", feature = "cuda")) {
        println!("[{label}] SKIP: non-sym device grow needs a real GPU backend (rocm/cuda)");
        return;
    }
    const EPS: f64 = 1e-4;
    let n_features = 3usize;
    let n_bins = 32usize;
    let max_depth = 4usize;
    let max_leaves = 8usize;
    let min_data_in_leaf = 1usize;
    let l2 = 3.0_f64;

    for &n in &[64usize, 300usize] {
        let (der1, weight, cindex) = fixture(n, n_features, n_bins);
        let scaled_l2 = scaled_l2_for(&weight, n, l2);

        // Unsampled: the score channels ARE the leaf channels (D-04 byte-unchanged).
        let dev = grow_nonsym_tree(
            policy, &der1, &weight, &der1, &weight, &cindex, n, n_bins, n_features, max_depth,
            max_leaves, min_data_in_leaf, scaled_l2, score_fn,
        )
        .expect("device non-sym grow must succeed on the clear-margin fixture");

        let cpu = cpu_leaf_wise(
            policy, &der1, &weight, &cindex, n, n_bins, n_features, max_depth, max_leaves,
            min_data_in_leaf, scaled_l2, score_fn,
        );

        // (A) STRUCTURE — integer-exact.
        assert_eq!(
            dev.step_nodes, cpu.step_nodes,
            "[{label} n={n}] device step_nodes must match CPU leaf-wise reference"
        );
        assert_eq!(
            dev.node_id_to_leaf_id, cpu.node_id_to_leaf_id,
            "[{label} n={n}] device node_id_to_leaf_id must match CPU reference"
        );
        assert_eq!(
            dev.splits, cpu.splits,
            "[{label} n={n}] device per-node (feature,bin) splits must match CPU reference"
        );
        assert_eq!(
            dev.leaf_of, cpu.leaf_of,
            "[{label} n={n}] device per-object leaf_of must match CPU reference"
        );

        // (B) LEAF VALUES — within ε=1e-4.
        let (abs, rel) = max_divergence(&dev.leaf_values, &cpu.leaf_values);
        println!(
            "[{label} n={n}] {} nodes, {} leaves; leaf-value max abs_div={abs:.3e} rel_div={rel:.3e} (bar={EPS:.0e})",
            dev.step_nodes.len(),
            dev.leaf_values.len(),
        );
        assert!(
            abs <= EPS || rel <= EPS,
            "[{label} n={n}] device leaf values exceeded ε=1e-4: abs={abs:.3e} rel={rel:.3e}"
        );
    }
}

/// GDC-03 (T05): the SESSION-level weighted oracle. The weighted-der substitution
/// lives in `GpuTrainSession::grow_one`'s nonsym arm (caller-side `w·der1`), so the
/// discriminating test drives the SESSION (not `grow_nonsym_tree` directly, which is
/// deliberately untouched) with a NON-uniform weight and compares against the CPU
/// leaf-wise reference fed the SAME weighted der. Pre-fix the session passed the RAW
/// der — structure and leaf values both diverge from this reference.
fn assert_session_weighted_matches(policy: NonsymPolicy, score_fn: u32, label: &str) {
    if !cfg!(any(feature = "rocm", feature = "cuda")) {
        println!("[{label}] SKIP: non-sym device grow needs a real GPU backend (rocm/cuda)");
        return;
    }
    const EPS: f64 = 1e-4;
    let n = 64usize;
    let n_features = 3usize;
    let n_bins = 32usize;
    let max_depth = 4usize;
    let l2 = 3.0_f64;

    let (der1, weight, cindex) = fixture(n, n_features, n_bins);
    assert!(
        weight.iter().any(|&w| (w - 1.0).abs() > 1e-12),
        "the fixture weight must be non-uniform or this oracle is vacuous"
    );
    let scaled_l2 = scaled_l2_for(&weight, n, l2);

    // Session der derivation: RMSE, approx == 0 ⇒ der1 = target, so the fixture's
    // der ramp doubles as the target.
    let target = der1.clone();
    let escore = if score_fn == SCORE_FN_COSINE {
        cb_compute::EScoreFunction::Cosine
    } else {
        cb_compute::EScoreFunction::L2
    };
    let (device_policy, max_leaves_cfg) = match policy {
        NonsymPolicy::Depthwise => (cb_compute::DeviceGrowPolicy::Depthwise, None),
        NonsymPolicy::Lossguide => (cb_compute::DeviceGrowPolicy::Lossguide, Some(8usize)),
    };
    let config = cb_compute::DeviceTrainConfig {
        grow_policy: device_policy,
        max_leaves: max_leaves_cfg,
        ..cb_compute::DeviceTrainConfig::default()
    };
    let mut session = crate::gpu_runtime::GpuTrainSession::begin(
        &cb_compute::Loss::Rmse,
        max_depth,
        true,
        1,
        escore,
        &cindex,
        &weight,
        n,
        n_features,
        n_bins,
        0.3,
        scaled_l2,
        &config,
    )
    .expect("begin must not error on a covered nonsym config")
    .expect("a covered nonsym config must open a session");
    let dev = session
        .grow_one(&vec![0.0_f64; n], &target, &[])
        .expect("session nonsym grow must succeed");

    // The CPU reference consumes the WEIGHTED der — exactly what the session arm
    // now feeds `grow_nonsym_tree` (`host_weighted_der1`).
    let weighted: Vec<f64> = der1.iter().zip(weight.iter()).map(|(&d, &w)| d * w).collect();
    let cpu = cpu_leaf_wise(
        policy,
        &weighted,
        &weight,
        &cindex,
        n,
        n_bins,
        n_features,
        max_depth,
        max_leaves_cfg.unwrap_or(usize::MAX),
        1,
        scaled_l2,
        score_fn,
    );

    assert_eq!(
        dev.splits, cpu.splits,
        "[{label}] session weighted splits must match the weighted CPU reference"
    );
    assert_eq!(
        dev.leaf_of, cpu.leaf_of,
        "[{label}] session weighted leaf_of must match the weighted CPU reference"
    );
    let (abs, rel) = max_divergence(&dev.leaf_values, &cpu.leaf_values);
    println!("[{label}] weighted session oracle: abs={abs:.3e} rel={rel:.3e} (bar={EPS:.0e})");
    assert!(
        abs <= EPS || rel <= EPS,
        "[{label}] session weighted leaf values exceeded ε=1e-4: abs={abs:.3e} rel={rel:.3e}"
    );
    drop(session);
}

#[test]
fn depthwise_weighted_matches_cpu_leaf_wise_l2() {
    assert_session_weighted_matches(NonsymPolicy::Depthwise, SCORE_FN_L2, "depthwise-weighted-l2");
}

#[test]
fn depthwise_weighted_matches_cpu_leaf_wise_cosine() {
    assert_session_weighted_matches(
        NonsymPolicy::Depthwise,
        SCORE_FN_COSINE,
        "depthwise-weighted-cosine",
    );
}

#[test]
fn lossguide_weighted_matches_cpu_leaf_wise_l2() {
    assert_session_weighted_matches(NonsymPolicy::Lossguide, SCORE_FN_L2, "lossguide-weighted-l2");
}

#[test]
fn lossguide_weighted_matches_cpu_leaf_wise_cosine() {
    assert_session_weighted_matches(
        NonsymPolicy::Lossguide,
        SCORE_FN_COSINE,
        "lossguide-weighted-cosine",
    );
}

#[test]
fn depthwise_matches_cpu_leaf_wise_l2() {
    assert_matches(NonsymPolicy::Depthwise, SCORE_FN_L2, "depthwise-l2");
}

#[test]
fn depthwise_matches_cpu_leaf_wise_cosine() {
    assert_matches(NonsymPolicy::Depthwise, SCORE_FN_COSINE, "depthwise-cosine");
}

#[test]
fn lossguide_matches_cpu_leaf_wise_l2() {
    assert_matches(NonsymPolicy::Lossguide, SCORE_FN_L2, "lossguide-l2");
}

#[test]
fn lossguide_matches_cpu_leaf_wise_cosine() {
    assert_matches(NonsymPolicy::Lossguide, SCORE_FN_COSINE, "lossguide-cosine");
}

// ─── FPP-12 (T08): the host bootstrap sample reaches the nonsym grower's SCORE channels ──

/// FPP-12: with a length-`n` `sample`, the non-symmetric device grower scores splits (and
/// the unsplit-gain baseline, and the Lossguide priority) over `der1 ⊙ sample` /
/// `weight ⊙ sample`, while LEAF values keep using the UNSAMPLED channels —
/// `Runtime::grow_tree_on_device`'s contract verbatim.
///
/// # Why the reference multiplies twice
///
/// The `der1` that reaches `grow_nonsym_tree` has already been through
/// `host_weighted_der1`, so on a weighted × sampled fit the score channel is
/// `w · der1 · s` and the score weight is `w · s`. That mirrors the oblivious resident
/// arm's nested `fold_weights_resident(fold_weights_resident(der1, weight), sample)`. A
/// reference that multiplied once would be chasing a phantom.
fn assert_sampled_matches(policy: NonsymPolicy, score_fn: u32, label: &str) {
    if !cfg!(any(feature = "rocm", feature = "cuda")) {
        println!("[{label}] SKIP: non-sym device grow needs a real GPU backend (rocm/cuda)");
        return;
    }
    const EPS: f64 = 1e-4;
    let n = 64usize;
    let n_features = 3usize;
    let n_bins = 32usize;
    let max_depth = 4usize;
    let max_leaves = 8usize;
    let min_data_in_leaf = 1usize;
    let l2 = 3.0_f64;

    let (der1, weight, cindex) = fixture(n, n_features, n_bins);
    let scaled_l2 = scaled_l2_for(&weight, n, l2);
    // ~30% DROPPED objects plus up- and down-weighted ones. The zeros are the
    // discriminating part: a dropped object must contribute NOTHING to any histogram.
    let sample: Vec<f64> = (0..n)
        .map(|i| match i % 10 {
            0 | 3 | 7 => 0.0,
            1 | 4 => 0.5,
            2 | 5 | 8 => 1.0,
            _ => 2.0,
        })
        .collect();
    let score_der1: Vec<f64> = der1.iter().zip(sample.iter()).map(|(&d, &s)| d * s).collect();
    let score_weight: Vec<f64> =
        weight.iter().zip(sample.iter()).map(|(&w, &s)| w * s).collect();

    let sampled = grow_nonsym_tree(
        policy, &der1, &weight, &score_der1, &score_weight, &cindex, n, n_bins, n_features,
        max_depth, max_leaves, min_data_in_leaf, scaled_l2, score_fn,
    )
    .expect("sampled non-sym grow must succeed");

    // (A) STRUCTURE is decided by the SAMPLED score channels.
    let all_sampled = grow_nonsym_tree(
        policy, &score_der1, &score_weight, &score_der1, &score_weight, &cindex, n, n_bins,
        n_features, max_depth, max_leaves, min_data_in_leaf, scaled_l2, score_fn,
    )
    .expect("all-sampled non-sym grow must succeed");
    assert_eq!(
        sampled.splits, all_sampled.splits,
        "[{label}] the node splits must be decided by the SAMPLED score channels"
    );
    assert_eq!(sampled.leaf_of, all_sampled.leaf_of, "[{label}] routing follows the sampled structure");

    // …and (A) is only meaningful if the sampled structure DIFFERS from the unsampled one.
    // Otherwise a fixture where both coincide would pass (A) vacuously.
    let unsampled = grow_nonsym_tree(
        policy, &der1, &weight, &der1, &weight, &cindex, n, n_bins, n_features, max_depth,
        max_leaves, min_data_in_leaf, scaled_l2, score_fn,
    )
    .expect("unsampled non-sym grow must succeed");
    assert_ne!(
        sampled.splits, unsampled.splits,
        "[{label}] the sampled and unsampled structures coincide — the sample never \
         reached the scorer, or this fixture cannot detect it"
    );

    // (B) LEAF VALUES come from the UNSAMPLED channels, so they must DIFFER from the
    // all-sampled grow. That difference IS the contract.
    let (leaf_abs, _r) = max_divergence(&sampled.leaf_values, &all_sampled.leaf_values);
    assert!(
        leaf_abs > EPS,
        "[{label}] sampled and all-sampled leaf values coincide (abs={leaf_abs:.3e}) — \
         leaf estimation must NOT see the sample"
    );

    // (C) …and they equal `calc_average` over each leaf's RAW der/weight, recomputed here
    // independently from the emitted routing.
    let leaf_count = sampled.leaf_values.len();
    let mut leaves: Vec<Vec<usize>> = vec![Vec::new(); leaf_count];
    for (obj, &leaf) in sampled.leaf_of.iter().enumerate() {
        if let Some(slot) = leaves.get_mut(leaf as usize) {
            slot.push(obj);
        }
    }
    let expected: Vec<f64> = leaves
        .iter()
        .map(|docs| {
            let ds: Vec<f64> = docs.iter().map(|&i| der1[i]).collect();
            let ws: Vec<f64> = docs.iter().map(|&i| weight[i]).collect();
            calc_average(sum_f64(&ds), sum_f64(&ws), scaled_l2)
        })
        .collect();
    let (abs, rel) = max_divergence(&sampled.leaf_values, &expected);
    println!("[{label}] sampled leaf oracle: abs={abs:.3e} rel={rel:.3e} (bar={EPS:.0e})");
    assert!(
        abs <= EPS || rel <= EPS,
        "[{label}] sampled leaf values must be calc_average over the RAW channels: \
         abs={abs:.3e} rel={rel:.3e}"
    );
}

#[test]
fn depthwise_matches_cpu_with_nontrivial_sample() {
    assert_sampled_matches(NonsymPolicy::Depthwise, SCORE_FN_L2, "depthwise-sampled-l2");
}

#[test]
fn lossguide_matches_cpu_with_nontrivial_sample() {
    // Lossguide additionally orders its priority queue by the per-node GAIN, which is
    // computed from the score channels — so this arm exercises the sample's effect on
    // expansion ORDER, not just on each node's chosen split.
    assert_sampled_matches(NonsymPolicy::Lossguide, SCORE_FN_COSINE, "lossguide-sampled-cosine");
}
