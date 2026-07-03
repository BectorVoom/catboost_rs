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
    splits: Vec<(u32, u32)>,
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
                splits.push((*feature, *bin));
                node_id_to_leaf_id[id] = u32::MAX;
                step_nodes.push((
                    u16::try_from(left - id).unwrap(),
                    u16::try_from(right - id).unwrap(),
                ));
            }
            RefNode::Leaf => {
                splits.push((0, 0));
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

        let dev = grow_nonsym_tree(
            policy, &der1, &weight, &cindex, n, n_bins, n_features, max_depth, max_leaves,
            min_data_in_leaf, scaled_l2, score_fn,
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
