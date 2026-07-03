//! Unit self-oracle for the Phase 12 Plan 03 (GPUT-18) device-fold NON-SYMMETRIC arm:
//! the transcribed `device_leaf_of_nonsym` pointer-walk (the load-bearing piece of the
//! `boosting.rs` `!dev_tree.step_nodes.is_empty()` fold arm) must assign each object to the
//! SAME distinct leaf as an independent replica of `cb_model::apply::leaf_index_nonsym` over
//! the identical hand-built node graph.
//!
//! Mounted as a sibling `#[path]` submodule of `boosting` (source/test separation, CLAUDE.md),
//! so it reaches the private `super::device_leaf_of_nonsym` + `super::Split` directly. The
//! END-TO-END "folds into `non_symmetric_trees`, `trees` stays empty" assertion runs through
//! `cb_train::train()` in `device_nonsym_fit_test.rs` (Task 3) — a src-mounted unit test
//! cannot instantiate `cb_model` (the `cb_train` dev-dep diamond, 12-02 SUMMARY), so this
//! file replicates the walk inline as its reference rather than importing `cb_model`.

use super::{device_leaf_of_nonsym, Split};

/// An independent replica of the non-symmetric walk (`leaf_index_nonsym`): a bounded
/// flat-node walk over `step_nodes`, halting on the zero side and reading the distinct
/// leaf id. Written FRESH here (not calling `device_leaf_of_nonsym`) so the test is a real
/// cross-check, not a tautology.
fn replica_walk(
    obj: usize,
    splits: &[Split],
    step_nodes: &[(u16, u16)],
    node_id_to_leaf_id: &[u32],
    features: &[Vec<f32>],
) -> Option<usize> {
    let mut index: i64 = 0;
    for _ in 0..=step_nodes.len() {
        let idx = usize::try_from(index).ok()?;
        let &(left, right) = step_nodes.get(idx)?;
        let split = splits.get(idx)?;
        let v = *features.get(split.feature).and_then(|c| c.get(obj))?;
        let passes = f64::from(v) > split.border;
        let diff: i64 = if passes { i64::from(right) } else { i64::from(left) };
        index += diff;
        if diff == 0 {
            let leaf = *node_id_to_leaf_id.get(idx)?;
            if leaf == u32::MAX {
                return None;
            }
            return usize::try_from(leaf).ok();
        }
    }
    None
}

/// Build a fixed 5-node non-symmetric graph:
/// - node 0 (interior): `f0 > 0.5`, children (1, 2)  → step (1, 2)
/// - node 1 (interior): `f1 > 0.5`, children (3, 4)  → step (2, 3)
/// - node 2 (leaf, id 0), node 3 (leaf, id 1), node 4 (leaf, id 2)
///
/// Routing: `f0 > 0.5` → leaf 0; `f0 <= 0.5 & f1 <= 0.5` → leaf 1;
/// `f0 <= 0.5 & f1 > 0.5` → leaf 2.
fn fixture() -> (Vec<Split>, Vec<(u16, u16)>, Vec<u32>) {
    let splits = vec![
        Split { feature: 0, border: 0.5 },
        Split { feature: 1, border: 0.5 },
        Split { feature: 0, border: 0.0 }, // inert leaf placeholder
        Split { feature: 0, border: 0.0 },
        Split { feature: 0, border: 0.0 },
    ];
    let step_nodes = vec![(1u16, 2u16), (2, 3), (0, 0), (0, 0), (0, 0)];
    let node_id_to_leaf_id = vec![u32::MAX, u32::MAX, 0, 1, 2];
    (splits, step_nodes, node_id_to_leaf_id)
}

#[test]
fn device_leaf_of_nonsym_matches_replica_walk() {
    let (splits, step_nodes, node_id_to_leaf_id) = fixture();
    // 6 objects spanning all three leaves.
    let f0 = vec![1.0_f32, 2.0, 0.0, 0.0, 0.2, 0.9];
    let f1 = vec![0.0_f32, 9.0, 0.0, 1.0, 0.7, 0.1];
    let features = vec![f0, f1];
    let n = 6usize;

    let mut seen = [false; 3];
    for obj in 0..n {
        let got = device_leaf_of_nonsym(obj, &splits, &step_nodes, &node_id_to_leaf_id, &features);
        let want = replica_walk(obj, &splits, &step_nodes, &node_id_to_leaf_id, &features);
        assert_eq!(got, want, "walk disagreement at obj {obj}");
        let leaf = got.expect("every object must reach a valid leaf");
        assert!(leaf < 3, "leaf id {leaf} out of range at obj {obj}");
        if let Some(s) = seen.get_mut(leaf) {
            *s = true;
        }
    }
    // The fixture objects exercise all three distinct leaves.
    assert_eq!(seen, [true, true, true], "all three leaves must be reachable");

    // Spot-check the routing semantics explicitly (independent of the replica).
    // obj 0: f0=1.0 > 0.5 → leaf 0.
    assert_eq!(
        device_leaf_of_nonsym(0, &splits, &step_nodes, &node_id_to_leaf_id, &features),
        Some(0)
    );
    // obj 3: f0=0.0 <= 0.5, f1=1.0 > 0.5 → leaf 2.
    assert_eq!(
        device_leaf_of_nonsym(3, &splits, &step_nodes, &node_id_to_leaf_id, &features),
        Some(2)
    );
    // obj 4: f0=0.2 <= 0.5, f1=0.7 > 0.5 → leaf 2.
    assert_eq!(
        device_leaf_of_nonsym(4, &splits, &step_nodes, &node_id_to_leaf_id, &features),
        Some(2)
    );
    // obj 2: f0=0.0 <= 0.5, f1=0.0 <= 0.5 → leaf 1.
    assert_eq!(
        device_leaf_of_nonsym(2, &splits, &step_nodes, &node_id_to_leaf_id, &features),
        Some(1)
    );
}

#[test]
fn malformed_graph_yields_none_not_panic() {
    // A cyclic graph: node 0's left diff is 0 but its leaf slot is the interior sentinel
    // (`u32::MAX`) — the walk halts on the zero side but finds no real leaf id → None
    // (the caller substitutes a checked leaf-0 fallback, never a panic, T-12-05).
    let splits = vec![Split { feature: 0, border: 0.5 }];
    let step_nodes = vec![(0u16, 0u16)];
    let node_id_to_leaf_id = vec![u32::MAX];
    let features = vec![vec![0.0_f32]];
    assert_eq!(
        device_leaf_of_nonsym(0, &splits, &step_nodes, &node_id_to_leaf_id, &features),
        None,
        "an interior-sentinel halt point must yield None, not a fabricated leaf"
    );

    // A self-loop (node 0 points back to itself with a non-zero diff on both sides) must be
    // rejected by the visit cap, not spin forever.
    let splits = vec![Split { feature: 0, border: 0.5 }, Split { feature: 0, border: 0.5 }];
    let step_nodes = vec![(1u16, 1u16), (u16::MAX, u16::MAX)];
    let node_id_to_leaf_id = vec![u32::MAX, u32::MAX];
    let features = vec![vec![0.0_f32]];
    assert_eq!(
        device_leaf_of_nonsym(0, &splits, &step_nodes, &node_id_to_leaf_id, &features),
        None,
        "a walk exceeding the node-count cap must terminate as None"
    );
}
