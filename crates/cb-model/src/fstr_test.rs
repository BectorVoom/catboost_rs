//! Unit tests for `fstr.rs` (FSTR-01: FIC-01/FIC-02/FIC-03 — CTR-aware
//! `interaction()` / `prediction_values_change()`). Sibling `#[path]` mount
//! (source/test separation, CLAUDE.md), mirroring `ctr_data.rs:58-61`.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use super::*;
use crate::ctr_data::{ECtrType, Prior};
use crate::model::{CtrSplit, Model, ModelSplit, NonSymmetricTree, ObliviousTree, Split};

// ── Shared fixtures ─────────────────────────────────────────────────────────

/// An all-empty, all-oblivious, float-only, `ctr_data: None` model — the
/// baseline every hand-built fixture below overrides one field of. Mirrors
/// `export/onnx_test.rs::empty_model`.
fn empty_model() -> Model {
    Model {
        oblivious_trees: Vec::new(),
        non_symmetric_trees: Vec::new(),
        region_trees: Vec::new(),
        bias: 0.0,
        float_feature_borders: Vec::new(),
        ctr_data: None,
        approx_dimension: 1,
        class_to_label: Vec::new(),
    }
}

/// A minimal `CtrSplit` over the given projection — mirrors
/// `export/onnx_test.rs::minimal_ctr_split`, parameterized by projection.
fn ctr_split(projection: cb_train::TProjection) -> CtrSplit {
    CtrSplit {
        projection,
        ctr_type: ECtrType::Borders,
        prior: Prior { num: 0.0, denom: 1.0 },
        target_border_idx: 0,
        border: 0.0,
        shift: 0.0,
        scale: 1.0,
    }
}

// ── FIC-01: cat_feature_count / flat_cat_index ──────────────────────────────

#[test]
fn cat_feature_count_zero_when_no_ctr_splits() {
    // AT-FIC01a: a model with only `ModelSplit::Float` splits (both tree
    // kinds) has `cat_feature_count == 0`.
    let mut model = empty_model();
    model.oblivious_trees.push(ObliviousTree {
        splits: vec![ModelSplit::Float(Split { feature: 0, border: 1.0 })],
        leaf_values: vec![0.0, 1.0],
        leaf_weights: vec![1.0, 1.0],
    });
    assert_eq!(cat_feature_count(&model), 0);
}

#[test]
fn cat_feature_count_single_ctr_split() {
    // AT-FIC01b: one Ctr split with `cat_features() == [2]` -> `2 + 1 == 3`.
    let mut model = empty_model();
    model.oblivious_trees.push(ObliviousTree {
        splits: vec![ModelSplit::Ctr(ctr_split(cb_train::TProjection::single(2)))],
        leaf_values: vec![0.0, 1.0],
        leaf_weights: vec![1.0, 1.0],
    });
    assert_eq!(cat_feature_count(&model), 3);
}

#[test]
fn cat_feature_count_combination_ctr_uses_max_member() {
    // AT-FIC01c: `cat_features() == [0, 3]` -> count reflects the MAX member
    // index (`3 + 1 == 4`), NOT the member count (`2`).
    let mut model = empty_model();
    model.oblivious_trees.push(ObliviousTree {
        splits: vec![ModelSplit::Ctr(ctr_split(cb_train::TProjection::from_features(&[0, 3])))],
        leaf_values: vec![0.0, 1.0],
        leaf_weights: vec![1.0, 1.0],
    });
    assert_eq!(cat_feature_count(&model), 4);
}

#[test]
fn cat_feature_count_max_across_both_tree_kinds() {
    // AT-FIC01d: a Ctr split (`cat_features() == [1]`) in an oblivious tree
    // AND a DIFFERENT Ctr split (`cat_features() == [4]`) in a non-symmetric
    // tree -> overall max `4 + 1 == 5`.
    let mut model = empty_model();
    model.oblivious_trees.push(ObliviousTree {
        splits: vec![ModelSplit::Ctr(ctr_split(cb_train::TProjection::single(1)))],
        leaf_values: vec![0.0, 1.0],
        leaf_weights: vec![1.0, 1.0],
    });
    model.non_symmetric_trees.push(NonSymmetricTree {
        tree_splits: vec![
            ModelSplit::Ctr(ctr_split(cb_train::TProjection::single(4))),
            // Leaf placeholder splits (WR-04 sentinel), matching the real
            // shape a `Model::from_trained` lift would produce.
            ModelSplit::Float(Split { feature: 0, border: 0.0 }),
            ModelSplit::Float(Split { feature: 0, border: 0.0 }),
        ],
        step_nodes: vec![(1, 2), (0, 0), (0, 0)],
        node_id_to_leaf_id: vec![0, 0, 1],
        leaf_values: vec![0.0, 1.0],
        leaf_weights: vec![1.0, 1.0],
    });
    assert_eq!(cat_feature_count(&model), 5);
}

#[test]
fn flat_cat_index_is_n_float_plus_local() {
    assert_eq!(flat_cat_index(5, 0), 5);
    assert_eq!(flat_cat_index(5, 2), 7);
}

// ── FIC-02: interaction() CTR-aware pairwise attribution ────────────────────

/// AT-FIC02a (regression, unit): a small float-only oblivious model (2 float
/// splits, 4 leaves with distinct values) — `interaction()` must return the
/// SAME triples this exact model produced BEFORE this task's CTR-aware
/// changes. The expected value below was captured by hand-deriving the
/// PRE-CHANGE `interaction()` arithmetic on this exact fixture (2-split
/// oblivious bit-indexed loop: `delta = -val0 + val1 + val2 - val3 = -1.0 +
/// 2.0 + 3.0 - 0.0 = 4.0`; single pair `(0, 1)` gets 100% of the total) —
/// this is a golden-value lock, not a tautology, since the arithmetic was
/// worked out independently of any new code.
#[test]
fn interaction_float_only_unchanged() {
    let mut model = empty_model();
    model.oblivious_trees.push(ObliviousTree {
        splits: vec![
            ModelSplit::Float(Split { feature: 0, border: 1.0 }),
            ModelSplit::Float(Split { feature: 1, border: 2.0 }),
        ],
        leaf_values: vec![1.0, 2.0, 3.0, 0.0],
        leaf_weights: vec![1.0, 1.0, 1.0, 1.0],
    });
    assert_eq!(interaction(&model), vec![(0, 1, 100.0)]);
}

/// AT-FIC02b (unit): a tiny oblivious tree with one `Float` split and one
/// `Ctr` split (single cat feature) at the two split levels. `n_float == 1`
/// (only float split feature `0`), so `flat_cat_index(1, 0) == 1`. Hand-worked
/// `delta` (same bit-indexed formula as AT-FIC02a): `-1.0 + 2.0 + 3.0 - 0.0 =
/// 4.0`; `side1.len() == 1` (simple CTR) so NO division — the single
/// resulting pair gets 100% of the total.
#[test]
fn interaction_float_ctr_pair_single_cat() {
    let mut model = empty_model();
    model.oblivious_trees.push(ObliviousTree {
        splits: vec![
            ModelSplit::Float(Split { feature: 0, border: 1.0 }),
            ModelSplit::Ctr(ctr_split(cb_train::TProjection::single(0))),
        ],
        leaf_values: vec![1.0, 2.0, 3.0, 0.0],
        leaf_weights: vec![1.0, 1.0, 1.0, 1.0],
    });
    assert_eq!(interaction(&model), vec![(0, flat_cat_index(1, 0), 100.0)]);
}

/// AT-FIC02c (unit): a tiny oblivious tree with a `Float` split at the FIRST
/// level and a combination-CTR split (`cat_features() == [0, 1]`) at the
/// SECOND level. `n_float == 1`. Same `delta = 4.0` as above; `side0.len() ==
/// 1` (float), `side1.len() == 2` (combination CTR) -> EACH resulting pair
/// gets `delta.abs() / (1 * 2) == 2.0`, which is 50% of the total (`2.0 +
/// 2.0 == 4.0`).
#[test]
fn interaction_combination_ctr_splits_equally() {
    let mut model = empty_model();
    model.oblivious_trees.push(ObliviousTree {
        splits: vec![
            ModelSplit::Float(Split { feature: 0, border: 1.0 }),
            ModelSplit::Ctr(ctr_split(cb_train::TProjection::from_features(&[0, 1]))),
        ],
        leaf_values: vec![1.0, 2.0, 3.0, 0.0],
        leaf_weights: vec![1.0, 1.0, 1.0, 1.0],
    });
    assert_eq!(
        interaction(&model),
        vec![
            (0, flat_cat_index(1, 0), 50.0),
            (0, flat_cat_index(1, 1), 50.0),
        ]
    );
}

/// AT-FIC02e (unit, non-symmetric arm — MANDATORY, per plan-checker pass #3
/// CRITICAL-1): a hand-built non-symmetric `Model` with TWO `ModelSplit::Ctr`
/// splits at DIFFERENT node depths on the SAME root-to-leaf path:
/// `cat_features() == [0, 1]` at the shallower (root) node, `cat_features()
/// == [1, 2]` at a deeper node (the root's left child) — a PARTIAL overlap on
/// cat feature `1`.
///
/// Tree shape (node indices):
/// ```text
///        0 (Ctr[0,1])
///       / \
///      1   2 (terminal leaf, path len 1 -> contributes no pairs)
///  (Ctr[1,2])
///     /  \
///    3    4
///  (leaf) (leaf)
/// ```
/// Leaf values: node 3 (`L0`'s LEFT child) = `L = +3.0`, node 4 (`L0`'s RIGHT
/// child) = `R = +1.0` — SAME sign, unequal magnitude (plan-checker pass #3
/// CRITICAL-1's corrected worked example: an opposite-signed pair would make
/// the correct and the sign-dropping-buggy implementations coincide
/// numerically, defeating this test's purpose).
///
/// Hand-derivation (upstream two-stage semantics, `feature_str.cpp:226-295` +
/// `calc_fstr.cpp:343-414`; sign convention: left child sign `-1`, right
/// child sign `+1`, freshly assigned per node): the two CTR splits are
/// DISTINCT internal features (different projections), so the per-tree SIGNED
/// internal-pair sum is `sign_L*L + sign_R*R` with `sign_L = (-1)*(-1) = +1`
/// (leaf 3's path) and `sign_R = (-1)*(+1) = -1` (leaf 4's path):
/// `3.0 - 1.0 = 2.0`; per-tree `.abs()` → the internal pair's score is `2.0`.
/// A sign-dropping/abs-per-leaf-then-sum bug would instead compute `|L| +
/// |R| = 4.0` — genuinely different, so this test still fails hard under
/// that regression class. The `CalcFeatureInteraction` expansion then
/// distributes `score / (2*2) = 0.5` per cross-product cell of `{0,1} x
/// {1,2}`; the `(1,1)` cell (both resolve to cat feature `1`) is a SELF-CELL
/// and is dropped from the OUTPUT — but the pair's FULL `2.0` still enters
/// `totalEffect` (upstream `totalEffect += effect` sits outside the
/// cross-product loops). The three surviving cells `(0,1)`, `(0,2)`, `(1,2)`
/// therefore each score `0.5 / 2.0 * 100 = 25.0` — the percentages sum to
/// 75, NOT 100, because the dropped self-cell mass deflates the total (the
/// same upstream behavior AT-FIC02d's oracle fixture locks: its five scores
/// sum to ~92.5). **`n_float == 1` here** (NOT `0`): every leaf node's
/// `tree_splits` entry is the standard `Float { feature: 0, .. }` placeholder
/// sentinel (the same convention the real trainer emits,
/// `cb-train/src/tree.rs:1718`), and the EXISTING, unchanged `feature_count`
/// counts it via `float_feature()` like any other `Float` split (it has no
/// notion of the `(0, 0)` leaf-sentinel special case) — so
/// `feature_count(model) == 1` even though no split is ever actually resolved
/// as float feature 0 in the DFS walk. `flat_cat_index(1, c) == 1 + c`.
#[test]
fn interaction_non_symmetric_two_ctr_splits_partial_overlap_self_pair() {
    let root_ctr = ctr_split(cb_train::TProjection::from_features(&[0, 1]));
    let deep_ctr = ctr_split(cb_train::TProjection::from_features(&[1, 2]));
    let leaf_placeholder = || ModelSplit::Float(Split { feature: 0, border: 0.0 });

    let mut model = empty_model();
    model.non_symmetric_trees.push(NonSymmetricTree {
        tree_splits: vec![
            ModelSplit::Ctr(root_ctr),  // node 0
            ModelSplit::Ctr(deep_ctr),  // node 1
            leaf_placeholder(),         // node 2 (terminal, R0)
            leaf_placeholder(),         // node 3 (terminal, LL = L)
            leaf_placeholder(),         // node 4 (terminal, LR = R)
        ],
        step_nodes: vec![(1, 2), (2, 3), (0, 0), (0, 0), (0, 0)],
        node_id_to_leaf_id: vec![0, 0, 0, 1, 2],
        leaf_values: vec![0.0, 3.0, 1.0], // leaf0=R0 (unused), leaf1=L, leaf2=R
        leaf_weights: vec![1.0, 1.0, 1.0],
    });

    // n_float == 1: `feature_count(model)` counts the leaf-sentinel `Float
    // { feature: 0, .. }` placeholders like any other Float split (see the
    // doc comment above).
    let n_float = 1usize;
    // score 2.0 per internal pair; 0.5 per surviving cell; totalEffect = 2.0
    // (the dropped (1,1) self-cell's 0.5 stays in the denominator).
    let expected_score = 0.5_f64 / 2.0_f64 * 100.0;
    assert_eq!(
        interaction(&model),
        vec![
            (flat_cat_index(n_float, 0), flat_cat_index(n_float, 1), expected_score),
            (flat_cat_index(n_float, 0), flat_cat_index(n_float, 2), expected_score),
            (flat_cat_index(n_float, 1), flat_cat_index(n_float, 2), expected_score),
        ]
    );
}

/// AT-FIC02f (unit, oblivious arm — the denominator lock the fstr_ctr oracle
/// exposed): ONE oblivious tree whose two split levels are a simple CTR
/// (`cat_features() == [0]`) and a combination CTR (`cat_features() == [0,
/// 1]`) — DISTINCT internal features with a PARTIAL flat overlap on cat
/// feature `0`. `n_float == 0`, so flats are `c0 = 0`, `c1 = 1`. Same `delta
/// = 4.0` as AT-FIC02a. Expansion: cells `(0,0)` (SELF — dropped from output
/// only) and `(0,1)` (`delta/2 = 2.0`); `totalEffect = 4.0` (the FULL pair
/// score, upstream `totalEffect += effect`). The single returned pair
/// therefore scores `2.0 / 4.0 * 100 = 50.0` — a
/// drop-self-cells-from-the-denominator bug would return `100.0` instead
/// (exactly the pre-fix behavior the fstr_ctr oracle caught as a uniform
/// ~1.081x inflation of every score).
#[test]
fn interaction_partial_overlap_keeps_self_cell_mass_in_total() {
    let mut model = empty_model();
    model.oblivious_trees.push(ObliviousTree {
        splits: vec![
            ModelSplit::Ctr(ctr_split(cb_train::TProjection::single(0))),
            ModelSplit::Ctr(ctr_split(cb_train::TProjection::from_features(&[0, 1]))),
        ],
        leaf_values: vec![1.0, 2.0, 3.0, 0.0],
        leaf_weights: vec![1.0, 1.0, 1.0, 1.0],
    });
    assert_eq!(interaction(&model), vec![(0, 1, 50.0)]);
}

/// AT-FIC02g (unit, oblivious arm — internal-feature grouping): a tree with a
/// `Float` split and TWO borders of the SAME CTR internal feature (identical
/// projection / type / prior / shift / scale, different `border` — upstream
/// `TFeature` equality is border-less). The two same-internal levels' pair is
/// skipped OUTRIGHT (upstream `srcFeature1 == srcFeature2 → continue` in
/// `CalcMostInteractingFeatures`) — it contributes to NEITHER the output NOR
/// `totalEffect` — so the two surviving (Float × Ctr-border) pairs' scores
/// sum to exactly 100. An include-same-internal-pairs-in-the-total bug would
/// make the sum land strictly below 100.
#[test]
fn interaction_same_internal_ctr_borders_pair_skipped_entirely() {
    let mut base = ctr_split(cb_train::TProjection::single(0));
    base.border = 0.25;
    let mut second_border = base.clone();
    second_border.border = 0.75;

    let mut model = empty_model();
    model.oblivious_trees.push(ObliviousTree {
        splits: vec![
            ModelSplit::Float(Split { feature: 0, border: 1.0 }),
            ModelSplit::Ctr(base),
            ModelSplit::Ctr(second_border),
        ],
        leaf_values: vec![1.0, 2.0, 3.0, 0.0, 2.0, 5.0, 0.0, 4.0],
        leaf_weights: vec![1.0; 8],
    });

    let pairs = interaction(&model);
    // Every returned pair is (float 0, flat cat 1); the same-internal
    // (Ctr b=0.25, Ctr b=0.75) level pair contributed nothing anywhere.
    assert!(pairs.iter().all(|&(a, b, _)| (a, b) == (0, 1)), "pairs: {pairs:?}");
    let total: f64 = pairs.iter().map(|&(_, _, s)| s).sum();
    assert!((total - 100.0).abs() < 1e-12, "scores must sum to 100, got {total}");
}

// ── FIC-03: prediction_values_change() CTR-aware redistribution ────────────

/// AT-FIC03a (regression, unit): the SAME 2-float-split, 4-leaf model as
/// AT-FIC02a. The expected values were hand-derived from the PRE-CHANGE
/// `pvc_accumulate_oblivious` bit-indexed formula (`avrg`/`dif` over each
/// `(leaf, leaf ^ (1<<bit))` pair): `res[0] == 0.5 + 4.5 == 5.0`,
/// `res[1] == 2.0 + 2.0 == 4.0`, then `ConvertToPercents` scales to sum 100 —
/// a golden-value lock (Rust-evaluated, so float rounding matches exactly),
/// not a tautology.
#[test]
fn pvc_float_only_unchanged() {
    let mut model = empty_model();
    model.oblivious_trees.push(ObliviousTree {
        splits: vec![
            ModelSplit::Float(Split { feature: 0, border: 1.0 }),
            ModelSplit::Float(Split { feature: 1, border: 2.0 }),
        ],
        leaf_values: vec![1.0, 2.0, 3.0, 0.0],
        leaf_weights: vec![1.0, 1.0, 1.0, 1.0],
    });
    let expected = vec![5.0_f64 / 9.0 * 100.0, 4.0_f64 / 9.0 * 100.0];
    assert_eq!(prediction_values_change(&model), expected);
}

/// AT-FIC03b (unit): a tiny oblivious tree with ONE `Ctr` split (simple,
/// single cat feature) and non-zero leaf weights on both sides. Hand-worked
/// `dif` (same `avrg`/`dif` formula, `val=[1.0,4.0]`, `weight=[1.0,1.0]`):
/// `avrg = 2.5`, `dif = (1-2.5)^2 + (4-2.5)^2 = 4.5`. `n_float == 0` (no
/// `Float` splits), `cat_feature_count == 1` -> `res` has length `1`, so the
/// FULL `dif` lands in `res[flat_cat_index(0, 0)] == res[0]`, and after
/// `convert_to_percents` the single non-zero slot is exactly `100.0`.
#[test]
fn pvc_simple_ctr_full_redistribution() {
    let mut model = empty_model();
    model.oblivious_trees.push(ObliviousTree {
        splits: vec![ModelSplit::Ctr(ctr_split(cb_train::TProjection::single(0)))],
        leaf_values: vec![1.0, 4.0],
        leaf_weights: vec![1.0, 1.0],
    });
    let pvc = prediction_values_change(&model);
    assert_eq!(pvc, vec![100.0]);
}

/// AT-FIC03e (unit): `prediction_values_change_with_data` recomputes per-leaf
/// weights from the provided columns via the APPLY path (upstream
/// `CollectLeavesStatistics` — the `data=pool` mode) instead of the stored
/// `leaf_weights`. Same 2-float-split tree as AT-FIC03a (stored weights all
/// `1.0` → no-data PVC is `[5/9, 4/9] * 100`), but the 3 provided documents
/// land in leaves `0`, `3`, `2` → recomputed weights `[1, 0, 1, 1]`:
/// - bit 0 (feature 0): pair `(0,1)` skipped (`count2 == 0`); pair `(2,3)`:
///   `avrg = 1.5`, `dif = 1.5² + 1.5² = 4.5` → `res[0] = 4.5`.
/// - bit 1 (feature 1): pair `(0,2)`: `avrg = 2.0`, `dif = 1 + 1 = 2.0` →
///   `res[1] = 2.0`; pair `(1,3)` skipped (`count1 == 0`).
/// Percentages: `[4.5, 2.0] / 6.5 * 100` — provably different from the
/// stored-weights answer, so a fall-back-to-stored-weights bug fails hard.
#[test]
fn pvc_with_data_uses_recomputed_leaf_statistics() {
    let mut model = empty_model();
    model.oblivious_trees.push(ObliviousTree {
        splits: vec![
            ModelSplit::Float(Split { feature: 0, border: 1.0 }),
            ModelSplit::Float(Split { feature: 1, border: 1.0 }),
        ],
        leaf_values: vec![1.0, 2.0, 3.0, 0.0],
        leaf_weights: vec![1.0, 1.0, 1.0, 1.0],
    });
    // doc0: (0, 0) -> leaf 0; doc1: (2, 2) -> leaf 3; doc2: (0, 2) -> leaf 2.
    let float_cols: Vec<Vec<f32>> = vec![vec![0.0, 2.0, 0.0], vec![0.0, 2.0, 2.0]];

    let stats = crate::apply::collect_leaves_statistics(&model, &float_cols, &[]);
    assert_eq!(stats, vec![vec![1.0, 0.0, 1.0, 1.0]]);

    let expected = vec![4.5_f64 / 6.5 * 100.0, 2.0_f64 / 6.5 * 100.0];
    assert_eq!(prediction_values_change_with_data(&model, &float_cols, &[]), expected);
    // The stored-weights mode is UNCHANGED by the new plumbing (AT-FIC03a's
    // exact expectation, re-asserted here against the same model).
    assert_eq!(
        prediction_values_change(&model),
        vec![5.0_f64 / 9.0 * 100.0, 4.0_f64 / 9.0 * 100.0]
    );
}

/// AT-FIC03c (unit): a tiny oblivious tree with ONE combination-CTR split
/// (`cat_features() == [0, 1]`). Same `dif == 4.5` as AT-FIC03b (the
/// `avrg`/`dif` computation is UNCHANGED regardless of split kind) —
/// equal-split redistribution (NO cross-product, unlike FIC-02, since PVC
/// redistributes a SINGLE split's own effect): `res[flat_cat_index(0,0)] ==
/// res[flat_cat_index(0,1)] == dif / 2.0 == 2.25`, each `50.0%` after
/// `convert_to_percents` (`2.25 + 2.25 == 4.5` total).
#[test]
fn pvc_combination_ctr_equal_split() {
    let mut model = empty_model();
    model.oblivious_trees.push(ObliviousTree {
        splits: vec![ModelSplit::Ctr(ctr_split(cb_train::TProjection::from_features(&[0, 1])))],
        leaf_values: vec![1.0, 4.0],
        leaf_weights: vec![1.0, 1.0],
    });
    let pvc = prediction_values_change(&model);
    assert_eq!(pvc, vec![50.0, 50.0]);
}

// ── FL-01: injected final-error closure (metric-agnostic core) ──────────────

/// A hand-written RMSE `GetFinalError` closure: `sqrt(mean((a − t)^2))`. Routes
/// its element sum through [`cb_core::sum_f64`] (D-08). Used to prove
/// `loss_function_change` is driven by the *injected* closure, not the retained
/// hard-coded Logloss.
fn rmse_final_error(approx: &[f64], target: &[f64]) -> f64 {
    let n = approx.len();
    if n == 0 {
        return 0.0;
    }
    let sq: Vec<f64> = approx
        .iter()
        .zip(target.iter())
        .map(|(&a, &t)| (a - t) * (a - t))
        .collect();
    (cb_core::sum_f64(&sq) / n as f64).sqrt()
}

/// AT-FL-01 (`loss_change_uses_injected_final_error`): the generalized
/// `loss_function_change` computes each per-feature score as
/// `final_error(approx − shap_f) − final_error(approx)` using the SUPPLIED
/// closure. With an RMSE closure the result must equal the RMSE degradation —
/// provably different from the retained Logloss path (which would give the old
/// binary-cross-entropy delta). The expected vector is recomputed in-test from
/// `predict_raw` + `shap_values` so the assertion pins the closure, not a magic
/// number.
#[test]
fn loss_change_uses_injected_final_error() {
    // Tiny oblivious tree over 2 float features → non-trivial per-feature SHAP.
    let mut model = empty_model();
    model.oblivious_trees.push(ObliviousTree {
        splits: vec![
            ModelSplit::Float(Split { feature: 0, border: 1.0 }),
            ModelSplit::Float(Split { feature: 1, border: 1.0 }),
        ],
        leaf_values: vec![1.0, 2.0, 3.0, 0.5],
        leaf_weights: vec![1.0, 1.0, 1.0, 1.0],
    });
    // 4 documents landing across the leaves; regression-style targets.
    let cols: Vec<Vec<f32>> = vec![vec![0.0, 2.0, 0.0, 2.0], vec![0.0, 0.0, 2.0, 2.0]];
    let labels = vec![1.0_f64, 0.0, 2.5, 1.5];
    let n_features = 2;

    // Independent recomputation of the expected LFC vector with the RMSE closure.
    let approx = crate::apply::predict_raw(&model, &cols);
    let shap = shap_values(&model, &cols, n_features);
    let base = rmse_final_error(&approx, &labels);
    let expected: Vec<f64> = (0..n_features)
        .map(|feature| {
            let approx_f: Vec<f64> = (0..approx.len())
                .map(|obj| approx[obj] - shap[obj][feature])
                .collect();
            rmse_final_error(&approx_f, &labels) - base
        })
        .collect();

    let got = loss_function_change(&model, &cols, &labels, n_features, rmse_final_error);
    assert_eq!(got.len(), n_features);
    for (f, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-12,
            "feature {f}: injected RMSE closure score {g} != recomputed {e}"
        );
    }

    // Guard: the RMSE result must genuinely differ from the retained Logloss
    // wrapper's result (else the test could pass even if the closure were
    // ignored and Logloss hard-coded).
    let logloss = loss_function_change_logloss(&model, &cols, &labels, n_features);
    assert!(
        got.iter().zip(logloss.iter()).any(|(r, l)| (r - l).abs() > 1e-9),
        "RMSE closure result must differ from the Logloss wrapper: rmse={got:?} logloss={logloss:?}"
    );
}
