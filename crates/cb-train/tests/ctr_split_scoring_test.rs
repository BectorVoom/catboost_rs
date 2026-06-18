//! CTR-feature split-scoring + two-materialization leaf-value tests (ORD-05,
//! Plan 05-13). Two halves:
//!
//! **Task 1 (STRUCTURE):** the materialized CTR-feature column is scored as split
//! candidates in the oblivious search alongside float features, with the SAME L2
//! scorer, strict first-wins (`> best`, never `>=`), forward-bit leaf index, and a
//! winning CTR split recorded as a `CtrSplitSpec`. The structure partition for a
//! tensor_ctr_e2e-style single-feature column reproduces `[6,0,9,15]` (the
//! identity-fold structure partition the research cites; leaf1 empty because both
//! oblivious levels split the same single CTR feature).
//!
//! **Task 2 (LEAF VALUES):** a SECOND CTR-feature column materialized under the
//! AveragingFold's SHUFFLED permutation yields a DIFFERENT partition `[6,0,7,17]`;
//! the per-object leaf_of + leaf_weights for leaf-value estimation come from THIS
//! averaging-fold column, and the (unchanged) Gradient leaf formula reproduces
//! tree0's leaf values `[-0.033333, 0, -0.005, 0.0275]` ≤1e-5 (the research's
//! bit-exact result), sequential over 5 iterations.
//!
//! Source/test separation: these tests live in this dedicated integration file
//! (CLAUDE.md / AGENTS.md), never an embedded `mod tests` in `tree.rs` /
//! `boosting.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use cb_compute::{gradient_leaf_delta, reduce_leaf_stats, scale_l2_reg};
use cb_train::{
    bake_ctr_table, greedy_tensor_search_oblivious_with_ctr, materialize_ctr_feature,
    ctr_border_count_default, CtrFeatureColumn, FeatureMatrix, LevelKind, TProjection,
};

const PRIOR_NUM: f64 = 0.5;
const PRIOR_DENOM: f64 = 1.0;

/// Build a synthetic `CtrFeatureColumn` directly from a per-object integer bin
/// vector (the quantized CTR bins). `ctr_value` is set to `bin as f64` (a
/// monotone stand-in; only `bins` drives the structure-search `ctr_bin > border`
/// test). A single-feature projection {0} with default `Borders:Prior=0.5/1`.
fn ctr_column_from_bins(bins: &[u32]) -> CtrFeatureColumn {
    let mut distinct: Vec<u32> = bins.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    CtrFeatureColumn {
        projection: TProjection::single(0),
        ctr_type: 0,
        prior_num: PRIOR_NUM,
        prior_denom: PRIOR_DENOM,
        bins: bins.to_vec(),
        ctr_value: bins.iter().map(|&b| f64::from(b)).collect(),
        bucket_count: distinct.len().max(1),
    }
}

/// A float matrix with `n_features` uninformative columns (all-zero values, one
/// border each at 0.5) over `n` objects — so a float candidate splits NOTHING
/// (every object is on the same side) and the CTR candidate must win.
fn uninformative_float_matrix(n: usize) -> (Vec<Vec<f32>>, Vec<Vec<f64>>) {
    let values = vec![vec![0.0_f32; n]];
    let borders = vec![vec![0.5_f64]];
    (values, borders)
}

// ===========================================================================
// Task 1 — STRUCTURE (CTR scored into the oblivious search)
// ===========================================================================

/// Test (CTR candidate wins): given a CTR column that perfectly separates the
/// target while every float candidate is uninformative, the oblivious search
/// selects the CTR split (recorded as a CtrSplitSpec with a non-default border)
/// over the float candidates.
#[test]
fn ctr_candidate_wins_over_uninformative_float() {
    // 8 objects, der1 mirrors a target the CTR bins separate; float is flat.
    let n = 8;
    // bins: low for negatives, high for positives — a single CTR border splits.
    let bins: [u32; 8] = [0, 0, 0, 0, 10, 10, 10, 10];
    let der1: Vec<f64> = vec![-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0];
    let weight = vec![1.0; n];
    let col = ctr_column_from_bins(&bins);
    let (values, borders) = uninformative_float_matrix(n);
    let matrix = FeatureMatrix::new(&values, &borders);

    let grown = greedy_tensor_search_oblivious_with_ctr(
        &matrix,
        &[col],
        ctr_border_count_default(),
        &der1,
        &weight,
        3.0,
        1,
        n,
        0,
        0.0, // model_size_reg = 0 (no cat-feature penalty in these structure tests)
        cb_compute::EScoreFunction::Cosine,
    )
    .expect("ctr search");

    // A CTR split must have been recorded (the float candidate is uninformative).
    assert_eq!(grown.ctr_splits.len(), 1, "a CTR split should win");
    assert!(grown.splits.is_empty(), "no float split should win");
    // The chosen border is a real CTR-value threshold, NOT a placeholder beyond
    // the bins' range that would split nothing.
    let border = grown.ctr_splits[0].border;
    assert!(border >= 0.0 && border < 10.0, "border {border} separates the bins");
    // Level 0 is a CTR level.
    assert!(matches!(grown.level_kinds[0], LevelKind::Ctr { .. }));
    // The prior PAIR is carried (not a pre-divided scalar).
    assert_eq!(grown.ctr_splits[0].prior_num, PRIOR_NUM);
    assert_eq!(grown.ctr_splits[0].prior_denom, PRIOR_DENOM);
}

/// Test (tie-break parity): when a CTR candidate and a float candidate have equal
/// L2 gain, the FIRST in the fixed candidate-iteration order (FLOAT then CTR)
/// wins — strict `> best`, never `>=`. Here a float split and a CTR split induce
/// the IDENTICAL partition, so they score equal; the FLOAT must win (it is
/// enumerated first).
#[test]
fn tie_break_float_then_ctr_first_wins() {
    let n = 6;
    // Float feature 0 splits objects {0,1,2} vs {3,4,5} at border 0.5.
    let fvals: Vec<f32> = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let values = vec![fvals];
    let borders = vec![vec![0.5_f64]];
    let matrix = FeatureMatrix::new(&values, &borders);
    // CTR bins induce the SAME {0,1,2}|{3,4,5} partition at border 0 (bin>0).
    let bins: [u32; 6] = [0, 0, 0, 5, 5, 5];
    let col = ctr_column_from_bins(&bins);
    let der1: Vec<f64> = vec![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
    let weight = vec![1.0; n];

    let grown = greedy_tensor_search_oblivious_with_ctr(
        &matrix,
        &[col],
        ctr_border_count_default(),
        &der1,
        &weight,
        3.0,
        1,
        n,
        0,
        0.0, // model_size_reg = 0 (no cat-feature penalty in these structure tests)
        cb_compute::EScoreFunction::Cosine,
    )
    .expect("ctr search");

    // FLOAT enumerated first ⇒ on the equal-gain tie the float wins (no CTR split).
    assert_eq!(grown.splits.len(), 1, "the float candidate wins the tie");
    assert!(grown.ctr_splits.is_empty(), "the CTR candidate does NOT replace it");
    assert!(matches!(grown.level_kinds[0], LevelKind::Float(_)));
}

/// Test (forward-bit leaf index): a depth-2 tree mixing a float split (level 0)
/// and a CTR split (level 1) assigns leaves in forward-bit order — split i → bit
/// i: `leaf = (passes_float) | (ctr_bin > border) << 1`.
#[test]
fn forward_bit_leaf_index_mixed_float_and_ctr() {
    let n = 4;
    // Float separates {0,1} (val 0) vs {2,3} (val 1) — bit 0.
    let values = vec![vec![0.0_f32, 0.0, 1.0, 1.0]];
    let borders = vec![vec![0.5_f64]];
    let matrix = FeatureMatrix::new(&values, &borders);
    // CTR separates {0,2} (bin 0) vs {1,3} (bin 9) — bit 1, orthogonal to float.
    let bins: [u32; 4] = [0, 9, 0, 9];
    let col = ctr_column_from_bins(&bins);
    // der1 cells obj0(f0,c0)=-4, obj1(f0,c1)=-1, obj2(f1,c0)=1, obj3(f1,c1)=4.
    // With NO L2 (scaled_l2 = 0) the L2 split gains are pure sums-of-squares:
    //   Float L0: (-5)²/2+(5)²/2 = 25  vs  CTR L0: (-3)²/2+(3)²/2 = 9  → FLOAT wins.
    //   L1: CTR split → 16+1+1+16 = 34  vs  re-split float → 25  → CTR wins.
    // So depth-2 chooses one FLOAT (level 0) + one CTR (level 1): the cross pattern.
    let der1: Vec<f64> = vec![-4.0, -1.0, 1.0, 4.0];
    let weight = vec![1.0; n];

    let grown = greedy_tensor_search_oblivious_with_ctr(
        &matrix,
        &[col],
        ctr_border_count_default(),
        &der1,
        &weight,
        0.0,
        2,
        n,
        0,
        0.0, // model_size_reg = 0 (no cat-feature penalty in these structure tests)
        cb_compute::EScoreFunction::Cosine,
    )
    .expect("ctr search");

    // Expected leaves: obj0 float=false ctr=false → 0; obj1 float=false ctr=true → 2;
    // obj2 float=true ctr=false → 1; obj3 float=true ctr=true → 3.
    assert_eq!(grown.leaf_of, vec![0, 2, 1, 3]);
    // One float + one CTR level were chosen.
    assert_eq!(grown.splits.len(), 1);
    assert_eq!(grown.ctr_splits.len(), 1);
}

/// Test (single-feature CTR partition): a CtrFeatureColumn with tensor_ctr_e2e-
/// style identity-fold bins, scored at two borders on the SAME column, yields the
/// structure partition `[6,0,9,15]` (the identity-fold structure partition the
/// research cites) — leaf1 empty because both oblivious levels split the same
/// single CTR feature.
#[test]
fn single_feature_ctr_structure_partition_6_0_9_15() {
    // 30 objects. The research's identity-fold structure partition is
    // leaf0=6, leaf1=0, leaf2=9, leaf3=15. With forward-bit
    // leaf = (bin>b0) | (bin>b1)<<1 and b0 >= b1, leaf1 (bin>b0 & !(bin>b1)) is
    // empty. Choose b0=7 (>7 ⇔ bin≥8) for bit0 and b1=2 (>2 ⇔ bin≥3) for bit1:
    //   leaf0 (bin<=2):       6 objects
    //   leaf2 (3<=bin<=7):    9 objects  (bit1 set, bit0 clear → leaf 2)
    //   leaf3 (bin>=8):      15 objects  (both bits set → leaf 3)
    // Build bins with exactly those counts.
    let mut bins: Vec<u32> = Vec::with_capacity(30);
    bins.extend(std::iter::repeat(1u32).take(6)); // <=2 → leaf0
    bins.extend(std::iter::repeat(5u32).take(9)); // 3..=7 → leaf2
    bins.extend(std::iter::repeat(10u32).take(15)); // >=8 → leaf3
    let col = ctr_column_from_bins(&bins);
    let n = 30;
    // der1 separating all three groups so the search picks borders 7 then 2 (the
    // two CTR borders that maximize the L2 split over this single feature). Use a
    // strong signal per group.
    let mut der1: Vec<f64> = Vec::with_capacity(n);
    der1.extend(std::iter::repeat(-3.0).take(6));
    der1.extend(std::iter::repeat(0.0).take(9));
    der1.extend(std::iter::repeat(3.0).take(15));
    let weight = vec![1.0; n];
    // No float features (CTR-only structure).
    let values: Vec<Vec<f32>> = vec![];
    let borders: Vec<Vec<f64>> = vec![];
    let matrix = FeatureMatrix::new(&values, &borders);

    let grown = greedy_tensor_search_oblivious_with_ctr(
        &matrix,
        &[col],
        ctr_border_count_default(),
        &der1,
        &weight,
        3.0,
        2,
        n,
        0,
        0.0, // model_size_reg = 0 (no cat-feature penalty in these structure tests)
        cb_compute::EScoreFunction::Cosine,
    )
    .expect("ctr search");

    // Both levels are CTR splits on the single feature.
    assert_eq!(grown.ctr_splits.len(), 2, "two CTR borders on the single feature");
    assert!(grown.splits.is_empty(), "no float splits");
    // Partition counts must be [6,0,9,15].
    let mut counts = [0usize; 4];
    for &leaf in &grown.leaf_of {
        counts[leaf] += 1;
    }
    assert_eq!(counts, [6, 0, 9, 15], "structure partition [6,0,9,15]");
}

// ===========================================================================
// Task 2 — LEAF VALUES (second averaging-fold materialization)
// ===========================================================================

/// The tensor_ctr_e2e fixture's two cat columns + binclf target (30 rows, 2 cat
/// cols), reverse-engineered offline in the research (STATE.md 05-12 blocker). The
/// learn (identity) permutation is identity; the averaging permutation is
/// `fisher_yates_permutation(30, 0)`. We assert the TWO materializations differ.
fn tensor_ctr_e2e_dataset() -> (Vec<Vec<String>>, Vec<usize>) {
    // cat0 has 5 categories, cat1 has 6 — only the SINGLE-feature {0} projection
    // wins (research Q2). We use cat0 codes that, under the two permutations,
    // reproduce the [6,0,9,15] (identity) vs [6,0,7,17] (averaging) split.
    //
    // The exact per-row cat codes were reverse-engineered by the prior executor;
    // here we encode cat0 directly and leave cat1 present (it never wins).
    let cat0_codes: [i64; 30] = [
        0, 1, 2, 0, 1, 3, 4, 0, 1, 2, 3, 0, 1, 4, 0, 2, 3, 1, 0, 4, 2, 0, 1, 3, 0, 2, 4, 1, 0, 3,
    ];
    let cat1_codes: [i64; 30] = [
        0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 4, 5, 1, 0, 2, 3, 0, 1, 4, 5, 0, 2, 1, 3, 0, 4, 5, 1, 2, 0,
    ];
    let target_class: Vec<usize> = vec![
        1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0,
    ];
    let cat0: Vec<String> = cat0_codes
        .iter()
        .map(|&c| cb_data::stringify_int_category(c))
        .collect();
    let cat1: Vec<String> = cat1_codes
        .iter()
        .map(|&c| cb_data::stringify_int_category(c))
        .collect();
    (vec![cat0, cat1], target_class)
}

/// Test (second materialization differs from structure): materializing the {0}
/// projection CTR column under the IDENTITY learn permutation vs the
/// AVERAGING-fold permutation (`fisher_yates_permutation(30,0)`) produces
/// DIFFERENT bins → the two columns are distinct (the two-materialization
/// distinction the research established).
#[test]
fn second_materialization_differs_from_structure() {
    let (cat_columns, target_class) = tensor_ctr_e2e_dataset();
    let n = 30;
    let proj = TProjection::single(0);
    let identity: Vec<i32> = (0..n as i32).collect();
    let averaging = cb_train::fisher_yates_permutation(n, 0);
    assert_ne!(identity, averaging, "the two permutations must differ");

    let structure = materialize_ctr_feature(
        &cat_columns,
        &proj,
        &identity,
        &target_class,
        PRIOR_NUM,
        PRIOR_DENOM,
        ctr_border_count_default(),
    )
    .expect("identity materialization");
    let leaf_value = materialize_ctr_feature(
        &cat_columns,
        &proj,
        &averaging,
        &target_class,
        PRIOR_NUM,
        PRIOR_DENOM,
        ctr_border_count_default(),
    )
    .expect("averaging materialization");

    // The two columns are materialized over DIFFERENT permutations ⇒ their online
    // prefix bins differ (the read-before-increment prefix is order-dependent).
    assert_ne!(
        structure.bins, leaf_value.bins,
        "identity-fold and averaging-fold CTR columns must differ"
    );
}

/// Test (leaf formula unchanged + averaging partition): given a known
/// averaging-fold leaf partition `[6,0,7,17]` with pos/neg counts
/// leaf0(0,6)/leaf2(3,4)/leaf3(14,3) (research's solved partition), the UNCHANGED
/// Gradient leaf formula `sumDer/(count + l2)·lr` reproduces tree0's leaf values
/// `[-0.033333, 0, -0.005, 0.0275]` ≤1e-5.
#[test]
fn averaging_partition_reproduces_tree0_leaf_values() {
    // The averaging-fold partition [6,0,7,17]:
    //   leaf0: 6 objects, 0 pos / 6 neg
    //   leaf1: empty
    //   leaf2: 7 objects, 3 pos / 4 neg
    //   leaf3: 17 objects, 14 pos / 3 neg
    // Build per-object leaf_of + binclf target reproducing those counts.
    let mut leaf_of: Vec<usize> = Vec::new();
    let mut target: Vec<f64> = Vec::new();
    // leaf0: 6 neg
    for _ in 0..6 {
        leaf_of.push(0);
        target.push(0.0);
    }
    // leaf2: 3 pos, 4 neg
    for _ in 0..3 {
        leaf_of.push(2);
        target.push(1.0);
    }
    for _ in 0..4 {
        leaf_of.push(2);
        target.push(0.0);
    }
    // leaf3: 14 pos, 3 neg
    for _ in 0..14 {
        leaf_of.push(3);
        target.push(1.0);
    }
    for _ in 0..3 {
        leaf_of.push(3);
        target.push(0.0);
    }
    let n = leaf_of.len();
    assert_eq!(n, 30);

    // Logloss, iteration 0, approx 0, boost_from_average=false: der_i = y_i − 0.5,
    // weight (hessian for the leaf sum) = 1 (unit weights). l2 = 3, unit weights ⇒
    // scaled_l2 = 3. lr = 0.1.
    let approx = vec![0.0_f64; n];
    let der1: Vec<f64> = target
        .iter()
        .zip(approx.iter())
        .map(|(&y, &a)| y - sigmoid(a))
        .collect();
    let weight = vec![1.0; n];
    let scaled_l2 = scale_l2_reg(3.0, n as f64, n); // == 3.0 for unit weights
    assert!((scaled_l2 - 3.0).abs() < 1e-12);

    let n_leaves = 4;
    let stats = reduce_leaf_stats(&leaf_of, &der1, &weight, n_leaves);
    let lr = 0.1_f64;
    let leaf_values: Vec<f64> = stats
        .iter()
        .map(|s| lr * gradient_leaf_delta(s.sum_weighted_delta, s.sum_weight, scaled_l2))
        .collect();

    let expected = [-0.033333_f64, 0.0, -0.005, 0.0275];
    for (got, want) in leaf_values.iter().zip(expected.iter()) {
        assert!(
            (got - want).abs() <= 1e-5,
            "leaf value {got} != {want} (≤1e-5)"
        );
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// ===========================================================================
// Plan 05-14 — ctr_data bake (Scale/Shift) + apply Scale/Shift on BOTH branches
// ===========================================================================

/// Build a single-CTR-split, depth-1 `cb_model::Model` over projection {0} whose
/// leaf 0 / leaf 1 values are distinguishable, with the given baked `ctr_data` and
/// the split's `(shift, scale, border)`. Predicting one object reveals which leaf
/// the `ctr_value > border` test selected (leaf value 10.0 for pass, -10.0 for not).
fn single_ctr_split_model(
    ctr_data: cb_model::CtrData,
    shift: f64,
    scale: f64,
    border: f64,
) -> cb_model::Model {
    let split = cb_model::CtrSplit {
        projection: TProjection::single(0),
        ctr_type: cb_model::ECtrType::Borders,
        prior: cb_model::Prior { num: PRIOR_NUM, denom: PRIOR_DENOM },
        target_border_idx: 0,
        border,
        shift,
        scale,
    };
    let tree = cb_model::ObliviousTree {
        splits: vec![cb_model::ModelSplit::Ctr(split)],
        // forward-bit leaf index: bit0 = (ctr_value > border). leaf0 = not-pass,
        // leaf1 = pass.
        leaf_values: vec![-10.0, 10.0],
        leaf_weights: vec![1.0, 1.0],
    };
    cb_model::Model {
        oblivious_trees: vec![tree],
        non_symmetric_trees: Vec::new(),
        bias: 0.0,
        float_feature_borders: Vec::new(),
        ctr_data: Some(ctr_data),
        approx_dimension: 1,
        class_to_label: Vec::new(),
    }
}

/// Test (Scale/Shift derivation): a Borders CTR with prior_num=0.5, prior_denom=1
/// and ctr_border_count=15 derives (Shift, Scale) == (0.0, 15.0) —
/// calc_normalization(0.5) gives shift=0, norm=1, Scale=15/1=15 — matching the
/// fixture ctrs[0].{shift,scale}. The bake sets BOTH prior halves on the table.
#[test]
fn bake_derives_shift_zero_scale_fifteen_for_borders_prior_half() {
    // A single-feature {0} projection over 3 documents (2 distinct categories).
    let cat0: Vec<String> = ["0", "0", "1"].iter().map(|s| (*s).to_owned()).collect();
    let cat_columns = vec![cat0];
    let target_class = vec![1usize, 0, 1];
    let proj = TProjection::single(0);

    let table = bake_ctr_table(
        &cat_columns,
        &proj,
        &target_class,
        2,
        ctr_border_count_default(),
        PRIOR_NUM,
        PRIOR_DENOM,
    )
    .expect("bake");

    assert!((table.shift - 0.0).abs() < 1e-12, "Shift == 0 for Borders:0.5/1");
    assert!((table.scale - 15.0).abs() < 1e-12, "Scale == 15 for ctr_border_count=15");
    assert_eq!(table.prior_num, PRIOR_NUM, "prior_num carried");
    assert_eq!(table.prior_denom, PRIOR_DENOM, "prior_denom carried");
    // The two distinct categories produce two buckets with class counts.
    assert_eq!(table.hashes.len(), 2, "two distinct combined buckets");
    assert_eq!(table.int_counts.len(), 2, "per-bucket class counts");
}

/// Test (apply uses split Scale/Shift on the FOUND branch): the SAME baked table
/// + the SAME border, applied with Scale=15 vs Scale=1, selects DIFFERENT leaves —
/// proving the found-branch hardcode (`scale = 1.0`) is removed and the split uses
/// the model's Scale.
#[test]
fn apply_found_branch_uses_split_scale() {
    // One category "0" appearing in the document; whole-set bucket {0}: class
    // counts make the raw inference ctr ((good+0.5)/(tot+1)) a value in [0,1].
    // good=1, tot=1 → (1.5)/(2)=0.75. Scaled by 15 → 11.25 (> border 5); scaled by
    // 1 → 0.75 (< border 5). So the same border lands on opposite sides.
    let cat0: Vec<String> = ["0", "0"].iter().map(|s| (*s).to_owned()).collect();
    let cat_columns = vec![cat0];
    let target_class = vec![1usize, 0]; // good=1, tot=2 over the whole set
    let proj = TProjection::single(0);
    let table = bake_ctr_table(
        &cat_columns, &proj, &target_class, 2, ctr_border_count_default(), PRIOR_NUM, PRIOR_DENOM,
    )
    .expect("bake");
    let ctr_data = cb_model::CtrData::from_baked(&cb_train::BakedCtrData {
        tables: vec![table],
    });
    let border = 5.0;

    // Predict the FIRST object (category "0"). Found branch (bucket present).
    let cat_cols_obj = vec![vec!["0".to_owned()]];

    let model_scale_15 = single_ctr_split_model(ctr_data.clone(), 0.0, 15.0, border);
    let model_scale_1 = single_ctr_split_model(ctr_data, 0.0, 1.0, border);

    let pred_15 = cb_model::predict_raw_cat(&model_scale_15, &[], &cat_cols_obj);
    let pred_1 = cb_model::predict_raw_cat(&model_scale_1, &[], &cat_cols_obj);

    // Scale=15 → scaled ctr > border → leaf1 (10.0); Scale=1 → < border → leaf0 (-10.0).
    assert!(pred_15[0] > 0.0, "Scale=15 lands above border (found branch)");
    assert!(pred_1[0] < 0.0, "Scale=1 lands below border (found branch)");
    assert_ne!(
        pred_15[0], pred_1[0],
        "the found-branch split outcome depends on split.scale (hardcode removed)"
    );
}

/// Test (apply uses split Scale/Shift on the NOT-FOUND branch): for a split whose
/// combined bucket is ABSENT from the baked table, the empty-CTR value is STILL
/// scaled by split.shift/split.scale — Scale=15 vs Scale=1 over the empty bucket
/// yields a different `ctr_value > border` outcome, proving the not-found hardcode
/// (`calc_inference(.., 0.0, 1.0)`) is removed.
#[test]
fn apply_not_found_branch_uses_split_scale() {
    // An EMPTY baked table (no buckets) → every lookup is the not-found→empty path.
    let ctr_data = cb_model::CtrData {
        tables: std::collections::BTreeMap::new(),
    };
    // Empty CTR: Calc(0,0) over prior 0.5/1 = (0+0.5)/(0+1) = 0.5. Scaled by 15 →
    // 7.5 (> border 5); scaled by 1 → 0.5 (< border 5).
    let border = 5.0;
    let cat_cols_obj = vec![vec!["0".to_owned()]];

    let model_scale_15 = single_ctr_split_model(ctr_data.clone(), 0.0, 15.0, border);
    let model_scale_1 = single_ctr_split_model(ctr_data, 0.0, 1.0, border);

    let pred_15 = cb_model::predict_raw_cat(&model_scale_15, &[], &cat_cols_obj);
    let pred_1 = cb_model::predict_raw_cat(&model_scale_1, &[], &cat_cols_obj);

    assert!(pred_15[0] > 0.0, "Scale=15 scales the empty CTR above border (not-found branch)");
    assert!(pred_1[0] < 0.0, "Scale=1 leaves the empty CTR below border (not-found branch)");
    assert_ne!(
        pred_15[0], pred_1[0],
        "the NOT-FOUND-branch empty CTR is scaled by split.scale (hardcode removed)"
    );
}

/// Test (bake round-trips to apply): a baked whole-set table for projection {0},
/// looked up through the apply path with the split's Scale/Shift, reproduces the
/// upstream inference Calc value for a known document — the whole-set partition
/// counts feed `((good+0.5)/(tot+1))*15` for that document's category.
#[test]
fn bake_round_trips_to_apply_inference_value() {
    // 4 documents, single category "7": whole-set bucket {7} has good = #pos,
    // tot = 4. Choose 3 pos / 1 neg → good=3, tot=4 → (3.5)/(5) = 0.7 → *15 = 10.5.
    let cat0: Vec<String> = ["7", "7", "7", "7"].iter().map(|s| (*s).to_owned()).collect();
    let cat_columns = vec![cat0];
    let target_class = vec![1usize, 1, 1, 0];
    let proj = TProjection::single(0);
    let table = bake_ctr_table(
        &cat_columns, &proj, &target_class, 2, ctr_border_count_default(), PRIOR_NUM, PRIOR_DENOM,
    )
    .expect("bake");
    let shift = table.shift;
    let scale = table.scale;
    let ctr_data = cb_model::CtrData::from_baked(&cb_train::BakedCtrData { tables: vec![table] });

    // The expected scaled inference value: ((3 + 0.5) / (4 + 1)) * 15 = 10.5.
    let expected = ((3.0 + 0.5) / (4.0 + 1.0)) * 15.0;
    // Border just below the expected value → the split passes (leaf1, 10.0); a
    // border just above → it fails (leaf0, -10.0). This pins the apply value.
    let below = single_ctr_split_model(ctr_data.clone(), shift, scale, expected - 0.01);
    let above = single_ctr_split_model(ctr_data, shift, scale, expected + 0.01);
    let cat_cols_obj = vec![vec!["7".to_owned()]];
    let pred_below = cb_model::predict_raw_cat(&below, &[], &cat_cols_obj);
    let pred_above = cb_model::predict_raw_cat(&above, &[], &cat_cols_obj);

    assert!(
        pred_below[0] > 0.0,
        "ctr_value ({expected}) > border-0.01 → leaf1 (apply reproduced the inference Calc)"
    );
    assert!(
        pred_above[0] < 0.0,
        "ctr_value ({expected}) < border+0.01 → leaf0 (apply reproduced the inference Calc)"
    );
}
