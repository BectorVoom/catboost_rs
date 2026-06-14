//! Unit tests for one-hot vs CTR encoding-path selection (ORD-04, RESEARCH
//! Pitfall 3). The threshold is the parity landmine: one-hot is INCLUSIVE at
//! `cardinality == one_hot_max_size` and EXCLUSIVE above it (CTR), with a
//! constant column (`cardinality <= 1`) skipped — reproducing the upstream
//! `AddOneHotFeatures` skip predicate `(count > max) || (count <= 1)`
//! (`greedy_tensor_search.cpp:171-197`). Test names carry the `one_hot_threshold`
//! substring so the plan's `cargo test -p cb-train one_hot_threshold` selects
//! all of them.

use crate::candidates::{
    learn_set_cardinality, one_hot_max_size_default, route_categorical, route_column,
    tensor_ctr_candidates, EncodingPath,
};

#[test]
fn one_hot_threshold_at_max_is_inclusive_one_hot() {
    // count == one_hot_max_size (=3) → ONE-HOT (inclusive boundary,
    // greedy_tensor_search.cpp:182 skip-if-`>`).
    assert_eq!(route_categorical(3, 3), EncodingPath::OneHot);
}

#[test]
fn one_hot_threshold_above_max_is_ctr() {
    // count == one_hot_max_size + 1 (=4) → CTR path (exclusive boundary,
    // deferred to later waves).
    assert_eq!(route_categorical(4, 3), EncodingPath::Ctr);
}

#[test]
fn one_hot_threshold_just_above_one_is_one_hot() {
    // count == 2 with one_hot_max_size == 3 → ONE-HOT (1 < count <= max).
    assert_eq!(route_categorical(2, 3), EncodingPath::OneHot);
}

#[test]
fn one_hot_threshold_at_one_is_skip() {
    // count == 1 → SKIP (constant column, skip-if-`<=1`).
    assert_eq!(route_categorical(1, 3), EncodingPath::Skip);
}

#[test]
fn one_hot_threshold_at_zero_is_skip() {
    // count == 0 (empty column) → SKIP (skip-if-`<=1`).
    assert_eq!(route_categorical(0, 3), EncodingPath::Skip);
}

#[test]
fn one_hot_threshold_default_max_two_boundary() {
    // Against the upstream default one_hot_max_size == 2
    // (cat_feature_options.cpp:231-232): count==2 inclusive one-hot, count==3 CTR.
    let max = one_hot_max_size_default();
    assert_eq!(max, 2);
    assert_eq!(route_categorical(2, max), EncodingPath::OneHot);
    assert_eq!(route_categorical(3, max), EncodingPath::Ctr);
    assert_eq!(route_categorical(1, max), EncodingPath::Skip);
}

#[test]
fn one_hot_threshold_cardinality_is_learn_set_unique_count() {
    // Cardinality is the LEARN-SET distinct-value count via calc_cat_feature_hash
    // first-seen bins. Three distinct values with repeats → cardinality 3.
    let column = ["a", "b", "a", "c", "b", "a"];
    let card = learn_set_cardinality(&column).expect("cardinality must count");
    assert_eq!(card, 3);
    // With one_hot_max_size == 3 a cardinality-3 column is one-hot.
    assert_eq!(route_categorical(card, 3), EncodingPath::OneHot);
}

#[test]
fn one_hot_threshold_route_column_end_to_end() {
    // route_column counts then routes. cat0-like column of cardinality EXACTLY
    // one_hot_max_size (=3) → ONE-HOT; cat1-like of one_hot_max_size+1 (=4) → CTR
    // (the one_hot_cat fixture's documented cat0/cat1 split, RESEARCH Pitfall 3).
    let cat0 = ["x", "y", "z", "x", "y"]; // cardinality 3
    let cat1 = ["p", "q", "r", "s", "p"]; // cardinality 4
    assert_eq!(
        route_column(&cat0, 3).expect("route cat0"),
        EncodingPath::OneHot
    );
    assert_eq!(
        route_column(&cat1, 3).expect("route cat1"),
        EncodingPath::Ctr
    );
}

#[test]
fn one_hot_threshold_single_distinct_value_skipped() {
    // A column with one distinct value (cardinality 1) is SKIP regardless of max.
    let column = ["only", "only", "only"];
    let card = learn_set_cardinality(&column).expect("cardinality must count");
    assert_eq!(card, 1);
    assert_eq!(route_categorical(card, 5), EncodingPath::Skip);
}

#[test]
fn projection_tensor_ctr_candidates_two_eligible_features_complexity_two() {
    // The tensor_ctr fixture surface: two cat features above one_hot_max_size=1
    // (cardinalities 5 and 4), max_ctr_complexity=2 → 2 SimpleCtrs + 1
    // CombinationCtr (the 2-feature tensor). Both eligible (cardinality > 1).
    let cardinalities = [5u32, 4];
    let cands = tensor_ctr_candidates(&cardinalities, 1, 2);
    assert_eq!(cands.len(), 3, "2 simple + 1 combination");
    let simple = cands.iter().filter(|c| c.is_simple).count();
    let combos = cands.iter().filter(|c| !c.is_simple).count();
    assert_eq!(simple, 2, "one SimpleCtr per eligible feature");
    assert_eq!(combos, 1, "one 2-feature CombinationCtr");
    // The combination projection spans both eligible features (positions 0,1).
    let combo = cands.iter().find(|c| !c.is_simple).expect("a combination");
    assert_eq!(combo.projection.cat_features(), &[0, 1]);
}

#[test]
fn projection_tensor_ctr_candidates_complexity_one_only_simple() {
    // max_ctr_complexity=1 → only SimpleCtrs, never a tensor.
    let cardinalities = [5u32, 4];
    let cands = tensor_ctr_candidates(&cardinalities, 1, 1);
    assert_eq!(cands.len(), 2);
    assert!(cands.iter().all(|c| c.is_simple), "complexity 1 → no tensors");
}

#[test]
fn projection_tensor_ctr_candidates_exclude_one_hot_features() {
    // A low-cardinality (one-hot) feature is NOT CTR-eligible: cardinality 2 with
    // one_hot_max_size=2 routes one-hot, so only the cardinality-5 feature is
    // eligible → a single SimpleCtr, no combination.
    let cardinalities = [2u32, 5];
    let cands = tensor_ctr_candidates(&cardinalities, 2, 2);
    assert_eq!(cands.len(), 1, "only the high-cardinality feature is CTR-eligible");
    assert!(cands[0].is_simple);
}

#[test]
fn projection_tensor_ctr_candidates_no_eligible_features_empty() {
    // All features one-hot / skip → no CTR candidates.
    let cardinalities = [2u32, 1, 2];
    assert!(tensor_ctr_candidates(&cardinalities, 2, 4).is_empty());
}
