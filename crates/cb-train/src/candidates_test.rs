//! Unit tests for one-hot vs CTR encoding-path selection (ORD-04, RESEARCH
//! Pitfall 3). The threshold is the parity landmine: one-hot is INCLUSIVE at
//! `cardinality == one_hot_max_size` and EXCLUSIVE above it (CTR), with a
//! constant column (`cardinality <= 1`) skipped — reproducing the upstream
//! `AddOneHotFeatures` skip predicate `(count > max) || (count <= 1)`
//! (`greedy_tensor_search.cpp:171-197`). Test names carry the `one_hot_threshold`
//! substring so the plan's `cargo test -p cb-train one_hot_threshold` selects
//! all of them.

use crate::candidates::{
    learn_set_cardinality, one_hot_max_size_default, route_categorical, route_column, EncodingPath,
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
