//! BUG-BTMV / SPEC-BTMV-02 — the baked BinarizedTargetMeanValue table must match
//! catboost 1.2.10's own committed `ctr_data`, bucket for bucket, BY HASH.
//!
//! B01 proves INTERNAL correctness against a hand-derivation. This proves
//! UPSTREAM agreement — a systematically wrong-but-self-consistent derivation
//! would satisfy B01 and fail here.
//!
//! The gate is at TABLE level, not prediction level, so a divisor error reports
//! a ratio rather than "predictions differ".
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]


use std::path::PathBuf;

use cb_backend::CpuBackend;
use cb_compute::{LeafMethod, Loss};
use cb_data::stringify_int_category;
use cb_model::Model as CbModel;
use cb_oracle::{compare_stage, load_f64_vec, load_model_json, Stage};
use cb_train::{
    materialize_ctr_feature, train_cat, BoostParams, EBootstrapType, ECtrType,
    EOverfittingDetectorType, TProjection,
};
use ndarray::Array2;
use ndarray_npy::read_npy;

const SCENARIO: &str = "ctr_btmv_simple";

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join(rel)
}

/// The two categorical columns as SoA `Vec<String>`, stringified via
/// `stringify_int_category` — the A4 plain-integer form upstream's Pool hashed
/// when the fixture was generated.
fn load_cat_columns() -> Vec<Vec<String>> {
    let x: Array2<i32> = read_npy(fixture(&format!("{SCENARIO}/X_cat.npy")))
        .unwrap_or_else(|e| panic!("{SCENARIO}/X_cat.npy must load as int32 [N,2]: {e:?}"));
    (0..x.ncols())
        .map(|fi| {
            x.column(fi)
                .iter()
                .map(|&code| stringify_int_category(i64::from(code)))
                .collect()
        })
        .collect()
}

/// The fixture's pinned config, EVERY field explicit (Pitfall-6 discipline: a
/// changed builder default must not silently alter what this gate exercises).
fn counter_params() -> BoostParams {
    BoostParams {
        loss: Loss::Logloss,
        iterations: 5,
        depth: 2,
        learning_rate: 0.1,
        l2_leaf_reg: 3.0,
        random_strength: 0.0,
        boost_from_average: false,
        leaf_method: LeafMethod::Gradient,
        bootstrap_type: EBootstrapType::No,
        subsample: 1.0,
        bagging_temperature: 0.0,
        random_seed: 0,
        od_type: EOverfittingDetectorType::None,
        od_pval: 0.0,
        od_wait: 0,
        use_best_model: false,
        eval_metric: None,
        auto_learning_rate: false,
        one_hot_max_size: 1,
        permutation_count: 1,
        fold_len_multiplier: 2.0,
        simple_ctr: ECtrType::BinarizedTargetMeanValue,
        simple_ctr_priors: vec![0.5],
        counter_calc_method: cb_train::counter_calc_method_default(),
        boosting_type: cb_train::boosting_type_default(),
        max_ctr_complexity: 1,
        combinations_ctr: cb_train::combinations_ctr_default(),
        combinations_ctr_priors: vec![0.5],
        score_function: cb_train::score_function_default(),
        has_time: false,
        feature_weights: cb_train::feature_weights_default(),
        first_feature_use_penalties: cb_train::first_feature_use_penalties_default(),
        per_object_feature_penalties: cb_train::per_object_feature_penalties_default(),
        penalties_coefficient: cb_train::penalties_coefficient_default(),
        monotone_constraints: cb_train::monotone_constraints_default(),
        grow_policy: cb_train::grow_policy_default(),
        max_leaves: cb_train::max_leaves_default(),
        min_data_in_leaf: cb_train::min_data_in_leaf_default(),
    }
}

/// Train the fixture through production `train_cat`.
fn fit() -> (
    cb_train::Model,
    cb_train::BakedCtrData,
    Vec<Vec<String>>,
    Vec<Vec<f64>>,
) {
    let cat_cols = load_cat_columns();
    let model_json = load_model_json(&fixture(&format!("{SCENARIO}/model.json")))
        .unwrap_or_else(|e| panic!("{SCENARIO}/model.json must load: {e:?}"));
    let borders = model_json.float_feature_borders();
    let target = load_f64_vec(&fixture(&format!("{SCENARIO}/y.npy"))).unwrap();

    let (trained, baked) = train_cat(
        &CpuBackend,
        &[], // categorical-only fixture: zero float columns
        &borders,
        &cat_cols,
        &target,
        &[],
        &counter_params(),
        None,
    )
    .unwrap_or_else(|e| panic!("Counter-CTR training failed: {e:?}"));

    (trained, baked, cat_cols, borders)
}

/// One real upstream bucket, after the empty-slot sentinel is dropped.
struct UpstreamBucket {
    hash: u64,
    sum: f64,
    count: i64,
}

/// Upstream's committed BTMV `ctr_data`, read from the RAW `hash_map` field.
///
/// `cb_oracle`'s `bucket_counts()` neither strips the sentinel nor exposes the
/// hashes, so the raw field is read directly.
fn upstream_btmv_buckets() -> Vec<UpstreamBucket> {
    let mj = load_model_json(&fixture(&format!("{SCENARIO}/model.json")))
        .unwrap_or_else(|e| panic!("{SCENARIO}/model.json must load: {e:?}"));
    let (_key, table) = mj
        .ctr_data
        .iter()
        .find(|(k, _)| k.contains("\"type\":\"BinarizedTargetMeanValue\""))
        .expect("the committed model.json must carry a BinarizedTargetMeanValue ctr_data entry");

    assert_eq!(table.hash_stride, 3, "a mean CTR table is (hash, Sum, Count)");

    table
        .hash_map
        .chunks_exact(3)
        .filter_map(|chunk| {
            // Upstream emits the hash as a STRING.
            let hash: u64 = chunk[0]
                .as_str()
                .expect("hash must be a string")
                .parse()
                .expect("hash must parse as u64");
            // SKIP the empty-slot sentinel: upstream's dense hash map leads with a
            // u64::MAX slot whose payload is stale memory (here 3, 7 — duplicating
            // a real bucket). Including it inflates the count total from 30 to 37.
            if hash == u64::MAX {
                return None;
            }
            Some(UpstreamBucket {
                hash,
                sum: chunk[1].as_f64().expect("Sum must be a number"),
                count: chunk[2].as_i64().expect("Count must be an integer"),
            })
        })
        .collect()
}

#[test]
fn baked_btmv_table_matches_upstream_ctr_data_bucket_for_bucket() {
    let (_trained, baked, _cat_cols, _borders) = fit();

    let ours: Vec<_> = baked
        .tables
        .iter()
        .filter(|t| t.ctr_type == ECtrType::BinarizedTargetMeanValue.as_i8())
        .collect();
    assert_eq!(
        ours.len(),
        1,
        "expected exactly ONE baked BTMV table; got {}. More than one means the \
         trainer chose a second BTMV projection — a STRUCTURAL divergence from \
         upstream (which committed exactly one, on cat feature 1). STOP AND REPORT.",
        ours.len()
    );
    let ours = ours[0];
    assert_eq!(
        ours.projection.cat_features(),
        &[1usize],
        "upstream's committed CTR is on cat_feature_index 1; ours is on {:?}. A \
         different chosen projection is a STRUCTURAL parity finding, NOT a test to \
         weaken. STOP AND REPORT.",
        ours.projection.cat_features()
    );

    let upstream = upstream_btmv_buckets();

    // Structural guards BEFORE comparing values.
    assert_eq!(
        upstream.len(),
        5,
        "expected 5 real buckets after dropping the u64::MAX sentinel \
         (cat1 cardinality is 5)"
    );
    assert_eq!(
        upstream.iter().map(|b| b.count).sum::<i64>(),
        30,
        "the real buckets' Counts must sum to n_rows = 30; a different total means \
         the sentinel filter is wrong"
    );
    assert_eq!(
        ours.mean.len(),
        ours.hashes.len(),
        "one (Sum, Count) per bucket"
    );
    assert_eq!(ours.hashes.len(), upstream.len(), "bucket-count mismatch");

    // ANTI-VACUITY: a degenerate corpus would satisfy the comparison trivially.
    assert!(
        upstream.iter().any(|b| b.sum != b.count as f64 && b.sum != 0.0),
        "no upstream bucket has Sum != Count and Sum != 0 — a degenerate corpus \
         would satisfy this comparison trivially"
    );

    // BY HASH, not by index: our first-seen bucket order need not equal upstream's
    // map order. Exact `==` on the Sums is legitimate — with the correct divisor
    // every upstream Sum is a small integer, exactly representable in f32 and f64.
    // Do NOT relax this to an epsilon compare.
    for (i, &h) in ours.hashes.iter().enumerate() {
        let u = upstream.iter().find(|b| b.hash == h).unwrap_or_else(|| {
            panic!(
                "baked hash {h} is absent from upstream's committed ctr_data {:?}. \
                 A hash-set mismatch is a STRUCTURAL finding (different projection \
                 or different category hashing), NOT a divisor bug. STOP AND REPORT.",
                upstream.iter().map(|b| b.hash).collect::<Vec<_>>()
            )
        });
        assert_eq!(
            ours.mean[i].1, u.count,
            "bucket {h}: Count {} != upstream {}",
            ours.mean[i].1, u.count
        );
        assert_eq!(
            f64::from(ours.mean[i].0),
            u.sum,
            "bucket {h}: Sum {} != upstream {}. Ratio {:.6} — a ratio of exactly \
             0.5 means the bake divided by `classes` (2) instead of `classes - 1` \
             (1); see online_ctr.cpp:914.",
            ours.mean[i].0,
            u.sum,
            f64::from(ours.mean[i].0) / u.sum
        );
    }
}

#[test]
fn upstream_btmv_sums_are_integers_which_pins_the_divisor_at_one() {
    // Fixture-side evidence for the divisor, INDEPENDENT of our implementation.
    let upstream = upstream_btmv_buckets();
    assert!(!upstream.is_empty(), "no upstream buckets — gate would be vacuous");
    for b in &upstream {
        assert_eq!(
            b.sum,
            b.sum.round(),
            "for binclf targetClass ∈ {{0,1}}, so Sum = Σ targetClass / \
             targetBorderCount is an integer IFF targetBorderCount == 1. \
             Upstream's committed Sum {} is not an integer, which would contradict \
             the whole-set divisor being `targetClassesCount - 1` \
             (online_ctr.cpp:914).",
            b.sum
        );
    }
}
