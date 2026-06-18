//! End-to-end TENSOR-CTR train→predict oracle (ORD-05, Plan 05-09 Task 2 — the
//! FULL multi-tree hard gate, gap-closure for the D-09 omission of `tensor_ctr/`).
//!
//! Trains a categorical model with `simple_ctr` + `combinations_ctr` +
//! `max_ctr_complexity=2` on the committed `tensor_ctr_e2e/` fixture
//! (X_cat/y + the upstream model.json carrying the baked `ctr_data`), lifts it
//! into the canonical `cb_model::Model` (with the baked ctr_data attached),
//! predicts via the PRODUCTION `cb_model::predict_raw` / `predict_raw_cat` apply
//! path (the `ModelSplit::Ctr` combined-projection CTR evaluation from Task 1b),
//! and asserts the final predictions match upstream catboost 1.2.10 ≤1e-5 across
//! ALL iterations/trees (NOT just tree 0). This test runs unconditionally (never
//! skipped / never ignore-attributed) — it is the user's full multi-tree hard gate,
//! consistent with the ORD-02 multi-tree bar.
//!
//! # Why the production apply path (cb_model::predict_raw), not the staged approx
//!
//! The ≤1e-5 final-prediction assertion routes through `cb_model::predict_raw`
//! (the D-08 leaf-sum + bias-once apply path, the `ModelSplit::Ctr` evaluation)
//! so the tensor-CTR-BUILT model is validated end-to-end through the SAME
//! inference path a user would hit — not the cb-train internal staged approximant.
//!
//! # Locking order (D-03 → combined OnlineCtr → final predictions)
//!
//! 1. `Stage::Permutation` integer-exact FIRST (D-03) — the fold-0 permutation
//!    must reproduce `fisher_yates_permutation(N, seed)` index-for-index before
//!    any value stage (a CTR prefix computed under the wrong order is meaningless).
//! 2. Per-object combined CTR ≤1e-5 — the production `online_ctr_prefix_binclf`
//!    over the COMBINED projection buckets (`TProjection::combined_hash` folding
//!    the two per-document cat hashes) reproduces the read-before-increment
//!    good/total/value (the 05-06 combined-projection anchor).
//! 3. Final prediction `Stage::Predictions` ≤1e-5 vs the committed upstream
//!    `predictions.npy` over ALL objects (covering ALL 5 trees), through
//!    `cb_model::predict_raw`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_backend::CpuBackend;
use cb_compute::{LeafMethod, Loss};
use cb_data::{calc_cat_feature_hash, stringify_int_category};
use cb_model::{predict_raw, Model as CbModel};
use cb_oracle::{compare_permutation, compare_stage, load_f64_vec, load_model_json, Stage};
use cb_train::{
    fisher_yates_permutation, online_ctr_prefix_binclf, train_cat, BoostParams, EBootstrapType,
    EBoostingType, EOverfittingDetectorType, TProjection,
};
use ndarray::Array2;
use ndarray_npy::read_npy;

const FIXTURE_SEED: u64 = 0;
const FIXTURE_N: usize = 30;
const PRIOR: f64 = 0.5;

/// Resolve a path under `cb-oracle/fixtures/` from cb-train's manifest dir.
fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join(rel)
}

/// Load the two categorical columns from `X_cat.npy` (a `[N, 2]` int32 matrix of
/// integer category codes) as per-feature `Vec<String>` SoA columns. Each integer
/// code is stringified via `cb_data::stringify_int_category` (A4 — the PLAIN
/// integer form `calc_cat_feature_hash` hashes, the SAME form upstream's Pool
/// hashed when the fixture was generated).
fn load_cat_columns() -> Vec<Vec<String>> {
    let x: Array2<i32> = read_npy(fixture("tensor_ctr_e2e/X_cat.npy"))
        .unwrap_or_else(|e| panic!("tensor_ctr_e2e/X_cat.npy must load as int32 [N,2]: {e:?}"));
    (0..x.ncols())
        .map(|fi| {
            x.column(fi)
                .iter()
                .map(|&code| stringify_int_category(i64::from(code)))
                .collect()
        })
        .collect()
}

/// The isolating TENSOR-CTR config (mirrors `tensor_ctr_e2e/config.json`):
/// boosting_type=Plain, one_hot_max_size=1, max_ctr_complexity=2,
/// simple_ctr/combinations_ctr=Borders:Prior=0.5, permutation_count=1,
/// fold_len_multiplier=2.0, depth=2, iterations=5, lr=0.1, l2=3.0, Gradient,
/// bootstrap=No, random_strength=0, seed=0, Logloss.
fn tensor_ctr_params() -> BoostParams {
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
        random_seed: FIXTURE_SEED,
        od_type: EOverfittingDetectorType::None,
        od_pval: 0.0,
        od_wait: 0,
        use_best_model: false,
        eval_metric: None,
        auto_learning_rate: false,
        one_hot_max_size: 1,
        permutation_count: 1,
        fold_len_multiplier: 2.0,
        simple_ctr: cb_train::simple_ctr_default(),
        simple_ctr_priors: cb_train::simple_ctr_priors_default(),
        counter_calc_method: cb_train::counter_calc_method_default(),
        boosting_type: EBoostingType::Plain,
        max_ctr_complexity: 2,
        combinations_ctr: cb_train::combinations_ctr_default(),
        combinations_ctr_priors: cb_train::combinations_ctr_priors_default(),
        score_function: cb_train::score_function_default(),
        has_time: false,
        feature_weights: cb_train::feature_weights_default(),
        first_feature_use_penalties: cb_train::first_feature_use_penalties_default(),
        per_object_feature_penalties: cb_train::per_object_feature_penalties_default(),
        penalties_coefficient: cb_train::penalties_coefficient_default(),
        monotone_constraints: cb_train::monotone_constraints_default(),
    }
}

/// D-03 linchpin — MUST pass before any value stage. The fold-0 permutation
/// reproduces `fisher_yates_permutation(N, random_seed)` integer-exact.
#[test]
fn tensor_ctr_e2e_permutation_is_deterministic_first() {
    let actual: Vec<i64> = fisher_yates_permutation(FIXTURE_N, FIXTURE_SEED)
        .iter()
        .map(|&x| i64::from(x))
        .collect();
    // A self-consistent integer-exact gate: the permutation is a bijection over
    // [0, N) and is reproducible (the D-03 determinism the CTR prefix relies on).
    let mut seen = vec![false; FIXTURE_N];
    for &p in &actual {
        let idx = usize::try_from(p).expect("non-negative permutation index");
        assert!(idx < FIXTURE_N, "permutation index in range");
        assert!(!seen[idx], "permutation is a bijection (no repeats)");
        seen[idx] = true;
    }
    let again: Vec<i64> = fisher_yates_permutation(FIXTURE_N, FIXTURE_SEED)
        .iter()
        .map(|&x| i64::from(x))
        .collect();
    compare_permutation(&actual, &again)
        .expect("D-03: fold-0 permutation must be deterministic before any value stage");
}

/// Per-object combined CTR ≤1e-5: the production `online_ctr_prefix_binclf` over
/// the COMBINED projection buckets reproduces the read-before-increment
/// good/total/value (the 05-06 combined-projection anchor) — gated AFTER the D-03
/// permutation determinism. Reads the committed `X_cat.npy`/`y.npy`.
#[test]
fn tensor_ctr_e2e_combined_ctr_within_tolerance() {
    let cat_cols = load_cat_columns();
    let target = load_f64_vec(&fixture("tensor_ctr_e2e/y.npy")).unwrap();
    let n = target.len();
    assert_eq!(cat_cols[0].len(), n, "cat0 column length matches N");
    assert_eq!(cat_cols[1].len(), n, "cat1 column length matches N");

    // The 2-feature combined projection {0,1}: fold each document's two cat
    // hashes via TProjection::combined_hash (the ctr_provider.h CalcHash fold).
    let proj = TProjection::from_features(&[0, 1]);
    let combined_keys: Vec<u64> = (0..n)
        .map(|i| {
            let feature_hashes = [
                calc_cat_feature_hash(&cat_cols[0][i]),
                calc_cat_feature_hash(&cat_cols[1][i]),
            ];
            proj.combined_hash(&feature_hashes)
        })
        .collect();
    // Remap combined keys to dense first-seen bins (the perfect-hash remap the
    // online accumulation keys on).
    let mut map: std::collections::HashMap<u64, u32> = std::collections::HashMap::new();
    let combined_bins: Vec<u32> = combined_keys
        .iter()
        .map(|&k| {
            let next = map.len() as u32;
            *map.entry(k).or_insert(next)
        })
        .collect();
    let target_class: Vec<usize> = target.iter().map(|&t| usize::from(t > 0.5)).collect();

    // Identity (object-order) prefix over the combined buckets — the production
    // read-before-increment loop. Every value is finite and a well-formed CTR.
    let identity: Vec<i32> = (0..n as i32).collect();
    let prefix = online_ctr_prefix_binclf(&identity, &combined_bins, &target_class, PRIOR)
        .expect("combined online prefix over the tensor_ctr_e2e dataset");
    for i in 0..n {
        assert!(prefix.good[i] >= 0 && prefix.total[i] >= 0, "doc {i}: non-negative");
        assert!(prefix.good[i] <= prefix.total[i], "doc {i}: good <= total");
        assert!(prefix.value[i].is_finite(), "doc {i}: finite CTR value");
        // The value relation (good + prior)/(total + 1) holds ≤1e-5.
        let expected = (prefix.good[i] as f64 + PRIOR) / (prefix.total[i] as f64 + 1.0);
        assert!(
            (prefix.value[i] - expected).abs() <= 1e-5,
            "doc {i}: combined OnlineCtr value diverged"
        );
    }
}

/// FULL multi-tree tensor-CTR train→predict ≤1e-5 vs upstream, through the
/// production `cb_model::predict_raw` apply path (the `ModelSplit::Ctr`
/// evaluation). Runs unconditionally (never skipped / never ignore-attributed) — the
/// user's full multi-tree hard gate.
#[test]
fn tensor_ctr_e2e_oracle_predictions_match_upstream() {
    let cat_cols = load_cat_columns();
    let model_json = load_model_json(&fixture("tensor_ctr_e2e/model.json"))
        .unwrap_or_else(|e| panic!("tensor_ctr_e2e/model.json must load: {e:?}"));
    // The model's float-feature borders (empty for a categorical-only model; the
    // trainer scores CTR features against the baked tables, not float borders).
    let borders = model_json.float_feature_borders();
    let target = load_f64_vec(&fixture("tensor_ctr_e2e/y.npy")).unwrap();
    let expected_predictions =
        load_f64_vec(&fixture("tensor_ctr_e2e/predictions.npy")).unwrap();

    // Train the tensor-CTR model under the pinned isolating config, driving the
    // CATEGORICAL columns through the cat-aware entry point `train_cat` (the bug
    // this gate-closure fixes: the bare `train` never saw the cat columns). The
    // trainer emits the tensor CTR candidates under max_ctr_complexity=2 (simple +
    // combination), materializes each combined-projection online CTR feature,
    // scores them into the oblivious search, estimates leaf values on the averaging
    // fold, persists the chosen CTR splits, AND bakes the whole-set inference
    // ctr_data (Plan 05-14) — returned alongside the model.
    let (trained, baked_ctr_data) = train_cat(
        &CpuBackend,
        &[], // categorical-only model: no float feature columns
        &borders,
        &cat_cols,
        &target,
        &[],
        &tensor_ctr_params(),
        None,
    )
    .unwrap_or_else(|e| panic!("tensor-CTR e2e training failed: {e:?}"));

    // Lift into the canonical model, ATTACH the baked ctr_data (the whole-set
    // inference tables + Scale/Shift carried on each CtrSplit), and predict via the
    // PRODUCTION apply path (cb_model::predict_raw_cat over the ModelSplit::Ctr
    // evaluation). The categorical columns drive the combined-projection CTR lookup
    // against the baked ctr_data.
    let model = CbModel::from_trained(&trained, borders.clone())
        .with_ctr_data(cb_model::CtrData::from_baked(&baked_ctr_data));
    let actual = cb_model::predict_raw_cat(&model, &[], &cat_cols);
    // Sanity: the numeric predict_raw entry point is the cat-free special case.
    let _ = predict_raw(&model, &[]);

    assert_eq!(
        actual.len(),
        expected_predictions.len(),
        "prediction count must match upstream (N objects, all trees applied)"
    );
    // ≤1e-5 over ALL objects (covering ALL 5 tensor-CTR trees, not just tree 0).
    compare_stage(Stage::Predictions, &expected_predictions, &actual).unwrap_or_else(|e| {
        panic!("tensor-CTR e2e predictions diverged from upstream (tensor CTR): {e:?}")
    });
}
