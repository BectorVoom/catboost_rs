//! T24 / FPP-21 — the ORDERED device fit end-to-end against upstream `catboost==1.2.10`.
//!
//! The device sibling of `ordered_boost_e2e_oracle_test` (which trains the SAME fixture on
//! `CpuBackend` and must be left untouched). Trains `ordered_boost_e2e/` through
//! `GpuBackend`, lifts into the canonical model and predicts via the production
//! `cb_model::predict_raw`, asserting ≤1e-5 against the frozen upstream predictions.
//!
//! # Why this fixture is reachable now (V-8 re-evaluated, as T24 requires)
//!
//! The prior phase ruled `ordered_boost_e2e/` device-ineligible *solely* because it pins
//! `boost_from_average: true`. FPP-02 (T09) removed that clause, and FPP-20 (T23) removed the
//! unconditional ordered decline in `GpuTrainSession::begin`. So no new fixture generation was
//! needed — the check T24 asks for came back positive.
//!
//! Its shape is also the exact worked example the segment math is documented against: `n = 30`
//! with `fold_len_multiplier = 2.0` gives boundaries `[1,2,4,8,16,30]`, i.e. the five segments
//! `[(1,2),(2,4),(4,8),(8,16),(16,30)]` (`cb-train/src/fold.rs`), scored as the five
//! permutation PREFIXES of length 2, 4, 8, 16, 30.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use cb_compute::{LeafMethod, Loss};
use cb_train::{
    BoostParams, EBootstrapType, EBoostingType, EOverfittingDetectorType,
};

/// The isolating ORDERED config — the SAME values `ordered_boost_e2e/config.json` pins and
/// `ordered_boost_e2e_oracle_test::ordered_params` uses, so the device and CPU arms differ in
/// BACKEND only.
fn ordered_params() -> BoostParams {
    BoostParams {
        loss: Loss::Rmse,
        iterations: 5,
        depth: 2,
        learning_rate: 0.1,
        l2_leaf_reg: 3.0,
        random_strength: 0.0,
        boost_from_average: true,
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
        one_hot_max_size: cb_train::one_hot_max_size_default(),
        permutation_count: 1,
        fold_len_multiplier: 2.0,
        simple_ctr: cb_train::simple_ctr_default(),
        simple_ctr_priors: cb_train::simple_ctr_priors_default(),
        counter_calc_method: cb_train::counter_calc_method_default(),
        boosting_type: EBoostingType::Ordered,
        max_ctr_complexity: cb_train::max_ctr_complexity_default(),
        combinations_ctr: cb_train::combinations_ctr_default(),
        combinations_ctr_priors: cb_train::combinations_ctr_priors_default(),
        score_function: cb_compute::EScoreFunction::L2,
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

#[cfg(any(feature = "rocm", feature = "cuda"))]
mod device {
    use std::path::PathBuf;

    use super::ordered_params;
    use cb_backend::GpuBackend;
    use cb_model::{predict_raw, Model as CbModel};
    use cb_oracle::{load_f64_vec, load_model_json};
    use cb_train::train;
    use ndarray::Array2;
    use ndarray_npy::read_npy;

    fn fixture(rel: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("cb-oracle")
            .join("fixtures")
            .join("ordered_boost_e2e")
            .join(rel)
    }

    pub fn run() {
        let x: Array2<f32> =
            read_npy(fixture("X.npy")).expect("ordered_boost_e2e/X.npy must load as f32 [N,F]");
        let columns: Vec<Vec<f32>> = (0..x.ncols()).map(|fi| x.column(fi).to_vec()).collect();
        let model_json = load_model_json(&fixture("model.json"))
            .unwrap_or_else(|e| panic!("ordered_boost_e2e/model.json must load: {e:?}"));
        let borders = model_json.float_feature_borders();
        let target = load_f64_vec(&fixture("y.npy")).unwrap();
        let expected = load_f64_vec(&fixture("predictions.npy")).unwrap();

        let params = ordered_params();
        assert!(
            matches!(params.boosting_type, cb_train::EBoostingType::Ordered),
            "this oracle is about ORDERED boosting; a Plain config would silently test the \
             already-covered path"
        );

        let trained = train(
            &GpuBackend::default(),
            &columns,
            &borders,
            &target,
            &[],
            &params,
            None,
        )
        .unwrap_or_else(|e| panic!("ordered device train failed: {e:?}"));

        // Arm routing: the oblivious grower produced one tree per iteration and neither
        // non-symmetric arm fired.
        assert_eq!(trained.oblivious_trees.len(), params.iterations);
        assert!(
            trained.non_symmetric_trees.is_empty() && trained.region_trees.is_empty(),
            "an ordered fit must ride the oblivious arm"
        );

        // ─── READ THIS BEFORE TRUSTING A GREEN RUN ──────────────────────────────────────
        // These assertions do NOT prove the DEVICE ran. An ordered fit that the gate declines
        // falls back to the CPU grower, which produces the same oblivious shape and passes the
        // ≤1e-5 bar below by construction (`ordered_boost_e2e_oracle_test` is exactly that
        // test). This oracle was in fact silently passing on the CPU fallback until the host
        // gate's `ordered_learning_perm.is_none()` clause was removed.
        //
        // `TrainedModel` exposes no activation flag, so there is nothing to assert here.
        // Activation is verified OUT OF BAND:
        //
        //     CB_GPU_PROF=1 cargo test -p cb-train --no-default-features --features rocm \
        //         --test device_ordered_fit_test -- --nocapture
        //
        // which must print one line per iteration:
        //
        //     CB_GPU_PROF tree ORDERED n=30 nf=2 bins=32 depth=2 segments=5
        //
        // (verified on gfx1151: 5 lines, `begin covered=true`, max |Δpred| = 3.735e-8).
        // `segments=5` is itself meaningful — it is the documented `n=30, multiplier=2.0`
        // segmentation `[(1,2),(2,4),(4,8),(8,16),(16,30)]`. Zero lines means the CPU
        // fallback ran and this oracle proved nothing about the device.
        // ────────────────────────────────────────────────────────────────────────────────

        let model = CbModel::from_trained(&trained, borders);
        let actual = predict_raw(&model, &columns);
        assert_eq!(actual.len(), expected.len());
        let mut max_abs = 0.0_f64;
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            let abs = (a - e).abs();
            max_abs = max_abs.max(abs);
            assert!(
                abs <= 1e-5,
                "obj {i}: ordered device prediction {a} vs upstream {e} exceeds ≤1e-5 \
                 (|Δ|={abs:.3e})"
            );
        }
        println!("[device-ordered-e2e] max |Δpred| = {max_abs:.3e} (bar 1e-5)");
    }
}

#[test]
fn device_ordered_fit_matches_upstream() {
    #[cfg(any(feature = "rocm", feature = "cuda"))]
    device::run();
    #[cfg(not(any(feature = "rocm", feature = "cuda")))]
    {
        let _ = ordered_params();
        eprintln!("SKIP device_ordered_fit_matches_upstream: needs rocm/cuda");
    }
}
