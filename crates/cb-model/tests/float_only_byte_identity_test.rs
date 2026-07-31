//! T00 / SPEC-OH-31 — the float-only `.cbm` byte-identity baseline.
//!
//! # Why this exists
//!
//! Every later "the float-only path is unchanged" claim (SPEC-OH-31 / A2, T29,
//! T29b) is only meaningful against bytes frozen BEFORE any production change.
//! Captured at the plan-base SHA recorded in the fixture `README.md`; if the
//! baseline were regenerated after a change it would degenerate into a
//! self-comparison and prove nothing.
//!
//! # The fixture is FROZEN
//!
//! No later task may regenerate `float_only_byte_identity/`. The capture fn below
//! is `#[ignore]`d precisely so a routine `cargo test` can never rewrite it.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_backend::CpuBackend;
use cb_compute::{EScoreFunction, LeafMethod, Loss};
use cb_model::{save_cbm, Model as CbModel};
use cb_train::{train, BoostParams, EBootstrapType, EGrowPolicy, EOverfittingDetectorType};
use ndarray::Array2;
use ndarray_npy::{read_npy, write_npy};

/// The frozen baseline fixture root.
fn baseline_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join("float_only_byte_identity")
}

const N_ROWS: usize = 512;
const N_FEATURES: usize = 4;

/// Deterministic LCG — the inputs must be reproducible without a `rand` dep so the
/// capture is a pure function of this file.
struct Lcg(u64);

impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

/// The pinned float-only inputs: `N_ROWS x N_FEATURES` features plus an RMSE target.
fn generate_inputs() -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = Lcg(20_260_731);
    let cols: Vec<Vec<f64>> = (0..N_FEATURES)
        .map(|_| (0..N_ROWS).map(|_| rng.next_f64()).collect())
        .collect();
    let target = (0..N_ROWS)
        .map(|i| 0.7 * cols[0][i] - 0.4 * cols[1][i] + 0.2 * cols[2][i])
        .collect();
    (cols, target)
}

/// Every confound pinned OFF. `bootstrap_type = No` and `random_strength = 0`
/// keep the RNG stream out of the picture entirely, so a byte difference can only
/// come from the tree/serialization path this plan touches.
fn pinned_float_only_params() -> BoostParams {
    BoostParams {
        loss: Loss::Rmse,
        iterations: 3,
        depth: 3,
        learning_rate: 0.3,
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
        one_hot_max_size: cb_train::one_hot_max_size_default(),
        permutation_count: cb_train::permutation_count_default(),
        fold_len_multiplier: cb_train::fold_len_multiplier_default(),
        simple_ctr: cb_train::simple_ctr_default(),
        simple_ctr_priors: cb_train::simple_ctr_priors_default(),
        counter_calc_method: cb_train::counter_calc_method_default(),
        boosting_type: cb_train::boosting_type_default(),
        max_ctr_complexity: cb_train::max_ctr_complexity_default(),
        combinations_ctr: cb_train::combinations_ctr_default(),
        combinations_ctr_priors: cb_train::combinations_ctr_priors_default(),
        score_function: EScoreFunction::L2,
        has_time: false,
        feature_weights: cb_train::feature_weights_default(),
        first_feature_use_penalties: cb_train::first_feature_use_penalties_default(),
        per_object_feature_penalties: cb_train::per_object_feature_penalties_default(),
        penalties_coefficient: cb_train::penalties_coefficient_default(),
        monotone_constraints: cb_train::monotone_constraints_default(),
        grow_policy: EGrowPolicy::SymmetricTree,
        max_leaves: cb_train::max_leaves_default(),
        min_data_in_leaf: cb_train::min_data_in_leaf_default(),
    }
}

/// Borders for the pinned inputs — derived from the data, so the fit is a pure
/// function of `generate_inputs()`.
fn borders_for(cols: &[Vec<f64>]) -> Vec<Vec<f64>> {
    cols.iter()
        .map(|col| cb_data::select_borders_greedy_logsum(col, 32, false))
        .collect()
}

/// Run the pinned float-only fit and return the serialized `.cbm` bytes.
fn fit_and_serialize(cols: &[Vec<f64>], target: &[f64]) -> Vec<u8> {
    let feature_values: Vec<Vec<f32>> = cols
        .iter()
        .map(|c| c.iter().map(|&v| v as f32).collect())
        .collect();
    let feature_borders = borders_for(cols);
    let weights = vec![1.0f64; target.len()];

    let trained = train(
        &CpuBackend,
        &feature_values,
        &feature_borders,
        target,
        &weights,
        &pinned_float_only_params(),
        None,
    )
    .expect("the pinned float-only fit must succeed");

    let model = CbModel::from_trained(&trained, feature_borders);
    let tmp = std::env::temp_dir().join("cb_float_only_byte_identity_probe.cbm");
    save_cbm(&model, &tmp).expect("save_cbm must succeed");
    let bytes = std::fs::read(&tmp).expect("read back the serialized model");
    let _ = std::fs::remove_file(&tmp);
    bytes
}

/// CAPTURE ONLY — freezes the baseline. Run ONCE, at the plan-base SHA, with
/// `-- --ignored`. `#[ignore]`d so no routine test run can silently rewrite the
/// very bytes SPEC-OH-31 compares against.
#[test]
#[ignore = "capture-only: run once at the plan-base SHA to freeze the fixture"]
fn capture_float_only_baseline() {
    let dir = baseline_dir();
    let inputs = dir.join("inputs");
    std::fs::create_dir_all(&inputs).expect("create fixture dirs");

    let (cols, target) = generate_inputs();

    let mut x = Array2::<f64>::zeros((N_ROWS, N_FEATURES));
    for (f, col) in cols.iter().enumerate() {
        for (i, &v) in col.iter().enumerate() {
            x[[i, f]] = v;
        }
    }
    write_npy(inputs.join("X.npy"), &x).expect("write X.npy");
    write_npy(
        inputs.join("y.npy"),
        &ndarray::Array1::from(target.clone()),
    )
    .expect("write y.npy");

    let bytes = fit_and_serialize(&cols, &target);
    std::fs::write(dir.join("baseline.cbm"), &bytes).expect("write baseline.cbm");

    let sha = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map_or_else(|| "UNKNOWN".to_owned(), |s| s.trim().to_owned());

    std::fs::write(
        dir.join("README.md"),
        format!(
            "# float-only `.cbm` byte-identity baseline (SPEC-OH-31 / T00)\n\
             \n\
             **PLAN_BASE_SHA: `{sha}`**\n\
             \n\
             Captured BEFORE any one-hot production change, by\n\
             `float_only_byte_identity_test::capture_float_only_baseline`\n\
             (`#[ignore]`d; run with `-- --ignored`).\n\
             \n\
             ## THIS FIXTURE IS FROZEN\n\
             \n\
             No later task may regenerate it. Regenerating it after a production\n\
             change turns SPEC-OH-31 into a self-comparison that proves nothing —\n\
             the whole point is that these bytes predate the change.\n\
             \n\
             ## Contents\n\
             \n\
             - `baseline.cbm` — the pinned float-only fit, serialized.\n\
             - `inputs/X.npy` — `{N_ROWS} x {N_FEATURES}` float64 features.\n\
             - `inputs/y.npy` — `{N_ROWS}` float64 RMSE target.\n\
             \n\
             ## Pinned configuration\n\
             \n\
             RMSE, `iterations=3`, `depth=3`, `learning_rate=0.3`,\n\
             `l2_leaf_reg=3.0`, `random_strength=0`, `boost_from_average=false`,\n\
             `bootstrap_type=No`, `random_seed=0`, `score_function=L2`,\n\
             `grow_policy=SymmetricTree`, `border_count=32`.\n\
             \n\
             `bootstrap_type=No` + `random_strength=0` keep the RNG stream out of\n\
             the comparison entirely, so any byte difference is attributable to the\n\
             tree-construction or serialization path this plan touches.\n"
        ),
    )
    .expect("write README.md");
}

/// SPEC-OH-31 / A2: re-running the pinned float-only fit on the CURRENT tree must
/// reproduce the frozen plan-base bytes exactly.
///
/// A one-hot change that leaked into the float path — a reordered split lift, an
/// extra RNG draw, a stray `CatFeatures`/`OneHotFeatures` FlatBuffers section —
/// shows up here as a byte difference.
#[test]
fn float_only_cbm_bytes_match_the_frozen_plan_base_baseline() {
    let dir = baseline_dir();
    let baseline_path = dir.join("baseline.cbm");
    let expected = std::fs::read(&baseline_path).unwrap_or_else(|e| {
        panic!(
            "frozen baseline missing at {} ({e}). Capture it FIRST, at the plan-base \
             SHA: cargo test -p cb-model --test float_only_byte_identity_test -- --ignored",
            baseline_path.display()
        )
    });

    // Re-derive the inputs from the FROZEN npy files, not from `generate_inputs()`,
    // so a change to the generator cannot silently move the goalposts.
    let x: Array2<f64> = read_npy(dir.join("inputs/X.npy")).expect("frozen X.npy");
    let y: ndarray::Array1<f64> = read_npy(dir.join("inputs/y.npy")).expect("frozen y.npy");
    let cols: Vec<Vec<f64>> = (0..x.ncols()).map(|f| x.column(f).to_vec()).collect();
    let target: Vec<f64> = y.to_vec();

    let actual = fit_and_serialize(&cols, &target);

    assert_eq!(
        actual.len(),
        expected.len(),
        "float-only .cbm LENGTH changed ({} -> {}): the float path is not \
         byte-identical to the plan base (SPEC-OH-31)",
        expected.len(),
        actual.len()
    );
    if actual != expected {
        let first_diff = actual
            .iter()
            .zip(expected.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        panic!(
            "float-only .cbm BYTES changed at offset {first_diff} (len {}): the \
             float path is not byte-identical to the plan base (SPEC-OH-31 / A2). \
             A one-hot change has leaked into the float-only path.",
            expected.len()
        );
    }
}

/// The frozen fixture must carry its provenance — without the plan-base SHA there
/// is no way to tell a genuine baseline from one regenerated after a change.
#[test]
fn frozen_baseline_records_its_plan_base_sha() {
    let readme = std::fs::read_to_string(baseline_dir().join("README.md"))
        .expect("the frozen baseline must carry a README.md");
    assert!(
        readme.contains("PLAN_BASE_SHA"),
        "README.md must record the plan-base SHA the bytes were captured at"
    );
    assert!(
        readme.contains("FROZEN"),
        "README.md must state that the fixture is frozen"
    );
}
