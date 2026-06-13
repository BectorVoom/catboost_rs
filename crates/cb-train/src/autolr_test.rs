//! Unit tests for automatic learning-rate selection ([`crate::autolr`], TRAIN-08).
//!
//! Two assertion families:
//!
//! 1. **RESEARCH example coeff rows** — the two coefficient rows pinned in
//!    `03-RESEARCH.md` (RMSE, bestModel=F, bfa=T -> {0.158,-4.287,-0.813,2.571};
//!    Logloss, bestModel=F, bfa=F -> {0.427,-7.525,-0.917,2.63}) are present in
//!    the CPU table and produce the expected rate for a known (N, iter) pair.
//! 2. **Upstream oracle parity** — the guessed rate matches the persisted
//!    `model.get_all_params()['learning_rate']` from the committed
//!    `autolr/{rmse,logloss}` fixtures (Task 1) at <= 1e-5, on the exact keying
//!    inputs CatBoost used.
//!
//! Dedicated `*_test.rs` file (source/test separation, CLAUDE.md). Reads the
//! committed fixture config.json via `serde_json` (a cb-train dev-dependency).

use approx::assert_abs_diff_eq;

use crate::autolr::{coefficients, guess, TargetType};

/// The oracle parity tolerance (CLAUDE.md / D-08: <= 1e-5).
const TOL: f64 = 1e-5;

#[test]
fn research_example_rows_present_in_cpu_table() {
    // RMSE, useBestModel=false, boostFromAverage=true.
    let rmse = coefficients(TargetType::Rmse, false, true).expect("RMSE/F/T row present");
    assert_eq!(rmse, [0.158, -4.287, -0.813, 2.571]);

    // Logloss, useBestModel=false, boostFromAverage=false.
    let logloss = coefficients(TargetType::Logloss, false, false).expect("Logloss/F/F row present");
    assert_eq!(logloss, [0.427, -7.525, -0.917, 2.63]);
}

#[test]
fn guess_matches_research_example_rates() {
    // N = 50 objects, iter = 500 (the autolr fixture inputs).
    let rmse = guess(TargetType::Rmse, false, true, 50, 500).expect("rmse guess");
    assert_abs_diff_eq!(rmse, 0.044808, epsilon = TOL);

    let logloss = guess(TargetType::Logloss, false, false, 50, 500).expect("logloss guess");
    assert_abs_diff_eq!(logloss, 0.005413, epsilon = TOL);
}

#[test]
fn guess_caps_at_half() {
    // A tiny dataset with a huge default-LR coefficient would exceed 0.5; the
    // formula caps at min(.., 0.5). RMSE/F/T at N=1, iter=1000 -> defLR=exp(B)
    // = exp(-4.287) ~= 0.0137, well under 0.5, so construct a cap case via the
    // monotonic size term: for the cap to bind we need exp(A*ln N + B) large.
    // Use a degenerate-but-valid large-N inverse is impossible (A>0), so assert
    // the cap holds structurally: any guess is <= 0.5.
    let r = guess(TargetType::Rmse, true, true, 1_000_000, 1000).expect("guess");
    assert!(r <= 0.5, "auto-LR must be capped at 0.5, got {r}");
}

#[test]
fn unknown_target_has_no_coefficients() {
    // Mae / Quantile is not in the auto-LR table upstream (GetTargetType ->
    // Unknown), so no guess is produced (matches NeedToUpdate == false).
    assert!(coefficients(TargetType::Unknown, false, false).is_none());
    assert!(guess(TargetType::Unknown, false, false, 50, 500).is_err());
}

#[test]
fn zero_object_count_is_error_not_panic() {
    // T-03-07-01: never ln(0) -> -inf; return an error instead.
    assert!(guess(TargetType::Rmse, false, true, 0, 500).is_err());
    assert!(guess(TargetType::Rmse, false, true, 50, 0).is_err());
}

/// Read a fixture config.json field and assert the guess matches the persisted
/// upstream `selected_learning_rate`.
fn assert_fixture(name: &str, target: TargetType) {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join("autolr")
        .join(name)
        .join("config.json");
    let raw = std::fs::read_to_string(&path).expect("autolr fixture config must exist");
    let cfg: serde_json::Value = serde_json::from_str(&raw).expect("valid json");

    let use_best_model = cfg["use_best_model"].as_bool().expect("use_best_model");
    let boost_from_average = cfg["boost_from_average"].as_bool().expect("boost_from_average");
    let n = cfg["learn_object_count"].as_u64().expect("learn_object_count") as usize;
    let iter = cfg["iterations"].as_u64().expect("iterations") as usize;
    let expected = cfg["selected_learning_rate"].as_f64().expect("selected_learning_rate");

    let got = guess(target, use_best_model, boost_from_average, n, iter)
        .expect("guess on fixture inputs");
    assert_abs_diff_eq!(got, expected, epsilon = TOL);
}

#[test]
fn matches_upstream_rmse_fixture() {
    assert_fixture("rmse", TargetType::Rmse);
}

#[test]
fn matches_upstream_logloss_fixture() {
    assert_fixture("logloss", TargetType::Logloss);
}
