//! Automatic learning-rate selection (TRAIN-08).
//!
//! A bit-faithful port of upstream CatBoost's `TAutoLRParamsGuesser`
//! (`catboost-master/catboost/libs/train_lib/options_helper.cpp:116-262`).
//!
//! # When it fires
//!
//! Upstream `UpdateLearningRate` (`options_helper.cpp:269-288`) invokes the
//! guesser ONLY when `learning_rate`, `leaf_estimation_method`,
//! `leaf_estimation_iterations`, and `l2_leaf_reg` are ALL unset — and only when
//! the `(target, task, useBestModel, boostFromAverage)` key is present in the
//! coefficient table (`NeedToUpdate`). This module owns the table + the scalar
//! formula; the boosting loop (`boosting.rs`) owns the gating decision and calls
//! [`guess`] before the loop when the gate is open.
//!
//! # The formula (`GetLearningRate`, :252-262)
//!
//! With coefficients `{A,B,C,D}` (`DatasetSizeCoeff`, `DatasetSizeConst`,
//! `IterCountCoeff`, `IterCountConst`):
//!
//! ```text
//! custIter = exp(C * ln(iterCount)  + D)
//! defIter  = exp(C * ln(1000)       + D)
//! defLR    = exp(A * ln(objectCount)+ B)
//! lr       = round(min(defLR * custIter / defIter, 0.5), 6)
//! ```
//!
//! `round(x, 6)` is upstream's `Round` (`options_helper.cpp:15-18`):
//! `round(x * 1e6) / 1e6` with banker-free `round()` (round-half-away-from-zero).
//!
//! # Parity / safety discipline
//!
//! This is a pure host scalar; no float SUM is involved (no `cb_core::sum_f64`
//! routing needed — D-08 grep is about summation, not `exp`/`ln`). Degenerate
//! inputs (`object_count == 0` or `iter_count == 0`) would make `ln(0) = -inf`;
//! [`guess`] guards `> 0` and returns [`CbError`] instead (T-03-07-01). No
//! `unwrap` / `expect` / `panic` / `[]`-indexing.

use cb_core::{CbError, CbResult};

/// Auto-LR target classification (upstream `ETargetType`, restricted to the
/// Phase-3 losses). `GetTargetType` (`options_helper.cpp:181-194`) maps
/// `Logloss`/`MultiLogloss`/`MultiCrossEntropy` -> `Logloss`, `RMSE` -> `RMSE`,
/// everything else (e.g. MAE/Quantile) -> [`TargetType::Unknown`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetType {
    /// RMSE regression.
    Rmse,
    /// Logloss / cross-entropy binary classification.
    Logloss,
    /// A loss not present in the auto-LR table (no guess is produced — matches
    /// upstream `NeedToUpdate == false`).
    Unknown,
}

/// Upstream `TLearningRateCoefficients` `{A, B, C, D}` ==
/// `{DatasetSizeCoeff, DatasetSizeConst, IterCountCoeff, IterCountConst}`.
type Coeffs = [f64; 4];

/// The CPU coefficient table (`TAutoLRParamsGuesser::TAutoLRParamsGuesser`,
/// `options_helper.cpp:198-219`), keyed by
/// `(target, useBestModel, boostFromAverage)`. This phase is CPU-only
/// (`task_type == CPU`); the GPU rows are intentionally omitted (Phase 7).
///
/// A `None` lookup means the key is absent from the upstream table, so no
/// learning rate is guessed (upstream `NeedToUpdate` returns `false`).
#[must_use]
pub fn coefficients(
    target: TargetType,
    use_best_model: bool,
    boost_from_average: bool,
) -> Option<Coeffs> {
    // (target, useBestModel, boostFromAverage) -> {A, B, C, D}
    match (target, use_best_model, boost_from_average) {
        // --- Logloss (CPU) --------------------------------------------------
        (TargetType::Logloss, true, true) => Some([0.246, -5.127, -0.451, 0.978]),
        (TargetType::Logloss, false, true) => Some([0.408, -7.299, -0.928, 2.701]),
        (TargetType::Logloss, true, false) => Some([0.247, -5.158, -0.435, 0.934]),
        (TargetType::Logloss, false, false) => Some([0.427, -7.525, -0.917, 2.63]),
        // --- RMSE (CPU) -----------------------------------------------------
        (TargetType::Rmse, true, true) => Some([0.157, -4.062, -0.61, 1.557]),
        (TargetType::Rmse, false, true) => Some([0.158, -4.287, -0.813, 2.571]),
        (TargetType::Rmse, true, false) => Some([0.189, -4.383, -0.623, 1.439]),
        (TargetType::Rmse, false, false) => Some([0.178, -4.473, -0.76, 2.133]),
        // Unknown target / any other combination is not in the table.
        (TargetType::Unknown, _, _) => None,
    }
}

/// Upstream `Round(number, precision)` (`options_helper.cpp:15-18`):
/// `round(number * 10^precision) / 10^precision`, where `round` is C++
/// `std::round` (round-half-away-from-zero). Rust's `f64::round` has the same
/// half-away-from-zero rule, so this matches bit-for-bit on the LR domain.
#[must_use]
fn round_to(number: f64, precision: i32) -> f64 {
    let multiplier = 10f64.powi(precision);
    (number * multiplier).round() / multiplier
}

/// Guess the learning rate via the upstream coefficient table + exp/log/round
/// formula (`GetLearningRate`, `options_helper.cpp:252-262`).
///
/// `learn_object_count` is the learn-pool object count `N`; `iter_count` is the
/// configured `iterations`. The result is `round(min(defLR * custIter / defIter,
/// 0.5), 6)`.
///
/// # Errors
///
/// - [`CbError::OutOfRange`] if `learn_object_count == 0` or `iter_count == 0`
///   (would otherwise be `ln(0) = -inf`; T-03-07-01).
/// - [`CbError::Degenerate`] if the `(target, useBestModel, boostFromAverage)`
///   key is absent from the table (the loss is not auto-LR eligible — matches
///   upstream `NeedToUpdate == false`).
pub fn guess(
    target: TargetType,
    use_best_model: bool,
    boost_from_average: bool,
    learn_object_count: usize,
    iter_count: usize,
) -> CbResult<f64> {
    if learn_object_count == 0 {
        return Err(CbError::OutOfRange(
            "auto-LR: learn_object_count must be > 0".to_string(),
        ));
    }
    if iter_count == 0 {
        return Err(CbError::OutOfRange(
            "auto-LR: iter_count must be > 0".to_string(),
        ));
    }

    let [a, b, c, d] = coefficients(target, use_best_model, boost_from_average).ok_or_else(|| {
        CbError::Degenerate("auto-LR: no coefficient row for this target/task key".to_string())
    })?;

    let object_count = learn_object_count as f64;
    let iterations = iter_count as f64;

    let custom_iteration_constant = (c * iterations.ln() + d).exp();
    let default_iteration_constant = (c * 1000f64.ln() + d).exp();
    let default_learning_rate = (a * object_count.ln() + b).exp();

    let lr = default_learning_rate * custom_iteration_constant / default_iteration_constant;
    Ok(round_to(lr.min(0.5), 6))
}

#[cfg(test)]
#[path = "autolr_test.rs"]
mod tests;
