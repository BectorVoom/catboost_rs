//! Prediction-type transforms (LOSS-06): map a slice of raw `approx` values
//! (`RawFormulaVal` logits, from [`crate::apply::predict_raw`]) to the requested
//! output type.
//!
//! # Source of truth (RESEARCH Pattern 3 — `PrepareEval`, `eval_helpers.cpp`)
//!
//! The Python `predict(prediction_type=…)` dispatcher is the spec (D-13 fixtures
//! come from it). The binary / single-dimension path
//! (`eval_helpers.cpp:352-496`):
//!
//! | Type | Formula (binary, 1-dim) | exp used |
//! |------|-------------------------|----------|
//! | `RawFormulaVal` | identity (= raw approx) | — |
//! | `Probability` | two columns `[1 - sigmoid(a), sigmoid(a)]` | `std::exp` (vector overload) |
//! | `LogProbability` | two columns `[-log(1+exp(a)), -log(1+exp(-a))]` | `std::exp` |
//! | `Class` | `approx > 0` (default `binClassLogitThreshold`) | — |
//! | `Exponent` | `exp(approx)` | `FastExp` (table/SSE/AVX) |
//!
//! `Probability` / `LogProbability` use `f64::exp` — the Python oracle uses the
//! `std::exp` vector overloads there (`CalcSigmoid` / `CalcLogSigmoid`,
//! `eval_processing.h:103-141`), which `f64::exp` matches exactly. `Exponent` uses
//! `f64::exp` too, accepting that upstream's `CalcExponent` →
//! `FastExpWithInfInplace` is a table approximation (`fast_exp.cpp:33-49`); the
//! `<= 1e-5` parity gate absorbs the FastExp gap (RESEARCH Pitfall 3 / assumption
//! A2 — verified against the committed `exponent.npy` fixture). `Class` uses
//! threshold `0` (RESEARCH Pitfall 4; Phase-4 fixtures set no custom probability
//! border).
//!
//! `Probability` and `LogProbability` emit TWO columns per object (class-0 then
//! class-1), flattened row-major to match the upstream binary `predict` output
//! (`eval_helpers.cpp:393`).

/// The in-scope prediction types (RESEARCH D-10 — uncertainty types deferred to
/// Phase 6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictionType {
    /// The raw model score / logit (identity transform).
    RawFormulaVal,
    /// Class probabilities: two columns `[1 - sigmoid(a), sigmoid(a)]`.
    Probability,
    /// Log class probabilities: two columns
    /// `[-log(1+exp(a)), -log(1+exp(-a))]`.
    LogProbability,
    /// Predicted class label: `1.0` when `approx > 0`, else `0.0`.
    Class,
    /// `exp(approx)` (e.g. for Poisson-style exponentiated scores).
    Exponent,
}

/// The default binary-class logit threshold (`eval_helpers.cpp:329`, Pitfall 4):
/// `0` unless a probability border is configured (never in Phase 4).
const BIN_CLASS_LOGIT_THRESHOLD: f64 = 0.0;

/// `sigmoid(a) = 1 / (1 + exp(-a))`, the binary `Probability` positive-class
/// probability (`eval_processing.h:103-110` `CalcSigmoid`, `std::exp`).
#[must_use]
fn sigmoid(approx: f64) -> f64 {
    1.0 / (1.0 + (-approx).exp())
}

/// Apply `prediction_type` to a slice of raw `approx` logits, returning the
/// flattened (row-major) output (LOSS-06).
///
/// For single-column types (`RawFormulaVal`, `Class`, `Exponent`) the output has
/// one value per object. For the two-column types (`Probability`,
/// `LogProbability`) the output has `2 * approx.len()` values: object `i`'s
/// `[class-0, class-1]` pair at indices `2*i` and `2*i + 1` (matching upstream's
/// binary `predict` row-major layout, `eval_helpers.cpp:393`).
#[must_use]
pub fn apply_prediction_type(prediction_type: PredictionType, approx: &[f64]) -> Vec<f64> {
    match prediction_type {
        // Identity (`eval_helpers.cpp:490`).
        PredictionType::RawFormulaVal => approx.to_vec(),
        // Two columns `[1 - p, p]`, `p = sigmoid(a)` (`eval_helpers.cpp:391`).
        PredictionType::Probability => {
            let mut out = Vec::with_capacity(approx.len() * 2);
            for &a in approx {
                let p = sigmoid(a);
                out.push(1.0 - p);
                out.push(p);
            }
            out
        }
        // Two columns `[-log(1+exp(a)), -log(1+exp(-a))]` — the log-sigmoid of the
        // negative and positive logit (`CalcLogSigmoid`, `eval_processing.h:131-141`).
        PredictionType::LogProbability => {
            let mut out = Vec::with_capacity(approx.len() * 2);
            for &a in approx {
                out.push(-(1.0 + a.exp()).ln());
                out.push(-(1.0 + (-a).exp()).ln());
            }
            out
        }
        // `approx > threshold` (default threshold 0, `eval_helpers.cpp:413-414`).
        PredictionType::Class => approx
            .iter()
            .map(|&a| if a > BIN_CLASS_LOGIT_THRESHOLD { 1.0 } else { 0.0 })
            .collect(),
        // `exp(approx)` — upstream uses FastExp; `f64::exp` is within 1e-5 (A2,
        // `eval_helpers.cpp:420` -> `CalcExponent`, `eval_processing.h:30-33`).
        PredictionType::Exponent => approx.iter().map(|&a| a.exp()).collect(),
    }
}
