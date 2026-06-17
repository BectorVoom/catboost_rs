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

use cb_core::sum_f64;

/// The in-scope prediction types: the Phase-4 deterministic transforms plus the
/// Phase-6.4 uncertainty types (`RmseWithUncertainty` / `VirtEnsembles` /
/// `TotalUncertainty` — LOSS-06, the Phase-4 D-10 deferral closed).
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
    /// RMSEWithUncertainty single-model predict (LOSS-06, the Phase-4 D-10
    /// deferral closed): two columns `[mean = approx[0], variance =
    /// exp(2*approx[1])]`. The variance is `CalcSquaredExponent(approx[1]) =
    /// exp(2*x)` — NOT `exp(x)` and NOT `x²` (RESEARCH Pitfall 6,
    /// `eval_processing.h:47`); the log-scale dim is `0.5*log(variance)`.
    /// Consumes the DIM-MAJOR 2-dim raw approx (`[mean(0..n), log-scale(n..2n)]`).
    RmseWithUncertainty,
    /// Virtual-ensembles predict (LOSS-06): the `V` per-ensemble `[mean,
    /// variance=exp(2*log-scale)]` pairs. Consumes the OBJECT-MAJOR `(n, V, 2)`
    /// virtual-ensemble matrix from [`crate::apply_virtual_ensembles`]; only the
    /// odd (log-scale) dims are `exp(2*x)`-transformed in place
    /// (`eval_helpers.cpp:428-444`). VE-aware — apply via
    /// [`apply_ve_prediction_type`].
    VirtEnsembles,
    /// Total-uncertainty predict (LOSS-06): the `[mean, knowledgeUncertainty,
    /// dataUncertainty]` regression-uncertainty decomposition
    /// (`CalcRegressionUncertaitny`, `eval_helpers.cpp:209-269`). Consumes the
    /// OBJECT-MAJOR `(n, V, 2)` virtual-ensemble matrix. VE-aware — apply via
    /// [`apply_ve_prediction_type`].
    TotalUncertainty,
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
        // RMSEWithUncertainty single-model predict (LOSS-06): the DIM-MAJOR 2-dim
        // raw approx `[mean(0..n), log-scale(n..2n)]` -> OBJECT-MAJOR `(n, 2)`
        // `[mean, variance = exp(2*log-scale)]` (`eval_helpers.cpp:422-427`;
        // `CalcSquaredExponent = exp(2x)`, Pitfall 6). `n = approx.len() / 2`; an
        // odd length yields empty (guarded — no panic).
        PredictionType::RmseWithUncertainty => {
            let n = approx.len() / 2;
            if approx.len() != 2 * n {
                return Vec::new();
            }
            let mut out = Vec::with_capacity(approx.len());
            for i in 0..n {
                let mean = approx.get(i).copied().unwrap_or(0.0);
                let log_scale = approx.get(n + i).copied().unwrap_or(0.0);
                out.push(mean);
                out.push((2.0 * log_scale).exp());
            }
            out
        }
        // VirtEnsembles / TotalUncertainty consume the VE matrix, not a single
        // approx — they are VE-aware (`apply_ve_prediction_type`). On the
        // single-approx path they pass the input through unchanged (no panic).
        PredictionType::VirtEnsembles | PredictionType::TotalUncertainty => approx.to_vec(),
    }
}

/// Apply a VIRTUAL-ENSEMBLE uncertainty prediction transform (LOSS-06) to the
/// OBJECT-MAJOR `(n, V, dim)` virtual-ensemble matrix produced by
/// [`crate::apply_virtual_ensembles`] — value at `i*(V*dim) + e*dim + d` is object
/// `i`'s ensemble `e` dimension `d` RAW approx (dim 0 mean, dim 1 log-scale for
/// RMSEWithUncertainty).
///
/// - [`PredictionType::VirtEnsembles`]: identity over the matrix with every
///   ODD dimension (the log-scale dims) → `exp(2*x)` IN PLACE, so each ensemble
///   reads `[mean, variance]` (`eval_helpers.cpp:428-444`). Output shape
///   `(n, V, dim)`.
/// - [`PredictionType::TotalUncertainty`]: per object over its `V` ensembles
///   (`dimShift = dim`, `CalcRegressionUncertaitny`, `eval_helpers.cpp:209-269`):
///   `mean = (1/V) Σ_e approx[e*dim]`,
///   `knowledgeUncertainty = (1/V) Σ_e (approx[e*dim] - mean)²` (epistemic),
///   `dataUncertainty = (1/V) Σ_e exp(2*approx[e*dim+1])` (aleatoric); output
///   `(n, 3)` `[mean, knowledgeUncertainty, dataUncertainty]`.
/// - any other [`PredictionType`]: the matrix unchanged (no panic).
///
/// All per-ensemble Σ route through [`cb_core::sum_f64`] (D-08). `n` is derived
/// from `ve.len() / (virtual_ensembles_count * approx_dimension)`; a zero `V` /
/// `dim` or a non-conforming length returns empty (guarded — no div-by-zero /
/// panic). All access is checked `.get` (`indexing_slicing` deny).
#[must_use]
pub fn apply_ve_prediction_type(
    prediction_type: PredictionType,
    ve: &[f64],
    virtual_ensembles_count: usize,
    approx_dimension: usize,
) -> Vec<f64> {
    let v = virtual_ensembles_count;
    let dim = approx_dimension;
    if v == 0 || dim == 0 {
        return Vec::new();
    }
    let block = v.saturating_mul(dim);
    if block == 0 || ve.len() % block != 0 {
        return Vec::new();
    }
    let n = ve.len() / block;
    match prediction_type {
        // The 2V-row matrix with the log-scale dims (odd `d`) -> exp(2*x).
        PredictionType::VirtEnsembles => {
            let mut out = Vec::with_capacity(ve.len());
            for i in 0..n {
                for e in 0..v {
                    for d in 0..dim {
                        let idx = i.saturating_mul(block).saturating_add(e * dim).saturating_add(d);
                        let raw = ve.get(idx).copied().unwrap_or(0.0);
                        // dim 0 = mean (identity); dim 1 = log-scale -> exp(2*x).
                        if d % 2 == 1 {
                            out.push((2.0 * raw).exp());
                        } else {
                            out.push(raw);
                        }
                    }
                }
            }
            out
        }
        // [mean, knowledgeUncertainty, dataUncertainty] per object (dimShift=dim).
        PredictionType::TotalUncertainty => {
            let v_f = v as f64;
            let mut out = Vec::with_capacity(n * 3);
            for i in 0..n {
                // Gather the per-ensemble mean (dim 0) and exp(2*log-scale) (dim 1).
                let means: Vec<f64> = (0..v)
                    .map(|e| {
                        let idx = i.saturating_mul(block).saturating_add(e * dim);
                        ve.get(idx).copied().unwrap_or(0.0)
                    })
                    .collect();
                let data_terms: Vec<f64> = (0..v)
                    .map(|e| {
                        let idx = i.saturating_mul(block).saturating_add(e * dim).saturating_add(1);
                        let log_scale = ve.get(idx).copied().unwrap_or(0.0);
                        (2.0 * log_scale).exp()
                    })
                    .collect();
                // mean = (1/V) Σ_e mean_e.
                let mean = sum_f64(&means) / v_f;
                // knowledgeUncertainty = (1/V) Σ_e (mean_e - mean)^2 (epistemic).
                let sq: Vec<f64> = means.iter().map(|&m| (m - mean) * (m - mean)).collect();
                let knowledge = sum_f64(&sq) / v_f;
                // dataUncertainty = (1/V) Σ_e exp(2*log-scale_e) (aleatoric).
                let data = sum_f64(&data_terms) / v_f;
                out.push(mean);
                out.push(knowledge);
                out.push(data);
            }
            out
        }
        // Not a VE transform: the matrix unchanged (no panic).
        PredictionType::RawFormulaVal
        | PredictionType::Probability
        | PredictionType::LogProbability
        | PredictionType::Class
        | PredictionType::Exponent
        | PredictionType::RmseWithUncertainty => ve.to_vec(),
    }
}

/// Which multiclass prediction transform a multi-dimensional model uses (LOSS-02).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultiClassKind {
    /// MultiClass (softmax): `Probability` = softmax over the per-object dimension
    /// slice (the probabilities sum to 1); `Class` = argmax over dim.
    Softmax,
    /// MultiClassOneVsAll: `Probability` = per-dimension sigmoid (the probabilities
    /// do NOT sum to 1); `Class` = argmax over dim.
    OneVsAll,
    /// MultiLogloss / MultiCrossEntropy (multilabel, LOSS-02): `Probability` =
    /// per-dimension sigmoid of the raw approx — each label dimension is an
    /// INDEPENDENT binary probability (the values do NOT sum to 1; there is no
    /// softmax coupling and no single "winning" class). Identical `Probability`
    /// transform to [`MultiClassKind::OneVsAll`]; named separately because the
    /// dimensions are label columns, not mutually-exclusive classes.
    MultiLabel,
}

/// The per-object softmax over `slice` (max-subtracted, `eval_processing.h:18`) —
/// the MultiClass `Probability` transform's per-object normalizer. Reproduces the
/// MAX-SUBTRACTION before `exp` so a large logit cannot overflow to `Inf`/`NaN`
/// (T-6.2-02); `f64::exp` (A2). An empty slice returns empty.
#[must_use]
fn softmax(slice: &[f64]) -> Vec<f64> {
    if slice.is_empty() {
        return Vec::new();
    }
    let mut max_a = f64::NEG_INFINITY;
    for &a in slice {
        if a > max_a {
            max_a = a;
        }
    }
    let exps: Vec<f64> = slice.iter().map(|&a| (a - max_a).exp()).collect();
    let mut sum = 0.0_f64;
    for &e in &exps {
        sum += e;
    }
    exps.iter().map(|&e| e / sum).collect()
}

/// `sigmoid(a) = 1 / (1 + exp(-a))` for the OneVsAll per-dimension `Probability`.
#[must_use]
fn sigmoid_pos(a: f64) -> f64 {
    1.0 / (1.0 + (-a).exp())
}

/// Apply a multiclass prediction transform to the DIMENSION-MAJOR raw approx
/// `approx[d * n + i]` (length `approx_dimension * n`), returning the flattened
/// OBJECT-MAJOR output (row-major, object then dim) — matching upstream's
/// `predict(prediction_type)` `(n, dim)` layout (LOSS-02, RESEARCH A4).
///
/// - `Probability`: per object, softmax over its `k` dimensions (MultiClass) or
///   per-dimension sigmoid (OneVsAll); emits `k` values per object (`n*k` total).
/// - `Class`: per object, the argmax dimension mapped through `class_to_label`
///   (so the ORIGINAL label is recovered, Pitfall 4); emits ONE value per object.
/// - `RawFormulaVal`: the raw approx transposed dim-major → object-major.
///
/// `class_to_label[c]` is the original label for class index `c`; an empty map
/// falls back to the raw class index. `n = approx.len() / approx_dimension`. A
/// zero `approx_dimension` returns empty (guarded — no div-by-zero/panic).
#[must_use]
pub fn apply_multiclass_prediction(
    prediction_type: PredictionType,
    kind: MultiClassKind,
    approx: &[f64],
    approx_dimension: usize,
    class_to_label: &[f64],
) -> Vec<f64> {
    if approx_dimension == 0 || approx.is_empty() {
        return Vec::new();
    }
    let n = approx.len() / approx_dimension;
    // Gather each object's k-dimensional slice (dim-major -> object view).
    let object_slice = |i: usize| -> Vec<f64> {
        (0..approx_dimension)
            .map(|d| approx.get(d * n + i).copied().unwrap_or(0.0))
            .collect()
    };
    match prediction_type {
        // Raw approx, transposed dim-major -> object-major (n, dim).
        PredictionType::RawFormulaVal => {
            let mut out = Vec::with_capacity(approx.len());
            for i in 0..n {
                out.extend(object_slice(i));
            }
            out
        }
        PredictionType::Probability => {
            let mut out = Vec::with_capacity(n * approx_dimension);
            for i in 0..n {
                let slice = object_slice(i);
                match kind {
                    MultiClassKind::Softmax => out.extend(softmax(&slice)),
                    // OneVsAll and MultiLabel share the per-dimension sigmoid
                    // Probability transform (each dimension an independent binary
                    // probability; no softmax coupling).
                    MultiClassKind::OneVsAll | MultiClassKind::MultiLabel => {
                        out.extend(slice.iter().map(|&a| sigmoid_pos(a)));
                    }
                }
            }
            out
        }
        // Argmax over dim -> original label via class_to_label (Pitfall 4).
        PredictionType::Class => {
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let slice = object_slice(i);
                let mut best_dim = 0usize;
                let mut best_val = f64::NEG_INFINITY;
                for (d, &v) in slice.iter().enumerate() {
                    if v > best_val {
                        best_val = v;
                        best_dim = d;
                    }
                }
                let label = class_to_label
                    .get(best_dim)
                    .copied()
                    .unwrap_or(best_dim as f64);
                out.push(label);
            }
            out
        }
        // LogProbability / Exponent and the uncertainty types are not multiclass
        // transforms in scope; fall back to the raw object-major approx (no NaN,
        // no panic). The uncertainty types apply via `apply_prediction_type`
        // (RmseWithUncertainty) / `apply_ve_prediction_type` (VirtEnsembles /
        // TotalUncertainty), not this multiclass entry point.
        PredictionType::LogProbability
        | PredictionType::Exponent
        | PredictionType::RmseWithUncertainty
        | PredictionType::VirtEnsembles
        | PredictionType::TotalUncertainty => {
            let mut out = Vec::with_capacity(approx.len());
            for i in 0..n {
                out.extend(object_slice(i));
            }
            out
        }
    }
}
