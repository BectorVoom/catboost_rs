//! Overfitting detection / early stopping (TRAIN-06) — a pure host state machine
//! porting `catboost/libs/overfitting_detector/overfitting_detector.cpp` plus the
//! `NStatistics::Wilcoxon` signed-rank statistic
//! (`library/cpp/statistics/{statistics.h,detail.h}`).
//!
//! # Source of truth
//!
//! - `overfitting_detector.cpp:37-208` — `TOverfittingDetectorBase`
//!   (`IsActive()` iff `Threshold>0`; `IsNeedStop()` iff `!IsEmpty &&
//!   CurrentPValue < Threshold`), `TOverfittingDetectorIncToDec::AddError`
//!   (running `LocalMax`, exponentially-forgotten `ExpectedInc` over the last
//!   `ITERATION_FORGET=2000` errors, `LAMBDA_FORGET=0.99`; p-value
//!   `exp(-LAMBDA_SCALE / max(ExpectedInc/max(LocalMax-Last,EPS), EPS))`,
//!   `LAMBDA_SCALE=0.5`, `EPS=1e-10`, fired only once `IterationsFromLocalMax >=
//!   IterationsWait`), `TOverfittingDetectorWilcoxon::AddError` (deltas
//!   `LastError - err` AFTER the local max; p-value once `>= IterationsWait`
//!   deltas). `Iter` == `IncToDec` with the threshold forced to `1.0`. For a LOSS
//!   metric `maxIsOptimal=false`, so `err` is negated (a decreasing loss is an
//!   increasing score).
//! - `detail.h:WilcoxonTestWithSign` — signed-rank statistic over deltas sorted
//!   by absolute value, average-rank tie handling, normal-approximation p-value
//!   via the standard normal CDF `Phi`.
//!
//! # Parity & safety discipline
//!
//! No external stats crate (Don't-Hand-Roll: port the Wilcoxon semantics). The
//! `erf` powering `Phi` is the standard W. J. Cody rational-Chebyshev primitive
//! (`~1e-16`, the same shape as the libm `erf` upstream links against) — a math
//! primitive, not the Wilcoxon statistic. Degenerate inputs surface as
//! [`CbError`], never a panic; deny-lints (`unwrap`/`indexing_slicing`) hold.

use cb_core::{CbError, CbResult};

#[cfg(test)]
#[path = "overfit_test.rs"]
mod tests;

/// The overfitting-detector type (`EOverfittingDetectorType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EOverfittingDetectorType {
    /// No detection (always inactive, never stops).
    None,
    /// IncToDec (the default): the increasing-to-decreasing p-value detector.
    IncToDec,
    /// Iter: `IncToDec` with the threshold forced to `1.0` (stops `od_wait`
    /// iterations after the best).
    Iter,
    /// Wilcoxon signed-rank statistic over post-local-max deltas.
    Wilcoxon,
}

// IncToDec constants (overfitting_detector.cpp:161-164).
const LAMBDA_FORGET: f64 = 0.99;
const ITERATION_FORGET: usize = 2000;
const LAMBDA_SCALE: f64 = 0.5;
const EPS: f64 = 1e-10;

/// The internal per-type state of the detector.
#[derive(Debug)]
enum DetectorState {
    /// `None` — always inactive.
    Inactive,
    /// `IncToDec` / `Iter` (Iter forces the threshold to 1.0).
    IncToDec(IncToDecState),
    /// `Wilcoxon`.
    Wilcoxon(WilcoxonState),
}

/// IncToDec running state (`TOverfittingDetectorIncToDec`).
#[derive(Debug, Default)]
struct IncToDecState {
    /// `Errors` deque, most-recent at the front (`push_front`).
    errors: std::collections::VecDeque<f64>,
    local_max: f64,
    expected_inc: f64,
    last_error: f64,
    iterations_from_local_max: usize,
}

/// Wilcoxon running state (`TOverfittingDetectorWilcoxon`).
#[derive(Debug, Default)]
struct WilcoxonState {
    deltas_after_local_max: Vec<f64>,
    last_error: f64,
    local_max: f64,
}

/// The overfitting-detection state machine (`IOverfittingDetector`).
///
/// `maxIsOptimal` is fixed to `false` for the loss metrics this phase covers
/// (RMSE / Logloss) — a DECREASING loss is an improving (increasing) score, so
/// `add_error` negates the incoming loss to match upstream.
#[derive(Debug)]
pub struct OverfittingDetector {
    state: DetectorState,
    /// `Threshold` — `<= 0` means inactive (`IsActive()` iff `Threshold > 0`).
    threshold: f64,
    /// `IterationsWait` (`od_wait`).
    iterations_wait: usize,
    /// `IsEmpty` — no error has been added yet.
    is_empty: bool,
    /// `CurrentPValue` (starts at `1.0`).
    current_pvalue: f64,
    /// `MaxIsOptimal` — `false` for a loss metric (this phase). Kept explicit so
    /// the port reads 1:1 against upstream.
    max_is_optimal: bool,
}

impl OverfittingDetector {
    /// Construct a detector mirroring `CreateOverfittingDetector`
    /// (`overfitting_detector.cpp:185-208`).
    ///
    /// `threshold` is the `AutoStopPValue` (`od_pval`); `iterations_wait` is
    /// `od_wait`. `has_test` records whether an eval set is present. `Iter` forces
    /// the threshold to `1.0`. Wilcoxon / IncToDec adopt `0` (inactive) when there
    /// is no test (`hasTest ? threshold : 0`), and Wilcoxon additionally requires
    /// a test when the threshold is positive.
    ///
    /// `max_is_optimal` is `false` (loss metric) for every Phase-3 caller.
    ///
    /// # Errors
    /// [`CbError::Degenerate`] for a Wilcoxon detector with a positive threshold
    /// but no test set (`CB_ENSURE(hasTest || threshold == 0)`).
    pub fn new(
        detector_type: EOverfittingDetectorType,
        threshold: f64,
        iterations_wait: usize,
        has_test: bool,
    ) -> CbResult<Self> {
        // hasTest ? threshold : 0 (overfitting_detector.cpp:84,123).
        let effective_threshold = if has_test { threshold } else { 0.0 };

        let (state, threshold, max_is_optimal) = match detector_type {
            EOverfittingDetectorType::None => (DetectorState::Inactive, 0.0, false),
            EOverfittingDetectorType::IncToDec => (
                DetectorState::IncToDec(IncToDecState::default()),
                effective_threshold,
                false,
            ),
            // Iter == IncToDec with threshold forced to 1.0
            // (overfitting_detector.cpp:195-198). The forced 1.0 is NOT gated on
            // hasTest (upstream passes the literal 1.0 through).
            EOverfittingDetectorType::Iter => {
                (DetectorState::IncToDec(IncToDecState::default()), 1.0, false)
            }
            EOverfittingDetectorType::Wilcoxon => {
                // CB_ENSURE(hasTest || threshold == 0) (overfitting_detector.cpp:85).
                if !has_test && threshold != 0.0 {
                    return Err(CbError::Degenerate(
                        "Wilcoxon overfitting detector: no test provided, cannot check overfitting"
                            .to_owned(),
                    ));
                }
                (
                    DetectorState::Wilcoxon(WilcoxonState::default()),
                    effective_threshold,
                    false,
                )
            }
        };

        Ok(Self {
            state,
            threshold,
            iterations_wait,
            is_empty: true,
            current_pvalue: 1.0,
            max_is_optimal,
        })
    }

    /// `IsActive()` — `Threshold > 0` (`overfitting_detector.cpp:67-69`).
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.threshold > 0.0
    }

    /// `GetCurrentPValue()`.
    #[must_use]
    pub fn current_pvalue(&self) -> f64 {
        self.current_pvalue
    }

    /// `IsNeedStop()` — `(!IsEmpty) && (CurrentPValue < Threshold)`
    /// (`overfitting_detector.cpp:63-65`).
    #[must_use]
    pub fn is_need_stop(&self) -> bool {
        (!self.is_empty) && (self.current_pvalue < self.threshold)
    }

    /// `AddError(err)` — feed one eval-metric value (the loss). No-op when the
    /// detector is inactive (`Threshold <= 0`), matching both the `None` detector
    /// and the early `if (Threshold <= 0.0) return;` guard in the IncToDec /
    /// Wilcoxon `AddError`.
    pub fn add_error(&mut self, err: f64) {
        if self.threshold <= 0.0 {
            return;
        }
        // maxIsOptimal=false (loss): err = -err (overfitting_detector.cpp:91-92,
        // 130-131).
        let err = if self.max_is_optimal { err } else { -err };

        match &mut self.state {
            DetectorState::Inactive => {}
            DetectorState::IncToDec(s) => {
                Self::inctodec_add_error(
                    s,
                    &mut self.is_empty,
                    &mut self.current_pvalue,
                    self.iterations_wait,
                    err,
                );
            }
            DetectorState::Wilcoxon(s) => {
                Self::wilcoxon_add_error(
                    s,
                    &mut self.is_empty,
                    &mut self.current_pvalue,
                    self.iterations_wait,
                    err,
                );
            }
        }
    }

    /// `TOverfittingDetectorIncToDec::AddError` + `UpdatePValue`
    /// (`overfitting_detector.cpp:127-174`).
    fn inctodec_add_error(
        s: &mut IncToDecState,
        is_empty: &mut bool,
        current_pvalue: &mut f64,
        iterations_wait: usize,
        err: f64,
    ) {
        if *is_empty || err > s.local_max {
            if *is_empty {
                *is_empty = false;
                s.expected_inc = 0.0;
            }
            s.local_max = err;
            s.iterations_from_local_max = 0;
        } else {
            s.iterations_from_local_max += 1;
        }

        // Errors.push_front(err); pop_back beyond ITERATION_FORGET.
        s.errors.push_front(err);
        if s.errors.len() > ITERATION_FORGET {
            s.errors.pop_back();
        }

        // ExpectedInc *= LAMBDA_FORGET; then max over the forgotten history.
        s.expected_inc *= LAMBDA_FORGET;
        let mut cur_mult = 1.0;
        for &e in &s.errors {
            s.expected_inc = s.expected_inc.max(cur_mult * (err - e));
            cur_mult *= LAMBDA_FORGET;
        }

        s.last_error = err;

        // UpdatePValue (overfitting_detector.cpp:167-174).
        if s.iterations_from_local_max >= iterations_wait {
            let ratio = s.expected_inc / (s.local_max - s.last_error).max(EPS);
            *current_pvalue = (-LAMBDA_SCALE / ratio.max(EPS)).exp();
        } else {
            *current_pvalue = 1.0;
        }
    }

    /// `TOverfittingDetectorWilcoxon::AddError` + `UpdatePValue`
    /// (`overfitting_detector.cpp:89-113`).
    fn wilcoxon_add_error(
        s: &mut WilcoxonState,
        is_empty: &mut bool,
        current_pvalue: &mut f64,
        iterations_wait: usize,
        err: f64,
    ) {
        if *is_empty || err > s.local_max {
            *is_empty = false;
            s.deltas_after_local_max.clear();
            s.local_max = err;
        } else {
            s.deltas_after_local_max.push(s.last_error - err);
        }
        s.last_error = err;

        // UpdatePValue (overfitting_detector.cpp:107-113).
        if s.deltas_after_local_max.len() >= iterations_wait {
            *current_pvalue = wilcoxon(&s.deltas_after_local_max);
        } else {
            *current_pvalue = 1.0;
        }
    }
}

/// Tracks the best (lowest-loss) iteration for `use_best_model`. Ties keep the
/// FIRST (earliest) best — upstream's strict-`<` improvement check first-wins.
#[derive(Debug, Default)]
pub struct BestModelTracker {
    best_iteration: Option<usize>,
    best_error: f64,
    next_index: usize,
}

impl BestModelTracker {
    /// A fresh tracker with no observed errors.
    #[must_use]
    pub fn new() -> Self {
        Self {
            best_iteration: None,
            best_error: f64::INFINITY,
            next_index: 0,
        }
    }

    /// Feed the iteration's eval-metric (loss). The lowest loss wins; a strictly
    /// smaller loss updates the best (ties do NOT replace the earlier best).
    pub fn add_error(&mut self, err: f64) {
        let idx = self.next_index;
        self.next_index += 1;
        if self.best_iteration.is_none() || err < self.best_error {
            self.best_error = err;
            self.best_iteration = Some(idx);
        }
    }

    /// The best iteration index so far (`None` if no error was added).
    #[must_use]
    pub fn best_iteration(&self) -> Option<usize> {
        self.best_iteration
    }
}

/// `NStatistics::Wilcoxon` over the difference samples (`detail.h:163-191` +
/// `WilcoxonTestWithSign`): drop zeros, sort by absolute value, accumulate the
/// signed-rank statistic `w` with average ranks for ties, and return the
/// two-sided normal-approximation p-value `(1 - Phi(|x|)) * 2`.
///
/// Returns `0.5` (the neutral p-value) on an empty / all-zero input, matching
/// upstream's `TStatTestResult(0.5, 0)` early returns.
#[must_use]
fn wilcoxon(deltas: &[f64]) -> f64 {
    if deltas.is_empty() {
        return 0.5;
    }
    // Keep nonzero deltas (statistics.h:173-176 `if (*it != 0)`).
    let mut v: Vec<f64> = deltas.iter().copied().filter(|&d| d != 0.0).collect();
    if v.is_empty() {
        return 0.5;
    }
    // Sort by absolute value (WilcoxonComparator: fabs(a) < fabs(b)).
    v.sort_by(|a, b| {
        a.abs()
            .partial_cmp(&b.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let size = v.len() as f64;
    let mut denominator = size * (size + 1.0) * (2.0 * size + 1.0);
    let mut w = 0.0f64;

    // Average-rank tie handling over blocks of |value|-equal entries
    // (detail.h:170-190). Indices are 0-based here; the rank uses the upstream
    // `(blockFirstIndex + blockLastIndex + 2) / 2` average-rank formula.
    let n = v.len();
    let mut block_first = 0usize;
    while block_first < n {
        let mut block_last = block_first;
        let first_val = v.get(block_first).copied().unwrap_or(0.0);
        // Extend the block while the next entry is relatively-equal in abs value.
        while let Some(&next_val) = v.get(block_last + 1) {
            if !relative_equal(next_val, first_val) {
                break;
            }
            block_last += 1;
        }
        let rank = (block_first as f64 + block_last as f64 + 2.0) / 2.0;
        for &val in v.iter().take(block_last + 1).skip(block_first) {
            if val > 0.0 {
                w += rank;
            }
        }
        let block_size = (block_last - block_first + 1) as f64;
        denominator -= block_size * (block_size - 1.0) * (block_size + 1.0) * 0.5;
        block_first = block_last + 1;
    }

    if denominator <= 0.0 {
        // Upstream throws; here a degenerate denominator yields the neutral
        // p-value rather than a panic (T-03-05-01).
        return 0.5;
    }
    let denominator = (denominator / 24.0).sqrt();
    let x = (w - size * (size + 1.0) / 4.0) / denominator;
    // Phi(0, 1, |x|, continuity=false) then two-sided (1 - res) * 2.
    let res = phi(x.abs());
    (1.0 - res) * 2.0
}

/// Relative-equal comparator (`detail.h:36-44`): `|x-y| < EPS*max(|x|,|y|)`,
/// `EPS = 16*f64::EPSILON`. The Wilcoxon block-grouping uses it over abs values,
/// so the comparison is between the (already abs-sorted) raw signed values whose
/// magnitudes are equal.
fn relative_equal(x: f64, y: f64) -> bool {
    if x == 0.0 && y == 0.0 {
        return true;
    }
    let eps = 16.0 * f64::EPSILON;
    (x - y).abs() < eps * x.abs().max(y.abs())
}

/// Standard normal CDF `Phi(x) = (1 + erf(x / sqrt(2))) / 2` (`detail.h:60-64`).
fn phi(x: f64) -> f64 {
    (1.0 + erf(x / std::f64::consts::SQRT_2)) / 2.0
}

/// The error function `erf(x)` via the W. J. Cody rational-Chebyshev
/// approximation (the algorithm libm's `erf` uses; absolute error `~1e-16`). A
/// numeric primitive — NOT the Wilcoxon statistic — so the Don't-Hand-Roll rule
/// (port the Wilcoxon SEMANTICS, no stats crate) is honoured. `erfc` is derived
/// from the same kernels for the tail. Polynomials are evaluated inline by
/// Horner's method (no array indexing — `indexing_slicing` deny-lint clean); the
/// `excessive_precision` lint is intentionally allowed because the literals are
/// the published Cody reference coefficients (trimming them degrades accuracy).
#[allow(clippy::excessive_precision)]
fn erf(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 0.5 {
        // |x| < 0.5: erf(x) = x * P(x^2) / Q(x^2). Q is monic (leading 1).
        let z = x * x;
        let p = [
            1.857_777_061_846_031_526_730e-1,
            3.161_123_743_870_565_596_947e0,
            1.138_641_541_510_501_556_495e2,
            3.774_852_376_853_020_208_137e2,
            3.209_377_589_138_469_472_562e3,
        ];
        let q = [
            1.0,
            2.360_129_095_234_412_093_499e1,
            2.440_246_379_344_441_733_056e2,
            1.282_616_526_077_372_275_645e3,
            2.844_236_833_439_170_622_273e3,
        ];
        x * horner(&p, z) / horner(&q, z)
    } else {
        // |x| >= 0.5: erf(x) = sign(x) * (1 - erfc(|x|)).
        let c = erfc_large(ax);
        if x < 0.0 {
            c - 1.0
        } else {
            1.0 - c
        }
    }
}

/// `erfc(|x|)` for `|x| >= 0.5` via Cody's two rational-Chebyshev regions
/// (`0.5 <= |x| < 4` and `|x| >= 4`), scaled by `exp(-x^2)`. Inline Horner
/// (no array indexing); reference coefficients (precision lint allowed).
#[allow(clippy::excessive_precision)]
fn erfc_large(ax: f64) -> f64 {
    if ax < 4.0 {
        // P/Q coefficients in descending degree, evaluated by explicit Horner
        // steps (no array indexing). Q is monic (leading 1).
        let p = [
            2.153_115_354_744_038_463_96e-8,
            5.641_884_969_886_700_891_30e-1,
            8.883_149_794_388_375_337_98e0,
            6.611_919_063_714_162_948_60e1,
            2.986_351_381_974_001_311_71e2,
            8.819_522_212_417_690_888_43e2,
            1.712_047_612_634_070_625_88e3,
            2.051_078_377_826_071_535_88e3,
            1.231_392_727_220_350_703_47e3,
        ];
        let q = [
            1.0,
            1.574_492_611_070_983_473_06e1,
            1.176_939_508_913_124_993_88e2,
            5.371_811_018_620_098_575_01e2,
            1.621_389_574_566_690_189_53e3,
            3.290_799_235_733_459_627_29e3,
            4.362_619_090_143_247_158_82e3,
            3.439_367_674_143_721_637_46e3,
            1.231_392_727_220_350_703_47e3,
        ];
        let num = horner(&p, ax);
        let den = horner(&q, ax);
        (-ax * ax).exp() * num / den
    } else {
        let z = 1.0 / (ax * ax);
        let p = [
            -1.631_538_713_730_209_785_36e-2,
            -3.051_326_456_376_409_578_67e-1,
            -3.603_448_999_498_044_394_07e-1,
            -1.257_817_261_112_292_462_37e-1,
            -1.608_378_514_874_227_663_05e-2,
            -6.587_491_615_298_378_032_70e-4,
        ];
        let q = [
            1.0,
            2.568_520_192_289_822_421_36e0,
            1.872_952_849_923_460_604_85e0,
            5.279_051_029_514_284_122_19e-1,
            6.051_834_131_244_131_912_29e-2,
            2.335_204_976_268_691_853_99e-3,
        ];
        let r = z * horner(&p, z) / horner(&q, z);
        // 1/sqrt(pi) (FRAC_1_SQRT_PI is still unstable on this toolchain).
        let inv_sqrt_pi = std::f64::consts::FRAC_2_SQRT_PI / 2.0;
        (-ax * ax).exp() / ax * (inv_sqrt_pi + r)
    }
}

/// Evaluate a polynomial whose `coeffs` are in DESCENDING degree by Horner's
/// method, without panicking indexing (the iterator fold visits each coefficient
/// in order). `horner([a,b,c], x) = (a*x + b)*x + c`.
fn horner(coeffs: &[f64], x: f64) -> f64 {
    coeffs.iter().fold(0.0, |acc, &c| acc * x + c)
}
