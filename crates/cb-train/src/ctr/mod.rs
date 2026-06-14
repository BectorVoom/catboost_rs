//! Plain-mode (whole-set) CTR target statistics (ORD-03, D-06) — all six CTR
//! types computed as a whole-set statistic with no per-object prefix, the math
//! locked BEFORE the ordered (read-before-increment) per-object prefix of
//! Wave 5 (D-06 key isolation).
//!
//! # Module layout (RESEARCH Recommended Structure)
//!
//! - [`online`] — whole-set per-bucket class-count / target-sum accumulation
//!   (`online_ctr.cpp` read-before-increment template, run to completion).
//! - [`calc_ctr`] — the online (`+1` denom) and inference (`+PriorDenom`)
//!   CTR-value quantizers as SEPARATE functions (Pitfall 1).
//! - [`final_ctr`] — the whole-set `CalcFinalCtrsImpl` table build, one table per
//!   CTR type, with the Counter (max bucket) vs FeatureFreq (total sample)
//!   denominator distinction (Pitfall 4).
//!
//! # The six CTR types (D-07)
//!
//! [`ECtrType`] mirrors the upstream `ECtrType` (`ctr_type.h`) discriminants
//! bit-for-bit (`Borders=0 … FeatureFreq=5`). It is defined HERE (not imported
//! from `cb-model`'s generated bindings) because `cb-train` is BELOW `cb-model`
//! in the dependency graph — `cb-model` depends on `cb-train`, so `cb-train`
//! cannot depend on `cb-model`. The two enums share the same i8 discriminants, so
//! the `cb-model` serde maps losslessly between them.
//!
//! # Categorical hashing (D Carried-Forward)
//!
//! Every CTR bucket identity is keyed on [`cb_data::calc_cat_feature_hash`] +
//! [`cb_data::PerfectHash`] — NEVER a model's `ctr_data` hash_map (RESEARCH
//! Anti-Pattern; STATE.md Plan 02-04 Rule-1 fix). See [`online::accumulate_online`].

#[path = "online.rs"]
pub mod online;
#[path = "calc_ctr.rs"]
pub mod calc_ctr;
#[path = "final_ctr.rs"]
pub mod final_ctr;

#[cfg(test)]
#[path = "online_test.rs"]
mod online_test;
#[cfg(test)]
#[path = "calc_ctr_test.rs"]
mod calc_ctr_test;
#[cfg(test)]
#[path = "final_ctr_test.rs"]
mod final_ctr_test;

/// The six CTR types (`ECtrType`, `ctr_type.h`), mirroring the upstream i8
/// discriminants bit-for-bit (the SAME values as `cb-model`'s generated
/// `ECtrType` — `cb-train` is below `cb-model` in the dependency graph, so the
/// enum is duplicated here, not imported, and the two map losslessly).
///
/// - [`Borders`](ECtrType::Borders) / [`Buckets`](ECtrType::Buckets): per-bucket
///   class counts (`N[0]`/`N[1]` for binclf).
/// - [`BinarizedTargetMeanValue`](ECtrType::BinarizedTargetMeanValue): mean of
///   `targetClass / targetBorderCount`.
/// - [`FloatTargetMeanValue`](ECtrType::FloatTargetMeanValue): mean of the raw
///   target (final-CTR path ONLY, Pitfall 5).
/// - [`Counter`](ECtrType::Counter): bucket total / MAX bucket total.
/// - [`FeatureFreq`](ECtrType::FeatureFreq): bucket total / total sample count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i8)]
pub enum ECtrType {
    /// Class-count borders CTR (`= 0`): `N[1] / (N[0] + N[1])` for binclf.
    Borders = 0,
    /// Class-count buckets CTR (`= 1`): per-class bucket counts.
    Buckets = 1,
    /// Binarized-target mean (`= 2`): `Add(targetClass / targetBorderCount)`.
    BinarizedTargetMeanValue = 2,
    /// Raw float-target mean (`= 3`): `Add(target)` — final-CTR path only.
    FloatTargetMeanValue = 3,
    /// Counter CTR (`= 4`): bucket total / MAX bucket total (Pitfall 4).
    Counter = 4,
    /// Feature-frequency CTR (`= 5`): bucket total / total sample count.
    FeatureFreq = 5,
}

impl ECtrType {
    /// The upstream i8 discriminant (`ctr_type.h`), shared with `cb-model`'s
    /// generated `ECtrType`. Lets the `cb-model` serde map between the two.
    #[must_use]
    pub fn as_i8(self) -> i8 {
        self as i8
    }

    /// Reconstruct from the upstream i8 discriminant; `None` for an unknown
    /// value (checked — no panic).
    #[must_use]
    pub fn from_i8(value: i8) -> Option<Self> {
        match value {
            0 => Some(Self::Borders),
            1 => Some(Self::Buckets),
            2 => Some(Self::BinarizedTargetMeanValue),
            3 => Some(Self::FloatTargetMeanValue),
            4 => Some(Self::Counter),
            5 => Some(Self::FeatureFreq),
            _ => None,
        }
    }

    /// The default priors for this CTR type (`cat_feature_options.cpp:118-138`):
    /// Borders/Buckets/BinarizedTargetMeanValue get THREE unit-denominator
    /// priors `{0/1, 0.5/1, 1/1}`; Counter/FeatureFreq/FloatTargetMeanValue get a
    /// single `{0/1}` prior (RESEARCH "Default priors per CTR type").
    #[must_use]
    pub fn default_priors(self) -> Vec<calc_ctr::Prior> {
        match self {
            Self::Borders | Self::Buckets | Self::BinarizedTargetMeanValue => vec![
                calc_ctr::Prior::unit(0.0),
                calc_ctr::Prior::unit(0.5),
                calc_ctr::Prior::unit(1.0),
            ],
            Self::FloatTargetMeanValue | Self::Counter | Self::FeatureFreq => {
                vec![calc_ctr::Prior::unit(0.0)]
            }
        }
    }
}

/// The `CounterCalcMethod` (`cat_feature_options.cpp:234`): whether the Counter
/// denominator includes test documents. The default is [`SkipTest`](CounterCalcMethod::SkipTest)
/// (Pitfall 4 — pinned EXPLICITLY, never auto). In the whole-learn-set build of
/// this wave there are no test documents, so the flag does not change the counts;
/// it is recorded on the model for the tensor-CTR path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CounterCalcMethod {
    /// Skip test documents in the Counter denominator (the upstream default).
    #[default]
    SkipTest,
    /// Include the full dataset (learn + test) in the Counter denominator.
    Full,
}

pub use calc_ctr::{
    calc_ctr_inference, calc_ctr_online, calc_ctr_online_bin, calc_normalization, Prior,
};
pub use final_ctr::{build_final_ctr, FinalCtrTable};
pub use online::{
    accumulate_online, online_ctr_prefix_binclf, ordered_ctr_per_permutation,
    OnlineCtrAccumulator, OnlineCtrPrefix, OrderedCtrPrefix, TCtrHistory, TCtrMeanHistory,
    SIMPLE_CLASSES_COUNT,
};
