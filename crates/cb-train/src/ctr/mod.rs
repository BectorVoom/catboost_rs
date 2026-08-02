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
#[path = "ctr_feature.rs"]
pub mod ctr_feature;
#[path = "bake.rs"]
pub mod bake;

#[cfg(test)]
#[path = "online_test.rs"]
mod online_test;
#[cfg(test)]
#[path = "calc_ctr_test.rs"]
mod calc_ctr_test;
#[cfg(test)]
#[path = "final_ctr_test.rs"]
mod final_ctr_test;
#[cfg(test)]
#[path = "mod_test.rs"]
mod mod_test;

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

    /// The number of target borders this CTR type binarizes the target into
    /// (`GetTargetBorderCount`, `ctr_helper.h:34-42`).
    ///
    /// - [`Buckets`](Self::Buckets) keeps the full `target_classes_count`
    ///   (one bucket per class);
    /// - [`BinarizedTargetMeanValue`](Self::BinarizedTargetMeanValue) and
    ///   [`Counter`](Self::Counter) are fixed at `1` — neither binarizes the
    ///   target at all;
    /// - every other type uses `target_classes_count - 1` (the border count
    ///   between classes), saturating at `0` so a degenerate
    ///   `target_classes_count == 0` cannot underflow.
    ///
    /// For binary classification (`target_classes_count == 2`) this is `2` for
    /// `Buckets` and `1` for every other CPU-legal type.
    ///
    /// # Which path uses this
    ///
    /// This mirrors `GetTargetBorderCount` (`ctr_helper.h:34-42`), which upstream
    /// calls on the **ONLINE** path only: for CTR-data allocation sizing
    /// (`online_ctr.cpp:738/741`) and for the CLASS prefix types (`:777`). It is
    /// **NOT** the whole-set bake's divisor — see
    /// [`final_ctr_target_border_count`], which is `targetClassesCount - 1`
    /// unconditionally (`online_ctr.cpp:914`). The two DIFFER for
    /// [`Buckets`](Self::Buckets). Do not substitute one for the other (BUG-BTMV).
    ///
    /// NOTE: this helper currently has **no production caller** — the online
    /// allocation path does not yet route through it. Do not "fix" that by wiring
    /// it into the bake.
    #[must_use]
    pub const fn target_border_count(self, target_classes_count: usize) -> usize {
        match self {
            Self::Buckets => target_classes_count,
            Self::BinarizedTargetMeanValue | Self::Counter => 1,
            Self::Borders | Self::FloatTargetMeanValue | Self::FeatureFreq => {
                target_classes_count.saturating_sub(1)
            }
        }
    }

    /// Whether this CTR type is computable on the CPU training path
    /// (`IsSupportedCtrType(ETaskType::CPU, …)`, `restrictions.h:18-48`).
    ///
    /// Exactly [`FloatTargetMeanValue`](Self::FloatTargetMeanValue) and
    /// [`FeatureFreq`](Self::FeatureFreq) are GPU-only; the other four are legal.
    /// A configuration naming a CPU-illegal type must be rejected with a typed
    /// error before any accumulation happens (SPEC-CTRT-03).
    #[must_use]
    pub const fn is_cpu_supported(self) -> bool {
        match self {
            Self::Borders | Self::Buckets | Self::BinarizedTargetMeanValue | Self::Counter => true,
            Self::FloatTargetMeanValue | Self::FeatureFreq => false,
        }
    }

    /// Whether this CTR type is accumulated as a permutation-dependent
    /// read-before-increment ONLINE PREFIX, rather than as a whole-set tally
    /// (`IsPermutationDependentCtrType`, `ctr_type.cpp:43-56`).
    ///
    /// [`Counter`](Self::Counter) and [`FeatureFreq`](Self::FeatureFreq) are
    /// permutation-INdependent: their bucket totals are counted over the whole
    /// set, so the learn permutation cannot change them. Every other type reads
    /// the prefix strictly before its own document is folded in.
    ///
    /// [`FloatTargetMeanValue`](Self::FloatTargetMeanValue) is classified here
    /// for match totality only — it is CPU-illegal
    /// ([`is_cpu_supported`](Self::is_cpu_supported)) and is rejected upstream of
    /// every caller of this method.
    #[must_use]
    pub const fn is_online_prefix(self) -> bool {
        match self {
            Self::Borders
            | Self::Buckets
            | Self::BinarizedTargetMeanValue
            | Self::FloatTargetMeanValue => true,
            Self::Counter | Self::FeatureFreq => false,
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

/// The target border count used by the **WHOLE-SET** CTR bake
/// (`CalcFinalCtrsImpl`, `online_ctr.cpp:914`).
///
/// ```text
/// online_ctr.cpp:914   int targetBorderCount = targetClassesCount - 1;
/// online_ctr.cpp:920   elem.Add(static_cast<float>(targetClass[z]) / targetBorderCount);
/// ```
///
/// # This is NOT [`ECtrType::target_border_count`]
///
/// Upstream has **two** target-border-count rules and they are not the same
/// function:
///
/// | path | rule | upstream site |
/// |---|---|---|
/// | whole-set bake | `targetClassesCount - 1`, **type-independent** | `online_ctr.cpp:914` (this fn) |
/// | online, mean | `targetClassesCount - 1`, passed as a literal | `online_ctr.cpp:762` ([`online::online_mean_prefix`]) |
/// | online, alloc + class types | `GetTargetBorderCount(ctrInfo, …)`, **type-dependent** | `online_ctr.cpp:738/741, :777` ([`ECtrType::target_border_count`]) |
///
/// `GetTargetBorderCount` is NEVER called inside `CalcFinalCtrsImpl`. The two
/// rules agree for Borders / `BinarizedTargetMeanValue` / Counter at binary
/// classification and **DIFFER for [`Buckets`](ECtrType::Buckets)** (the helper
/// returns `target_classes_count`, this returns `target_classes_count - 1`), so
/// substituting one for the other is undetectable at binclf and wrong at
/// multiclass. **BUG-BTMV was the bake passing `target_classes_count` itself.**
///
/// # The `.max(1)` floor
///
/// [`online::accumulate_online`] rejects `target_border_count == 0` with a typed
/// error, so a single-class corpus would begin returning `CbError::Degenerate`
/// without the floor. Behavior at `target_classes_count == 1` is identical either
/// way (every `target_class` is 0, so every `Sum` is 0). At
/// `target_classes_count == 0` the floor flips `Err(Degenerate)` to `Ok` with a
/// degenerate table — unreachable, the sole production caller hard-codes 2. Same
/// idiom as [`online::online_mean_prefix`]. Upstream divides by 0 here and is
/// undefined.
///
/// # Deliberately NOT shared with `online_mean_prefix`
///
/// The two expressions coincide because upstream's two independent code paths
/// happen to use the same rule (`:762` and `:914`), not because they are one
/// rule. Merging them would hide a future upstream divergence. See PLAN §0.
fn final_ctr_target_border_count(target_classes_count: usize) -> usize {
    target_classes_count.saturating_sub(1).max(1)
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
pub use bake::{bake_ctr_table, BakedCtrData, BakedCtrTable};
pub use ctr_feature::{materialize_ctr_feature, CtrFeatureColumn};
pub use final_ctr::{build_final_ctr, FinalCtrTable};
pub use online::{
    accumulate_online, online_class_prefix, online_class_prefix_column, online_counter_column,
    online_ctr_prefix_binclf, online_mean_prefix, ordered_ctr_per_permutation,
    OnlineCtrAccumulator, OnlineCtrPrefix, OnlineMeanPrefix, OrderedCtrPrefix, TCtrHistory,
    TCtrMeanHistory,
    SIMPLE_CLASSES_COUNT,
};
