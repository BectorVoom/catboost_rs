//! Unit tests for [`crate::ctr::ctr_feature`]'s per-document CTR-bin quantizer.
//!
//! # DCTR-04 — the `BinarizedTargetMeanValue` `(ctr + shift) / norm` correction
//!
//! Upstream's online quantizer is
//! `CalcCTR(sum, count, prior, shift, norm, borderCount)
//!  = ((sum + prior) / (count + 1) + shift) / norm * borderCount`
//! (`online_ctr.h:128-131`, called from `CalcOnlineCTRMean`,
//! `online_ctr.cpp:483-489`), with `(shift, norm) = CalcNormalization(prior)`
//! (`online_ctr.cpp:102-111` — [`crate::ctr::calc_normalization`]).
//!
//! The `quantize_in_f32` arm of `materialize_ctr_feature` (the BTMV arm) applied
//! only the raw `ctr * borderCount`, dropping `(ctr + shift) / norm`. The
//! correction is a **provable no-op** for every artifact committed to this
//! repository — every prior here is in `[0, 1]` and `calc_normalization` is the
//! identity `(0.0, 1.0)` there (DCTR-05, pinned by
//! `calc_ctr_test::calc_normalization_is_identity_for_every_repo_prior`) — so it
//! is only observable at an **out-of-range prior**, which is exactly what the
//! test below drives.
//!
//! ## Why `prior = 2.0`, and why the column is THREE documents wide
//!
//! At `p = 2.0`, `calc_normalization(2.0) == (0.0, 2.0)` (note: the shift is
//! `-min(0, p)`, i.e. `-0.0` at `p = 0`; comparisons here are IEEE `==`, never
//! `f64::to_bits`).
//!
//! One single-bucket column, classes `[1, 0, 0]`, identity permutation. `norm`
//! below is `calc_normalization`'s `2.0`; `denom` is the CTR denominator
//! `count + 1` — the two are DIFFERENT quantities and the pre-fix code shadowed
//! the former's name with the latter's value.
//!
//! | doc | prefix read | `ctr` | correct `(ctr + shift)/norm · 15` | uncorrected `ctr · 15` | `denom`-conflated `(ctr/denom) · 15` |
//! |---|---|---|---|---|---|
//! | 0 | `sum 0.0, count 0` | `(0+2)/(0+1) = 2.0` | `15.0` ⇒ **15** | `30.0` ⇒ clamp **15** | `(2.0/1.0)·15 = 30` ⇒ clamp **15** |
//! | 1 | `sum 1.0, count 1` | `(1+2)/(1+1) = 1.5` | `11.25` ⇒ **11** | `22.5` ⇒ trunc 22 ⇒ clamp **15** | `(1.5/2.0)·15 = 11.25` ⇒ **11** |
//! | 2 | `sum 1.0, count 2` | `(1+2)/(2+1) = 1.0` | `7.5` ⇒ **7** | `15.0` ⇒ **15** | `(1.0/3.0)·15 = 5.0` ⇒ **5** |
//!
//! - **Document 0 is masked** and cannot discriminate anything: the clamp maps the
//!   uncorrected `30` back onto `15`, which is also the correct value. It is
//!   asserted anyway, as a documented non-detector.
//! - **Document 1** separates correct from uncorrected (`11` vs `15`) but NOT from
//!   the `denom` conflation — at `count = 1` the two denominators coincide at
//!   `2.0`. This is the assertion the plan predicted.
//! - **Document 2** separates all three (`7` vs `15` vs `5`). It exists because the
//!   two-document construction the plan specified cannot police the shadowing trap
//!   the refactor half of DCTR-04 exists to clear.
//!
//! The prefix state each document lands on is **verified by construction** below
//! through `ctr_value` (the raw `calc_ctr_online` prefix value, which DCTR-04 does
//! not touch), never assumed.

use crate::ctr::calc_ctr::calc_normalization;
use crate::ctr::ctr_feature::materialize_ctr_feature;
use crate::ctr::ECtrType;
use crate::projection::TProjection;

/// DCTR-04: the BTMV quantizer must apply `(ctr + shift) / norm` before scaling
/// by the border count, in `f32`, matching upstream's all-`float` `CalcCTR`.
///
/// Driven at `prior = 2.0 / 1.0`, the smallest integral out-of-`[0, 1]` prior at
/// which the normalization is not the identity, over a three-document
/// single-bucket column (see the module header for the per-document table).
#[test]
fn btmv_bin_applies_shift_and_norm_at_out_of_range_prior() {
    // `calc_normalization(2.0)` is the whole reason this prior is observable.
    // IEEE `==` (PartialEq), NOT `f64::to_bits`: `shift` is `-min(0, p)`, so the
    // in-range priors produce `-0.0` and a bit comparison would be wrong.
    assert_eq!(
        calc_normalization(2.0),
        (0.0, 2.0),
        "prior 2.0 is outside [0,1], so (shift, norm) is not the identity"
    );

    // One cat column, three documents sharing ONE bucket, identity permutation.
    // Document 0 carries class 1, so after its read-before-increment fold the
    // bucket history is (Sum = 1.0f32, Count = 1) — the state document 1 reads.
    // Document 1 carries class 0, leaving (Sum = 1.0f32, Count = 2) for document 2.
    let cat_columns = vec![vec!["a".to_owned(), "a".to_owned(), "a".to_owned()]];
    let projection = TProjection::single(0);
    let permutation: Vec<i32> = vec![0, 1, 2];
    let target_class: Vec<usize> = vec![1, 0, 0];
    let ctr_border_count = 15usize;

    let column = materialize_ctr_feature(
        &cat_columns,
        &projection,
        &permutation,
        &target_class,
        2.0,
        1.0,
        ctr_border_count,
        ECtrType::BinarizedTargetMeanValue,
        0,
        &[],
    )
    .expect("materialize a three-document BTMV column at prior 2.0");

    // --- verify the prefix states BY CONSTRUCTION (not by assumption) --------
    // `ctr_value` is `calc_ctr_online(sum, count, prior)` = (sum + 2) / (count + 1)
    // and is untouched by DCTR-04, so it is an independent witness of which
    // document landed on which prefix state.
    let ctr0 = column.ctr_value.first().copied().expect("three ctr values");
    let ctr1 = column.ctr_value.get(1).copied().expect("three ctr values");
    let ctr2 = column.ctr_value.get(2).copied().expect("three ctr values");
    assert_eq!(
        ctr0, 2.0,
        "document 0 must read the empty prefix (sum = 0, count = 0) ⇒ (0 + 2) / (0 + 1)"
    );
    assert_eq!(
        ctr1, 1.5,
        "document 1 must read the (sum = 1.0, count = 1) prefix ⇒ (1 + 2) / (1 + 1); \
         if this fails, widen the column until some document reaches that state — \
         do NOT weaken the bin assertions below to the clamped value"
    );
    assert_eq!(
        ctr2, 1.0,
        "document 2 must read the (sum = 1.0, count = 2) prefix ⇒ (1 + 2) / (2 + 1)"
    );

    // --- the masked case (documented, deliberately non-discriminating) -------
    let bin0 = column.bins.first().copied().expect("three bins");
    assert_eq!(
        bin0, 15,
        "document 0: corrected (2.0 / 2.0) * 15 = 15.0; the uncorrected 30.0 also \
         clamps to 15, so this assertion does NOT discriminate the fix"
    );

    // --- the load-bearing, unclamped case ------------------------------------
    let bin1 = column.bins.get(1).copied().expect("three bins");
    assert_eq!(
        bin1, 11,
        "document 1: ((1.5 + 0.0) / 2.0) * 15 = 11.25 ⇒ bin 11. The uncorrected \
         form gives 1.5 * 15 = 22.5 ⇒ trunc 22 ⇒ clamped to 15, so 11 vs 15 is the \
         unmasked DCTR-04 detector"
    );

    // --- the norm-vs-denom separator ----------------------------------------
    let bin2 = column.bins.get(2).copied().expect("three bins");
    assert_eq!(
        bin2, 7,
        "document 2: ((1.0 + 0.0) / 2.0) * 15 = 7.5 ⇒ bin 7. The uncorrected form \
         gives 1.0 * 15 = 15; dividing by the CTR denominator (count + 1 = 3) \
         instead of calc_normalization's norm (2.0) gives 5. Only the correct form \
         yields 7"
    );
}
