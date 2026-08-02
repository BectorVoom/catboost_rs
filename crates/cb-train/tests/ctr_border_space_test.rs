//! BUG-CTRB / SPEC-CTRB-02 — the training-side and apply-side CTR split
//! decisions must agree for the SAME document.
//!
//! # The contract under test
//!
//! Let `v` be a document's CTR value in APPLY space and `b` the chosen bin-space
//! border index. Then, for every `v` the quantizer can produce:
//!
//! ```text
//!   TRAINING decides:  trunc(v) > b                 (tree.rs:2600, boosting.rs:1938)
//!   APPLY    decides:  v > CtrSplitSpec.border      (apply.rs:189)
//!   => REQUIRED:       CtrSplitSpec.border in (b, b+1), specifically (b+1) - 2^-20
//! ```
//!
//! `bin = trunc(v)` is not assumed: it is `materialize_ctr_feature`'s own
//! quantization step, and this test RE-DERIVES `v` from the column's own
//! `ctr_value` and proves the relation against the real column before using it.
//!
//! # Why not an end-to-end parity assertion
//!
//! `ctr_counter_simple_oracle_test` fails at `max |diff| = 2.687e-1` — a number
//! consistent with a dozen unrelated defects. That is the SECONDARY gate. This
//! test names the disagreeing document, its bin, both borders and both booleans,
//! so the defect is localized rather than merely detected.
//!
//! # Why this gate must exist at all
//!
//! The eleven existing CTR oracles are green by DATA-DEPENDENT COINCIDENCE: none
//! of their documents happens to land exactly on a chosen border. They are a
//! non-regression gate only and cannot prove this defect fixed.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use cb_compute::EScoreFunction;
use cb_data::stringify_int_category;
use cb_train::{
    calc_normalization, ctr_border_count_default, greedy_tensor_search_oblivious_with_ctr,
    materialize_ctr_feature, ECtrType, FeatureMatrix, LevelKind, TProjection,
};

/// PIN — NON-TUNABLE. The prior is fixed at `0.5 / 1.0` because the apply-space
/// reconstruction below is bit-exact ONLY at `norm == 1.0`:
/// `calc_ctr_online_bin` computes `(ctr + shift) / norm * border_count` while
/// this test computes `(ctr + shift) * (border_count / norm)`, and
/// `(a/n)*b != a*(b/n)` in general. `calc_normalization(0.5) == (0.0, 1.0)`, so
/// both reduce to `ctr * border_count` and the premise holds exactly.
///
/// Corpus tuning (rows, category values) is permitted; changing the prior is NOT.
const PRIOR_NUM: f64 = 0.5;
const PRIOR_DENOM: f64 = 1.0;

/// A deterministic categorical corpus. 24 rows over 6 category values, with a
/// mixed binclf target so the per-bucket class counts spread across bins.
fn corpus() -> (Vec<Vec<String>>, Vec<usize>, Vec<i32>) {
    let n = 24usize;
    let col: Vec<String> = (0..n)
        .map(|i| stringify_int_category((i % 6) as i64))
        .collect();
    // Deliberately uneven so different buckets land in different CTR bins.
    let target_class: Vec<usize> = (0..n).map(|i| usize::from((i * 7) % 5 < 2)).collect();
    let permutation: Vec<i32> = (0..n as i32).collect();
    (vec![col], target_class, permutation)
}

/// A float matrix whose single feature splits NOTHING (all-zero values, one
/// border at 0.5), so the CTR candidate must win the level.
fn uninformative_float_matrix(n: usize) -> (Vec<Vec<f32>>, Vec<Vec<f64>>) {
    (vec![vec![0.0_f32; n]], vec![vec![0.5_f64]])
}

/// Materialize the CTR column, grow a depth-1 tree over it, and return
/// `(column, apply-space values, bin-space border, persisted border)`.
fn materialize_and_grow() -> (cb_train::CtrFeatureColumn, Vec<f64>, f64, f64) {
    let (cat_columns, target_class, permutation) = corpus();
    let n = permutation.len();
    let ctr_border_count = ctr_border_count_default();
    let proj = TProjection::from_features(&[0]);

    let col = materialize_ctr_feature(
        &cat_columns,
        &proj,
        &permutation,
        &target_class,
        PRIOR_NUM,
        PRIOR_DENOM,
        ctr_border_count,
        ECtrType::Borders,
        0,
    )
    .expect("materialize_ctr_feature must succeed");

    // Re-derive APPLY-space values from the column's own ctr_value.
    let (shift, norm) = calc_normalization(PRIOR_NUM / PRIOR_DENOM);
    assert_eq!(
        (shift, norm),
        (0.0, 1.0),
        "this reconstruction is bit-exact ONLY at norm == 1.0 ((a/n)*b != a*(b/n)); \
         the prior is pinned at 0.5/1.0 and is NOT tunable — see the PRIOR_NUM doc"
    );
    let scale = ctr_border_count as f64 / norm;
    let v: Vec<f64> = col.ctr_value.iter().map(|&c| (c + shift) * scale).collect();

    // PREMISE: the bin IS the truncated apply-space value. Proved, not assumed.
    for i in 0..n {
        assert_eq!(
            v[i].trunc() as u32,
            col.bins[i],
            "doc {i}: the bin<->value relation `bin = trunc(v)` \
             (ctr_feature.rs quantization step 4) does not hold; this test's \
             premise is invalid — investigate before touching any border"
        );
    }

    let (values, borders) = uninformative_float_matrix(n);
    let matrix = FeatureMatrix::new(&values, &borders);
    let der1: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
    let weight = vec![1.0_f64; n];

    let grown = greedy_tensor_search_oblivious_with_ctr(
        &matrix,
        &[col.clone()],
        ctr_border_count,
        &der1,
        &weight,
        3.0,
        1,
        n,
        0,
        0.0,
        EScoreFunction::Cosine,
        &[],
    )
    .expect("depth-1 CTR-aware growth must succeed");

    assert_eq!(
        grown.ctr_splits.len(),
        1,
        "no CTR split won; this gate would be vacuous"
    );
    assert!(
        grown.splits.is_empty(),
        "a float split won instead of the CTR candidate; this gate would be vacuous"
    );

    let LevelKind::Ctr {
        border: bin_border, ..
    } = grown.level_kinds[0]
    else {
        panic!("the single chosen level must be a CTR level");
    };

    // SPEC-CTRB-03 — THE ONLY POSSIBLE DETECTOR.
    assert_eq!(
        bin_border,
        bin_border.trunc(),
        "LevelKind::Ctr.border must stay in BIN space (integral). Its consumer \
         `assign_leaf_of_averaging` (boosting.rs:1938) compares it against the u32 \
         `col.bins`, so converting it here is arithmetically a NO-OP and NO oracle \
         can catch it — this assertion is the only detector. Units contract."
    );
    assert!(
        bin_border >= 0.0 && bin_border < ctr_border_count as f64,
        "the bin-space border must be a valid bin index"
    );

    let value_border = grown.ctr_splits[0].border;
    (col, v, bin_border, value_border)
}

#[test]
fn train_and_apply_agree_for_a_document_whose_bin_equals_the_border() {
    let (col, v, bin_border, value_border) = materialize_and_grow();

    // ANTI-VACUITY: at least one document must sit strictly inside the chosen
    // border's bin, or this corpus cannot exercise SPEC-CTRB-02 at all.
    let on_border: Vec<usize> = (0..col.bins.len())
        .filter(|&i| f64::from(col.bins[i]) == bin_border && v[i] > bin_border)
        .collect();
    assert!(
        !on_border.is_empty(),
        "no document lands strictly inside the chosen border's bin (border {bin_border}) — \
         this corpus cannot exercise SPEC-CTRB-02; widen the corpus, do NOT weaken \
         the assertion"
    );

    // THE AGREEMENT ASSERTION, over EVERY document.
    for i in 0..col.bins.len() {
        let training = f64::from(col.bins[i]) > bin_border; // tree.rs:2600 / boosting.rs:1938
        let apply = v[i] > value_border; // apply.rs:189
        assert_eq!(
            training, apply,
            "doc {i}: bin {} vs bin-space border {bin_border}: training={training}, \
             but apply value {} vs persisted border {value_border}: apply={apply}. \
             The persisted CtrSplitSpec.border is in BIN space; every consumer reads \
             it as a VALUE-space threshold (SPEC-CTRB-02).",
            col.bins[i], v[i]
        );
    }
}

#[test]
fn persisted_border_brackets_the_chosen_bin() {
    let (_col, _v, bin_border, value_border) = materialize_and_grow();

    // The cheapest possible statement of the whole defect.
    assert!(
        bin_border < value_border && value_border < bin_border + 1.0,
        "the persisted border must lie strictly inside (b, b+1) so that \
         `v > border` reproduces `trunc(v) > b`; got bin_border={bin_border}, \
         persisted border={value_border}"
    );
}

#[test]
fn every_candidate_border_is_a_valid_bin_index() {
    // Characterization: the structure search enumerates candidates as integer
    // bin indices in `0..ctr_border_count`. Green today and after the fix.
    let (_col, _v, bin_border, _value_border) = materialize_and_grow();
    let ctr_border_count = ctr_border_count_default();
    assert!(
        (0..ctr_border_count).any(|b| (b as f64 - bin_border).abs() < f64::EPSILON),
        "the chosen bin-space border {bin_border} must equal some border_idx in \
         0..{ctr_border_count}"
    );
}
