//! Unit tests for [`super`] tensor-CTR projection enumeration, the combined
//! projection hash fold, and the `max_ctr_complexity` gate (ORD-05). Test names
//! carry the `projection` substring (the module path) so `cargo test -p cb-train
//! projection` selects all of them.

use super::{
    calc_hash, enumerate_projections, fold_cat_hash, max_ctr_complexity_default, TProjection,
};
use cb_data::calc_cat_feature_hash;

#[test]
fn projection_max_ctr_complexity_default_is_four() {
    // cat_feature_options.cpp:231-232 default.
    assert_eq!(max_ctr_complexity_default(), 4);
}

#[test]
fn projection_enumeration_count_complexity_one_only_simple() {
    // max_ctr_complexity == 1 → only SimpleCtrs (one per feature), no pairs.
    let projs = enumerate_projections(3, 1);
    assert_eq!(projs.len(), 3, "3 features → 3 simple CTRs at complexity 1");
    assert!(
        projs.iter().all(TProjection::is_simple),
        "complexity 1 emits only simple (length-1) projections"
    );
    assert_eq!(projs[0].cat_features(), &[0]);
    assert_eq!(projs[1].cat_features(), &[1]);
    assert_eq!(projs[2].cat_features(), &[2]);
}

#[test]
fn projection_enumeration_count_complexity_two_includes_pairs() {
    // max_ctr_complexity == 2 over 3 features → 3 simple + C(3,2)=3 pairs = 6.
    let projs = enumerate_projections(3, 2);
    assert_eq!(projs.len(), 6, "3 simple + 3 pairs at complexity 2");
    let simple = projs.iter().filter(|p| p.is_simple()).count();
    let combos = projs.iter().filter(|p| p.is_combination()).count();
    assert_eq!(simple, 3, "three single-feature projections");
    assert_eq!(combos, 3, "three 2-feature combinations");
    // The pairs are the sorted unordered pairs {0,1},{0,2},{1,2}.
    let pairs: Vec<&[usize]> = projs
        .iter()
        .filter(|p| p.is_combination())
        .map(TProjection::cat_features)
        .collect();
    assert_eq!(pairs, vec![&[0usize, 1][..], &[0, 2][..], &[1, 2][..]]);
}

#[test]
fn projection_enumeration_count_complexity_three_includes_triples() {
    // max_ctr_complexity == 3 over 3 features → 3 simple + 3 pairs + 1 triple = 7.
    let projs = enumerate_projections(3, 3);
    assert_eq!(projs.len(), 7, "3 simple + 3 pairs + 1 triple at complexity 3");
    let triples: Vec<&[usize]> = projs
        .iter()
        .filter(|p| p.full_projection_length() == 3)
        .map(TProjection::cat_features)
        .collect();
    assert_eq!(triples, vec![&[0usize, 1, 2][..]], "the single 3-feature combination");
}

#[test]
fn projection_gate_bounds_length_by_complexity() {
    // The GetFullProjectionLength gate: NO enumerated projection ever exceeds
    // max_ctr_complexity, even when more features are available (the DoS bound,
    // T-05-06-01). 5 features, complexity 2 → max length 2.
    let projs = enumerate_projections(5, 2);
    assert!(
        projs.iter().all(|p| p.full_projection_length() <= 2),
        "no projection longer than max_ctr_complexity"
    );
    // 5 simple + C(5,2)=10 pairs = 15.
    assert_eq!(projs.len(), 15);
}

#[test]
fn projection_complexity_zero_emits_nothing() {
    assert!(enumerate_projections(4, 0).is_empty(), "complexity 0 → no CTRs");
    assert!(enumerate_projections(0, 4).is_empty(), "no features → no CTRs");
}

#[test]
fn projection_from_features_sorts_and_dedups() {
    // AddCatFeature keeps CatFeatures sorted; IsRedundant dedups.
    let p = TProjection::from_features(&[2, 0, 2, 1]);
    assert_eq!(p.cat_features(), &[0, 1, 2], "sorted + de-duplicated");
    assert_eq!(p.full_projection_length(), 3);
    assert!(p.is_combination());
}

#[test]
fn projection_with_added_extends_sorted() {
    let p = TProjection::single(1);
    assert!(p.is_simple());
    let q = p.with_added(0);
    assert_eq!(q.cat_features(), &[0, 1], "extension stays sorted");
    assert!(q.is_combination());
    // Re-adding an existing member is a no-op (dedup).
    let r = q.with_added(1);
    assert_eq!(r.cat_features(), &[0, 1]);
}

#[test]
fn projection_calc_hash_magic_mult_is_exact() {
    // CalcHash(a,b) = MAGIC_MULT * (a + MAGIC_MULT * b), MAGIC_MULT=0x4906ba494954cb65.
    // CalcHash(0, 0) == 0 (the seed of an empty fold).
    assert_eq!(calc_hash(0, 0), 0);
    // CalcHash(0, 1) == MAGIC_MULT * (0 + MAGIC_MULT) == MAGIC_MULT^2 (wrapping).
    let magic: u64 = 0x4906_ba49_4954_cb65;
    assert_eq!(calc_hash(0, 1), magic.wrapping_mul(magic));
}

#[test]
fn projection_fold_cat_hash_sign_extends() {
    // A ui32 hash with the top bit set folds in its SIGN-EXTENDED form
    // ((ui64)(int)) — the upper 32 bits become all-ones, NOT zero.
    let high_bit: u32 = 0x8000_0001;
    let folded = fold_cat_hash(0, high_bit);
    // Expected: CalcHash(0, sign_extend(0x80000001)) = CalcHash(0, 0xFFFFFFFF80000001).
    let signext: u64 = ((high_bit as i32) as i64) as u64;
    assert_eq!(signext, 0xFFFF_FFFF_8000_0001);
    assert_eq!(folded, calc_hash(0, signext));
    // A small hash (top bit clear) zero-extends (== its plain value).
    let small: u32 = 0x0000_0007;
    assert_eq!(fold_cat_hash(0, small), calc_hash(0, 7));
}

#[test]
fn projection_known_two_feature_combined_hash() {
    // A KNOWN 2-feature combined hash value, derived from the committed
    // calc_cat_feature_hash port: feature0 = "a", feature1 = "x".
    //   h0 = calc_cat_feature_hash("a") = 3489901818
    //   h1 = calc_cat_feature_hash("x") = 3510494104
    //   combined = CalcHash(CalcHash(0, signext(h0)), signext(h1)) = 13609484770549027626
    let h0 = calc_cat_feature_hash("a");
    let h1 = calc_cat_feature_hash("x");
    assert_eq!(h0, 3_489_901_818, "anchor cat-hash for \"a\"");
    assert_eq!(h1, 3_510_494_104, "anchor cat-hash for \"x\"");

    let proj = TProjection::from_features(&[0, 1]);
    let feature_hashes = [h0, h1];
    let combined = proj.combined_hash(&feature_hashes);
    assert_eq!(
        combined, 13_609_484_770_549_027_626,
        "known 2-feature combined projection hash"
    );

    // A simple (single-feature) projection over feature0 folds exactly one hash.
    let simple = TProjection::single(0);
    let simple_hash = simple.combined_hash(&feature_hashes);
    assert_eq!(
        simple_hash, 16_670_418_538_649_264_618,
        "known simple (single-feature) projection hash"
    );
    // The combined key differs from either simple key (the tensor keyspace).
    assert_ne!(combined, simple_hash);
}

#[test]
fn projection_combined_hash_order_is_sorted_members() {
    // The fold visits projection members in sorted order; constructing the same
    // projection from differently-ordered inputs yields the same combined hash.
    let h = [10u32, 20, 30];
    let a = TProjection::from_features(&[2, 0]);
    let b = TProjection::from_features(&[0, 2]);
    assert_eq!(a, b, "same sorted member set");
    assert_eq!(a.combined_hash(&h), b.combined_hash(&h));
}

#[test]
fn projection_combined_hash_skips_out_of_range_members() {
    // A member index beyond the supplied feature_hashes is skipped defensively
    // (checked .get) — no panic, no OOB.
    let proj = TProjection::from_features(&[0, 5]);
    let h = [7u32]; // only feature 0 present
    // Folds only feature 0; feature 5 is skipped.
    let expected = TProjection::single(0).combined_hash(&h);
    assert_eq!(proj.combined_hash(&h), expected);
}
