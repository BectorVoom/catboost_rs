//! Unit tests for the text feature calcers ([`crate::text_calcers`]).
//!
//! BoW is the only target-INDEPENDENT calcer (no online estimation), so its math
//! is a pure function of a digitized document (`cb_data::text::text::TText`) and the
//! calcer's active feature-id set. These tests pin the lockstep two-pointer merge
//! transcribed from upstream `TBagOfWordsCalcer::Compute` (`bow.cpp:7-21`):
//! presence (0/1) per active token id, NOT a count, walked over a TText and active
//! ids BOTH sorted ascending.

use cb_data::text::text::TText;

use crate::text_calcers::bag_of_words_compute;

/// Helper: build a `TText` from explicit `(tokenId, count)` pairs by replaying
/// the raw token-id multiset through the production `TText::from_token_ids`
/// constructor (so the test never hand-fabricates the sorted-RLE invariant).
fn ttext_from_pairs(pairs: &[(u32, u32)]) -> TText {
    let mut ids: Vec<u32> = Vec::new();
    for &(token, count) in pairs {
        for _ in 0..count {
            ids.push(token);
        }
    }
    TText::from_token_ids(ids)
}

/// The plan's worked example: `text=[(1,2),(3,1)]`, `active_ids=[0,1,2,3]` yields
/// `[0,1,0,1]` — presence (token present at id 1 and 3), not the count.
#[test]
fn bow_presence_not_count_over_active_ids() {
    let text = ttext_from_pairs(&[(1, 2), (3, 1)]);
    let active_ids = [0u32, 1, 2, 3];
    let out = bag_of_words_compute(&text, &active_ids).expect("well-formed BoW compute");
    assert_eq!(out, vec![0.0, 1.0, 0.0, 1.0]);
}

/// Empty text yields an all-zero vector of width = active_ids.len() (the
/// document contains none of the active tokens). No panic on empty input (V5).
#[test]
fn bow_empty_text_is_all_zero_width_active_ids() {
    let text = TText::default();
    let active_ids = [0u32, 1, 2, 3, 4];
    let out = bag_of_words_compute(&text, &active_ids).expect("empty text is well-formed");
    assert_eq!(out, vec![0.0; 5]);
}

/// Zero active tokens yields an empty output regardless of the text (FeatureCount
/// == 0). No panic.
#[test]
fn bow_no_active_tokens_is_empty_output() {
    let text = ttext_from_pairs(&[(0, 1), (5, 3)]);
    let out = bag_of_words_compute(&text, &[]).expect("zero active tokens is well-formed");
    assert!(out.is_empty());
}

/// FeatureCount == active_ids.len(): every active id maps to exactly one output
/// cell, in active-id order. Here all active ids are present.
#[test]
fn bow_feature_count_equals_active_id_count_all_present() {
    let text = ttext_from_pairs(&[(2, 1), (5, 1), (9, 1)]);
    let active_ids = [2u32, 5, 9];
    let out = bag_of_words_compute(&text, &active_ids).expect("all present");
    assert_eq!(out.len(), active_ids.len());
    assert_eq!(out, vec![1.0, 1.0, 1.0]);
}

/// The two-pointer merge must advance the text iterator past tokens BELOW the
/// current active id (a token present in the doc but absent from the active set
/// is skipped, not mismatched). `text` carries token 0 and 7 which are NOT active;
/// the active ids 1,3,5 are absent from the doc -> all zero; active id 7 present.
#[test]
fn bow_lockstep_skips_text_tokens_below_active_id() {
    // doc tokens: 0,4,7 ; active: 1,3,5,7  -> only 7 present.
    let text = ttext_from_pairs(&[(0, 1), (4, 1), (7, 1)]);
    let active_ids = [1u32, 3, 5, 7];
    let out = bag_of_words_compute(&text, &active_ids).expect("lockstep merge");
    assert_eq!(out, vec![0.0, 0.0, 0.0, 1.0]);
}

/// A document token GREATER than the current active id yields 0 for that active
/// id (the active token is absent), and the merge then continues. active id 4 is
/// absent (doc jumps 2 -> 6), so its cell is 0; 2 and 6 are present.
#[test]
fn bow_active_id_absent_when_text_overshoots() {
    let text = ttext_from_pairs(&[(2, 1), (6, 1)]);
    let active_ids = [2u32, 4, 6];
    let out = bag_of_words_compute(&text, &active_ids).expect("overshoot merge");
    assert_eq!(out, vec![1.0, 0.0, 1.0]);
}

/// Counts > 1 still emit a single presence `1.0` (binary, never the count).
#[test]
fn bow_repeated_token_is_still_binary_presence() {
    let text = ttext_from_pairs(&[(3, 9)]);
    let out = bag_of_words_compute(&text, &[3u32]).expect("repeat -> presence");
    assert_eq!(out, vec![1.0]);
}
