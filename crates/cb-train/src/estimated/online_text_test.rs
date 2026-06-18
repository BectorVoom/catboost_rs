//! Unit tests for the online (ordered) text-feature prefix
//! ([`crate::estimated::online_text`]).
//!
//! These pin the read-before-update prefix discipline (D-03): the first document
//! in the permutation sees an EMPTY prefix; each later document sees only its
//! predecessors; the visiting order is the permutation; and a malformed
//! permutation/length yields a typed error, never a panic.

use cb_data::text::text::TText;

use crate::estimated::online_text::{offline_text_features, online_text_prefix, OnlineTextCalcer};

/// Build a `TText` from explicit `(tokenId, count)` pairs via the production
/// constructor (never hand-fabricating the sorted-RLE invariant).
fn ttext(pairs: &[(u32, u32)]) -> TText {
    let mut ids: Vec<u32> = Vec::new();
    for &(token, count) in pairs {
        for _ in 0..count {
            ids.push(token);
        }
    }
    TText::from_token_ids(ids)
}

/// The first document in permutation order sees the EMPTY prefix: for binary
/// NaiveBayes that is the symmetric softmax = 0.5 (the read-before-update
/// no-leakage anchor — doc 0's encoding does not depend on its own label).
#[test]
fn naive_bayes_first_prefix_is_half_identity_perm() {
    let texts = vec![
        ttext(&[(0, 1), (1, 1)]),
        ttext(&[(0, 2)]),
        ttext(&[(1, 3)]),
    ];
    let classes = vec![1usize, 0, 1];
    let perm = vec![0i32, 1, 2];
    let out = online_text_prefix(OnlineTextCalcer::NaiveBayes, &perm, &texts, &classes, 2)
        .expect("nb prefix");
    // Width 1 (binary).
    assert_eq!(out.columns.len(), 1);
    // doc 0 = first in order = empty prefix = 0.5.
    assert!((out.columns[0][0] - 0.5).abs() < 1e-12);
    assert!((out.encoding_in_order[0][0] - 0.5).abs() < 1e-12);
}

/// The encoding is OBJECT-indexed, not permutation-indexed: under a non-identity
/// permutation `[2,0,1]`, the FIRST visited document (object 2) gets the
/// empty-prefix 0.5 stored at `columns[0][2]`, and `encoding_in_order[0]` is
/// object 2's encoding.
#[test]
fn naive_bayes_object_indexed_under_nonidentity_perm() {
    let texts = vec![
        ttext(&[(0, 1)]),
        ttext(&[(1, 1)]),
        ttext(&[(0, 1), (1, 1)]),
    ];
    let classes = vec![1usize, 0, 1];
    let perm = vec![2i32, 0, 1];
    let out = online_text_prefix(OnlineTextCalcer::NaiveBayes, &perm, &texts, &classes, 2)
        .expect("nb prefix");
    // First visited = object 2 -> empty prefix -> 0.5 at columns[0][2].
    assert!((out.columns[0][2] - 0.5).abs() < 1e-12);
    // encoding_in_order is in PERMUTATION order: position 0 = object 2 = 0.5.
    assert!((out.encoding_in_order[0][0] - 0.5).abs() < 1e-12);
    // All three objects got an encoding.
    assert_eq!(out.encoding_in_order.len(), 3);
}

/// BM25 width = numClasses; first document (empty prefix, no term frequencies)
/// scores all-zero.
#[test]
fn bm25_first_prefix_all_zero_width_num_classes() {
    let texts = vec![ttext(&[(0, 1)]), ttext(&[(1, 1)])];
    let classes = vec![1usize, 0];
    let perm = vec![0i32, 1];
    let out =
        online_text_prefix(OnlineTextCalcer::Bm25, &perm, &texts, &classes, 2).expect("bm25 prefix");
    assert_eq!(out.columns.len(), 2);
    // doc 0 first -> empty prefix -> zero scores.
    assert!((out.columns[0][0]).abs() < 1e-12);
    assert!((out.columns[1][0]).abs() < 1e-12);
}

/// A length mismatch (permutation vs texts) is a typed error, not a panic (V5).
#[test]
fn length_mismatch_is_error() {
    let texts = vec![ttext(&[(0, 1)])];
    let classes = vec![1usize];
    let perm = vec![0i32, 1]; // longer than texts/classes
    assert!(
        online_text_prefix(OnlineTextCalcer::NaiveBayes, &perm, &texts, &classes, 2).is_err()
    );
}

/// An out-of-range permutation index is a typed error, not a panic (V5).
#[test]
fn out_of_range_permutation_index_is_error() {
    let texts = vec![ttext(&[(0, 1)]), ttext(&[(1, 1)])];
    let classes = vec![1usize, 0];
    let perm = vec![0i32, 5]; // index 5 out of range
    assert!(
        online_text_prefix(OnlineTextCalcer::Bm25, &perm, &texts, &classes, 2).is_err()
    );
}

/// The prefix grows monotonically: under the identity permutation, the NaiveBayes
/// encoding for later documents reflects MORE accumulated state than earlier ones
/// (the encodings differ from the constant 0.5 once class state diverges) — a
/// smoke check that Update actually advances the prefix between Computes.
#[test]
fn prefix_advances_between_computes() {
    // Two classes with clearly separated vocab so the prefix becomes informative.
    let texts = vec![
        ttext(&[(0, 3)]), // class 1 doc
        ttext(&[(1, 3)]), // class 0 doc
        ttext(&[(0, 3)]), // class 1 doc again
    ];
    let classes = vec![1usize, 0, 1];
    let perm = vec![0i32, 1, 2];
    let out = online_text_prefix(OnlineTextCalcer::NaiveBayes, &perm, &texts, &classes, 2)
        .expect("nb prefix");
    // doc 0: empty prefix -> 0.5.
    assert!((out.columns[0][0] - 0.5).abs() < 1e-12);
    // doc 2: prefix has seen class-1 token 0 and class-0 token 1; its encoding
    // must differ from 0.5 (the prefix is informative).
    assert!((out.columns[0][2] - 0.5).abs() > 1e-6, "doc2={}", out.columns[0][2]);
}

/// The OFFLINE whole-set estimate sees EVERY document (including the query) in
/// the state: unlike the online prefix, doc 0's offline NaiveBayes encoding is
/// NOT the empty-prefix 0.5 — the whole-set state is target-aware over all docs.
#[test]
fn offline_whole_set_sees_every_document() {
    let texts = vec![
        ttext(&[(0, 3)]), // class 1 doc
        ttext(&[(1, 3)]), // class 0 doc
        ttext(&[(0, 3)]), // class 1 doc
    ];
    let classes = vec![1usize, 0, 1];
    let cols = offline_text_features(OnlineTextCalcer::NaiveBayes, &texts, &classes, 2)
        .expect("offline nb");
    assert_eq!(cols.len(), 1, "binary NaiveBayes width 1");
    assert_eq!(cols[0].len(), 3);
    // doc 0 offline is computed against the COMPLETE state -> not the empty-prefix
    // 0.5 (token 0 is strongly class-1 in the whole set).
    assert!((cols[0][0] - 0.5).abs() > 1e-6, "offline doc0={}", cols[0][0]);
}

/// Offline length mismatch is a typed error, not a panic (V5).
#[test]
fn offline_length_mismatch_is_error() {
    let texts = vec![ttext(&[(0, 1)])];
    let classes = vec![1usize, 0]; // longer than texts
    assert!(offline_text_features(OnlineTextCalcer::Bm25, &texts, &classes, 2).is_err());
}
