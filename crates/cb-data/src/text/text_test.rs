//! Unit tests for the `TText` sorted-RLE data type (`text.rs`).
//!
//! Covers the upstream `TText(TVector<ui32>&&)` sort-then-RLE-collapse
//! constructor (`text.h:169-179`). Source/test separation (INFRA-06).

use super::text::{TText, TokenCount};

fn pairs(t: &TText) -> Vec<(u32, u32)> {
    t.pairs().iter().map(|p| (p.token, p.count)).collect()
}

#[test]
fn empty_input_yields_empty_ttext() {
    let t = TText::from_token_ids(Vec::new());
    assert!(t.is_empty());
    assert_eq!(t.len(), 0);
    assert!(t.pairs().is_empty());
}

#[test]
fn single_token_yields_one_pair_with_count_one() {
    let t = TText::from_token_ids(vec![7]);
    assert_eq!(pairs(&t), vec![(7, 1)]);
}

#[test]
fn sorts_ascending_then_rle_collapses_duplicates() {
    // Behavior contract: [3,1,3,1] -> [(1,2),(3,2)].
    let t = TText::from_token_ids(vec![3, 1, 3, 1]);
    assert_eq!(pairs(&t), vec![(1, 2), (3, 2)]);
}

#[test]
fn unsorted_input_is_sorted_before_rle() {
    // [5,1,5,1,1] -> sort [1,1,1,5,5] -> [(1,3),(5,2)].
    let t = TText::from_token_ids(vec![5, 1, 5, 1, 1]);
    assert_eq!(pairs(&t), vec![(1, 3), (5, 2)]);
}

#[test]
fn matches_fixture_ttext_for_bad_bad_awful() {
    // From fixtures/text_tokenizer/ttext.json (NaiveBayes path, Word dict):
    // document "bad bad awful" -> ids [0,0,2] -> TText [(0,2),(2,1)].
    let t = TText::from_token_ids(vec![0, 0, 2]);
    assert_eq!(pairs(&t), vec![(0, 2), (2, 1)]);
}

#[test]
fn pairs_are_always_ascending_by_token() {
    let t = TText::from_token_ids(vec![9, 2, 5, 2, 9, 0]);
    let toks: Vec<u32> = t.pairs().iter().map(|p| p.token).collect();
    let mut sorted = toks.clone();
    sorted.sort_unstable();
    assert_eq!(toks, sorted);
}

#[test]
fn token_count_is_copy_and_eq() {
    let a = TokenCount {
        token: 1,
        count: 2,
    };
    let b = a;
    assert_eq!(a, b);
}
