//! Unit tests for the text feature calcers ([`crate::text_calcers`]).
//!
//! BoW is the only target-INDEPENDENT calcer (no online estimation), so its math
//! is a pure function of a digitized document (`cb_data::text::text::TText`) and the
//! calcer's active feature-id set. These tests pin the lockstep two-pointer merge
//! transcribed from upstream `TBagOfWordsCalcer::Compute` (`bow.cpp:7-21`):
//! presence (0/1) per active token id, NOT a count, walked over a TText and active
//! ids BOTH sorted ascending.

use cb_data::text::text::TText;

use crate::text_calcers::{
    bag_of_words_compute, bm25_compute, naive_bayes_compute, Bm25State, NaiveBayesState,
};

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

// ---------------------------------------------------------------------------
// NaiveBayes (multinomial log-prob + softmax, priors 0.5)
// ---------------------------------------------------------------------------

const TOL: f64 = 1e-12;

/// Binary NaiveBayes width = 1 (`BaseFeatureCount(2) = 1`, Pitfall 5).
#[test]
fn naive_bayes_binary_width_is_one() {
    let state = NaiveBayesState::new(2);
    assert_eq!(state.feature_count(), 1);
    let out = naive_bayes_compute(&state, &TText::default()).expect("empty compute");
    assert_eq!(out.len(), 1);
}

/// Multiclass NaiveBayes width = numClasses (`BaseFeatureCount(3) = 3`).
#[test]
fn naive_bayes_multiclass_width_is_num_classes() {
    let state = NaiveBayesState::new(3);
    assert_eq!(state.feature_count(), 3);
    let out = naive_bayes_compute(&state, &TText::default()).expect("empty compute");
    assert_eq!(out.len(), 3);
}

/// The EMPTY-prefix binary case: with no documents seen yet, both classes have
/// identical log-probabilities, so softmax = [0.5, 0.5] and the single emitted
/// feature is 0.5. This is exactly the first per-prefix online encoding the
/// instrumented dump records (doc 0 → 0.5), the load-bearing read-before-update
/// anchor.
#[test]
fn naive_bayes_empty_prefix_binary_is_half() {
    let state = NaiveBayesState::new(2);
    // Any text: with empty per-class state both classes are symmetric.
    let text = ttext_from_pairs(&[(0, 1), (2, 1), (5, 1)]);
    let out = naive_bayes_compute(&state, &text).expect("empty-prefix compute");
    assert!((out[0] - 0.5).abs() < TOL, "got {}", out[0]);
}

/// Hand-derived two-class LogProb + softmax. After UPDATE with one class-0 doc
/// `[(0,2),(2,1)]` and one class-1 doc `[(1,1),(2,2)]`, compute over text
/// `[(0,1),(2,1)]`. We replicate the upstream formula in the test by an
/// independent closed-form computation and assert the production matches.
#[test]
fn naive_bayes_logprob_softmax_hand_derived() {
    let mut state = NaiveBayesState::new(2);
    let doc0 = ttext_from_pairs(&[(0, 2), (2, 1)]); // class 0
    let doc1 = ttext_from_pairs(&[(1, 1), (2, 2)]); // class 1
    state.update(0, &doc0);
    state.update(1, &doc1);

    // Prefix state now:
    //   Frequencies[0] = {0:2, 2:1}, ClassDocs[0]=1, ClassTotalTokens[0]=3
    //   Frequencies[1] = {1:1, 2:2}, ClassDocs[1]=1, ClassTotalTokens[1]=3
    //   NumSeenTokens = |{0,1,2}| = 3
    let prior = 0.5;
    let num_seen = 3.0;
    let seen_prior = 1.0;
    let text = ttext_from_pairs(&[(0, 1), (2, 1)]);

    // Independent reference: LogProb per class (naive_bayesian.cpp:14-44).
    let logprob = |freq: &[(u32, u64)], class_docs: f64, class_tokens: f64| -> f64 {
        let mut value = (class_docs + prior).ln();
        let mut ctc = class_tokens + prior * (num_seen + seen_prior);
        let mut text_len = 0.0;
        for &(tok, cnt) in &[(0u32, 1u64), (2u32, 1u64)] {
            text_len += cnt as f64;
            let found = freq.iter().find(|&&(t, _)| t == tok).map(|&(_, c)| c);
            let mut num = prior;
            match found {
                Some(c) => num += c as f64,
                None => ctc += prior,
            }
            value += (cnt as f64) * num.ln();
        }
        value -= text_len * ctc.ln();
        value
    };
    let lp0 = logprob(&[(0, 2), (2, 1)], 1.0, 3.0);
    let lp1 = logprob(&[(1, 1), (2, 2)], 1.0, 3.0);
    // softmax
    let m = lp0.max(lp1);
    let e0 = (lp0 - m).exp();
    let e1 = (lp1 - m).exp();
    let expected0 = e0 / (e0 + e1);

    let out = naive_bayes_compute(&state, &text).expect("hand-derived compute");
    assert!(
        (out[0] - expected0).abs() < 1e-9,
        "naive_bayes out {} != expected {}",
        out[0],
        expected0
    );
    // The binary feature is the class-0 softmax probability.
    let _ = text; // text reused above
}

/// Softmax sums to 1 across classes for multiclass (internal consistency).
#[test]
fn naive_bayes_multiclass_softmax_normalized() {
    let mut state = NaiveBayesState::new(3);
    state.update(0, &ttext_from_pairs(&[(0, 3)]));
    state.update(1, &ttext_from_pairs(&[(1, 2)]));
    state.update(2, &ttext_from_pairs(&[(2, 5)]));
    let out =
        naive_bayes_compute(&state, &ttext_from_pairs(&[(0, 1), (1, 1)])).expect("ml compute");
    let total: f64 = out.iter().sum();
    assert!((total - 1.0).abs() < 1e-12, "softmax total {total}");
}

/// Zero classes is a typed error, not a panic (V5).
#[test]
fn naive_bayes_zero_classes_is_error() {
    let state = NaiveBayesState::new(0);
    assert!(naive_bayes_compute(&state, &TText::default()).is_err());
}

// ---------------------------------------------------------------------------
// BM25 (class-as-document IDF + saturation, k=1.5, b=0.75, truncate=1e-3)
// ---------------------------------------------------------------------------

/// BM25 width = numClasses always (`BaseFeatureCount = numClasses`, Pitfall 5).
#[test]
fn bm25_width_is_num_classes() {
    let state = Bm25State::new(2);
    assert_eq!(state.feature_count(), 2);
    let out = bm25_compute(&state, &TText::default()).expect("empty compute");
    assert_eq!(out.len(), 2);
}

/// An empty document yields all-zero BM25 scores (no terms to score).
#[test]
fn bm25_empty_text_is_all_zero() {
    let mut state = Bm25State::new(2);
    state.update(0, &ttext_from_pairs(&[(0, 3)]));
    state.update(1, &ttext_from_pairs(&[(1, 2)]));
    let out = bm25_compute(&state, &TText::default()).expect("empty doc");
    assert_eq!(out, vec![0.0, 0.0]);
}

/// Hand-derived BM25 score. After UPDATE with class-0 `[(0,2),(2,1)]` and class-1
/// `[(1,1),(2,3)]`, compute over `[(2,1)]` (token 2 present in both classes).
#[test]
fn bm25_score_hand_derived() {
    let mut state = Bm25State::new(2);
    state.update(0, &ttext_from_pairs(&[(0, 2), (2, 1)]));
    state.update(1, &ttext_from_pairs(&[(1, 1), (2, 3)]));
    // Prefix state:
    //   Frequencies[0]={0:2,2:1}, ClassTotalTokens[0]=3
    //   Frequencies[1]={1:1,2:3}, ClassTotalTokens[1]=4
    //   TotalTokens = 1 (seed) + 3 + 4 = 8
    let k = 1.5_f64;
    let b = 0.75_f64;
    let eps = 1e-3_f64;
    let num_classes = 2.0_f64;
    let total_tokens = 8.0_f64;
    let mean_len = total_tokens / num_classes; // 4.0
    let class_len = [3.0, 4.0];

    // token 2: present in both classes -> nonZero = 2
    let inv = {
        let raw = ((num_classes - 2.0 + 0.5) / (2.0 + 0.5)).ln();
        raw.max(eps)
    };
    let score = |tf: f64, class_length: f64| -> f64 {
        if tf == 0.0 {
            0.0
        } else {
            tf * (k + 1.0) / (tf + k * (1.0 - b + b * mean_len / class_length))
        }
    };
    // tf for token 2: class0=1, class1=3
    let expected0 = inv * score(1.0, class_len[0]);
    let expected1 = inv * score(3.0, class_len[1]);

    let out = bm25_compute(&state, &ttext_from_pairs(&[(2, 1)])).expect("bm25 compute");
    assert!((out[0] - expected0).abs() < 1e-9, "bm25[0] {} != {}", out[0], expected0);
    assert!((out[1] - expected1).abs() < 1e-9, "bm25[1] {} != {}", out[1], expected1);
}

/// `total_tokens` seeds at 1 (`bm25.cpp:58`), so the empty-prefix
/// meanClassLength is 1/numClasses and BM25 compute never divides by zero.
#[test]
fn bm25_empty_prefix_no_divide_by_zero() {
    let state = Bm25State::new(2);
    // No update: ClassTotalTokens=[0,0]; a token absent from all classes ->
    // tf=0 -> Score returns 0, so no division by classLength=0 occurs.
    let out = bm25_compute(&state, &ttext_from_pairs(&[(0, 1)])).expect("empty-prefix bm25");
    assert_eq!(out, vec![0.0, 0.0]);
}

/// Zero classes is a typed error, not a panic (V5).
#[test]
fn bm25_zero_classes_is_error() {
    let state = Bm25State::new(0);
    assert!(bm25_compute(&state, &TText::default()).is_err());
}
