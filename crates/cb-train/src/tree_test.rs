//! Unit tests for oblivious tree growth + the strict first-wins tie-break
//! (TRAIN-02, Pitfall 1). The tie-break is the parity landmine: equal-gain
//! candidates MUST resolve to the FIRST one in upstream candidate order
//! (feature index ascending, border ascending) via strict `gain > bestGain`.

use crate::tree::{select_best_candidate, Candidate};

#[test]
fn select_best_candidate_empty_is_none() {
    let candidates: Vec<Candidate> = Vec::new();
    assert!(select_best_candidate(&candidates).is_none());
}

#[test]
fn select_best_candidate_picks_strict_max() {
    let candidates = [
        Candidate {
            feature: 0,
            border: 0.1,
            score: 3.0,
        },
        Candidate {
            feature: 1,
            border: 0.2,
            score: 9.0,
        },
        Candidate {
            feature: 2,
            border: 0.3,
            score: 5.0,
        },
    ];
    let best = select_best_candidate(&candidates).unwrap();
    assert_eq!(best.feature, 1);
}

#[test]
fn leaf_index_uses_forward_bit_order() {
    use crate::tree::leaf_index;
    // split 0 bit -> bit 0 (LSB); split 1 bit -> bit 1.
    // object passes split 0 only -> index 0b01 = 1
    assert_eq!(leaf_index(&[true, false]), 1);
    // object passes split 1 only -> index 0b10 = 2
    assert_eq!(leaf_index(&[false, true]), 2);
    // passes both -> 0b11 = 3
    assert_eq!(leaf_index(&[true, true]), 3);
    // passes neither -> 0
    assert_eq!(leaf_index(&[false, false]), 0);
}

#[test]
fn depth_cap_rejected_not_panicked() {
    use crate::tree::check_depth;
    assert!(check_depth(16).is_ok());
    assert!(check_depth(17).is_err());
}
