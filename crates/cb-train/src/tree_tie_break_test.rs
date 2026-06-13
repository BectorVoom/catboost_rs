//! The strict first-wins split tie-break tests (TRAIN-02, Pitfall 1) — the
//! parity landmine. Mounted at `tree::tie_break` so `cargo test -p cb-train
//! tree::tie_break` selects exactly these.
//!
//! Equal-gain candidates MUST resolve to the FIRST one in upstream candidate
//! order (feature index ascending, border ascending) via strict `gain >
//! bestGain`. A `>=` would flip the choice and diverge.

use crate::tree::{select_best_candidate, Candidate};

#[test]
fn first_wins_on_equal_gain() {
    // Three candidates, two of which tie at the max gain. `select_best_candidate`
    // iterates in upstream order and keeps the FIRST max via strict `>`.
    let candidates = vec![
        Candidate {
            feature: 0,
            border: 0.5,
            score: 10.0,
        },
        Candidate {
            feature: 1,
            border: 0.2,
            score: 10.0, // ties the first — must NOT replace it
        },
        Candidate {
            feature: 2,
            border: 0.9,
            score: 7.0,
        },
    ];
    let best = select_best_candidate(&candidates).expect("a candidate must win");
    assert_eq!(best.feature, 0);
    assert_eq!(best.border, 0.5);
}

#[test]
fn ge_would_flip_the_tie() {
    // Demonstrate that swapping strict `>` for `>=` would pick the LATER
    // equal-gain candidate — proving the strict `>` choice is load-bearing.
    let candidates = [
        Candidate {
            feature: 0,
            border: 0.5,
            score: 10.0,
        },
        Candidate {
            feature: 1,
            border: 0.2,
            score: 10.0,
        },
    ];

    let mut best_strict: Option<&Candidate> = None;
    let mut best_gain = f64::NEG_INFINITY;
    for c in &candidates {
        if c.score > best_gain {
            best_gain = c.score;
            best_strict = Some(c);
        }
    }

    let mut best_ge: Option<&Candidate> = None;
    let mut best_gain_ge = f64::NEG_INFINITY;
    for c in &candidates {
        if c.score >= best_gain_ge {
            best_gain_ge = c.score;
            best_ge = Some(c);
        }
    }

    assert_eq!(best_strict.unwrap().feature, 0, "strict > picks the first");
    assert_eq!(best_ge.unwrap().feature, 1, ">= picks the last");
    let prod = select_best_candidate(&candidates).unwrap();
    assert_eq!(prod.feature, best_strict.unwrap().feature);
}
