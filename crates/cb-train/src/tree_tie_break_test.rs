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

// ── SPEC-OH-06 (PLAN-CHECK MAJOR-6) — the float/one-hot tie-break ───────────

/// SPEC-OH-06 — an EXACT score tie between a float border and a one-hot value
/// must resolve to the FLOAT, because upstream enumerates `AddFloatFeatures`
/// before `AddOneHotFeatures` (`greedy_tensor_search.cpp:1020-1021`) and the
/// argmax is strict first-wins.
///
/// This is the MAJOR-6 guard: the property only holds if BOTH kinds go through
/// the ONE [`crate::tree::select_best`] over ONE flat vector in that order. A
/// second, hand-rolled scan for the one-hot candidates — even one that also uses
/// strict `>` — would compare the two kinds' winners AFTER the fact and could
/// let the one-hot side win a tie.
#[test]
fn float_and_one_hot_candidates_tie_breaks_to_float() {
    use crate::tree::{select_best, AnySplit, LevelCandidate, OneHotSplit, Split};

    // The production enumeration order: floats (feature asc x border asc) THEN
    // one-hots (cat feature asc x bin asc).
    let candidates = vec![
        LevelCandidate::Float(Candidate {
            feature: 0,
            border: 0.5,
            score: 1.0,
        }),
        LevelCandidate::Float(Candidate {
            feature: 1,
            border: 0.25,
            score: 42.0, // the tied maximum, enumerated FIRST
        }),
        LevelCandidate::OneHot {
            feature: 0,
            value: 3,
            score: 42.0, // ties it exactly — must NOT replace it
        },
    ];

    let best = select_best(&candidates).expect("a winner exists");
    assert_eq!(
        best.to_split(),
        AnySplit::Float(Split {
            feature: 1,
            border: 0.25
        }),
        "an exact float/one-hot tie resolves to the FLOAT (enumerated first)"
    );

    // …and a one-hot candidate that strictly EXCEEDS every float still wins, so
    // the rule above is a tie-break, not a float preference.
    let candidates = vec![
        LevelCandidate::Float(Candidate {
            feature: 1,
            border: 0.25,
            score: 42.0,
        }),
        LevelCandidate::OneHot {
            feature: 0,
            value: 3,
            score: 42.5,
        },
    ];
    assert_eq!(
        select_best(&candidates).expect("a winner exists").to_split(),
        AnySplit::OneHot(OneHotSplit {
            feature: 0,
            value: 3
        })
    );

    // Two one-hot candidates that tie resolve to the FIRST (cat feature asc x
    // bin asc) — the same strictness, applied within the kind.
    let candidates = vec![
        LevelCandidate::OneHot {
            feature: 0,
            value: 1,
            score: 7.0,
        },
        LevelCandidate::OneHot {
            feature: 1,
            value: 0,
            score: 7.0,
        },
    ];
    assert_eq!(
        select_best(&candidates).expect("a winner exists").to_split(),
        AnySplit::OneHot(OneHotSplit {
            feature: 0,
            value: 1
        })
    );
}
