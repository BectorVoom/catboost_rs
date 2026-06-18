//! Online (ordered) text-feature estimation — the per-fold read-before-update
//! prefix loop for the two target-AWARE text calcers, NaiveBayes and BM25 (D-03
//! leakage control).
//!
//! # The read-before-update prefix (D-03, mirrors `ctr/online.rs`)
//!
//! NaiveBayes and BM25 are `IOnlineFeatureEstimator`s: their per-document
//! encoding is computed from the class-frequency state accumulated from EARLIER
//! documents in the learn permutation ONLY, THEN the current document's
//! label/text updates that state. A document's encoding therefore never sees its
//! own label — the no-leakage property. This is the exact loop upstream runs in
//! `TTextBaseEstimator::ComputeOnlineFeatures`
//! (`base_text_feature_estimator.h:74-79`):
//!
//! ```text
//! for (line : learnPermutation) {
//!     text = ds.GetText(line);
//!     Compute(featureCalcer, text, line, ...);   // READ prefix state
//!     calcerVisitor.Update(target.Classes[line], text, &featureCalcer);  // THEN write
//! }
//! ```
//!
//! The Rust analog already exists for CTRs (`cb-train::ctr::online`,
//! `online_ctr_prefix_binclf` — "READ the prefix counts BEFORE incrementing").
//! This module is the SAME discipline for the text calcers; the visiting order is
//! the fold's learn permutation (`fold.rs`, the SAME permutation CTRs use — never
//! a freshly generated one, RESEARCH "Don't Hand-Roll").
//!
//! # Output is OBJECT-indexed
//!
//! Upstream writes `learnFeatures[f * samplesCount + line]` — the feature value
//! for document `line` is stored at its OBJECT index, not its permutation
//! position (`base_text_feature_estimator.h:77` + the `TOutputFloatIterator`
//! seeded at `features.data() + docId`). The columns this module returns are
//! therefore object-indexed (`columns[f][doc]`), ready to append to the
//! float-feature layout exactly like the offline BoW columns.
//!
//! # Parity discipline
//!
//! The calcer compute math (`cb_compute::naive_bayes_compute` /
//! `bm25_compute`) owns the `sum_f64`-routed float reductions; the integer
//! class-frequency state ([`cb_compute::NaiveBayesState`] / [`Bm25State`]) is
//! EXACT integer accumulation. This module only sequences Compute-then-Update and
//! narrows the `f64` calcer outputs to the `f32` estimated-feature storage type
//! (RESEARCH Pitfall 6). Checked `.get(..)` only; no `unwrap`/`expect`/panic/raw
//! index; no `anyhow`.

use cb_compute::{bm25_compute, naive_bayes_compute, Bm25State, NaiveBayesState};
use cb_core::{CbError, CbResult};
use cb_data::text::text::TText;

/// Compute the OFFLINE (whole-set) text-feature encodings for one calcer
/// (`TTextBaseEstimator::EstimateFeatureCalcer` + `Calc`,
/// `base_text_feature_estimator.h:118-162`): accumulate EVERY learn document's
/// class/text into the calcer state, then compute each document's encoding
/// against that COMPLETE state.
///
/// # When the offline (vs online) estimate is used
///
/// For `boosting_type=Plain`, the estimated features fed to the TREE SPLITS are
/// the offline whole-set estimate (`ComputeFeatures`, not `ComputeOnlineFeatures`);
/// the online read-before-update prefix ([`online_text_prefix`]) feeds the ordered
/// boosting leaf-approximation path. The FEAT-01 fixtures are Plain, so their
/// per-stage tree outputs are gated against this offline estimate. The offline
/// estimate is still target-AWARE (it accumulates labels) but is NOT
/// leakage-controlled — it is the whole-set estimate upstream builds the Plain
/// tree on.
///
/// Returns OBJECT-indexed columns (`columns[f][doc]`), `f32`-valued. Width = the
/// calcer's `feature_count()`.
///
/// # Errors
///
/// [`CbError::Degenerate`] if `texts` / `classes` length-mismatch, or the calcer
/// compute fails (zero classes).
pub fn offline_text_features(
    calcer: OnlineTextCalcer,
    texts: &[TText],
    classes: &[usize],
    num_classes: usize,
) -> CbResult<Vec<Vec<f32>>> {
    let n = texts.len();
    if num_classes == 0 {
        // Without this guard, an empty `texts` would return a width-1 all-zero
        // column set with no error (a silently-wrong shape for a zero-class
        // request), matching the bm25_compute/naive_bayes_compute precondition (WR-03).
        return Err(CbError::Degenerate(
            "offline_text_features: zero classes".to_owned(),
        ));
    }
    if classes.len() != n {
        return Err(CbError::Degenerate(
            "offline_text_features: texts / classes length mismatch".to_owned(),
        ));
    }

    let width = match calcer {
        OnlineTextCalcer::NaiveBayes => NaiveBayesState::new(num_classes).feature_count(),
        OnlineTextCalcer::Bm25 => Bm25State::new(num_classes).feature_count(),
    };
    let mut columns: Vec<Vec<f32>> = vec![vec![0.0_f32; n]; width];

    match calcer {
        OnlineTextCalcer::NaiveBayes => {
            // Accumulate the WHOLE learn set (base_text_feature_estimator.h:156-159).
            let mut state = NaiveBayesState::new(num_classes);
            for (doc, text) in texts.iter().enumerate() {
                let Some(&class) = classes.get(doc) else {
                    continue;
                };
                state.update(class, text);
            }
            // Compute each document against the complete state.
            for (doc, text) in texts.iter().enumerate() {
                let encoding = naive_bayes_compute(&state, text)?;
                for (f, &v) in encoding.iter().enumerate() {
                    if let Some(col) = columns.get_mut(f) {
                        if let Some(slot) = col.get_mut(doc) {
                            *slot = v as f32;
                        }
                    }
                }
            }
        }
        OnlineTextCalcer::Bm25 => {
            let mut state = Bm25State::new(num_classes);
            for (doc, text) in texts.iter().enumerate() {
                let Some(&class) = classes.get(doc) else {
                    continue;
                };
                state.update(class, text);
            }
            for (doc, text) in texts.iter().enumerate() {
                let encoding = bm25_compute(&state, text)?;
                for (f, &v) in encoding.iter().enumerate() {
                    if let Some(col) = columns.get_mut(f) {
                        if let Some(slot) = col.get_mut(doc) {
                            *slot = v as f32;
                        }
                    }
                }
            }
        }
    }

    Ok(columns)
}

/// Which target-aware online text calcer to estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnlineTextCalcer {
    /// Multinomial NaiveBayes (`naive_bayesian.cpp`): width
    /// `numClasses > 2 ? numClasses : 1`.
    NaiveBayes,
    /// BM25 (`bm25.cpp`): width `numClasses`.
    Bm25,
}

/// The object-indexed online-text estimated feature columns plus the
/// permutation-order per-prefix encoding trace (the instrumented online-order
/// anchor).
#[derive(Debug, Clone, PartialEq)]
pub struct OnlineTextPrefix {
    /// One estimated feature column per active calcer feature, OBJECT-indexed
    /// (`columns[f][doc]`), `f32`-valued (estimated features are `f32`, Pitfall
    /// 6). Width = the calcer's `feature_count()`.
    pub columns: Vec<Vec<f32>>,
    /// The per-document encoding in PERMUTATION order (`encoding_in_order[p]` is
    /// the full feature vector computed for the document at learn-order position
    /// `p`, BEFORE its own update). This is the exact sequence the instrumented
    /// `calcer_encoding` dump records over the `online_order` permutation — the
    /// per-prefix oracle anchor that localizes any leakage-order bug.
    pub encoding_in_order: Vec<Vec<f64>>,
}

/// Compute the online (ordered) text-feature encodings for one calcer over the
/// learn `permutation` with the read-before-update prefix (D-03).
///
/// - `permutation[p]` is the object index at learn-order position `p` (the fold's
///   `Fold::permutation`, NOT a fresh one).
/// - `texts[doc]` is object `doc`'s digitized [`TText`] (against the Word
///   dictionary; built once offline).
/// - `classes[doc]` is object `doc`'s binarized class in `[0, num_classes)`.
/// - `num_classes` is the number of target classes (2 for the binclf fixtures).
///
/// For each `p` in `0..permutation.len()`: read `doc = permutation[p]`, COMPUTE
/// the calcer output from the prefix state (documents at positions `< p`), store
/// it object-indexed at `columns[f][doc]` and in `encoding_in_order[p]`, THEN
/// UPDATE the state with `(classes[doc], texts[doc])`.
///
/// # Errors
///
/// [`CbError::Degenerate`] if `texts` / `classes` are shorter than the
/// permutation implies, a permutation index is out of range, or the calcer
/// compute fails (zero classes).
pub fn online_text_prefix(
    calcer: OnlineTextCalcer,
    permutation: &[i32],
    texts: &[TText],
    classes: &[usize],
    num_classes: usize,
) -> CbResult<OnlineTextPrefix> {
    let n = permutation.len();
    if texts.len() != n || classes.len() != n {
        return Err(CbError::Degenerate(
            "online_text_prefix: permutation / texts / classes length mismatch".to_owned(),
        ));
    }

    // The output width is fixed by the calcer + numClasses (BaseFeatureCount).
    let width = match calcer {
        OnlineTextCalcer::NaiveBayes => NaiveBayesState::new(num_classes).feature_count(),
        OnlineTextCalcer::Bm25 => Bm25State::new(num_classes).feature_count(),
    };

    let mut columns: Vec<Vec<f32>> = vec![vec![0.0_f32; n]; width];
    let mut encoding_in_order: Vec<Vec<f64>> = Vec::with_capacity(n);

    // Carry the accumulating prefix state for whichever calcer is active. Exactly
    // one of these is `Some`; the other stays `None`. (Two narrow Options keep the
    // state strongly typed without a trait object.)
    let mut nb_state = match calcer {
        OnlineTextCalcer::NaiveBayes => Some(NaiveBayesState::new(num_classes)),
        OnlineTextCalcer::Bm25 => None,
    };
    let mut bm_state = match calcer {
        OnlineTextCalcer::Bm25 => Some(Bm25State::new(num_classes)),
        OnlineTextCalcer::NaiveBayes => None,
    };

    for &doc_i in permutation {
        let doc = doc_i as usize;
        let Some(text) = texts.get(doc) else {
            return Err(CbError::Degenerate(
                "online_text_prefix: permutation index out of range for texts".to_owned(),
            ));
        };
        let Some(&class) = classes.get(doc) else {
            return Err(CbError::Degenerate(
                "online_text_prefix: permutation index out of range for classes".to_owned(),
            ));
        };

        // COMPUTE from the prefix state (read-before-update, D-03).
        let encoding: Vec<f64> = match calcer {
            OnlineTextCalcer::NaiveBayes => {
                let state = nb_state.as_ref().ok_or_else(|| {
                    CbError::Degenerate("online_text_prefix: missing NaiveBayes state".to_owned())
                })?;
                naive_bayes_compute(state, text)?
            }
            OnlineTextCalcer::Bm25 => {
                let state = bm_state.as_ref().ok_or_else(|| {
                    CbError::Degenerate("online_text_prefix: missing BM25 state".to_owned())
                })?;
                bm25_compute(state, text)?
            }
        };

        // Scatter the encoding OBJECT-indexed into the columns.
        for (f, &v) in encoding.iter().enumerate() {
            if let Some(col) = columns.get_mut(f) {
                if let Some(slot) = col.get_mut(doc) {
                    *slot = v as f32;
                }
            }
        }
        encoding_in_order.push(encoding);

        // THEN UPDATE with this document's label/text (learn set).
        match calcer {
            OnlineTextCalcer::NaiveBayes => {
                if let Some(state) = nb_state.as_mut() {
                    state.update(class, text);
                }
            }
            OnlineTextCalcer::Bm25 => {
                if let Some(state) = bm_state.as_mut() {
                    state.update(class, text);
                }
            }
        }
    }

    Ok(OnlineTextPrefix {
        columns,
        encoding_in_order,
    })
}
