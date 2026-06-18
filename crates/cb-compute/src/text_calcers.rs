//! Text feature calcers — the pure numeric primitives that turn a digitized
//! document (`cb_data::text::text::TText`) into estimated float feature columns.
//!
//! # Source of truth (D-04)
//!
//! This module transcribes the upstream CatBoost 1.2.10 text-feature calcer math
//! VERBATIM. The first (and only target-INDEPENDENT) calcer is BoW
//! (`TBagOfWordsCalcer`), transcribed from
//! `catboost-master/catboost/private/libs/text_features/bow.cpp:7-21`. The
//! remaining calcers (NaiveBayes, BM25) are target-aware online estimators landed
//! in later plans; they live behind the `cb-train` ordered-prefix seam, not here.
//!
//! # Summation routing (D-04 / D-08)
//!
//! Every parity-critical float SUM in this module routes through
//! [`cb_core::sum_f64`] in canonical order (the convention all `cb-compute` math
//! modules follow, see `score.rs`). BoW itself performs NO float reduction — it
//! emits a binary presence (0/1) per active token id, so there is no `sum_f64`
//! call in [`bag_of_words_compute`]; the convention is documented here so the
//! NaiveBayes/BM25 arms added later stay consistent. The output cell type is
//! `f32`-valued (estimated features are `f32`), represented as `f64` in this
//! pure-math layer and narrowed at the storage boundary (RESEARCH Pitfall 6).
//!
//! # Robustness (V5 / INFRA-02)
//!
//! Empty text yields an all-zero vector of width = `active_feature_ids.len()`;
//! zero active ids yield an empty vector. All access is via checked iteration; no
//! `unwrap`/`expect`/`panic`/raw index appears in this module (CLAUDE.md library
//! discipline). The lockstep walk cannot panic on any input.

use cb_core::CbResult;
use cb_data::text::text::TText;

/// Bag-of-Words binary-presence encoding — the target-INDEPENDENT calcer.
///
/// Transcribed VERBATIM from `TBagOfWordsCalcer::Compute` (`bow.cpp:7-21`): walk
/// the digitized document `text` and the calcer's `active_feature_ids` BOTH
/// sorted ascending by token id, in lockstep (a two-pointer merge). For each
/// active token id, emit `1.0` if that token id is present in the document and
/// `0.0` otherwise — presence, NOT the occurrence count.
///
/// The output width equals `active_feature_ids.len()` (upstream `FeatureCount()`
/// == number of active feature indices; for an untrimmed BoW calcer this is the
/// dictionary `NumTokens`). The output is in `active_feature_ids` order, one cell
/// per active id.
///
/// # Lockstep order is load-bearing (D-01 / SC-1)
///
/// The merge relies on BOTH inputs being sorted ascending by token id. A `TText`
/// is always sorted-ascending by construction
/// (`cb_data::text::text::TText::from_token_ids`); the caller MUST pass
/// `active_feature_ids` sorted ascending (the upstream `ActiveFeatureIndices` are
/// `Iota(0..NumTokens)` or a sorted `TrimFeatures` subset). A different ordering
/// silently shifts every output cell.
///
/// # Errors
///
/// Currently infallible for every well-formed input (empty text and zero active
/// ids are valid, not errors); the [`CbResult`] return type matches the
/// `cb-compute` calcer convention so the NaiveBayes/BM25 arms — which CAN fail on
/// a malformed class count — share one signature shape.
pub fn bag_of_words_compute(text: &TText, active_feature_ids: &[u32]) -> CbResult<Vec<f64>> {
    let pairs = text.pairs();
    // Two-pointer merge cursor into the sorted-ascending TText pairs.
    let mut text_cursor: usize = 0;

    let mut out: Vec<f64> = Vec::with_capacity(active_feature_ids.len());
    for &active_feature_id in active_feature_ids {
        // Advance the text iterator while the current document token id is
        // strictly BELOW the active feature id (`bow.cpp:9-11`:
        // `while (it != end && it->Token() < activeFeatureId) ++it;`). Checked
        // `.get(..)` keeps the library panic-free.
        while let Some(pair) = pairs.get(text_cursor) {
            if pair.token < active_feature_id {
                text_cursor += 1;
            } else {
                break;
            }
        }

        // `*out = (it == end || it->Token() > activeFeatureId) ? 0 : 1;`
        // (`bow.cpp:13-19`). Present iff the cursor sits on a token equal to the
        // active id (the `< active` ones were skipped above; a `> active` token —
        // or running off the end — means the active token is absent).
        let present = matches!(pairs.get(text_cursor), Some(pair) if pair.token == active_feature_id);
        out.push(if present { 1.0 } else { 0.0 });
    }

    Ok(out)
}
