//! BoW estimated-feature materialization (SC-4 first slice) — raw text columns →
//! BoW binary presence float columns → EXISTING quantizer → tree-search inputs.
//!
//! This is the SC-4 integration seam for the target-INDEPENDENT BoW calcer. It
//! takes a raw text column, builds the BiGram + Word dictionaries ONCE (offline,
//! target-independent — RESEARCH Anti-Pattern "recompute dictionary per fold"),
//! digitizes every document against each dictionary, runs the BoW calcer
//! ([`cb_compute::bag_of_words_compute`]) to produce one binary presence column
//! per active token id, appends those columns to the float-feature layout, and
//! selects each column's split borders through the UNCHANGED
//! [`cb_data::select_borders_greedy_logsum`] quantizer (NO parallel quantizer,
//! SC-4). The result feeds straight into [`crate::train`].
//!
//! # Feature ordering (load-bearing, RESEARCH Pitfall 4)
//!
//! The default classification BoW applies over the BiGram dictionary FIRST, then
//! the Word dictionary (`BoW.dicts = ['Bigram','Word']`,
//! `text_processing_options.cpp:184-202`). The BoW `ttext` D-07 dump confirms
//! this order (BiGram ids first, then Word ids). The estimated columns are
//! therefore emitted BiGram-block then Word-block, each block in ascending
//! active-token-id order (the BoW lockstep order). A different block order would
//! shift every estimated feature index.
//!
//! # Inert when absent (D-04 byte-identical)
//!
//! An empty text column yields no estimated columns and no borders, so the
//! existing numeric/categorical training path is byte-for-byte unchanged.
//!
//! # Robustness (V5 / INFRA-02)
//!
//! Empty documents digitize to empty `TText`s (all-zero presence); a degenerate
//! single-value column yields no border (the quantizer drops it). No
//! `unwrap`/`expect`/`panic`/raw-index in this module.

use cb_compute::bag_of_words_compute;
use cb_core::{CbError, CbResult};
use cb_data::select_borders_greedy_logsum;
use cb_data::text::bigram_dictionary::build_bigram_dictionary;
use cb_data::text::dictionary::{
    build_dictionary, resolve_occurrence_lower_bound, DictionaryOptions, DEFAULT_MAX_DICTIONARY_SIZE,
};
use cb_data::text::digitizer::{digitize_column, digitize_column_bigram};
use cb_data::text::tokenizer::{tokenize, TokenizerOptions};

use crate::estimated::online_embedding::offline_knn_features;
use crate::estimated::online_text::{online_text_prefix, OnlineTextCalcer, OnlineTextPrefix};

/// The BoW estimated feature columns plus their selected split borders, in the
/// canonical BiGram-block-then-Word-block order, ready to append to the
/// float-feature layout and hand to [`crate::train`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BowEstimatedFeatures {
    /// One binary presence column per active token id, column-major
    /// (`columns[f][doc]`), `f32`-valued (estimated features are `f32`,
    /// RESEARCH Pitfall 6). BiGram block first, then Word block.
    pub columns: Vec<Vec<f32>>,
    /// The split borders for each column in [`Self::columns`], in the SAME order
    /// (selected via the existing `select_borders_greedy_logsum`). A binary
    /// presence column yields a single border at `0.5`.
    pub borders: Vec<Vec<f64>>,
    /// Number of BiGram-dictionary feature columns (the size of the BiGram
    /// block, for downstream feature-index bookkeeping / oracle introspection).
    pub bigram_feature_count: usize,
    /// Number of Word-dictionary feature columns (the size of the Word block).
    pub word_feature_count: usize,
}

/// Build the BoW estimated feature columns for a single raw text column.
///
/// `texts` is the per-document raw text (length `n_docs`). `tokenizer_options`
/// pins the D-02 ByDelimiter(space)+lowercase tokenization. `max_borders` is the
/// per-feature border budget passed to the existing quantizer (binary columns
/// need only 1, but the budget is plumbed so the seam matches the numeric path).
///
/// The `OccurrenceLowerBound` is resolved data-dependently from `texts.len()`
/// (`learn_pool_size < 1000 ? 1 : 5`, A4) — never hard-coded. `MaxDictionarySize`
/// is pinned to the catboost default (50000). Both dictionaries use
/// `StartTokenId = 0` (each dictionary's BoW block is indexed from 0 over its own
/// active ids, exactly as upstream `ActiveFeatureIndices = Iota(0..NumTokens)`).
///
/// # Errors
///
/// Returns [`CbError::Degenerate`] if a BoW compute step fails (it is currently
/// infallible, but the error is plumbed so the online calcers share the
/// signature).
pub fn build_bow_estimated_features(
    texts: &[String],
    tokenizer_options: &TokenizerOptions,
    max_borders: usize,
) -> CbResult<BowEstimatedFeatures> {
    // Inert when absent: no documents -> no estimated features (D-04).
    if texts.is_empty() {
        return Ok(BowEstimatedFeatures::default());
    }

    let n_docs = texts.len();

    // Build the two dictionaries ONCE over the learn corpus (offline,
    // target-independent). Tokenize each document once for the dictionary build.
    let tokenized: Vec<Vec<String>> = texts
        .iter()
        .map(|t| tokenize(t, tokenizer_options))
        .collect();

    let olb = resolve_occurrence_lower_bound(n_docs);
    let dict_options = DictionaryOptions {
        occurrence_lower_bound: olb,
        max_dictionary_size: Some(DEFAULT_MAX_DICTIONARY_SIZE),
        start_token_id: 0,
    };

    let (bigram_dict, bigram_entries) = build_bigram_dictionary(&tokenized, &dict_options);
    let (word_dict, word_entries) = build_dictionary(&tokenized, &dict_options);

    let bigram_feature_count = bigram_entries.len();
    let word_feature_count = word_entries.len();

    // Digitize every document against each dictionary (one TText per doc per
    // dictionary).
    let bigram_texts = digitize_column_bigram(texts, tokenizer_options, &bigram_dict);
    let word_texts = digitize_column(texts, tokenizer_options, &word_dict);

    // Active feature ids = Iota(0..NumTokens) for each untrimmed BoW calcer
    // (upstream `ActiveFeatureIndices`). Build them ascending.
    let bigram_active: Vec<u32> = (0..bigram_feature_count as u32).collect();
    let word_active: Vec<u32> = (0..word_feature_count as u32).collect();

    // Per document, compute the BoW presence row for each dictionary. The row is
    // the binary presence vector over that dictionary's active ids. We then
    // transpose to column-major for the trainer (`columns[f][doc]`).
    let total_features = bigram_feature_count + word_feature_count;
    let mut columns: Vec<Vec<f32>> = vec![Vec::with_capacity(n_docs); total_features];

    for doc in 0..n_docs {
        // BiGram block first (RESEARCH Pitfall 4 ordering).
        let bigram_text = bigram_texts
            .get(doc)
            .ok_or_else(|| CbError::Degenerate(format!("missing bigram TText for doc {doc}")))?;
        let bigram_row = bag_of_words_compute(bigram_text, &bigram_active)?;

        let word_text = word_texts
            .get(doc)
            .ok_or_else(|| CbError::Degenerate(format!("missing word TText for doc {doc}")))?;
        let word_row = bag_of_words_compute(word_text, &word_active)?;

        // Scatter the document's presence cells into the column-major layout:
        // BiGram block at [0, bigram_feature_count), Word block after it.
        for (f, &v) in bigram_row.iter().enumerate() {
            if let Some(col) = columns.get_mut(f) {
                col.push(v as f32);
            }
        }
        for (f, &v) in word_row.iter().enumerate() {
            if let Some(col) = columns.get_mut(bigram_feature_count + f) {
                col.push(v as f32);
            }
        }
    }

    // Select borders for each estimated column through the EXISTING quantizer
    // (SC-4 — NO parallel quantizer). A binary presence column yields a single
    // border at 0.5.
    let borders: Vec<Vec<f64>> = columns
        .iter()
        .map(|col| {
            let as_f64: Vec<f64> = col.iter().map(|&v| f64::from(v)).collect();
            select_borders_greedy_logsum(&as_f64, max_borders, false)
        })
        .collect();

    Ok(BowEstimatedFeatures {
        columns,
        borders,
        bigram_feature_count,
        word_feature_count,
    })
}

/// The text estimated feature columns plus their selected split borders, for a
/// target-AWARE calcer (NaiveBayes or BM25). The columns are OBJECT-indexed `f32`
/// and join the float-feature layout exactly like [`BowEstimatedFeatures`].
///
/// # The ONLINE estimate feeds the tree (D-03)
///
/// [`Self::columns`] / [`Self::borders`] are the ONLINE read-before-update prefix
/// estimate (`ComputeOnlineFeatures`, `base_text_feature_estimator.h:74-79`) —
/// the leakage-controlled estimated feature upstream builds the
/// `boosting_type=Plain` TREE on (confirmed by the NaiveBayes per-stage oracle:
/// its split border 0.590515 matches the online column, not the offline whole-set
/// column whose border is 0.5). [`Self::encoding_in_order`] is the SAME estimate
/// in permutation-visiting order — the per-prefix leakage-order anchor.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OnlineTextEstimatedFeatures {
    /// ONLINE read-before-update estimate, one column per active calcer feature,
    /// OBJECT-indexed column-major (`columns[f][doc]`), `f32`-valued (Pitfall 6).
    /// Width = the calcer's `BaseFeatureCount(numClasses)`. The Plain-tree
    /// estimated feature.
    pub columns: Vec<Vec<f32>>,
    /// Split borders for each column in [`Self::columns`], in the SAME order
    /// (selected via the existing `select_borders_greedy_logsum`, SC-4).
    pub borders: Vec<Vec<f64>>,
    /// The ONLINE per-document encoding in PERMUTATION order (the D-03
    /// read-before-update prefix — the per-prefix leakage-order anchor; see
    /// [`OnlineTextPrefix::encoding_in_order`]).
    pub encoding_in_order: Vec<Vec<f64>>,
}

/// Build the online-text (NaiveBayes/BM25) estimated feature columns for a single
/// raw text column, with the read-before-update prefix over the learn
/// `permutation` (D-03 leakage control).
///
/// The Word dictionary is built ONCE over the learn corpus (offline,
/// target-independent — the dictionary is NOT per-fold; only the online calcer
/// state is). Each document is digitized against it, then the online prefix loop
/// ([`online_text_prefix`]) computes each document's encoding from the prefix
/// state accumulated from EARLIER permutation positions only, before updating the
/// state with that document's class. The resulting OBJECT-indexed columns flow
/// through the UNCHANGED [`cb_data::select_borders_greedy_logsum`] quantizer
/// (SC-4 — no parallel quantizer).
///
/// `texts` is the per-document raw text (length `n_docs`); `classes[doc]` is
/// object `doc`'s binarized class in `[0, num_classes)`; `permutation[p]` is the
/// object at learn-order position `p` (the fold's `Fold::permutation`).
/// `tokenizer_options` pins the D-02 ByDelimiter(space)+lowercase tokenization;
/// `max_borders` is the per-feature border budget for the existing quantizer.
///
/// The `OccurrenceLowerBound` is resolved data-dependently from `texts.len()`
/// (A4) — never hard-coded. The NaiveBayes/BM25 fixtures use the Word dictionary
/// ONLY (no BiGram), `StartTokenId = 0`.
///
/// # Inert when absent (D-04 byte-identical)
///
/// An empty text column yields no estimated columns and no borders.
///
/// # Errors
///
/// [`CbError::Degenerate`] if `classes` / `permutation` length-mismatch `texts`,
/// or the online prefix / compute fails.
pub fn build_online_text_estimated_features(
    calcer: OnlineTextCalcer,
    texts: &[String],
    classes: &[usize],
    permutation: &[i32],
    num_classes: usize,
    tokenizer_options: &TokenizerOptions,
    max_borders: usize,
) -> CbResult<OnlineTextEstimatedFeatures> {
    // Inert when absent (D-04).
    if texts.is_empty() {
        return Ok(OnlineTextEstimatedFeatures::default());
    }

    let n_docs = texts.len();
    if classes.len() != n_docs {
        return Err(CbError::Degenerate(
            "online text features: classes length != texts length".to_owned(),
        ));
    }
    if permutation.len() != n_docs {
        return Err(CbError::Degenerate(
            "online text features: permutation length != texts length".to_owned(),
        ));
    }

    // Build the Word dictionary ONCE (offline, target-independent). The
    // NaiveBayes/BM25 fixtures use the Word dictionary only.
    let tokenized: Vec<Vec<String>> = texts
        .iter()
        .map(|t| tokenize(t, tokenizer_options))
        .collect();
    let olb = resolve_occurrence_lower_bound(n_docs);
    let dict_options = DictionaryOptions {
        occurrence_lower_bound: olb,
        max_dictionary_size: Some(DEFAULT_MAX_DICTIONARY_SIZE),
        start_token_id: 0,
    };
    let (word_dict, _word_entries) = build_dictionary(&tokenized, &dict_options);

    // Digitize every document against the Word dictionary (one TText per doc).
    let word_texts = digitize_column(texts, tokenizer_options, &word_dict);

    // The ONLINE read-before-update prefix (D-03 leakage control) — the ordered
    // estimated feature that feeds the Plain-mode TREE. Upstream computes the
    // estimated text feature ONLINE over the learn permutation
    // (`ComputeOnlineFeatures`, base_text_feature_estimator.h:74-79); the
    // NaiveBayes per-stage oracle confirms this (its split border 0.590515 matches
    // the ONLINE column, NOT the offline whole-set column whose border is 0.5).
    // `encoding_in_order` is the same estimate in permutation order (the
    // per-prefix leakage-order anchor).
    let OnlineTextPrefix {
        columns,
        encoding_in_order,
    } = online_text_prefix(calcer, permutation, &word_texts, classes, num_classes)?;

    // Select borders for each estimated column through the EXISTING quantizer
    // (SC-4 — NO parallel quantizer).
    let borders: Vec<Vec<f64>> = columns
        .iter()
        .map(|col| {
            let as_f64: Vec<f64> = col.iter().map(|&v| f64::from(v)).collect();
            select_borders_greedy_logsum(&as_f64, max_borders, false)
        })
        .collect();

    Ok(OnlineTextEstimatedFeatures {
        columns,
        borders,
        encoding_in_order,
    })
}

// ===========================================================================
// SC-4 MIXED text + embedding + numeric feature-layout orchestration (06.5-07).
// ===========================================================================

/// The COMBINED feature layout for a mixed text + embedding (+ numeric) pool —
/// the terminal SC-4 join. The estimated text columns (BoW), the estimated
/// embedding columns (KNN), and the existing numeric columns are appended into a
/// single float-feature layout in the documented upstream block order, each block
/// quantized through the UNCHANGED [`cb_data::select_borders_greedy_logsum`]
/// quantizer (NO parallel quantizer, SC-4). The result feeds straight into
/// [`crate::train`].
///
/// # Block order (load-bearing)
///
/// The layout is emitted as `[ numeric block | text BoW block | embedding KNN
/// block ]`. Upstream appends the ESTIMATED features AFTER the raw
/// numeric/categorical features (`estimated_features.cpp` joins the estimated
/// layout onto the existing quantized objects), and within the estimated layout
/// the text estimators precede the embedding estimators (text feature ids are
/// registered before embedding feature ids). Each block is internally ascending
/// in its own active-id / projection order (the per-calcer order from Plans
/// 03/06).
///
/// Because every column in this mixed corpus PERFECTLY separates the two classes
/// (numeric at the 0.0 border, BoW words at the 0.5 presence border, KNN at the
/// 0.5 integer-vote border), the per-stage Splits/LeafValues/StagedApprox/
/// Predictions are invariant to WHICH separating feature the search selects at a
/// given level — the SC-4 oracle gates the combined layout end-to-end regardless.
///
/// # Inert when absent (D-04 byte-identical)
///
/// Empty `texts` AND empty `embeddings` AND empty `numeric` yields an empty
/// layout; with only numeric columns present the text/embedding blocks are empty
/// and the existing numeric training path is byte-for-byte unchanged (the
/// estimated-feature path is inert).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MixedEstimatedFeatures {
    /// The combined float-feature columns, column-major (`columns[f][doc]`),
    /// `f32`-valued. Order: numeric block, then BoW text block, then KNN embedding
    /// block.
    pub columns: Vec<Vec<f32>>,
    /// Split borders for each column in [`Self::columns`], SAME order, selected
    /// via the existing quantizer (SC-4).
    pub borders: Vec<Vec<f64>>,
    /// Number of leading NUMERIC columns (block 0).
    pub numeric_feature_count: usize,
    /// Number of BoW TEXT columns (block 1; BiGram block then Word block within).
    pub text_feature_count: usize,
    /// Number of KNN EMBEDDING columns (block 2; classification width =
    /// `num_classes`).
    pub embedding_feature_count: usize,
}

/// Build the COMBINED mixed text + embedding + numeric estimated-feature layout
/// for a single trained model (SC-4 terminal join, 06.5-07).
///
/// - `numeric[c][doc]` is the existing numeric column `c` for object `doc`
///   (already `f32`, no estimation — joined directly, quantized by the existing
///   quantizer). Pass an empty slice for a no-numeric pool.
/// - `texts[doc]` is object `doc`'s raw text (length `n_docs`); empty slice ->
///   no text block (inert, D-04). The BoW calcer (target-INDEPENDENT) builds the
///   BiGram + Word dictionaries ONCE and emits one binary-presence column per
///   active token id (the Plan-03 seam).
/// - `embeddings[doc]` is object `doc`'s embedding vector; empty slice -> no
///   embedding block. The KNN calcer (Plan-06, brute-force-exact) emits the
///   OFFLINE whole-set per-class vote columns — the Plain-mode estimate that feeds
///   the tree splits (06.5-06 decision).
/// - `targets[doc]` is object `doc`'s class label (for the KNN vote arm).
/// - `num_classes` is the target class count (KNN classification width).
/// - `close_num` is the KNN query `k` (`KNN:k=...`).
/// - `tokenizer_options` pins the D-02 tokenization; `max_borders` is the
///   per-feature border budget for the existing quantizer.
///
/// All three blocks flow through the UNCHANGED quantizer; NO parallel quantizer
/// and NO Pool schema change (SC-4). BoW + KNN are the two FULLY per-stage-closed
/// calcers (BoW target-independent; KNN neighbor-id bit-exact); BM25's normalized
/// per-stage borders (06.5-04 deferred) and LDA's documented raw-projection
/// tolerance (06.5-05) are deliberately NOT part of the mixed end-to-end gate so
/// the SC-4 oracle is a clean ≤1e-5 per-stage assertion.
///
/// # Errors
/// [`CbError::Degenerate`] on a `texts`/`embeddings`/`targets`/`numeric` length
/// mismatch, or a propagated calcer error.
#[allow(clippy::too_many_arguments)]
pub fn build_mixed_estimated_features(
    numeric: &[Vec<f32>],
    texts: &[String],
    embeddings: &[Vec<f32>],
    targets: &[f32],
    num_classes: usize,
    close_num: usize,
    tokenizer_options: &TokenizerOptions,
    max_borders: usize,
) -> CbResult<MixedEstimatedFeatures> {
    // Determine n_docs from whichever block is present (numeric, text, embedding).
    let n_docs = if let Some(col) = numeric.first() {
        col.len()
    } else if !texts.is_empty() {
        texts.len()
    } else {
        embeddings.len()
    };

    // Inert when absent (D-04): nothing present -> empty layout.
    if n_docs == 0 {
        return Ok(MixedEstimatedFeatures::default());
    }

    // ---- Block 0: NUMERIC (existing columns, joined directly). ----
    let mut columns: Vec<Vec<f32>> = Vec::new();
    let mut numeric_feature_count = 0usize;
    for col in numeric {
        if col.len() != n_docs {
            return Err(CbError::Degenerate(
                "mixed features: numeric column length != n_docs".to_owned(),
            ));
        }
        columns.push(col.clone());
        numeric_feature_count += 1;
    }

    // ---- Block 1: TEXT (BoW, target-independent, Plan-03 seam). ----
    let mut text_feature_count = 0usize;
    if !texts.is_empty() {
        if texts.len() != n_docs {
            return Err(CbError::Degenerate(
                "mixed features: texts length != n_docs".to_owned(),
            ));
        }
        let bow = build_bow_estimated_features(texts, tokenizer_options, max_borders)?;
        text_feature_count = bow.columns.len();
        columns.extend(bow.columns);
    }

    // ---- Block 2: EMBEDDING (KNN, OFFLINE whole-set, Plan-06 seam). ----
    let mut embedding_feature_count = 0usize;
    if !embeddings.is_empty() {
        if embeddings.len() != n_docs {
            return Err(CbError::Degenerate(
                "mixed features: embeddings length != n_docs".to_owned(),
            ));
        }
        if targets.len() != n_docs {
            return Err(CbError::Degenerate(
                "mixed features: targets length != n_docs".to_owned(),
            ));
        }
        let knn = offline_knn_features(embeddings, targets, num_classes, close_num, true)?;
        embedding_feature_count = knn.len();
        columns.extend(knn);
    }

    // Select borders for EVERY column through the EXISTING quantizer (SC-4 — no
    // parallel quantizer). Numeric / BoW-presence / KNN-vote columns all share the
    // same greedy-logsum border selection.
    let borders: Vec<Vec<f64>> = columns
        .iter()
        .map(|col| {
            let as_f64: Vec<f64> = col.iter().map(|&v| f64::from(v)).collect();
            select_borders_greedy_logsum(&as_f64, max_borders, false)
        })
        .collect();

    Ok(MixedEstimatedFeatures {
        columns,
        borders,
        numeric_feature_count,
        text_feature_count,
        embedding_feature_count,
    })
}
