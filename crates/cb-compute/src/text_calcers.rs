//! Text feature calcers — the pure numeric primitives that turn a digitized
//! document (`cb_data::text::text::TText`) into estimated float feature columns.
//!
//! # Source of truth (D-04)
//!
//! This module transcribes the upstream CatBoost 1.2.10 text-feature calcer math
//! VERBATIM. The first (and only target-INDEPENDENT) calcer is BoW
//! (`TBagOfWordsCalcer`), transcribed from
//! `catboost-master/catboost/private/libs/text_features/bow.cpp:7-21`.
//!
//! The two target-AWARE calcers — NaiveBayes (`TMultinomialNaiveBayes`,
//! `naive_bayesian.cpp:14-63`) and BM25 (`TBM25`, `bm25.cpp:12-83`) — are
//! computed as `IOnlineFeatureEstimator`s over the learn permutation with a
//! read-before-update prefix (D-03 leakage control). This module owns only their
//! PURE compute math given an accumulated class-frequency state
//! ([`NaiveBayesState`], [`Bm25State`]); the ordered per-fold prefix loop that
//! feeds them their prefix state lives behind the `cb-train` online-text seam
//! (`cb-train::estimated::online_text`). Splitting compute (here) from the prefix
//! wiring (cb-train) keeps the numeric primitive pure and reusable.
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

use std::collections::HashMap;

use cb_core::{sum_f64, CbError, CbResult};
use cb_data::text::text::TText;

/// NaiveBayes / BM25 default prior (`TMultinomialNaiveBayes::DEFAULT_PRIOR`,
/// `naive_bayesian.h:16`): `ClassPrior == TokenPrior == 0.5`.
pub const NAIVE_BAYES_DEFAULT_PRIOR: f64 = 0.5;

/// NaiveBayes seen-tokens prior (`TMultinomialNaiveBayes::SEEN_TOKENS_PRIOR`,
/// `naive_bayesian.h:17`): the `+1` added to `NumSeenTokens` in the denominator.
pub const NAIVE_BAYES_SEEN_TOKENS_PRIOR: u64 = 1;

/// BM25 default saturation parameter `k` (`bm25.h:24`).
pub const BM25_DEFAULT_K: f64 = 1.5;

/// BM25 default length-normalization parameter `b` (`bm25.h:25`).
pub const BM25_DEFAULT_B: f64 = 0.75;

/// BM25 default inverse-class-frequency truncation floor (`bm25.h:23`).
pub const BM25_DEFAULT_TRUNCATE_BORDER: f64 = 1e-3;

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

/// Accumulated per-class frequency state for the multinomial NaiveBayes calcer
/// (`TMultinomialNaiveBayes`, `naive_bayesian.h:65-75`). This is the state the
/// online prefix loop READS from (in [`naive_bayes_compute`]) before UPDATING it
/// with the current document's label/text ([`Self::update`]) — the
/// read-before-update prefix is the no-leakage property (D-03).
///
/// # Integer counts are EXACT (no `sum_f64`)
///
/// `frequencies[class][token]` (`TVector<TDenseHash<TTokenId,ui32>>`),
/// `class_docs[class]` (`TVector<ui32>`), `class_total_tokens[class]`
/// (`TVector<ui64>`), and `num_seen_tokens` (`ui64`) are EXACT integer
/// accumulation — they do NOT route through `sum_f64` (RESEARCH Anti-Pattern
/// caveat: only FLOAT sums do). Only the per-class log-probability reductions in
/// [`naive_bayes_compute`] route through [`cb_core::sum_f64`].
#[derive(Debug, Clone, PartialEq)]
pub struct NaiveBayesState {
    /// `Frequencies[class]`: token id → total occurrence count in that class
    /// (`naive_bayesian.h:73`, `TDenseHash<TTokenId, ui32>` per class).
    frequencies: Vec<HashMap<u32, u64>>,
    /// `ClassDocs[class]`: number of documents seen in each class
    /// (`naive_bayesian.h:71`, `TVector<ui32>`).
    class_docs: Vec<u64>,
    /// `ClassTotalTokens[class]`: total token count summed over each class's
    /// documents (`naive_bayesian.h:72`, `TVector<ui64>`).
    class_total_tokens: Vec<u64>,
    /// `NumSeenTokens`: size of the global distinct-token set seen so far
    /// (`naive_bayesian.h:70` + `TNaiveBayesVisitor::SeenTokens`,
    /// `naive_bayesian.cpp:131`).
    num_seen_tokens: u64,
    /// The global distinct-token set backing `num_seen_tokens`
    /// (`TNaiveBayesVisitor::SeenTokens`, `naive_bayesian.h:82`).
    seen_tokens: std::collections::HashSet<u32>,
    /// Number of target classes (`NumClasses`, `naive_bayesian.h:66`).
    num_classes: usize,
    /// `ClassPrior` (`naive_bayesian.h:67`), default 0.5.
    class_prior: f64,
    /// `TokenPrior` (`naive_bayesian.h:68`), default 0.5.
    token_prior: f64,
}

impl NaiveBayesState {
    /// A zeroed NaiveBayes state with `num_classes` classes and the default
    /// priors (`ClassPrior == TokenPrior == 0.5`), mirroring the
    /// `TMultinomialNaiveBayes` constructor (`naive_bayesian.h:19-34`) with empty
    /// `Frequencies` / zero `ClassDocs` / zero `ClassTotalTokens`.
    #[must_use]
    pub fn new(num_classes: usize) -> Self {
        Self {
            frequencies: vec![HashMap::new(); num_classes],
            class_docs: vec![0; num_classes],
            class_total_tokens: vec![0; num_classes],
            num_seen_tokens: 0,
            seen_tokens: std::collections::HashSet::new(),
            num_classes,
            class_prior: NAIVE_BAYES_DEFAULT_PRIOR,
            token_prior: NAIVE_BAYES_DEFAULT_PRIOR,
        }
    }

    /// The NaiveBayes output width (`BaseFeatureCount(numClasses)`,
    /// `naive_bayesian.h:38-40`): `numClasses > 2 ? numClasses : 1` (binary → 1,
    /// multiclass → `numClasses`).
    #[must_use]
    pub fn feature_count(&self) -> usize {
        if self.num_classes > 2 {
            self.num_classes
        } else {
            1
        }
    }

    /// UPDATE the state with one document's class label and text
    /// (`TNaiveBayesVisitor::Update`, `naive_bayesian.cpp:119-132`): for each
    /// `(token, count)` insert the token into the global seen set, add `count` to
    /// `Frequencies[class][token]` and to `ClassTotalTokens[class]`; then
    /// `ClassDocs[class] += 1` and `NumSeenTokens = |SeenTokens|`.
    ///
    /// Integer accumulation only (no `sum_f64`). An out-of-range `class` is
    /// ignored (checked access; the caller binarizes into `[0, num_classes)`).
    pub fn update(&mut self, class: usize, text: &TText) {
        let Some(class_counts) = self.frequencies.get_mut(class) else {
            return;
        };
        for pair in text.pairs() {
            self.seen_tokens.insert(pair.token);
            *class_counts.entry(pair.token).or_insert(0) += u64::from(pair.count);
        }
        if let Some(total) = self.class_total_tokens.get_mut(class) {
            for pair in text.pairs() {
                *total += u64::from(pair.count);
            }
        }
        if let Some(docs) = self.class_docs.get_mut(class) {
            *docs += 1;
        }
        self.num_seen_tokens = self.seen_tokens.len() as u64;
    }
}

/// Per-class multinomial log-probability for one document
/// (`TMultinomialNaiveBayes::LogProb`, `naive_bayesian.cpp:14-44`), VERBATIM:
///
/// ```text
/// value = log(classSamples + ClassPrior)
/// classTokensCount += TokenPrior * (NumSeenTokens + SEEN_TOKENS_PRIOR)
/// textLen = 0
/// for (token, count) in text:
///     textLen += count
///     num = TokenPrior + (freqTable[token] if present else 0)
///     if token absent from freqTable: classTokensCount += TokenPrior  // unseen-word adjust
///     value += count * log(num)
/// value -= textLen * log(classTokensCount)
/// ```
///
/// The two per-document FLOAT reductions — `value`'s `Σ count·log(num)` term and
/// `textLen`'s `Σ count` — route through [`cb_core::sum_f64`] in upstream
/// iteration order (D-04). `classTokensCount`'s prefix-independent base and its
/// per-unseen-token additions are simple running scalars matching the upstream
/// single-accumulator form.
fn naive_bayes_log_prob(
    state: &NaiveBayesState,
    freq_table: &HashMap<u32, u64>,
    class_samples: f64,
    class_tokens_count_base: f64,
    text: &TText,
) -> f64 {
    // value = log(classSamples + ClassPrior)   (naive_bayesian.cpp:21)
    let value_base = (class_samples + state.class_prior).ln();

    // classTokensCount += TokenPrior * (NumSeenTokens + SEEN_TOKENS_PRIOR)
    // (naive_bayesian.cpp:23)
    let class_tokens_count_seen = class_tokens_count_base
        + state.token_prior * (state.num_seen_tokens + NAIVE_BAYES_SEEN_TOKENS_PRIOR) as f64;

    // Σ count (textLen) and Σ count·log(num) (the value increment), both in
    // upstream token-iteration order → sum_f64 (D-04). The unseen-word
    // `classTokensCount += TokenPrior` is computed as `token_prior *
    // unseen_count` from a counted set, which is exact and order-free — this
    // avoids an order-dependent running scalar should `token_prior` ever become
    // per-token (WR-07).
    let mut text_len_terms: Vec<f64> = Vec::with_capacity(text.len());
    let mut value_terms: Vec<f64> = Vec::with_capacity(text.len());
    let mut unseen_count: u64 = 0;
    for pair in text.pairs() {
        let count = f64::from(pair.count);
        text_len_terms.push(count);

        // num = TokenPrior + freqTable[token] (if present)  (naive_bayesian.cpp:30-33)
        let mut num = state.token_prior;
        match freq_table.get(&pair.token) {
            Some(&freq) => num += freq as f64,
            None => unseen_count += 1, // unseen-word adjust (counted, summed once below)
        }
        // value += count * log(num)  (naive_bayesian.cpp:38)
        value_terms.push(count * num.ln());
    }
    let class_tokens_count = class_tokens_count_seen + state.token_prior * unseen_count as f64;

    let text_len = sum_f64(&text_len_terms);
    // value = value_base + Σ count·log(num) - textLen·log(classTokensCount)
    // (naive_bayesian.cpp:38-42)
    value_base + sum_f64(&value_terms) - text_len * class_tokens_count.ln()
}

/// In-place softmax over `vals` (`Softmax`, `helpers.h:33-47`): subtract the max
/// for numerical stability, exponentiate, divide by the total. Operates on `f64`
/// (downcast to `f32` happens at the estimated-feature storage boundary,
/// RESEARCH Pitfall 6). The `total` reduction routes through [`cb_core::sum_f64`]
/// (D-04); upstream uses a running `double total` — the canonical-order sum
/// matches it for the small per-class vectors here.
fn softmax_in_place(vals: &mut [f64]) -> CbResult<()> {
    let mut max_value = f64::NEG_INFINITY;
    for &v in vals.iter() {
        if v > max_value {
            max_value = v;
        }
    }
    // A non-finite max (empty input, all `-inf`, or a `NaN` that the `v >
    // max_value` comparison can never beat) means there is no valid
    // distribution to normalize. Surface a typed error rather than silently
    // returning the input un-softmaxed (raw log-probs or NaN as a
    // "probability"), which would poison the emitted feature (WR-04).
    if !max_value.is_finite() {
        return Err(CbError::Degenerate(format!(
            "softmax_in_place: non-finite max {max_value} (degenerate log-prob vector)"
        )));
    }
    let exps: Vec<f64> = vals.iter().map(|&v| (v - max_value).exp()).collect();
    let total = sum_f64(&exps);
    for (slot, e) in vals.iter_mut().zip(exps.iter()) {
        *slot = if total > 0.0 { e / total } else { *slot };
    }
    Ok(())
}

/// NaiveBayes per-document compute (`TMultinomialNaiveBayes::Compute`,
/// `naive_bayesian.cpp:47-63`): compute the per-class log-probabilities from the
/// accumulated prefix `state`, softmax them (`f64`), then emit the active
/// features. For binary (`num_classes == 2`) the single active feature is id `0`
/// → `logProbs[0]` after softmax; for multiclass all `num_classes` softmax
/// outputs are emitted (active ids `0..num_classes`).
///
/// The output width equals `state.feature_count()`
/// (`BaseFeatureCount(numClasses)`, Pitfall 5). Values are `f64` here, narrowed
/// to `f32` at the storage boundary by the caller.
///
/// # Errors
///
/// [`CbError::Degenerate`] if `state` has zero classes (no log-prob vector to
/// softmax / emit).
pub fn naive_bayes_compute(state: &NaiveBayesState, text: &TText) -> CbResult<Vec<f64>> {
    if state.num_classes == 0 {
        return Err(CbError::Degenerate(
            "naive_bayes_compute: state has zero classes".to_owned(),
        ));
    }

    // logProbs[c] = LogProb(Frequencies[c], ClassDocs[c], ClassTotalTokens[c], text)
    // (naive_bayesian.cpp:51-54)
    let mut log_probs: Vec<f64> = Vec::with_capacity(state.num_classes);
    for clazz in 0..state.num_classes {
        let (Some(freq), Some(&docs), Some(&tokens)) = (
            state.frequencies.get(clazz),
            state.class_docs.get(clazz),
            state.class_total_tokens.get(clazz),
        ) else {
            return Err(CbError::Degenerate(
                "naive_bayes_compute: per-class state length mismatch".to_owned(),
            ));
        };
        log_probs.push(naive_bayes_log_prob(
            state, freq, docs as f64, tokens as f64, text,
        ));
    }

    // Softmax(logProbs) (naive_bayesian.cpp:55).
    softmax_in_place(&mut log_probs)?;

    // ForEachActiveFeature emits logProbs[featureId] for featureId in
    // 0..FeatureCount (naive_bayesian.cpp:57-62). Active ids = Iota(0..width).
    let width = state.feature_count();
    let mut out: Vec<f64> = Vec::with_capacity(width);
    for feature_id in 0..width {
        let Some(&v) = log_probs.get(feature_id) else {
            return Err(CbError::Degenerate(
                "naive_bayes_compute: active feature id exceeds class count".to_owned(),
            ));
        };
        out.push(v);
    }
    Ok(out)
}

/// Accumulated per-class frequency state for the BM25 calcer (`TBM25`,
/// `bm25.h:42-51`). As with [`NaiveBayesState`], the online prefix loop READS
/// this (in [`bm25_compute`]) before UPDATING it ([`Self::update`]).
///
/// # Integer counts are EXACT (no `sum_f64`)
///
/// `frequencies[class][token]`, `class_total_tokens[class]` (`TVector<ui64>`),
/// and `total_tokens` (`ui64`) are EXACT integer accumulation. Only the per-class
/// score reductions in [`bm25_compute`] route through [`cb_core::sum_f64`].
///
/// # `total_tokens` starts at 1 (upstream)
///
/// `TBM25`'s constructor seeds `TotalTokens(1)` (`bm25.cpp:58`) — a `+1` floor so
/// the `meanClassLength = TotalTokens / NumClasses` denominator is never zero on
/// the empty prefix. This is reproduced exactly (`new` sets `total_tokens = 1`).
#[derive(Debug, Clone, PartialEq)]
pub struct Bm25State {
    /// `Frequencies[class]`: token id → total occurrence count in that class
    /// (`bm25.h:50`).
    frequencies: Vec<HashMap<u32, u64>>,
    /// `ClassTotalTokens[class]`: total token count per class (`bm25.h:49`).
    class_total_tokens: Vec<u64>,
    /// `TotalTokens`: total token count over all classes, seeded at 1
    /// (`bm25.cpp:58`, `bm25.h:48`).
    total_tokens: u64,
    /// Number of target classes (`NumClasses`, `bm25.h:43`).
    num_classes: usize,
    /// Saturation parameter `k` (`bm25.h:44`), default 1.5.
    k: f64,
    /// Length-normalization parameter `b` (`bm25.h:45`), default 0.75.
    b: f64,
    /// Inverse-class-frequency truncation floor (`bm25.h:46`), default 1e-3.
    truncate_border: f64,
    /// Precomputed `TruncatedInvClassFreq[0..=numClasses]`
    /// (`InitTruncatedInvClassFreq`, `bm25.cpp:40-44`): index = number of classes
    /// containing a term.
    truncated_inv_class_freq: Vec<f64>,
}

/// `CalcTruncatedInvClassFreq` (`bm25.cpp:12-14`):
/// `max(log((numClasses - classesWithTerm + 0.5)/(classesWithTerm + 0.5)), eps)`.
fn calc_truncated_inv_class_freq(num_classes: usize, classes_with_term: usize, eps: f64) -> f64 {
    let n = num_classes as f64;
    let c = classes_with_term as f64;
    let raw = ((n - c + 0.5) / (c + 0.5)).ln();
    raw.max(eps)
}

/// `Score` (`bm25.cpp:33-38`): `0` if `tf == 0`, else
/// `tf*(k+1)/(tf + k*(1 - b + b*meanLength/classLength))`.
fn bm25_score(term_freq: f64, k: f64, b: f64, mean_length: f64, class_length: f64) -> f64 {
    if term_freq == 0.0 {
        return 0.0;
    }
    term_freq * (k + 1.0) / (term_freq + k * (1.0 - b + b * mean_length / class_length))
}

impl Bm25State {
    /// A BM25 state with `num_classes` classes and the default parameters
    /// (`k = 1.5`, `b = 0.75`, `truncate = 1e-3`), mirroring the `TBM25`
    /// constructor (`bm25.cpp:46-64`): empty `Frequencies`, zero
    /// `ClassTotalTokens`, `TotalTokens = 1`, and the precomputed
    /// `TruncatedInvClassFreq` table.
    #[must_use]
    pub fn new(num_classes: usize) -> Self {
        let mut truncated_inv_class_freq = Vec::with_capacity(num_classes + 1);
        for classes_with_term in 0..=num_classes {
            truncated_inv_class_freq.push(calc_truncated_inv_class_freq(
                num_classes,
                classes_with_term,
                BM25_DEFAULT_TRUNCATE_BORDER,
            ));
        }
        Self {
            frequencies: vec![HashMap::new(); num_classes],
            class_total_tokens: vec![0; num_classes],
            total_tokens: 1, // bm25.cpp:58 TotalTokens(1)
            num_classes,
            k: BM25_DEFAULT_K,
            b: BM25_DEFAULT_B,
            truncate_border: BM25_DEFAULT_TRUNCATE_BORDER,
            truncated_inv_class_freq,
        }
    }

    /// The BM25 output width (`BaseFeatureCount(numClasses)`, `bm25.h:38-40`):
    /// always `numClasses` (Pitfall 5).
    #[must_use]
    pub fn feature_count(&self) -> usize {
        self.num_classes
    }

    /// UPDATE the state with one document's class label and text
    /// (`TBM25Visitor::Update`, `bm25.cpp:133-145`): for each `(token, count)` add
    /// `count` to `Frequencies[class][token]`, `ClassTotalTokens[class]`, and the
    /// global `TotalTokens`. Integer accumulation only (no `sum_f64`). Out-of-range
    /// `class` ignored (checked access).
    pub fn update(&mut self, class: usize, text: &TText) {
        let Some(class_counts) = self.frequencies.get_mut(class) else {
            return;
        };
        for pair in text.pairs() {
            *class_counts.entry(pair.token).or_insert(0) += u64::from(pair.count);
        }
        if let Some(total) = self.class_total_tokens.get_mut(class) {
            for pair in text.pairs() {
                *total += u64::from(pair.count);
            }
        }
        for pair in text.pairs() {
            self.total_tokens += u64::from(pair.count);
        }
    }
}

/// Extract per-class term frequencies for `term` from the prefix state
/// (`ExtractTermFreq`, `bm25.cpp:16-31`): fill `term_freq[class]` with
/// `Frequencies[class][term]` (or 0 if absent) and return the number of classes
/// that contain the term (`nonZeroCount`).
fn bm25_extract_term_freq(state: &Bm25State, term: u32, term_freq: &mut [u64]) -> usize {
    let mut non_zero_count = 0usize;
    for clazz in 0..state.num_classes {
        let f = state
            .frequencies
            .get(clazz)
            .and_then(|t| t.get(&term).copied())
            .unwrap_or(0);
        if let Some(slot) = term_freq.get_mut(clazz) {
            *slot = f;
        }
        if f != 0 {
            non_zero_count += 1;
        }
    }
    non_zero_count
}

/// BM25 per-document compute (`TBM25::Compute`, `bm25.cpp:66-83`): for each
/// `(token, count)` in the document extract the per-class term frequencies from
/// the accumulated prefix `state`, then accumulate
/// `scores[c] += TruncatedInvClassFreq[nonZero] * Score(tf_c, k, b, meanLen, classLen_c)`
/// over all tokens. `meanClassLength = TotalTokens / NumClasses`. The output
/// width equals `state.feature_count()` (= `numClasses`, Pitfall 5).
///
/// The per-class `scores[c]` reduction (a sum over the document's tokens) routes
/// through [`cb_core::sum_f64`] in upstream token-iteration order (D-04). Values
/// are `f64` here, narrowed to `f32` at the storage boundary by the caller.
///
/// # Errors
///
/// [`CbError::Degenerate`] if `state` has zero classes.
pub fn bm25_compute(state: &Bm25State, text: &TText) -> CbResult<Vec<f64>> {
    if state.num_classes == 0 {
        return Err(CbError::Degenerate(
            "bm25_compute: state has zero classes".to_owned(),
        ));
    }

    // meanClassLength = (double)TotalTokens / NumClasses  (bm25.cpp:69)
    let mean_class_length = state.total_tokens as f64 / state.num_classes as f64;

    // Per-class accumulation buffers of each token's contribution; reduced with
    // sum_f64 per class in upstream token order (D-04). scores[c] = Σ_token
    // TruncInvClassFreq[nonZero] * Score(...). Upstream accumulates in a running
    // `scores[clazz] += ...` over the token loop; the canonical-order sum over
    // the same per-token sequence matches it.
    let mut score_terms: Vec<Vec<f64>> = vec![Vec::with_capacity(text.len()); state.num_classes];
    let mut term_freq_in_class: Vec<u64> = vec![0; state.num_classes];

    for pair in text.pairs() {
        let non_zero_count = bm25_extract_term_freq(state, pair.token, &mut term_freq_in_class);
        // TruncatedInvClassFreq[nonZeroCount] (bm25.cpp:73). Index in 0..=numClasses.
        let inv_class_freq = state
            .truncated_inv_class_freq
            .get(non_zero_count)
            .copied()
            .unwrap_or_else(|| {
                // Defensive: recompute if the precomputed table is short (never in
                // practice — it is sized numClasses+1). Keeps the library panic-free.
                calc_truncated_inv_class_freq(state.num_classes, non_zero_count, state.truncate_border)
            });
        for clazz in 0..state.num_classes {
            let tf = term_freq_in_class.get(clazz).copied().unwrap_or(0) as f64;
            let class_length = state.class_total_tokens.get(clazz).copied().unwrap_or(0) as f64;
            let s = bm25_score(tf, state.k, state.b, mean_class_length, class_length);
            if let Some(terms) = score_terms.get_mut(clazz) {
                terms.push(inv_class_freq * s);
            }
        }
    }

    // ForEachActiveFeature emits scores[featureId] for 0..FeatureCount
    // (bm25.cpp:77-82). Active ids = Iota(0..numClasses).
    let mut out: Vec<f64> = Vec::with_capacity(state.num_classes);
    for clazz in 0..state.num_classes {
        let terms = score_terms.get(clazz).map_or(&[][..], Vec::as_slice);
        out.push(sum_f64(terms));
    }
    Ok(out)
}
