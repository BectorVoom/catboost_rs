//! Feature combinations / tensor CTRs (ORD-05): [`TProjection`] enumeration, the
//! combined projection hash, and the `max_ctr_complexity` gate — the final rung
//! of the additive categorical ladder (D-05).
//!
//! A tensor CTR is NOT new CTR math: per D-05 it is exactly the single-feature
//! online/ordered accumulation of 05-04/05-05 ([`crate::online_ctr_prefix_binclf`]
//! / [`crate::ordered_ctr_per_permutation`]) keyed on a COMBINED projection hash
//! instead of a single feature's hash. This module owns ONLY the projection
//! enumeration and the per-document combined-key fold; the accumulation underneath
//! is reused verbatim.
//!
//! # Source of truth
//!
//! - **`catboost/private/libs/algo/projection.h:61-145`** — `TProjection`
//!   (`CatFeatures` set), `GetFullProjectionLength` (`:138-144`): the projection
//!   "length" for the complexity gate is `CatFeatures.size() + (binFeatures? 1 :
//!   0)`. With only categorical members (this wave) it is simply the number of
//!   combined cat features.
//! - **`catboost/private/libs/algo/greedy_tensor_search.cpp:491-551`**
//!   (`AddTreeCtrs`): the greedy enumeration that extends each seen projection by
//!   one cat feature and emits it as a tensor-CTR candidate, gated at
//!   `proj.GetFullProjectionLength() > MaxTensorComplexity` (`:532-533`). A
//!   single cat feature (length 1) is a `SimpleCtr`; length ≥ 2 is a
//!   `CombinationCtr`.
//! - **`catboost/libs/model/ctr_provider.h:65-78`** (`CalcHash`): the combined
//!   document key is `result = 0; for featureIdx: result = CalcHash(result,
//!   (ui64)(int)hashedCatFeatures[featureIdx])`.
//! - **`catboost/libs/model/hash.h:11-14`** (`CalcHash(ui64 a, ui64 b)`): the
//!   two-argument fold `MAGIC_MULT * (a + MAGIC_MULT * b)` with
//!   `MAGIC_MULT = 0x4906ba494954cb65`.
//! - **`catboost/private/libs/options/cat_feature_options.cpp:231-232`**: default
//!   `max_ctr_complexity = 4` (pinned EXPLICITLY by the caller, never auto —
//!   RESEARCH Pitfall 6).
//!
//! # Parity discipline
//!
//! Per-feature hashes come from the single categorical-hash source
//! [`cb_data::calc_cat_feature_hash`] (NEVER a model's stored `ctr_data`
//! hash_map, RESEARCH Anti-Pattern). All arithmetic in the fold is `wrapping_*`
//! to match C++'s defined unsigned wraparound (no debug-overflow panic). The
//! enumeration bound is checked arithmetic, never an unbounded combinatorial
//! blow-up (T-05-06-01). Checked access only; no `unwrap`/`expect`/`panic`/raw
//! index in production; no `anyhow`.

// Tests live in a dedicated sibling file (source/test separation, CLAUDE.md /
// AGENTS.md). Mounted as a CHILD module of `projection` so the canonical filter
// `cargo test -p cb-train projection` selects them.
#[cfg(test)]
#[path = "projection_test.rs"]
mod tests;

/// The upstream default `max_ctr_complexity`
/// (`cat_feature_options.cpp:231-232`). Pinned EXPLICITLY by the caller (never
/// auto-selected — RESEARCH Pitfall 6); exposed so a [`crate::BoostParams`] /
/// fixture config can reference the canonical default without re-encoding the
/// magic number.
#[must_use]
pub const fn max_ctr_complexity_default() -> usize {
    4
}

/// The `CalcHash(ui64 a, ui64 b)` fold from `catboost/libs/model/hash.h:11-14`:
/// `MAGIC_MULT * (a + MAGIC_MULT * b)` with the low-collision magic multiplier.
/// All multiplies / adds wrap (C++ unsigned wraparound).
#[inline]
#[must_use]
pub fn calc_hash(a: u64, b: u64) -> u64 {
    // const static constexpr ui64 MAGIC_MULT = 0x4906ba494954cb65ull;
    const MAGIC_MULT: u64 = 0x4906_ba49_4954_cb65;
    // return MAGIC_MULT * (a + MAGIC_MULT * b);
    MAGIC_MULT.wrapping_mul(a.wrapping_add(MAGIC_MULT.wrapping_mul(b)))
}

/// Fold one document's per-feature ui32 categorical hash into the running
/// combined projection key, reproducing the `(ui64)(int)hashedCatFeatures[idx]`
/// cast of `ctr_provider.h:72` EXACTLY: the ui32 hash is reinterpreted as a
/// SIGNED 32-bit int (`as i32`) and then SIGN-EXTENDED to 64 bits (`as u64` of an
/// `i64`) before the [`calc_hash`] fold. The sign extension is load-bearing — a
/// hash with the top ui32 bit set folds in its sign-extended (upper-32-bits-set)
/// form, NOT its zero-extended form, so omitting it silently breaks parity for
/// half of all hashes.
#[inline]
#[must_use]
pub fn fold_cat_hash(running: u64, cat_hash: u32) -> u64 {
    // (ui64)(int)hashedCatFeatures[featureIdx]: ui32 -> int (i32) -> ui64 with
    // sign extension. `cat_hash as i32` reinterprets the bits as signed; `as i64`
    // sign-extends; `as u64` is the lossless bit reinterpretation C++'s implicit
    // `(ui64)` of a negative int performs.
    let extended = ((cat_hash as i32) as i64) as u64;
    calc_hash(running, extended)
}

/// A feature-combination projection over CATEGORICAL features (this wave's
/// `TProjection` surface, `projection.h:61-145`). It holds the sorted set of
/// combined cat-feature indices; length 1 is a `SimpleCtr`, length ≥ 2 a
/// `CombinationCtr`. Bin / one-hot projection members (`BinFeatures`,
/// `OneHotFeatures`) are out of scope for the categorical-only tensor-CTR fixture
/// and are not modeled here (their only effect on this wave would be the `+1`
/// `GetFullProjectionLength` addition, which the categorical-only path never
/// triggers).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TProjection {
    /// The sorted, de-duplicated set of combined categorical feature indices
    /// (`TProjection::CatFeatures`, kept sorted by `AddCatFeature`).
    cat_features: Vec<usize>,
}

impl TProjection {
    /// A single-feature (simple) projection over cat feature `feature`.
    #[must_use]
    pub fn single(feature: usize) -> Self {
        Self {
            cat_features: vec![feature],
        }
    }

    /// A projection over the given cat-feature indices, sorted and de-duplicated
    /// (`AddCatFeature` keeps `CatFeatures` sorted; `IsRedundant` rejects
    /// duplicates). A redundant input collapses to its distinct set.
    #[must_use]
    pub fn from_features(features: &[usize]) -> Self {
        let mut cat_features: Vec<usize> = features.to_vec();
        cat_features.sort_unstable();
        cat_features.dedup();
        Self { cat_features }
    }

    /// The sorted combined cat-feature indices.
    #[must_use]
    pub fn cat_features(&self) -> &[usize] {
        &self.cat_features
    }

    /// `GetFullProjectionLength` (`projection.h:138-144`): for the categorical-only
    /// projection this is simply the number of combined cat features (the `+1`
    /// bin/one-hot addition never fires in this wave).
    #[must_use]
    pub fn full_projection_length(&self) -> usize {
        self.cat_features.len()
    }

    /// Whether this projection is a SimpleCtr (single feature). Length 1.
    #[must_use]
    pub fn is_simple(&self) -> bool {
        self.cat_features.len() == 1
    }

    /// Whether this projection is a CombinationCtr (tensor, ≥ 2 features).
    #[must_use]
    pub fn is_combination(&self) -> bool {
        self.cat_features.len() >= 2
    }

    /// Extend this projection by one more cat feature, keeping `cat_features`
    /// sorted and de-duplicated (`TProjection::AddCatFeature` +
    /// `IsRedundant`-style dedup). Returns the extended projection.
    #[must_use]
    pub fn with_added(&self, feature: usize) -> Self {
        let mut cat_features = self.cat_features.clone();
        cat_features.push(feature);
        cat_features.sort_unstable();
        cat_features.dedup();
        Self { cat_features }
    }

    /// The COMBINED projection hash for ONE document, folding each member
    /// feature's per-document ui32 categorical hash via [`fold_cat_hash`]
    /// (`ctr_provider.h:65-78` `CalcHash`). `feature_hashes[f]` is document's ui32
    /// `calc_cat_feature_hash` for cat feature `f`; the fold visits the projection
    /// members in the projection's SORTED order (upstream iterates the projection's
    /// `CatFeatures` vector, which is kept sorted). Out-of-range member indices are
    /// skipped defensively (checked `.get`) — the caller supplies in-range members.
    ///
    /// A simple (single-feature) projection thus folds exactly one hash:
    /// `CalcHash(0, signext(hash))`, distinct from the bare feature hash but a
    /// stable function of it — the same combined-key keyspace tensor CTRs share.
    #[must_use]
    pub fn combined_hash(&self, feature_hashes: &[u32]) -> u64 {
        let mut result: u64 = 0;
        for &feature in &self.cat_features {
            if let Some(&hash) = feature_hashes.get(feature) {
                result = fold_cat_hash(result, hash);
            }
        }
        result
    }
}

/// Enumerate every tensor-CTR projection over `cat_feature_count` categorical
/// features bounded by `max_ctr_complexity` (the `AddTreeCtrs` greedy gate,
/// `greedy_tensor_search.cpp:491-551`, simplified to the from-empty enumeration
/// the fixture exercises). Returns ALL non-empty combinations of distinct cat
/// features whose `GetFullProjectionLength` is `<= max_ctr_complexity`, each as a
/// sorted [`TProjection`]:
///
/// - `max_ctr_complexity == 0` → no projections (degenerate).
/// - `max_ctr_complexity == 1` → only SimpleCtrs (one per feature).
/// - `max_ctr_complexity == 2` → SimpleCtrs + all unordered PAIRS.
/// - `max_ctr_complexity == k` → all non-empty subsets of size `1..=k`.
///
/// The enumeration is bounded: the gate `len > max_ctr_complexity` prunes deeper
/// subsets, so no unbounded combinatorial blow-up occurs (T-05-06-01). Results
/// are emitted in increasing length then lexicographic feature order (a stable,
/// deterministic order for the candidate list).
#[must_use]
pub fn enumerate_projections(cat_feature_count: usize, max_ctr_complexity: usize) -> Vec<TProjection> {
    let mut out: Vec<TProjection> = Vec::new();
    if max_ctr_complexity == 0 || cat_feature_count == 0 {
        return out;
    }
    // The deepest subset size is bounded by both the complexity gate and the
    // number of features available — checked `min`, no blow-up.
    let max_len = max_ctr_complexity.min(cat_feature_count);
    // Enumerate subsets by increasing length (length is GetFullProjectionLength
    // for the categorical-only projection, gated `<= max_ctr_complexity`).
    for len in 1..=max_len {
        enumerate_subsets_of_len(cat_feature_count, len, &mut out);
    }
    out
}

/// Append every sorted distinct-feature subset of size `len` over
/// `[0, cat_feature_count)` to `out`, in lexicographic order. A pure
/// combinatorial helper (no hashing) bounded by `len <= cat_feature_count`.
fn enumerate_subsets_of_len(cat_feature_count: usize, len: usize, out: &mut Vec<TProjection>) {
    if len == 0 || len > cat_feature_count {
        return;
    }
    // Lexicographic combinations via an index stack `indices[0] < indices[1] < …`.
    let mut indices: Vec<usize> = (0..len).collect();
    loop {
        out.push(TProjection {
            cat_features: indices.clone(),
        });
        // Advance to the next lexicographic combination: find the rightmost index
        // that can be incremented, bump it, then reset the suffix.
        let mut i = len;
        loop {
            if i == 0 {
                return;
            }
            i -= 1;
            // The max value position `i` may hold so the suffix still fits.
            let max_for_pos = cat_feature_count - len + i;
            if indices[i] < max_for_pos {
                indices[i] += 1;
                for j in (i + 1)..len {
                    indices[j] = indices[j - 1] + 1;
                }
                break;
            }
        }
    }
}
