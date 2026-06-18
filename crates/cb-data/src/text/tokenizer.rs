//! `ByDelimiter` tokenizer — splits a raw text string into tokens, with
//! optional lowercasing and empty-token skipping.
//!
//! Verbatim Rust transcription of upstream CatBoost's `ByDelimiter` split path
//! (`catboost-master/library/cpp/text_processing/tokenizer/tokenizer.cpp:188-254`,
//! 1.2.10 snapshot) with the default `TTokenizerOptions`
//! (`tokenizer/options.h:39-77`).
//!
//! # Scope (D-02) — defaults pinned to upstream
//!
//! Only the `ByDelimiter` separator path with the upstream defaults is
//! implemented, because that is exactly what the frozen FEAT-01 fixtures
//! exercise (`fixtures/text_embedding_inputs/meta.json`:
//! `"ByDelimiter(space) + lowercasing"`). The pinned defaults are:
//!
//! - `Delimiter = " "` (`options.h:48`)
//! - `SkipEmpty = true` (`options.h:50`) — drop empty tokens between
//!   consecutive delimiters (`tokenizer.cpp:203-204` `.SkipEmpty()`)
//! - `SplitBySet = false` (`options.h:49`) — split on the whole delimiter
//!   STRING, not on each character (`tokenizer.cpp:204` `SplitByString`)
//! - `Lowercasing` is applied per-token when enabled (`tokenizer.cpp:232-238`,
//!   `ProcessWordToken` → `ToLower`, `tokenizer.cpp:78-85`)
//!
//! The `BySense` separator (`TNlpTokenizer`), lemmatization, and number-process
//! policies are **out of scope** (deferred — not reachable by the FEAT-01/02
//! fixtures; CONTEXT D-02 / RESEARCH Open-Q4).
//!
//! # Do NOT crate-source a generic tokenizer (parity warning)
//!
//! Mirroring the `cat_hash.rs` discipline: a generic `unicode-segmentation` /
//! `tokenizers`-crate split would apply a *different* split rule (word
//! boundaries, punctuation handling, Unicode case-folding) and silently break
//! the bit-exact D-01 token stream that every downstream calcer index depends
//! on (SC-1, RESEARCH Pitfall 3). The split is therefore transcribed here, not
//! crate-sourced. Lowercasing uses Rust's `str::to_lowercase` (Unicode-aware,
//! matching upstream `ToLower` over the wide string for the ASCII corpus the
//! fixtures use).
//!
//! # Robustness (V5 / INFRA-02)
//!
//! Empty input yields an empty token vector (never a panic). No
//! `unwrap`/`expect`/`panic`/raw-index appears in this module.

/// Upstream default delimiter `" "` (`tokenizer/options.h:48`).
pub const DEFAULT_DELIMITER: &str = " ";

/// `ByDelimiter` tokenizer options, pinned to upstream `TTokenizerOptions`
/// defaults (`tokenizer/options.h:39-77`) for the surface the FEAT-01 fixtures
/// exercise (D-02).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerOptions {
    /// Split delimiter STRING (upstream `Delimiter`, default `" "`). Splitting
    /// is on the whole string (`SplitByString`), not per-character, because
    /// upstream `SplitBySet` defaults to `false` (`options.h:49`).
    pub delimiter: String,
    /// Lowercase each token after splitting (upstream `Lowercasing`,
    /// `tokenizer.cpp:79-81`). The classification text-processing default the
    /// fixtures use enables this.
    pub lowercasing: bool,
    /// Drop empty tokens (upstream `SkipEmpty`, default `true`,
    /// `options.h:50` / `tokenizer.cpp:203-204`).
    pub skip_empty: bool,
}

impl Default for TokenizerOptions {
    fn default() -> Self {
        // Upstream defaults (options.h:48-50) plus lowercasing, which the
        // classification text-processing path enables and the fixtures use.
        Self {
            delimiter: DEFAULT_DELIMITER.to_string(),
            lowercasing: true,
            skip_empty: true,
        }
    }
}

/// Tokenize `input` per the upstream `ByDelimiter` path
/// (`tokenizer.cpp:188-254`): split on the delimiter string, optionally skip
/// empty tokens, optionally lowercase each token.
///
/// Order of operations mirrors upstream exactly: `SplitByDelimiter` (split,
/// then `SkipEmpty`) first (`tokenizer.cpp:202-208`), then the per-token
/// lowercasing pass (`tokenizer.cpp:232-238`).
///
/// Empty input yields an empty `Vec`.
#[must_use]
pub fn tokenize(input: &str, options: &TokenizerOptions) -> Vec<String> {
    // `StringSplitter(input).SplitByString(delimiter)` (tokenizer.cpp:204/206).
    //
    // The empty-delimiter edge: upstream `StringSplitter::SplitByString("")`
    // is undefined for an empty delimiter; we treat an empty delimiter as "no
    // split" (the whole input as one token) to keep a total, panic-free
    // function. The fixtures pin a single-space delimiter, so this edge is not
    // exercised by the D-01 gate.
    let raw_tokens: Vec<&str> = if options.delimiter.is_empty() {
        if input.is_empty() {
            Vec::new()
        } else {
            vec![input]
        }
    } else {
        input.split(options.delimiter.as_str()).collect()
    };

    let mut tokens: Vec<String> = Vec::with_capacity(raw_tokens.len());
    for raw in raw_tokens {
        // `.SkipEmpty()` (tokenizer.cpp:203-204): drop empty tokens produced by
        // consecutive / leading / trailing delimiters.
        if options.skip_empty && raw.is_empty() {
            continue;
        }
        // `ProcessWordToken` → `ToLower` (tokenizer.cpp:79-81, 232-238): apply
        // lowercasing per token AFTER the split.
        if options.lowercasing {
            tokens.push(raw.to_lowercase());
        } else {
            tokens.push(raw.to_string());
        }
    }

    tokens
}

/// Convenience: tokenize with the upstream defaults (`ByDelimiter(" ")`,
/// lowercasing, `SkipEmpty`) — the exact configuration the FEAT-01 fixtures use.
#[must_use]
pub fn tokenize_default(input: &str) -> Vec<String> {
    tokenize(input, &TokenizerOptions::default())
}
