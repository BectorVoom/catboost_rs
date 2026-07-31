//! T00 / SPEC-OH-31 — the DEVICE float-only identity baseline.
//!
//! # Why this lives here and not in `crates/cb-backend/tests/`
//!
//! The artifacts this captures come from `pack_cindex` and
//! `PackedCindex::device_arrays`, which are `pub(crate)` inside a `pub(crate)`
//! module — an integration test under `tests/` links against the crate's PUBLIC
//! surface only and cannot reach them. A `#[cfg(test)] mod` declared in
//! `gpu_runtime/mod.rs` is a descendant of `gpu_runtime`, so it sees both. This
//! is the same placement the existing `session_residency` (`mod.rs:753`) and
//! `session_depth_gt1_test` (`mod.rs:760`) siblings use.
//!
//! # What T00 captures vs what T29b asserts
//!
//! T00 leaves ONLY the capture fn below, run once at the plan-base SHA. T29b
//! later adds the assertion fns to this same file. Splitting them this way is
//! what makes the device half of SPEC-OH-31 provable against PRE-change bytes
//! rather than degenerating into a self-comparison.
//!
//! `packed_cindex.json` is pure HOST bit-packing — no GPU is required to capture
//! or to compare it, so the float-only packing invariant is verifiable on any
//! machine. That matters because the one-hot work adds a real-cardinality array
//! alongside `TCFeature.folds`, and a change that perturbed the packed words for
//! a float-only pool would break every existing device oracle.

use std::path::PathBuf;

use super::cindex::pack_cindex;

/// The frozen device-baseline directory (a `device/` subdir of the T00 fixture).
fn device_baseline_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join("float_only_byte_identity")
        .join("device")
}

/// The pinned float-only quantized input: 4 features x 64 objects, bins cycling
/// through each feature's bucket count. Deterministic by construction — no RNG, so
/// the packed words are a pure function of these constants.
fn pinned_float_only_bins() -> (Vec<u32>, Vec<usize>, usize) {
    let n = 64usize;
    let n_buckets = vec![32usize, 32, 32, 32];
    let mut bins = Vec::with_capacity(n_buckets.len() * n);
    for (f, &buckets) in n_buckets.iter().enumerate() {
        for i in 0..n {
            bins.push(((i + f * 7) % buckets) as u32);
        }
    }
    (bins, n_buckets, n)
}

/// Serialize the packed cindex as stable JSON (hand-rolled: `cb-backend` has no
/// serde dependency and this must not add one for a test).
fn packed_cindex_json(words: &[u32], offsets: &[u32], shifts: &[u32], masks: &[u32]) -> String {
    let list = |v: &[u32]| {
        v.iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",")
    };
    format!(
        "{{\n  \"words\": [{}],\n  \"offsets\": [{}],\n  \"shifts\": [{}],\n  \"masks\": [{}]\n}}\n",
        list(words),
        list(offsets),
        list(shifts),
        list(masks)
    )
}

/// CAPTURE ONLY — freezes the device float-only baseline. Run ONCE, at the
/// plan-base SHA:
///
/// ```text
/// cargo test -p cb-backend --lib gpu_runtime::device_float_only_identity_test -- --ignored
/// ```
///
/// `#[ignore]`d so a routine `cargo test` can never rewrite the bytes T29b
/// compares against.
#[test]
#[ignore = "capture-only: run once at the plan-base SHA to freeze the fixture"]
fn capture_float_only_device_artifacts() {
    let dir = device_baseline_dir();
    std::fs::create_dir_all(&dir).expect("create device baseline dir");

    let (bins, n_buckets, n) = pinned_float_only_bins();
    let packed = pack_cindex(&bins, &n_buckets, n).expect("pack_cindex must succeed");
    let (offsets, shifts, masks) = packed
        .device_arrays()
        .expect("device_arrays must succeed");

    std::fs::write(
        dir.join("packed_cindex.json"),
        packed_cindex_json(&packed.words, &offsets, &shifts, &masks),
    )
    .expect("write packed_cindex.json");

    std::fs::write(
        dir.join("README.md"),
        "# Device float-only identity baseline (SPEC-OH-31 / T00)\n\
         \n\
         ## THIS FIXTURE IS FROZEN\n\
         \n\
         Captured at the plan-base SHA recorded in the parent directory's\n\
         `README.md`, BEFORE any one-hot production change. No later task may\n\
         regenerate it — a baseline regenerated after a change proves nothing.\n\
         \n\
         ## Contents\n\
         \n\
         - `packed_cindex.json` — `(words, offsets, shifts, masks)` for the pinned\n\
           float-only quantized input (4 features x 64 objects, 32 buckets each).\n\
           Pure HOST bit-packing: capturable and comparable with no GPU.\n\
         \n\
         ## Why it matters\n\
         \n\
         The one-hot work adds a separate real-cardinality array alongside\n\
         `TCFeature.folds` and threads `feature_lo`/`feature_hi` through the\n\
         scorer. Any of that leaking into the float-only packing would change\n\
         these words and break every existing device oracle. This fixture makes\n\
         that failure loud and immediate instead of subtle.\n\
         \n\
         ## Not captured here\n\
         \n\
         `scorer_winners.json` and `device_baseline.cbm` require a live GPU\n\
         session (`score_partition_over_binsums` needs a client and real bin\n\
         sums). They are captured on a GPU-enabled run; their absence is why\n\
         T29b's scorer assertion is gated on the artifact being present.\n",
    )
    .expect("write device README.md");
}

/// Guard: the pinned input must actually exercise multi-feature packing, or the
/// frozen artifact would be trivially stable and prove nothing about the packing
/// path the one-hot change touches.
#[test]
fn pinned_device_input_exercises_multi_feature_packing() {
    let (bins, n_buckets, n) = pinned_float_only_bins();
    assert!(
        n_buckets.len() > 1,
        "the pinned input must cover more than one feature"
    );
    assert_eq!(
        bins.len(),
        n_buckets.len() * n,
        "the plain cindex layout is exactly n_features * n cells"
    );

    let packed = pack_cindex(&bins, &n_buckets, n).expect("pack_cindex must succeed");
    let (offsets, shifts, masks) = packed.device_arrays().expect("device_arrays");
    assert_eq!(offsets.len(), n_buckets.len());
    assert_eq!(shifts.len(), n_buckets.len());
    assert_eq!(masks.len(), n_buckets.len());

    // At least two features must share a word group (a non-zero shift), otherwise
    // the packing is degenerate and would not detect an addressing regression.
    assert!(
        shifts.iter().any(|&s| s != 0),
        "the pinned input must produce at least one non-zero shift (shared word \
         group), else the packing artifact cannot detect an addressing change"
    );
}
