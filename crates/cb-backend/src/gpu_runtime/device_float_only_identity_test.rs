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
    let packed = pack_cindex(&bins, &n_buckets, &vec![false; n_buckets.len()], n)
        .expect("pack_cindex must succeed");
    let (offsets, shifts, masks, _one_hot_flags) = packed
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

    let packed = pack_cindex(&bins, &n_buckets, &vec![false; n_buckets.len()], n)
        .expect("pack_cindex must succeed");
    let (offsets, shifts, masks, _one_hot_flags) = packed.device_arrays().expect("device_arrays");
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

/// Parse a `"key": [a,b,c]` `u32` list out of the hand-rolled fixture JSON. `cb-backend`
/// has no serde dependency and must not gain one for a test, so the reader mirrors
/// [`packed_cindex_json`]'s writer exactly.
fn parse_u32_list(json: &str, key: &str) -> Vec<u32> {
    let needle = format!("\"{key}\": [");
    let start = json
        .find(&needle)
        .map(|i| i + needle.len())
        .unwrap_or_else(|| panic!("fixture JSON has no `{key}` list"));
    let rest = json.get(start..).unwrap_or("");
    let end = rest
        .find(']')
        .unwrap_or_else(|| panic!("fixture JSON `{key}` list is unterminated"));
    let body = rest.get(..end).unwrap_or("");
    body.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<u32>()
                .unwrap_or_else(|e| panic!("fixture JSON `{key}` entry {s:?} is not a u32: {e}"))
        })
        .collect()
}

/// T29b fn 1 / SPEC-OH-31 — the packed cindex for a FLOAT-ONLY pool is BIT-IDENTICAL to
/// the artifact frozen at the plan-base SHA.
///
/// This is the device half of the "the float-only path is unchanged" claim, and it is the
/// half the CPU's "the added loop range is empty" argument does NOT provide: on the device
/// the float bins and the one-hot bins share ONE concatenated feature axis, so there is no
/// analogous emptiness guarantee. The one-hot work adds a `one_hot` slice to `pack_cindex`
/// and a fourth `device_arrays()` array; either leaking into the packed words would break
/// every existing device oracle, and this makes that failure immediate and localized.
///
/// Regenerating the fixture is FORBIDDEN — a baseline regenerated after a change proves
/// nothing.
#[test]
fn packed_cindex_for_a_float_only_pool_is_bit_identical() {
    let path = device_baseline_dir().join("packed_cindex.json");
    let frozen = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "the frozen device baseline {} must be present (captured by T00): {e}",
            path.display()
        )
    });

    let (bins, n_buckets, n) = pinned_float_only_bins();
    let packed = pack_cindex(&bins, &n_buckets, &vec![false; n_buckets.len()], n)
        .expect("pack_cindex must succeed");
    let (offsets, shifts, masks, one_hot_flags) =
        packed.device_arrays().expect("device_arrays must succeed");

    assert_eq!(
        packed.words,
        parse_u32_list(&frozen, "words"),
        "the packed cindex WORDS for a float-only pool changed vs the plan-base baseline"
    );
    assert_eq!(
        offsets,
        parse_u32_list(&frozen, "offsets"),
        "the per-feature packed OFFSETS changed vs the plan-base baseline"
    );
    assert_eq!(
        shifts,
        parse_u32_list(&frozen, "shifts"),
        "the per-feature packed SHIFTS changed vs the plan-base baseline"
    );
    assert_eq!(
        masks,
        parse_u32_list(&frozen, "masks"),
        "the per-feature packed MASKS changed vs the plan-base baseline"
    );
    assert!(
        one_hot_flags.iter().all(|&f| f == 0),
        "a float-only pool must carry NO one-hot flag; the fourth `device_arrays()` array \
         is an ADDITION that leaves the first three byte-unchanged"
    );
}

/// T29b fn 2 / SPEC-OH-31 — the float-only SCORER winner is numerically unchanged by the
/// one-hot arm.
///
/// The scorer gained `one_hot` / `feature_lo` / `feature_hi` / `real_folds`. On the
/// float-only launch (`one_hot = false`, `feature_lo = 0`, `feature_hi = n_features`) the
/// added arithmetic must collapse to exactly today's: `lo == 0`, `hi == n_features *
/// n_bins`, `c = ABSOLUTE_POS`, the sentinel `= n_candidates`, and the eligibility test
/// back to `border < max_border` with `real_folds` never read.
///
/// This asserts identity of the kernel's OUTPUT, not byte-identity of the kernel: adding
/// parameters changes the generated source by construction, so only output identity is
/// testable. The reference is computed here from the same closed-form CPU fold the
/// `one_hot_split_score_test` sibling uses, so the two cannot drift.
///
/// Runs only where the resident scorer can run: it grid-strides over `CUBE_COUNT`, which
/// cubecl-cpu rejects outright.
#[test]
fn float_only_scorer_winner_is_numerically_identical_per_level() {
    if !cfg!(any(feature = "rocm", feature = "cuda")) {
        println!(
            "[T29b] SKIP float_only_scorer_winner_is_numerically_identical_per_level: the \
             resident scorer needs a real device (cubecl-cpu rejects the CUBE_COUNT builtin)"
        );
        return;
    }
    // What is asserted here — and NOT duplicated from the sibling test: on the float-only
    // arm `real_folds` must be UNREAD. Two launches over the same histogram, differing
    // ONLY in `real_folds` (a truthful array vs deliberate garbage), must produce the
    // IDENTICAL winner. If a future change let the float arm consult `real_folds`, the
    // garbage run would diverge and this fails loudly.
    //
    // (The value-level "the float winner equals an independent CPU threshold-fold argmax"
    // claim is asserted once, in `gpu_runtime::one_hot_split_score_test::\
    // float_only_scorer_output_is_numerically_identical_after_the_one_hot_arm`. A second
    // copy of that reference is exactly how two "identical" expectations drift.)
    let device = <crate::SelectedRuntime as cubecl::Runtime>::Device::default();
    let client = <crate::SelectedRuntime as cubecl::Runtime>::client(&device);

    let n_parts = 2usize;
    let n_features = 3usize;
    let n_bins = 8usize;
    let lambda = 2.0_f64;
    let mut bin_sums = vec![0u64; n_parts * n_features * n_bins * 2];
    for part in 0..n_parts {
        for f in 0..n_features {
            for b in 0..n_bins {
                let w = 1.0 + ((part * 5 + f * 3 + b) % 7) as f64;
                let d = ((part * 11 + f * 13 + b * 3) % 9) as f64 - 4.0;
                let base = part * (n_features * n_bins * 2) + (f * n_bins + b) * 2;
                let enc = |v: f64| ((v * crate::kernels::REDUCE_FIXEDPOINT_SCALE_F64).round()
                    as i64) as u64;
                if let Some(slot) = bin_sums.get_mut(base) {
                    *slot = enc(w);
                }
                if let Some(slot) = bin_sums.get_mut(base + 1) {
                    *slot = enc(d);
                }
            }
        }
    }

    let run = |real_folds: &[u32]| {
        let handle = client.create(cubecl::bytes::Bytes::from_elems(bin_sums.clone()));
        super::score_partition_over_binsums(
            &client,
            handle,
            n_parts,
            n_bins,
            n_bins,
            n_features,
            lambda,
            super::SCORE_FN_L2,
            real_folds,
            /* one_hot = */ false,
            /* feature_lo = */ 0,
            /* feature_hi = */ n_features,
        )
        .expect("the float-only scorer launch must succeed")
    };

    // Truthful bound vs deliberate garbage (`1` would exclude every border but bin 0 if
    // the float arm ever consulted it).
    let truthful = run(&vec![n_bins as u32; n_features]);
    let garbage = run(&[1u32, 1, 1]);

    let a = truthful.expect("a float-only level must produce a winner");
    let b = garbage.expect("the float arm must not be bounded by `real_folds`");
    assert_eq!(
        (a.feature_id, a.bin_id),
        (b.feature_id, b.bin_id),
        "the float-only winner changed when `real_folds` changed — the `one_hot == false` \
         arm must keep the unchanged `border < max_border` eligibility and never read \
         `real_folds` (SPEC-OH-31)"
    );
    assert_eq!(
        a.gain.to_bits(),
        b.gain.to_bits(),
        "the float-only GAIN must be bit-identical across the two `real_folds` values"
    );
}
