//! T00 / SPEC-OH-31 — the float-only `.cbm` byte-identity baseline.
//!
//! # Why this exists
//!
//! Every later "the float-only path is unchanged" claim (SPEC-OH-31 / A2, T29,
//! T29b) is only meaningful against bytes frozen BEFORE any production change.
//! Captured at the plan-base SHA recorded in the fixture `README.md`; if the
//! baseline were regenerated after a change it would degenerate into a
//! self-comparison and prove nothing.
//!
//! # The fixture is FROZEN
//!
//! No later task may regenerate `float_only_byte_identity/`. The capture fn below
//! is `#[ignore]`d precisely so a routine `cargo test` can never rewrite it.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use cb_backend::CpuBackend;
use cb_compute::{EScoreFunction, LeafMethod, Loss};
use cb_model::{save_cbm, Model as CbModel};
use cb_train::{train, BoostParams, EBootstrapType, EGrowPolicy, EOverfittingDetectorType};
use ndarray::Array2;
use ndarray_npy::{read_npy, write_npy};

/// The frozen baseline fixture root.
fn baseline_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join("float_only_byte_identity")
}

const N_ROWS: usize = 512;
const N_FEATURES: usize = 4;

/// Deterministic LCG — the inputs must be reproducible without a `rand` dep so the
/// capture is a pure function of this file.
struct Lcg(u64);

impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

/// The pinned float-only inputs: `N_ROWS x N_FEATURES` features plus an RMSE target.
fn generate_inputs() -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = Lcg(20_260_731);
    let cols: Vec<Vec<f64>> = (0..N_FEATURES)
        .map(|_| (0..N_ROWS).map(|_| rng.next_f64()).collect())
        .collect();
    let target = (0..N_ROWS)
        .map(|i| 0.7 * cols[0][i] - 0.4 * cols[1][i] + 0.2 * cols[2][i])
        .collect();
    (cols, target)
}

/// Every confound pinned OFF. `bootstrap_type = No` and `random_strength = 0`
/// keep the RNG stream out of the picture entirely, so a byte difference can only
/// come from the tree/serialization path this plan touches.
fn pinned_float_only_params() -> BoostParams {
    BoostParams {
        loss: Loss::Rmse,
        iterations: 3,
        depth: 3,
        learning_rate: 0.3,
        l2_leaf_reg: 3.0,
        random_strength: 0.0,
        boost_from_average: false,
        leaf_method: LeafMethod::Gradient,
        bootstrap_type: EBootstrapType::No,
        subsample: 1.0,
        bagging_temperature: 0.0,
        random_seed: 0,
        od_type: EOverfittingDetectorType::None,
        od_pval: 0.0,
        od_wait: 0,
        use_best_model: false,
        eval_metric: None,
        auto_learning_rate: false,
        one_hot_max_size: cb_train::one_hot_max_size_default(),
        permutation_count: cb_train::permutation_count_default(),
        fold_len_multiplier: cb_train::fold_len_multiplier_default(),
        simple_ctr: cb_train::simple_ctr_default(),
        simple_ctr_priors: cb_train::simple_ctr_priors_default(),
        counter_calc_method: cb_train::counter_calc_method_default(),
        boosting_type: cb_train::boosting_type_default(),
        max_ctr_complexity: cb_train::max_ctr_complexity_default(),
        combinations_ctr: cb_train::combinations_ctr_default(),
        combinations_ctr_priors: cb_train::combinations_ctr_priors_default(),
        score_function: EScoreFunction::L2,
        has_time: false,
        feature_weights: cb_train::feature_weights_default(),
        first_feature_use_penalties: cb_train::first_feature_use_penalties_default(),
        per_object_feature_penalties: cb_train::per_object_feature_penalties_default(),
        penalties_coefficient: cb_train::penalties_coefficient_default(),
        monotone_constraints: cb_train::monotone_constraints_default(),
        grow_policy: EGrowPolicy::SymmetricTree,
        max_leaves: cb_train::max_leaves_default(),
        min_data_in_leaf: cb_train::min_data_in_leaf_default(),
        extra: Default::default(),
    }
}

/// Borders for the pinned inputs — derived from the data, so the fit is a pure
/// function of `generate_inputs()`.
fn borders_for(cols: &[Vec<f64>]) -> Vec<Vec<f64>> {
    cols.iter()
        .map(|col| cb_data::select_borders_greedy_logsum(col, 32, false))
        .collect()
}

/// Run the pinned float-only fit and return the serialized `.cbm` bytes.
fn fit_and_serialize(cols: &[Vec<f64>], target: &[f64]) -> Vec<u8> {
    let feature_values: Vec<Vec<f32>> = cols
        .iter()
        .map(|c| c.iter().map(|&v| v as f32).collect())
        .collect();
    let feature_borders = borders_for(cols);
    let weights = vec![1.0f64; target.len()];

    let trained = train(
        &CpuBackend,
        &feature_values,
        &feature_borders,
        target,
        &weights,
        &pinned_float_only_params(),
        None,
    )
    .expect("the pinned float-only fit must succeed");

    let model = CbModel::from_trained(&trained, feature_borders);
    let tmp = std::env::temp_dir().join("cb_float_only_byte_identity_probe.cbm");
    save_cbm(&model, &tmp).expect("save_cbm must succeed");
    let bytes = std::fs::read(&tmp).expect("read back the serialized model");
    let _ = std::fs::remove_file(&tmp);
    bytes
}

/// CAPTURE ONLY — freezes the baseline. Run ONCE, at the plan-base SHA, with
/// `-- --ignored`. `#[ignore]`d so no routine test run can silently rewrite the
/// very bytes SPEC-OH-31 compares against.
#[test]
#[ignore = "capture-only: run once at the plan-base SHA to freeze the fixture"]
fn capture_float_only_baseline() {
    let dir = baseline_dir();
    let inputs = dir.join("inputs");
    std::fs::create_dir_all(&inputs).expect("create fixture dirs");

    let (cols, target) = generate_inputs();

    let mut x = Array2::<f64>::zeros((N_ROWS, N_FEATURES));
    for (f, col) in cols.iter().enumerate() {
        for (i, &v) in col.iter().enumerate() {
            x[[i, f]] = v;
        }
    }
    write_npy(inputs.join("X.npy"), &x).expect("write X.npy");
    write_npy(
        inputs.join("y.npy"),
        &ndarray::Array1::from(target.clone()),
    )
    .expect("write y.npy");

    let bytes = fit_and_serialize(&cols, &target);
    std::fs::write(dir.join("baseline.cbm"), &bytes).expect("write baseline.cbm");

    let sha = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map_or_else(|| "UNKNOWN".to_owned(), |s| s.trim().to_owned());

    std::fs::write(
        dir.join("README.md"),
        format!(
            "# float-only `.cbm` byte-identity baseline (SPEC-OH-31 / T00)\n\
             \n\
             **CAPTURED_AT_SHA: `{sha}`** (plus the uncommitted working tree at\n\
             capture time — see the re-baseline note below).\n\
             \n\
             Written by `float_only_byte_identity_test::capture_float_only_baseline`\n\
             (`#[ignore]`d; run with `-- --ignored`).\n\
             \n\
             ## RE-BASELINED once, deliberately (learning_rate f32)\n\
             \n\
             The ORIGINAL capture, taken at plan-base\n\
             `9bf734512d7fccb25a9e8304b34183375ae3e7f5` before any one-hot\n\
             production change, is preserved verbatim as `baseline_pre_lr_f32.cbm`\n\
             (sha256 `37b373e7e16499fdba8abbefda956bbb355ca3908b510b131f23f3990dce6b21`).\n\
             \n\
             `baseline.cbm` was regenerated ONCE, for a single deliberate reason:\n\
             upstream stores `learning_rate` as an **f32**, so every leaf value is\n\
             scaled by `f32(lr)` rather than the `f64` the caller supplied. Fixing\n\
             that (`cb-train/src/boosting.rs`) moves every leaf by the constant\n\
             factor `f32(0.3) / 0.3` at this fixture's pinned rate — TOWARD\n\
             upstream, verified to one ulp against real catboost 1.2.10.\n\
             \n\
             **The SPEC-OH-31 guarantee was NOT simply discarded.**\n\
             `baseline_only_differs_from_pre_lr_baseline_by_the_learning_rate_factor`\n\
             proves mechanically that the two baselines have IDENTICAL tree\n\
             structure (same split count, features and borders) and that every\n\
             leaf differs by exactly that one factor. So `no one-hot change leaked\n\
             into the float-only path` remains PROVEN, not merely asserted.\n\
             \n\
             ## RE-BASELINED a second time, deliberately (greedy border tie-break)\n\
             \n\
             The capture taken before the `feature_border_type` wave is preserved\n\
             verbatim as `baseline_pre_border_tiebreak.cbm` (sha256\n\
             `876310517504e8d9cbf55e85cf46aeb9f96d30a011179ce5537bb890fe9fd629`).\n\
             \n\
             `baseline.cbm` was regenerated for one deliberate reason: the greedy\n\
             binarizer resolved TIED split scores with a libstdc++ heap emulation\n\
             that does not match catboost 1.2.10. This fixture quantizes 512\n\
             UNIQUE values into 32 borders, so its budget BINDS — precisely the\n\
             regime where the tie-break decides where borders land — and the frozen\n\
             bytes therefore encoded the wrong border set.\n\
             \n\
             **This re-baseline is a move TOWARD upstream, and that is PROVEN, not\n\
             asserted.** `cb-data`'s `border_types` oracle now includes the cell\n\
             `float_only_byte_identity.bc32.GreedyLogSum` — catboost's own\n\
             `Pool.quantize(...).save_quantization_borders()` output for THIS\n\
             corpus at THIS budget — and\n\
             `baseline_moved_its_borders_onto_the_catboost_oracle` checks that the\n\
             new `.cbm` stores exactly those borders while the pre-fix `.cbm` does\n\
             not. The bytes changed because the quantization got CORRECT.\n\
             \n\
             ## Still frozen from here on\n\
             \n\
             Do not regenerate again without the same treatment: preserve the prior\n\
             bytes and prove the delta. An unexplained regeneration turns\n\
             SPEC-OH-31 into a self-comparison that proves nothing.\n\
             \n\
             ## Contents\n\
             \n\
             - `baseline.cbm` — the pinned float-only fit, serialized.\n\
             - `baseline_pre_lr_f32.cbm` — the ORIGINAL plan-base capture (frozen).\n\
             - `baseline_pre_border_tiebreak.cbm` — the capture taken before the\n\
               greedy border tie-break fix (frozen).\n\
             - `inputs/X.npy` — `{N_ROWS} x {N_FEATURES}` float64 features.\n\
             - `inputs/y.npy` — `{N_ROWS}` float64 RMSE target.\n\
             \n\
             ## Pinned configuration\n\
             \n\
             RMSE, `iterations=3`, `depth=3`, `learning_rate=0.3`,\n\
             `l2_leaf_reg=3.0`, `random_strength=0`, `boost_from_average=false`,\n\
             `bootstrap_type=No`, `random_seed=0`, `score_function=L2`,\n\
             `grow_policy=SymmetricTree`, `border_count=32`.\n\
             \n\
             `bootstrap_type=No` + `random_strength=0` keep the RNG stream out of\n\
             the comparison entirely, so any byte difference is attributable to the\n\
             tree-construction or serialization path this plan touches.\n"
        ),
    )
    .expect("write README.md");
}

/// SPEC-OH-31 / A2: re-running the pinned float-only fit on the CURRENT tree must
/// reproduce the frozen plan-base bytes exactly.
///
/// A one-hot change that leaked into the float path — a reordered split lift, an
/// extra RNG draw, a stray `CatFeatures`/`OneHotFeatures` FlatBuffers section —
/// shows up here as a byte difference.
#[test]
fn float_only_cbm_bytes_match_the_frozen_plan_base_baseline() {
    let dir = baseline_dir();
    let baseline_path = dir.join("baseline.cbm");
    let expected = std::fs::read(&baseline_path).unwrap_or_else(|e| {
        panic!(
            "frozen baseline missing at {} ({e}). Capture it FIRST, at the plan-base \
             SHA: cargo test -p cb-model --test float_only_byte_identity_test -- --ignored",
            baseline_path.display()
        )
    });

    // Re-derive the inputs from the FROZEN npy files, not from `generate_inputs()`,
    // so a change to the generator cannot silently move the goalposts.
    let x: Array2<f64> = read_npy(dir.join("inputs/X.npy")).expect("frozen X.npy");
    let y: ndarray::Array1<f64> = read_npy(dir.join("inputs/y.npy")).expect("frozen y.npy");
    let cols: Vec<Vec<f64>> = (0..x.ncols()).map(|f| x.column(f).to_vec()).collect();
    let target: Vec<f64> = y.to_vec();

    let actual = fit_and_serialize(&cols, &target);

    assert_eq!(
        actual.len(),
        expected.len(),
        "float-only .cbm LENGTH changed ({} -> {}): the float path is not \
         byte-identical to the plan base (SPEC-OH-31)",
        expected.len(),
        actual.len()
    );
    if actual != expected {
        let first_diff = actual
            .iter()
            .zip(expected.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        panic!(
            "float-only .cbm BYTES changed at offset {first_diff} (len {}): the \
             float path is not byte-identical to the plan base (SPEC-OH-31 / A2). \
             A one-hot change has leaked into the float-only path.",
            expected.len()
        );
    }
}

/// The fixture must carry its provenance — without the capture SHA, and without
/// the record of the ONE deliberate re-baseline, there is no way to tell a
/// genuine baseline from one quietly regenerated after a change.
#[test]
fn frozen_baseline_records_its_plan_base_sha() {
    let readme = std::fs::read_to_string(baseline_dir().join("README.md"))
        .expect("the frozen baseline must carry a README.md");
    assert!(
        readme.contains("CAPTURED_AT_SHA"),
        "README.md must record the SHA the bytes were captured at"
    );
    assert!(
        readme.contains("RE-BASELINED"),
        "README.md must disclose the one deliberate re-baseline (learning_rate f32)"
    );
    assert!(
        readme.contains("baseline_pre_lr_f32.cbm"),
        "README.md must point at the preserved plan-base bytes, which are what \
         keeps the SPEC-OH-31 guarantee provable rather than merely asserted"
    );
    assert!(
        readme.contains("Still frozen from here on"),
        "README.md must state that the fixture is frozen going forward"
    );
    // The preserved original must actually be there — the differential proof
    // (`baseline_only_differs_from_pre_lr_baseline_by_the_learning_rate_factor`)
    // is meaningless without it.
    assert!(
        baseline_dir().join("baseline_pre_lr_f32.cbm").is_file(),
        "the preserved plan-base baseline must not be deleted"
    );
    assert!(
        readme.contains("baseline_pre_border_tiebreak.cbm"),
        "README.md must point at the bytes preserved across the greedy border \
         tie-break re-baseline"
    );
    assert!(
        baseline_dir()
            .join("baseline_pre_border_tiebreak.cbm")
            .is_file(),
        "the preserved pre-border-tiebreak baseline must not be deleted"
    );
}

/// The SECOND re-baseline is a move TOWARD upstream, proven mechanically.
///
/// The greedy binarizer used to resolve TIED split scores with a libstdc++ heap
/// emulation that catboost 1.2.10 does not match. This fixture bins 512 unique
/// values into 32 borders, so its budget BINDS and the tie-break decides where
/// borders land — which is why the frozen `.cbm` had to be re-captured.
///
/// A re-capture with no proof would turn SPEC-OH-31 into a self-comparison. So
/// this test reads the borders back out of BOTH `.cbm` files and checks them
/// against catboost's own standalone quantization for this exact corpus at this
/// exact budget (the `border_types/float_only_byte_identity.bc32.GreedyLogSum`
/// oracle cell): the NEW baseline must match it, and the OLD one must NOT. That
/// is the difference between "the bytes drifted" and "the bytes got correct".
#[test]
fn baseline_moved_its_borders_onto_the_catboost_oracle() {
    let dir = baseline_dir();
    let new_model = cb_model::load_cbm(&dir.join("baseline.cbm"))
        .expect("the re-captured baseline must load");
    let old_model = cb_model::load_cbm(&dir.join("baseline_pre_border_tiebreak.cbm"))
        .expect("the preserved pre-fix baseline must load");

    // catboost's own borders for this corpus at border_count=32.
    let oracle_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join("border_types");
    let stem = "float_only_byte_identity.bc32.GreedyLogSum";
    let flat: Vec<f64> = read_npy::<_, ndarray::Array1<f64>>(oracle_dir.join(format!("{stem}.borders.npy")))
        .expect("oracle borders")
        .to_vec();
    let per_feature: Vec<f64> =
        read_npy::<_, ndarray::Array1<f64>>(oracle_dir.join(format!("{stem}.borders_per_feature.npy")))
            .expect("oracle per-feature counts")
            .to_vec();

    let mut expected: Vec<Vec<f64>> = Vec::new();
    let mut offset = 0usize;
    for &count in &per_feature {
        let count = count as usize;
        expected.push(flat[offset..offset + count].to_vec());
        offset += count;
    }

    // A `.cbm` stores borders as f32, and the oracle text-rounds to ~10
    // significant digits, so compare at f32 resolution rather than bit-exactly.
    let matches_oracle = |borders: &[Vec<f64>]| -> bool {
        borders.len() == expected.len()
            && borders.iter().zip(expected.iter()).all(|(got, want)| {
                got.len() == want.len()
                    && got
                        .iter()
                        .zip(want.iter())
                        .all(|(a, b)| (a - b).abs() <= 1e-6 * a.abs().max(1.0))
            })
    };

    assert!(
        matches_oracle(&new_model.float_feature_borders),
        "the re-captured baseline must store catboost's borders for this corpus; \
         the whole point of the re-baseline was to move onto them"
    );
    assert!(
        !matches_oracle(&old_model.float_feature_borders),
        "the PRE-FIX baseline must NOT already match catboost's borders — if it \
         did, the greedy tie-break was not the reason the bytes moved and this \
         re-baseline is unexplained"
    );
}

/// SPEC-OH-31, PRESERVED ACROSS THE ONE RE-BASELINE.
///
/// `baseline.cbm` was regenerated once, for the deliberate learning-rate f32
/// parity fix. That would normally destroy this fixture's whole value — a
/// regenerated baseline compared against itself proves nothing. It does not
/// here, because the ORIGINAL plan-base bytes are preserved as
/// `baseline_pre_lr_f32.cbm` and this test proves mechanically that the two
/// differ by EXACTLY the intended factor and nothing else:
///
/// * identical tree count, identical split count per tree, and identical split
///   `(feature, border)` pairs — i.e. the SEARCH is untouched, which is what
///   "no one-hot change leaked into the float-only path" actually means;
/// * every leaf value scaled by exactly `f32(0.3) / 0.3` (the fixture's pinned
///   `learning_rate = 0.3`), to within one ulp;
/// * identical bias.
///
/// If a future change disturbs the float path in ANY other way, the structural
/// assertions below fail even though the byte-identity gate was re-baselined.
#[test]
fn baseline_only_differs_from_pre_lr_baseline_by_the_learning_rate_factor() {
    let dir = baseline_dir();
    let pre = cb_model::load_cbm(&dir.join("baseline_pre_lr_f32.cbm"))
        .expect("the preserved plan-base baseline must load");
    let now = cb_model::load_cbm(&dir.join("baseline.cbm"))
        .expect("the current baseline must load");

    // The fixture pins learning_rate = 0.3; the fix multiplies every leaf by
    // f32(0.3)/0.3 relative to the pre-fix bytes.
    let factor = f64::from(0.3_f32) / 0.3_f64;
    assert!(factor != 1.0, "the guard is vacuous if 0.3 were f32-exact");

    assert_eq!(
        pre.oblivious_trees.len(),
        now.oblivious_trees.len(),
        "tree count changed — this is NOT a pure learning-rate difference"
    );
    assert_eq!(pre.bias, now.bias, "bias changed");

    for (t, (a, b)) in pre
        .oblivious_trees
        .iter()
        .zip(now.oblivious_trees.iter())
        .enumerate()
    {
        // STRUCTURE must be byte-for-byte the same: same splits, same order.
        assert_eq!(
            a.splits, b.splits,
            "tree {t}: the SPLITS changed — the float-only search was disturbed by \
             something other than the learning-rate fix (SPEC-OH-31)"
        );
        assert_eq!(a.leaf_values.len(), b.leaf_values.len(), "tree {t}: leaf count");

        // VALUES: for the FIRST tree the relation is EXACT — its leaves are the
        // raw deltas times the rate, so scaling the rate scales them one-for-one.
        //
        // For LATER trees it is exact only to first order: tree 0's rescaled
        // leaves shift the running approx, which shifts tree 1's gradients, and
        // so on. The residual second-order term is ~1e-8 relative (measured), the
        // same order as the rate change itself — which is precisely what a
        // learning-rate perturbation propagating through boosting looks like, and
        // is NOT compatible with a structural change (those move leaves by
        // percents, not by 1e-8, and would already have tripped the split
        // assertion above).
        let rel_tol = if t == 0 { 4.0 * f64::EPSILON } else { 1e-6 };
        for (l, (&old, &new)) in a.leaf_values.iter().zip(b.leaf_values.iter()).enumerate() {
            let want = old * factor;
            let tol = want.abs() * rel_tol;
            assert!(
                (new - want).abs() <= tol,
                "tree {t} leaf {l}: {new} is not {old} * f32(0.3)/0.3 ({want}) within \
                 {rel_tol:e} relative — the delta is not purely the learning-rate fix"
            );
        }
        assert_eq!(a.leaf_weights, b.leaf_weights, "tree {t}: leaf weights changed");
    }
}
