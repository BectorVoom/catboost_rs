//! Unit tests for [`crate::snapshot`] (ORCH-03). Kept in a dedicated `*_test.rs`
//! file per the source/test separation rule (D-17); no `#[cfg(test)] mod` lives in
//! `snapshot.rs` itself.

use crate::snapshot::{
    decode, dto_from_tree, encode, tree_from_dto, ObliviousTreeDto, SnapshotConfig, SplitDto,
    TrainSnapshot, SNAPSHOT_FORMAT_VERSION,
};
use crate::tree::Split;
use crate::ObliviousTree;
use cb_core::CbError;

/// TASK-03 scaffold: the config type exists and carries the two upstream-mirroring
/// fields.
#[test]
fn snapshot_config_carries_path_and_interval() {
    let cfg = SnapshotConfig {
        snapshot_file: std::path::PathBuf::from("/tmp/snap.json"),
        snapshot_interval: std::time::Duration::from_secs(600),
    };
    assert_eq!(cfg.snapshot_file, std::path::PathBuf::from("/tmp/snap.json"));
    assert_eq!(cfg.snapshot_interval, std::time::Duration::from_secs(600));
}

/// A non-trivial snapshot: two trees of different depths, a length-6 `approx`,
/// non-zero RNG state and call count.
fn sample_snapshot() -> TrainSnapshot {
    TrainSnapshot {
        format_version: SNAPSHOT_FORMAT_VERSION,
        fingerprint: 0xDEAD_BEEF_CAFE_F00D,
        completed_iters: 2,
        bias: -0.125,
        approx_dimension: 1,
        approx: vec![0.5, -0.25, 0.0, 1.5, -2.0, 0.125],
        trees: vec![
            ObliviousTreeDto {
                splits: vec![SplitDto { feature: 0, border: 0.5 }],
                leaf_values: vec![0.1, -0.2],
                leaf_weights: vec![3.0, 3.0],
            },
            ObliviousTreeDto {
                splits: vec![
                    SplitDto { feature: 2, border: -1.25 },
                    SplitDto { feature: 1, border: 7.5 },
                ],
                leaf_values: vec![0.01, 0.02, -0.03, 0.04],
                leaf_weights: vec![1.0, 2.0, 2.0, 1.0],
            },
        ],
        rng_raw_state: [1, 3, 5, 7],
        rng_call_count: 11,
    }
}

/// ORCH-03-S3 / AT-S3: `decode(encode(s)) == s`, field for field.
#[test]
fn snapshot_roundtrip_is_lossless() {
    let s = sample_snapshot();
    let bytes = encode(&s).expect("encode must succeed for a finite snapshot");
    let back = decode(&bytes).expect("decode must accept what encode produced");
    assert_eq!(back, s);
}

/// ORCH-03-S3: an unknown `format_version` is REJECTED, not silently accepted —
/// otherwise a future format change would be read with today's field meanings.
#[test]
fn snapshot_decode_rejects_an_unknown_format_version() {
    let mut s = sample_snapshot();
    s.format_version = SNAPSHOT_FORMAT_VERSION + 1;
    let bytes = encode(&s).expect("encode does not police the version");

    match decode(&bytes) {
        Err(CbError::Snapshot(msg)) => {
            assert!(
                msg.contains("format_version"),
                "the error must name the offending field, got: {msg}"
            );
        }
        other => panic!("expected Err(Snapshot), got {other:?}"),
    }
}

/// Malformed bytes are a typed error, never a panic.
#[test]
fn snapshot_decode_rejects_garbage_without_panicking() {
    assert!(matches!(decode(b"not json at all"), Err(CbError::Snapshot(_))));
    assert!(matches!(decode(b""), Err(CbError::Snapshot(_))));
}

/// REGRESSION (found by the TASK-07 keystone oracle): `serde_json`'s decimal
/// round-trip for `f64` is NOT bit-exact. This exact leaf value — produced by a
/// real 4-iteration fit in `snapshot_resume_selfconsistency_oracle_test`'s corpus —
/// came back ONE ULP off, which made a resumed model differ from a straight-through
/// model in the last bit. The DTO therefore stores float BITS, and this test pins
/// that: it fails the moment anyone "simplifies" the `#[serde(with = ...)]`
/// attributes away.
#[test]
fn snapshot_roundtrip_is_bit_exact_for_a_value_json_decimals_perturb() {
    // The value whose decimal round-trip lost a bit, plus neighbours and the
    // adversarial classics.
    let hostile = [
        f64::from_bits(13_817_809_268_918_714_368),
        f64::from_bits(13_800_530_458_377_599_204),
        f64::from_bits(4_582_262_501_124_370_888),
        0.1,
        -0.0,
        f64::MIN_POSITIVE,
        f64::MAX,
        1.0 / 3.0,
    ];

    let mut s = sample_snapshot();
    s.approx = hostile.to_vec();
    s.bias = hostile[0];
    s.trees[0].leaf_values = hostile.to_vec();
    s.trees[0].leaf_weights = hostile.to_vec();
    s.trees[0].splits[0].border = hostile[1];

    let back = decode(&encode(&s).expect("encode")).expect("decode");

    for (i, (a, b)) in s.approx.iter().zip(back.approx.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "approx[{i}] must round-trip BIT-exactly ({a:?} -> {b:?}); a one-ULP drift here              silently breaks resume equality"
        );
    }
    assert_eq!(s.bias.to_bits(), back.bias.to_bits());
    assert_eq!(
        s.trees[0].splits[0].border.to_bits(),
        back.trees[0].splits[0].border.to_bits()
    );
    assert_eq!(back, s);
}

/// A non-finite `approx` / leaf value means the fit has already diverged, and a
/// checkpoint of a diverged fit can only resume into more divergence. Encoding
/// refuses it at the point where the run is still in hand.
#[test]
fn snapshot_encode_rejects_non_finite_values() {
    for (label, mutate) in [
        ("approx", (|s: &mut TrainSnapshot| s.approx[2] = f64::NAN) as fn(&mut TrainSnapshot)),
        ("bias", |s: &mut TrainSnapshot| s.bias = f64::INFINITY),
        ("leaf_values", |s: &mut TrainSnapshot| s.trees[0].leaf_values[1] = f64::NEG_INFINITY),
        ("leaf_weights", |s: &mut TrainSnapshot| s.trees[1].leaf_weights[0] = f64::NAN),
        ("border", |s: &mut TrainSnapshot| s.trees[0].splits[0].border = f64::NAN),
    ] {
        let mut s = sample_snapshot();
        mutate(&mut s);
        assert!(
            matches!(encode(&s), Err(CbError::Snapshot(_))),
            "a non-finite {label} must be refused at encode time"
        );
    }
}

/// The DTO conversions are inverse over the scoped (float-only) tree shape.
#[test]
fn tree_dto_conversion_roundtrips_a_float_only_tree() {
    let tree = ObliviousTree {
        splits: vec![
            Split { feature: 3, border: 0.25 },
            Split { feature: 0, border: -4.5 },
        ],
        ctr_splits: Vec::new(),
        one_hot_splits: Vec::new(),
        level_kinds: Vec::new(),
        leaf_values: vec![1.0, 2.0, 3.0, 4.0],
        leaf_weights: vec![5.0, 6.0, 7.0, 8.0],
    };
    let dto = dto_from_tree(&tree).expect("a float-only tree is snapshottable");
    assert_eq!(tree_from_dto(&dto), tree);
}

/// A tree carrying categorical structure is NOT representable by the slice-1 DTO.
/// Refusing here is the second line of defence behind TASK-06's scope guard: if a
/// categorical tree ever reached the snapshot writer, dropping its `ctr_splits` /
/// `one_hot_splits` / `level_kinds` would produce a checkpoint that silently
/// resumes into a DIFFERENT model.
#[test]
fn tree_dto_conversion_refuses_categorical_structure() {
    let base = ObliviousTree {
        splits: vec![Split { feature: 0, border: 0.5 }],
        ctr_splits: Vec::new(),
        one_hot_splits: Vec::new(),
        level_kinds: Vec::new(),
        leaf_values: vec![1.0, 2.0],
        leaf_weights: vec![1.0, 1.0],
    };

    let mut one_hot = base.clone();
    one_hot.one_hot_splits.push(crate::tree::OneHotSplit { feature: 0, value: 1 });
    assert!(
        matches!(dto_from_tree(&one_hot), Err(CbError::Snapshot(_))),
        "a one-hot split must not be silently dropped"
    );

    let mut kinds = base.clone();
    kinds.level_kinds.push(crate::tree::LevelKind::Float(0));
    assert!(
        matches!(dto_from_tree(&kinds), Err(CbError::Snapshot(_))),
        "a non-empty level_kinds order must not be silently dropped"
    );

    // The float-only base itself stays acceptable — the guard rejects the
    // categorical fields, not every tree.
    assert!(dto_from_tree(&base).is_ok());
}

// ---------------------------------------------------------------------------
// TASK-05 — compat fingerprint (ORCH-03-S4)
// ---------------------------------------------------------------------------

use crate::snapshot::{check_resume, fingerprint};
use crate::{BoostParams, EBoostingType};
use cb_compute::{EScoreFunction, LeafMethod, Loss};

/// The baseline scoped-regime params every sensitivity row mutates ONE field of.
fn base_params() -> BoostParams {
    BoostParams {
        loss: Loss::Rmse,
        iterations: 5,
        depth: 3,
        learning_rate: 0.1,
        l2_leaf_reg: 3.0,
        random_strength: 0.0,
        boost_from_average: true,
        leaf_method: LeafMethod::Gradient,
        bootstrap_type: cb_train_bootstrap_no(),
        subsample: 1.0,
        bagging_temperature: 0.0,
        random_seed: 42,
        od_type: crate::EOverfittingDetectorType::None,
        od_pval: 0.0,
        od_wait: 0,
        use_best_model: false,
        eval_metric: None,
        auto_learning_rate: false,
        one_hot_max_size: 2,
        permutation_count: crate::permutation_count_default(),
        fold_len_multiplier: crate::fold_len_multiplier_default(),
        simple_ctr: crate::simple_ctr_default(),
        simple_ctr_priors: crate::simple_ctr_priors_default(),
        counter_calc_method: crate::counter_calc_method_default(),
        boosting_type: EBoostingType::Plain,
        max_ctr_complexity: 0,
        combinations_ctr: crate::combinations_ctr_default(),
        combinations_ctr_priors: crate::combinations_ctr_priors_default(),
        score_function: EScoreFunction::Cosine,
        has_time: false,
        feature_weights: crate::feature_weights_default(),
        first_feature_use_penalties: crate::first_feature_use_penalties_default(),
        per_object_feature_penalties: crate::per_object_feature_penalties_default(),
        penalties_coefficient: crate::penalties_coefficient_default(),
        monotone_constraints: vec![0, 0],
        grow_policy: crate::EGrowPolicy::SymmetricTree,
        max_leaves: crate::max_leaves_default(),
        min_data_in_leaf: 1,
    }
}

fn cb_train_bootstrap_no() -> crate::EBootstrapType {
    crate::EBootstrapType::No
}

fn base_borders() -> Vec<Vec<f64>> {
    vec![vec![0.5, 1.5], vec![-1.0, 0.0, 1.0]]
}

fn base_target() -> Vec<f64> {
    vec![1.0, 0.0, 1.0, 1.0, 0.0, 0.0]
}

fn base_weights() -> Vec<f64> {
    vec![1.0; 6]
}

fn fp(params: &BoostParams, n: usize, borders: &[Vec<f64>], target: &[f64]) -> u64 {
    fingerprint(params, n, borders, target, &base_weights())
}

/// `fp` with an explicit weight column — the field-17 sensitivity test's entry
/// point (every other test holds the weights fixed at [`base_weights`]).
fn fp_w(
    params: &BoostParams,
    n: usize,
    borders: &[Vec<f64>],
    target: &[f64],
    weights: &[f64],
) -> u64 {
    fingerprint(params, n, borders, target, weights)
}

fn base_fp() -> u64 {
    fp(&base_params(), 6, &base_borders(), &base_target())
}

/// ORCH-03-S4 / AT-S4: identical inputs hash identically, twice in a row and from
/// two independently constructed `BoostParams`.
#[test]
fn fingerprint_is_deterministic() {
    assert_eq!(base_fp(), base_fp());
    let a = base_params();
    let b = base_params();
    assert_eq!(
        fp(&a, 6, &base_borders(), &base_target()),
        fp(&b, 6, &base_borders(), &base_target())
    );
}

/// ORCH-03-S4 / AT-S4, the CRITICAL-1 + CRITICAL-3 completeness test: EVERY hashed
/// field, mutated ALONE, must move the fingerprint. A field that does not appear
/// here (or does not move the hash) is a field whose change a resume would silently
/// accept — which is the entire failure mode this fingerprint exists to prevent.
#[test]
fn fingerprint_is_sensitive_to_every_hashed_field() {
    let rows: Vec<(&str, Box<dyn Fn() -> u64>)> = vec![
        // 1. loss DISCRIMINANT
        ("loss discriminant", Box::new(|| {
            let mut p = base_params();
            p.loss = Loss::Logloss;
            fp(&p, 6, &base_borders(), &base_target())
        })),
        // 1b. CRITICAL-3: same discriminant, changed PAYLOAD
        ("Quantile alpha payload", Box::new(|| {
            let mut p = base_params();
            p.loss = Loss::Quantile { alpha: 0.9, delta: 1e-6 };
            fp(&p, 6, &base_borders(), &base_target())
        })),
        ("Lq q payload", Box::new(|| {
            let mut p = base_params();
            p.loss = Loss::Lq { q: 3.0 };
            fp(&p, 6, &base_borders(), &base_target())
        })),
        ("Huber delta payload", Box::new(|| {
            let mut p = base_params();
            p.loss = Loss::Huber { delta: 2.0 };
            fp(&p, 6, &base_borders(), &base_target())
        })),
        ("Expectile alpha payload", Box::new(|| {
            let mut p = base_params();
            p.loss = Loss::Expectile { alpha: 0.7 };
            fp(&p, 6, &base_borders(), &base_target())
        })),
        ("Focal gamma payload", Box::new(|| {
            let mut p = base_params();
            p.loss = Loss::Focal { alpha: 0.25, gamma: 2.0 };
            fp(&p, 6, &base_borders(), &base_target())
        })),
        ("Tweedie variance_power payload", Box::new(|| {
            let mut p = base_params();
            p.loss = Loss::Tweedie { variance_power: 1.5 };
            fp(&p, 6, &base_borders(), &base_target())
        })),
        ("MultiQuantile alpha payload", Box::new(|| {
            let mut p = base_params();
            p.loss = Loss::MultiQuantile { alpha: vec![0.1], delta: 1e-6 };
            fp(&p, 6, &base_borders(), &base_target())
        })),
        // 2..6
        ("iterations", Box::new(|| {
            let mut p = base_params();
            p.iterations = 6;
            fp(&p, 6, &base_borders(), &base_target())
        })),
        ("depth", Box::new(|| {
            let mut p = base_params();
            p.depth = 4;
            fp(&p, 6, &base_borders(), &base_target())
        })),
        ("learning_rate", Box::new(|| {
            let mut p = base_params();
            p.learning_rate = 0.2;
            fp(&p, 6, &base_borders(), &base_target())
        })),
        ("l2_leaf_reg", Box::new(|| {
            let mut p = base_params();
            p.l2_leaf_reg = 4.0;
            fp(&p, 6, &base_borders(), &base_target())
        })),
        ("random_seed", Box::new(|| {
            let mut p = base_params();
            p.random_seed = 43;
            fp(&p, 6, &base_borders(), &base_target())
        })),
        // 7..13 — the CRITICAL-1 additions
        ("boosting_type", Box::new(|| {
            let mut p = base_params();
            p.boosting_type = EBoostingType::Ordered;
            fp(&p, 6, &base_borders(), &base_target())
        })),
        ("leaf_method", Box::new(|| {
            let mut p = base_params();
            p.leaf_method = LeafMethod::Newton;
            fp(&p, 6, &base_borders(), &base_target())
        })),
        ("score_function", Box::new(|| {
            let mut p = base_params();
            p.score_function = EScoreFunction::L2;
            fp(&p, 6, &base_borders(), &base_target())
        })),
        ("min_data_in_leaf", Box::new(|| {
            let mut p = base_params();
            p.min_data_in_leaf = 2;
            fp(&p, 6, &base_borders(), &base_target())
        })),
        ("monotone_constraints", Box::new(|| {
            let mut p = base_params();
            p.monotone_constraints = vec![1, 0];
            fp(&p, 6, &base_borders(), &base_target())
        })),
        ("boost_from_average", Box::new(|| {
            let mut p = base_params();
            p.boost_from_average = false;
            fp(&p, 6, &base_borders(), &base_target())
        })),
        ("auto_learning_rate", Box::new(|| {
            let mut p = base_params();
            p.auto_learning_rate = true;
            fp(&p, 6, &base_borders(), &base_target())
        })),
        // 14..16 — the data inputs
        ("n", Box::new(|| fp(&base_params(), 7, &base_borders(), &base_target()))),
        ("a feature border", Box::new(|| {
            let mut b = base_borders();
            b[1][2] = 1.25;
            fp(&base_params(), 6, &b, &base_target())
        })),
        ("a target value", Box::new(|| {
            let mut t = base_target();
            t[3] = 0.0;
            fp(&base_params(), 6, &base_borders(), &t)
        })),
        // 17. the EFFECTIVE per-object weight column. A `class_weights` /
        // `auto_class_weights` / `scale_pos_weight` change reaches the trainer
        // ONLY through this vector — none of them is a `BoostParams` field — so
        // without this the resume of a re-weighted run is silently accepted and
        // the first K trees are fit against a different weighting than the rest.
        ("a weight value", Box::new(|| {
            let mut w = base_weights();
            w[2] = 5.0;
            fp_w(&base_params(), 6, &base_borders(), &base_target(), &w)
        })),
        // The LENGTH prefix, independently of the values: a shorter column must
        // not collide with a longer one that shares its prefix.
        ("the weight column length", Box::new(|| {
            let w = vec![1.0; 5];
            fp_w(&base_params(), 6, &base_borders(), &base_target(), &w)
        })),
    ];

    let base = base_fp();
    for (label, compute) in rows {
        assert_ne!(
            compute(),
            base,
            "changing `{label}` alone must change the fingerprint — otherwise a resume \
             against a snapshot written with the other value is silently accepted"
        );
    }
}

/// A changed `monotone_constraints` LENGTH (not just an element) must also move the
/// hash: a length-only change with identical leading elements is exactly what a
/// naive element-wise fold without a length feed would miss.
#[test]
fn fingerprint_is_sensitive_to_collection_lengths() {
    let base = base_fp();

    let mut p = base_params();
    p.monotone_constraints = vec![0, 0, 0];
    assert_ne!(fp(&p, 6, &base_borders(), &base_target()), base);

    // A border moved BETWEEN features (same flattened byte sequence, different
    // per-feature grouping) must not collide.
    let regrouped = vec![vec![0.5, 1.5, -1.0], vec![0.0, 1.0]];
    assert_ne!(fp(&base_params(), 6, &regrouped, &base_target()), base);
}

/// ORCH-03-S4: a fingerprint mismatch is REJECTED with a typed error; an equal pair
/// is accepted.
#[test]
fn check_resume_rejects_a_fingerprint_mismatch() {
    assert_eq!(check_resume(7, 7), Ok(()));
    match check_resume(7, 8) {
        Err(CbError::Snapshot(msg)) => {
            assert!(msg.contains('7') && msg.contains('8'), "the error must name both: {msg}");
        }
        other => panic!("expected Err(Snapshot), got {other:?}"),
    }
}
