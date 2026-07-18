//! Unit tests for [`crate::cbm`] — CTR-01 (combined bin classification),
//! CTR-02 (`CtrSplit` reconstruction), CTR-03 (model-parts tail parsing), and
//! CTR-05 (robustness / typed rejection). Source/test separation (CLAUDE.md /
//! AGENTS.md) — no test body lives in `cbm.rs` itself.
//!
//! T2/T3 read the COMMITTED `crates/cb-oracle/fixtures/ctr_load/simple.cbm`
//! (Phase 23 T0, FROZEN — never regenerated) as the ground-truth source for
//! both the `TModelTrees` core (T2) and the model-parts tail (T3). The exact
//! byte-level constants asserted below (hashes / int_counts /
//! counter_denominator) were dissected DIRECTLY off that committed file (not
//! copied from the research doc — the research doc's transcription of the
//! Borders blob has an off-by-one; the values here are self-consistent: each
//! Borders bucket's `n0 + n1` equals the co-located Counter table's bucket
//! total, which cross-validates the decode).

use std::path::PathBuf;

use flatbuffers::FlatBufferBuilder;

use super::{
    build_combined_bins, build_ctr_features, build_tctr_feature, ctr_split_from,
    ctr_split_to_global_index, BinKind, CtrIdentity,
};
use crate::ctr_data::{ctr_base_key, decode_ctr_model_parts, Prior};
use crate::ctr_data_generated::ncat_boost_fbs::{
    ECtrType as TailECtrType, TCtrValueTable, TCtrValueTableArgs,
    TFeatureCombination as TailFeatureCombination,
    TFeatureCombinationArgs as TailFeatureCombinationArgs, TModelCtrBase as TailModelCtrBase,
    TModelCtrBaseArgs as TailModelCtrBaseArgs,
};
use crate::error::ModelError;
use crate::model_generated::ncat_boost_fbs::{
    root_as_tmodel_core, ECtrType as CoreECtrType, TCtrFeature, TCtrFeatureArgs,
    TFeatureCombination, TFeatureCombinationArgs, TFloatFeature, TFloatFeatureArgs, TModelCtr,
    TModelCtrArgs, TModelCtrBase, TModelCtrBaseArgs, TModelTrees, TModelTreesArgs, TOneHotFeature,
    TOneHotFeatureArgs,
};
use crate::model::ModelSplit;
use crate::ECtrType;

// ── Shared fixture-file helpers ──────────────────────────────────────────────

/// Resolve the committed CTR-load oracle fixture directory (Phase 23 T0,
/// FROZEN — never regenerated in CI).
fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join("ctr_load")
        .join(name)
}

fn read_fixture(name: &str) -> Vec<u8> {
    std::fs::read(fixture(name)).unwrap_or_else(|e| panic!("{name} must load: {e:?}"))
}

/// Split a committed `.cbm` buffer into `(core, tail)` mirroring `decode_cbm`'s
/// own framing (magic + ui32 LE declared core size).
fn split_core_and_tail(buf: &[u8]) -> (&[u8], &[u8]) {
    assert_eq!(&buf[0..4], b"CBM1", "fixture must carry the CBM1 magic");
    let declared = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
    (&buf[8..8 + declared], &buf[8 + declared..])
}

// ── Synthetic `TModelTrees` builder (T1 one-hot case, T5 one-hot-split case) ─

/// One synthetic CTR feature spec for [`build_trees_buf`].
struct CtrFeatureSpec {
    cat_features: Vec<i32>,
    ctr_type: CoreECtrType,
    target_border_idx: i32,
    prior_num: f32,
    prior_denom: f32,
    shift: f32,
    scale: f32,
    borders: Vec<f32>,
}

/// Build a standalone `TModelTrees` FlatBuffers root (NOT wrapped in a full
/// `TModelCore` — `build_combined_bins`/`ctr_split_from`/`reconstruct_model`
/// only need `&TModelTrees`) with the given float borders, one-hot feature
/// value counts, and CTR feature specs. Optionally carries `tree_splits` /
/// `tree_sizes` / `leaf_values` so the buffer can also drive
/// `super::reconstruct_model` end-to-end (T5's one-hot-split case).
#[allow(clippy::too_many_arguments)]
fn build_trees_buf(
    float_borders: &[Vec<f32>],
    one_hot_values: &[Vec<i32>],
    ctr_specs: &[CtrFeatureSpec],
    tree_splits: &[i32],
    tree_sizes: &[i32],
    leaf_values: &[f64],
) -> Vec<u8> {
    let mut fbb = FlatBufferBuilder::new();

    let mut float_offsets = Vec::with_capacity(float_borders.len());
    for (idx, borders) in float_borders.iter().enumerate() {
        let borders_vec = fbb.create_vector(borders);
        let ff = TFloatFeature::create(
            &mut fbb,
            &TFloatFeatureArgs {
                Index: idx as i32,
                FlatIndex: idx as i32,
                Borders: Some(borders_vec),
                ..TFloatFeatureArgs::default()
            },
        );
        float_offsets.push(ff);
    }
    let float_features = fbb.create_vector(&float_offsets);

    let mut one_hot_offsets = Vec::with_capacity(one_hot_values.len());
    for (idx, values) in one_hot_values.iter().enumerate() {
        let values_vec = fbb.create_vector(values);
        let oh = TOneHotFeature::create(
            &mut fbb,
            &TOneHotFeatureArgs {
                Index: idx as i32,
                Values: Some(values_vec),
                ..TOneHotFeatureArgs::default()
            },
        );
        one_hot_offsets.push(oh);
    }
    let one_hot_features = fbb.create_vector(&one_hot_offsets);

    let mut ctr_offsets = Vec::with_capacity(ctr_specs.len());
    for spec in ctr_specs {
        let cat_vec = fbb.create_vector(&spec.cat_features);
        let combination = TFeatureCombination::create(
            &mut fbb,
            &TFeatureCombinationArgs {
                CatFeatures: Some(cat_vec),
                ..TFeatureCombinationArgs::default()
            },
        );
        let base = TModelCtrBase::create(
            &mut fbb,
            &TModelCtrBaseArgs {
                FeatureCombination: Some(combination),
                CtrType: spec.ctr_type,
                TargetBorderClassifierIdx: 0,
            },
        );
        let ctr = TModelCtr::create(
            &mut fbb,
            &TModelCtrArgs {
                Base: Some(base),
                TargetBorderIdx: spec.target_border_idx,
                PriorNum: spec.prior_num,
                PriorDenom: spec.prior_denom,
                Shift: spec.shift,
                Scale: spec.scale,
            },
        );
        let borders_vec = fbb.create_vector(&spec.borders);
        let cf = TCtrFeature::create(
            &mut fbb,
            &TCtrFeatureArgs {
                Ctr: Some(ctr),
                Borders: Some(borders_vec),
            },
        );
        ctr_offsets.push(cf);
    }
    let ctr_features = fbb.create_vector(&ctr_offsets);

    let splits_vec = fbb.create_vector(tree_splits);
    let sizes_vec = fbb.create_vector(tree_sizes);
    let leaf_values_vec = fbb.create_vector(leaf_values);

    let trees = TModelTrees::create(
        &mut fbb,
        &TModelTreesArgs {
            FloatFeatures: Some(float_features),
            OneHotFeatures: Some(one_hot_features),
            CtrFeatures: Some(ctr_features),
            TreeSplits: Some(splits_vec),
            TreeSizes: Some(sizes_vec),
            LeafValues: Some(leaf_values_vec),
            Scale: 1.0,
            ..TModelTreesArgs::default()
        },
    );
    fbb.finish(trees, None);
    fbb.finished_data().to_vec()
}

// ── T1 — CTR-01 combined bin classification ─────────────────────────────────

#[test]
fn classify_bins_places_float_onehot_ctr_ranges() {
    // 2 float features: borders [1.0, 2.0] and [5.0] -> 3 float bins.
    // 1 one-hot feature: 3 values -> 3 one-hot bins (COUNTED, not representable).
    // 1 ctr feature: 2 borders -> 2 ctr bins.
    let buf = build_trees_buf(
        &[vec![1.0, 2.0], vec![5.0]],
        &[vec![10, 20, 30]],
        &[CtrFeatureSpec {
            cat_features: vec![0],
            ctr_type: CoreECtrType::Borders,
            target_border_idx: 0,
            prior_num: 0.0,
            prior_denom: 1.0,
            shift: 0.0,
            scale: 1.0,
            borders: vec![0.1, 0.2],
        }],
        &[],
        &[],
        &[],
    );
    let trees = flatbuffers::root::<TModelTrees>(&buf).expect("verifying root parse");

    let combined = build_combined_bins(&trees).expect("classification must succeed");
    assert_eq!(combined.len(), 8, "3 float + 3 one-hot + 2 ctr bins");

    assert_eq!(
        combined[0],
        BinKind::Float {
            feature: 0,
            border: 1.0
        }
    );
    assert_eq!(
        combined[1],
        BinKind::Float {
            feature: 0,
            border: 2.0
        }
    );
    assert_eq!(
        combined[2],
        BinKind::Float {
            feature: 1,
            border: 5.0
        }
    );
    for i in 3..6 {
        assert_eq!(combined[i], BinKind::OneHot, "one-hot range at index {i}");
    }
    assert_eq!(
        combined[6],
        BinKind::Ctr {
            ctr_feature: 0,
            border_index: 0
        }
    );
    assert_eq!(
        combined[7],
        BinKind::Ctr {
            ctr_feature: 0,
            border_index: 1
        }
    );

    // Out-of-range: the caller (`reconstruct_model`) turns a missing `.get`
    // into `ModelError::Deserialize` — verified here at the classification
    // boundary that the table simply has no entry past its declared length.
    assert!(combined.get(8).is_none(), "index 8 is out of range");
}

#[test]
fn classify_bins_numeric_only_model_matches_float_only_order() {
    // No one-hot / ctr features: the combined table must be BYTE-IDENTICAL in
    // shape to the pre-CTR-load float-only classification (regression lock).
    let buf = build_trees_buf(&[vec![0.5, 1.5], vec![2.5]], &[], &[], &[], &[], &[]);
    let trees = flatbuffers::root::<TModelTrees>(&buf).expect("verifying root parse");
    let combined = build_combined_bins(&trees).expect("classification must succeed");
    assert_eq!(
        combined,
        vec![
            BinKind::Float {
                feature: 0,
                border: 0.5
            },
            BinKind::Float {
                feature: 0,
                border: 1.5
            },
            BinKind::Float {
                feature: 1,
                border: 2.5
            },
        ]
    );
}

// ── SAVE T1 — CtrFeatures grouping + combined global index ───────────────────

/// A `Borders` CTR split over `cats` at `border` with the given `scale` (the
/// remaining identity fields fixed so `cats`/`scale`/`border` drive grouping).
fn save_ctr_split(cats: &[usize], border: f64, scale: f64) -> crate::CtrSplit {
    crate::CtrSplit {
        projection: cb_train::TProjection::from_features(cats),
        ctr_type: ECtrType::Borders,
        prior: Prior { num: 0.0, denom: 1.0 },
        target_border_idx: 0,
        border,
        shift: 0.0,
        scale,
    }
}

/// A minimal `Model` carrying one oblivious tree whose splits are the given CTR
/// splits (float borders / leaves are irrelevant to grouping).
fn model_with_ctr_splits(splits: Vec<crate::CtrSplit>) -> crate::Model {
    let model_splits = splits.into_iter().map(crate::model::ModelSplit::Ctr).collect();
    crate::Model {
        oblivious_trees: vec![crate::ObliviousTree {
            splits: model_splits,
            leaf_values: Vec::new(),
            leaf_weights: Vec::new(),
        }],
        non_symmetric_trees: Vec::new(),
        region_trees: Vec::new(),
        bias: 0.0,
        float_feature_borders: Vec::new(),
        ctr_data: None,
        approx_dimension: 1,
        class_to_label: Vec::new(),
    }
}

#[test]
fn group_ctr_features_assigns_stable_indices() {
    // Identity A = proj[0], used at borders 6.0 and 1.0 (2 distinct borders);
    // Identity B = proj[1], used at border 2.0 (1 border). `scale` is held equal
    // so only (projection, border) vary — grouping must fold `border` OUT.
    let a_hi = save_ctr_split(&[0], 6.0, 15.0);
    let a_lo = save_ctr_split(&[0], 1.0, 15.0);
    let b = save_ctr_split(&[1], 2.0, 15.0);
    let model = model_with_ctr_splits(vec![a_hi.clone(), a_lo.clone(), b.clone()]);

    let plan = build_ctr_features(&model).expect("grouping must succeed");
    assert_eq!(plan.identities.len(), 2, "two distinct CTR identities");
    // Deterministic order by ctr_base_key: proj=[0] before proj=[1].
    assert_eq!(plan.identities[0].projection.cat_features(), &[0]);
    assert_eq!(
        plan.identities[0].borders,
        vec![1.0, 6.0],
        "distinct borders sorted ascending"
    );
    assert_eq!(plan.identities[1].projection.cat_features(), &[1]);
    assert_eq!(plan.identities[1].borders, vec![2.0]);

    // Combined global index = n_float_bins + Σ(preceding Borders.len()) +
    // border_index. With n_float_bins = 3: A's borders occupy [3, 4]; B's [5].
    let n_float_bins = 3;
    assert_eq!(
        ctr_split_to_global_index(&a_lo, n_float_bins, &plan).unwrap(),
        3
    );
    assert_eq!(
        ctr_split_to_global_index(&a_hi, n_float_bins, &plan).unwrap(),
        4
    );
    assert_eq!(ctr_split_to_global_index(&b, n_float_bins, &plan).unwrap(), 5);

    // A float split still maps via the existing float-only index (unchanged).
    let bins = super::build_bin_features(&[vec![0.5, 2.5], vec![1.5]]);
    let fs = crate::Split { feature: 1, border: 1.5 };
    assert_eq!(super::split_to_global_index(&fs, &bins).unwrap(), 2);
}

// ── SAVE T2 — TCtrFeature build round-trips via ctr_split_from ────────────────

#[test]
fn tctr_feature_roundtrips_via_ctr_split_from() {
    // Build ONE TCtrFeature from a CtrIdentity, then feed each border index back
    // through the EXISTING load-side `ctr_split_from` — proving T2 is the exact
    // inverse of CTR-02.
    let identity = CtrIdentity {
        projection: cb_train::TProjection::from_features(&[2, 0, 1]),
        ctr_type: ECtrType::Counter,
        prior: Prior { num: 0.5, denom: 1.0 },
        target_border_idx: 1,
        shift: 0.25,
        scale: 2.0,
        borders: vec![1.0, 3.5, 6.0],
    };
    let mut fbb = FlatBufferBuilder::new();
    let cf = build_tctr_feature(&mut fbb, &identity).expect("build_tctr_feature");
    let ctr_features = fbb.create_vector(&[cf]);
    let trees = TModelTrees::create(
        &mut fbb,
        &TModelTreesArgs {
            CtrFeatures: Some(ctr_features),
            Scale: 1.0,
            ..TModelTreesArgs::default()
        },
    );
    fbb.finish(trees, None);
    let buf = fbb.finished_data().to_vec();
    let trees = flatbuffers::root::<TModelTrees>(&buf).expect("verifying root parse");
    let tcf = trees.CtrFeatures().expect("CtrFeatures present").get(0);

    for (k, &b) in identity.borders.iter().enumerate() {
        let split = ctr_split_from(tcf, k).expect("ctr_split_from must succeed");
        assert_eq!(split.projection.cat_features(), &[0, 1, 2], "sorted + deduped");
        assert_eq!(split.ctr_type, ECtrType::Counter);
        assert_eq!(split.target_border_idx, 1);
        assert!((split.prior.num - 0.5).abs() < 1e-6);
        assert!((split.prior.denom - 1.0).abs() < 1e-6);
        assert!((split.shift - 0.25).abs() < 1e-6);
        assert!((split.scale - 2.0).abs() < 1e-6);
        assert!((split.border - b).abs() < 1e-6, "border {k}");
    }
}

// ── T2 — CTR-02 CtrSplit reconstruction from the core flatbuffer ────────────

#[test]
fn ctr_split_from_flatbuffer_maps_fields() {
    // Committed `simple.cbm` ctr_feature[0] (dissected from the FROZEN fixture,
    // see `dissect_ctr` in the Phase-23 execution evidence):
    //   ctr_type=Borders(0), cats=[0], target_border_idx=0, prior_num=0,
    //   prior_denom=1, shift=0, scale=15, borders=[0.999999, 5.999999].
    let buf = read_fixture("simple.cbm");
    let (core, _tail) = split_core_and_tail(&buf);
    let model_core = root_as_tmodel_core(core).expect("verifying TModelCore parse");
    let trees = model_core.ModelTrees().expect("ModelTrees present");
    let ctr_features = trees.CtrFeatures().expect("CtrFeatures present");
    assert!(ctr_features.len() >= 1);
    let tcf = ctr_features.get(0);

    let split = ctr_split_from(tcf, 0).expect("ctr_split_from must succeed");
    assert_eq!(split.projection.cat_features(), &[0]);
    assert_eq!(split.ctr_type, ECtrType::Borders);
    assert!((split.prior.num - 0.0).abs() < 1e-6);
    assert!((split.prior.denom - 1.0).abs() < 1e-6);
    assert_eq!(split.target_border_idx, 0);
    // Exact f32->f64 cast of the wire border (cross-checked against the
    // committed `simple.json`'s `features_info.ctrs[0].borders[0]`).
    assert!((split.border - 0.999_998_986_721_038_8).abs() < 1e-12, "got {}", split.border);
    assert!((split.shift - 0.0).abs() < 1e-6);
    assert!((split.scale - 15.0).abs() < 1e-6);

    // border_index=1 selects the SECOND border of the same feature.
    let split1 = ctr_split_from(tcf, 1).expect("ctr_split_from must succeed");
    assert!((split1.border - 5.999_999_046_325_684).abs() < 1e-12, "got {}", split1.border);
}

#[test]
fn ctr_split_from_sorts_and_dedups_combination_projection() {
    let cf = CtrFeatureSpec {
        cat_features: vec![2, 0, 2, 1],
        ctr_type: CoreECtrType::Counter,
        target_border_idx: 1,
        prior_num: 0.5,
        prior_denom: 1.0,
        shift: 0.25,
        scale: 2.0,
        borders: vec![3.5],
    };
    let buf = build_trees_buf(&[], &[], std::slice::from_ref(&cf), &[], &[], &[]);
    let trees = flatbuffers::root::<TModelTrees>(&buf).expect("verifying root parse");
    let tcf = trees.CtrFeatures().expect("CtrFeatures present").get(0);

    let split = ctr_split_from(tcf, 0).expect("ctr_split_from must succeed");
    assert_eq!(
        split.projection.cat_features(),
        &[0, 1, 2],
        "sorted + deduped"
    );
    assert_eq!(split.ctr_type, ECtrType::Counter);
    assert_eq!(split.target_border_idx, 1);
    assert!((split.prior.num - 0.5).abs() < 1e-6);
    assert!((split.prior.denom - 1.0).abs() < 1e-6);
    assert!((split.shift - 0.25).abs() < 1e-6);
    assert!((split.scale - 2.0).abs() < 1e-6);
    assert!((split.border - 3.5).abs() < 1e-6);
}

#[test]
fn ctr_split_from_out_of_range_border_index_is_typed_error() {
    let cf = CtrFeatureSpec {
        cat_features: vec![0],
        ctr_type: CoreECtrType::Borders,
        target_border_idx: 0,
        prior_num: 0.0,
        prior_denom: 1.0,
        shift: 0.0,
        scale: 1.0,
        borders: vec![1.0],
    };
    let buf = build_trees_buf(&[], &[], std::slice::from_ref(&cf), &[], &[], &[]);
    let trees = flatbuffers::root::<TModelTrees>(&buf).expect("verifying root parse");
    let tcf = trees.CtrFeatures().expect("CtrFeatures present").get(0);
    match ctr_split_from(tcf, 5) {
        Err(ModelError::Deserialize(_)) => {}
        other => panic!("out-of-range border_index must be Deserialize error, got {other:?}"),
    }
}

// ── T3 — CTR-03 ctr_data model-parts parsing ────────────────────────────────

#[test]
fn decode_ctr_parts_matches_dissected_simple() {
    let buf = read_fixture("simple.cbm");
    let (_core, tail) = split_core_and_tail(&buf);

    let ctr_data = decode_ctr_model_parts(tail).expect("decode_ctr_model_parts must succeed");
    assert_eq!(ctr_data.tables.len(), 3, "3 model-parts tables");

    // part0: (Borders, proj=[0]) — 5 buckets, tcc=2, cdenom=0.
    let key_borders_0 = ctr_base_key(ECtrType::Borders, &[0]);
    let t0 = ctr_data
        .tables
        .get(&key_borders_0)
        .expect("Borders proj=[0] table present");
    assert_eq!(t0.ctr_type, ECtrType::Borders);
    assert_eq!(t0.target_classes_count, 2);
    assert_eq!(t0.counter_denominator, 0);
    assert_eq!(t0.hashes.len(), 5, "bucket_count == 5 non-empty slots");
    // The non-empty IndexHashRaw idx set is exactly 0..bucket_count
    // (completeness) — dissected hash-per-idx order from the committed file.
    assert_eq!(t0.hashes[0], 0x3294_a5a4_9e7c_e3da);
    assert_eq!(t0.hashes[1], 0xc3a1_6f52_5643_1442);
    assert_eq!(t0.hashes[2], 0x5cdf_9cb0_2772_76df);
    assert_eq!(t0.hashes[3], 0xd186_2171_78b6_26ce);
    assert_eq!(t0.hashes[4], 0x93cd_3bc0_595b_a149);
    assert_eq!(
        t0.int_counts,
        vec![
            vec![36, 47],
            vec![84, 0],
            vec![0, 70],
            vec![58, 23],
            vec![53, 29],
        ]
    );

    // part1: (Counter, proj=[0]) — SAME 5-bucket keyspace, cdenom=84.
    let key_counter_0 = ctr_base_key(ECtrType::Counter, &[0]);
    let t1 = ctr_data
        .tables
        .get(&key_counter_0)
        .expect("Counter proj=[0] table present");
    assert_eq!(t1.ctr_type, ECtrType::Counter);
    assert_eq!(t1.counter_denominator, 84);
    assert_eq!(t1.hashes, t0.hashes, "identical proj=[0] keyspace as part0");
    assert_eq!(
        t1.int_counts,
        vec![vec![83], vec![84], vec![70], vec![81], vec![82]]
    );
    // Cross-validation: Borders' n0+n1 per bucket equals Counter's bucket
    // total — an internal consistency check confirming the decode is correct
    // (independent of any single transcribed number).
    for (b, counts) in t0.int_counts.iter().enumerate() {
        let total: i64 = counts.iter().sum();
        assert_eq!(total, t1.int_counts[b][0], "bucket {b} total mismatch");
    }

    // part2: (Borders, proj=[1]) — 4 buckets.
    let key_borders_1 = ctr_base_key(ECtrType::Borders, &[1]);
    let t2 = ctr_data
        .tables
        .get(&key_borders_1)
        .expect("Borders proj=[1] table present");
    assert_eq!(t2.hashes.len(), 4, "bucket_count == 4 non-empty slots");
    assert_eq!(
        t2.int_counts,
        vec![vec![24, 78], vec![50, 47], vec![87, 26], vec![70, 18]]
    );
}

#[test]
fn decode_ctr_parts_matches_dissected_combo() {
    let buf = read_fixture("combo.cbm");
    let (_core, tail) = split_core_and_tail(&buf);

    let ctr_data = decode_ctr_model_parts(tail).expect("decode_ctr_model_parts must succeed");
    assert_eq!(ctr_data.tables.len(), 4, "4 model-parts tables (combination adds one)");

    let key_combo = ctr_base_key(ECtrType::Borders, &[0, 1]);
    let combo_table = ctr_data
        .tables
        .get(&key_combo)
        .expect("Borders proj=[0,1] combination table present");
    assert_eq!(combo_table.hashes.len(), 20, "20 filled buckets of 64 slots");
}

// ── T4-adjacent regression lock (numeric .cbm still decodes with ctr_data:
//    None) is exercised in `cbm_oracle_test.rs`; the malformed-tail /
//    classification-boundary robustness lives here (CTR-05, T5). ──────────────

// ── T5 — CTR-05 robustness / typed rejection ────────────────────────────────

#[test]
fn decode_ctr_parts_truncated_tail_is_typed_error() {
    let buf = read_fixture("simple.cbm");
    let (_core, tail) = split_core_and_tail(&buf);
    // Drop the last 50 bytes — mid-part truncation.
    let truncated = &tail[..tail.len().saturating_sub(50)];
    match decode_ctr_model_parts(truncated) {
        Err(ModelError::Deserialize(_)) => {}
        other => panic!("truncated tail must be Deserialize error, got {other:?}"),
    }
}

#[test]
fn decode_ctr_parts_empty_tail_is_typed_error() {
    match decode_ctr_model_parts(&[]) {
        Err(ModelError::Deserialize(_)) => {}
        other => panic!("empty tail must be Deserialize error, got {other:?}"),
    }
}

/// Build a single-part tail buffer (`u32 count=1` + `u32 size` + the
/// `TCtrValueTable` bytes) from raw `IndexHashRaw` / `CTRBlob` byte vectors —
/// the CTR-05 malformed-part fixtures below hand-craft these to exercise the
/// gap / duplicate / width-mismatch / mean-type guards.
#[allow(clippy::too_many_arguments)]
fn build_one_part_tail(
    cat_features: &[i32],
    ctr_type: TailECtrType,
    target_classes_count: i32,
    counter_denominator: i32,
    index_hash_raw: &[u8],
    ctr_blob: &[u8],
) -> Vec<u8> {
    let mut fbb = FlatBufferBuilder::new();
    let cat_vec = fbb.create_vector(cat_features);
    let combination = TailFeatureCombination::create(
        &mut fbb,
        &TailFeatureCombinationArgs {
            CatFeatures: Some(cat_vec),
            ..TailFeatureCombinationArgs::default()
        },
    );
    let base = TailModelCtrBase::create(
        &mut fbb,
        &TailModelCtrBaseArgs {
            FeatureCombination: Some(combination),
            CtrType: ctr_type,
            TargetBorderClassifierIdx: 0,
        },
    );
    let ihr_vec = fbb.create_vector(index_hash_raw);
    let blob_vec = fbb.create_vector(ctr_blob);
    let vt = TCtrValueTable::create(
        &mut fbb,
        &TCtrValueTableArgs {
            ModelCtrBase: Some(base),
            IndexHashRaw: Some(ihr_vec),
            CTRBlob: Some(blob_vec),
            CounterDenominator: counter_denominator,
            TargetClassesCount: target_classes_count,
        },
    );
    fbb.finish(vt, None);
    let part = fbb.finished_data().to_vec();

    let mut tail = Vec::new();
    tail.extend_from_slice(&1u32.to_le_bytes()); // part_count = 1
    tail.extend_from_slice(&(part.len() as u32).to_le_bytes());
    tail.extend_from_slice(&part);
    tail
}

/// A 12-byte `IndexHashRaw` slot: `(u64 hash LE, u32 idx LE)`.
fn hash_slot(hash: u64, idx: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(12);
    out.extend_from_slice(&hash.to_le_bytes());
    out.extend_from_slice(&idx.to_le_bytes());
    out
}

#[test]
fn decode_ctr_parts_gap_in_bucket_indices_is_typed_error() {
    // Two non-empty slots with idx {0, 2} — a gap at 1 (bucket_count would be
    // 2, but idx 2 is out of range for that bucket_count).
    let mut ihr = Vec::new();
    ihr.extend_from_slice(&hash_slot(111, 0));
    ihr.extend_from_slice(&hash_slot(222, 2));
    let blob: Vec<u8> = [10i32, 20, 30, 40]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let tail = build_one_part_tail(&[0], TailECtrType::Borders, 2, 0, &ihr, &blob);
    match decode_ctr_model_parts(&tail) {
        Err(ModelError::Deserialize(_)) => {}
        other => panic!("gap in bucket indices must be Deserialize error, got {other:?}"),
    }
}

#[test]
fn decode_ctr_parts_duplicate_bucket_index_is_typed_error() {
    // Two non-empty slots BOTH claiming idx 0 — a duplicate.
    let mut ihr = Vec::new();
    ihr.extend_from_slice(&hash_slot(111, 0));
    ihr.extend_from_slice(&hash_slot(222, 0));
    let blob: Vec<u8> = [10i32, 20, 30, 40]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let tail = build_one_part_tail(&[0], TailECtrType::Borders, 2, 0, &ihr, &blob);
    match decode_ctr_model_parts(&tail) {
        Err(ModelError::Deserialize(_)) => {}
        other => panic!("duplicate bucket index must be Deserialize error, got {other:?}"),
    }
}

#[test]
fn decode_ctr_parts_blob_length_mismatch_is_typed_error() {
    // 1 non-empty bucket, TargetClassesCount=2 (width 2) but CTRBlob only
    // carries 1 i32 (should be 2) — a width mismatch.
    let ihr = hash_slot(111, 0);
    let blob: Vec<u8> = 7i32.to_le_bytes().to_vec();
    let tail = build_one_part_tail(&[0], TailECtrType::Borders, 2, 0, &ihr, &blob);
    match decode_ctr_model_parts(&tail) {
        Err(ModelError::Deserialize(_)) => {}
        other => panic!("CTRBlob width mismatch must be Deserialize error, got {other:?}"),
    }
}

#[test]
fn decode_ctr_parts_mean_ctr_type_is_typed_error() {
    let ihr = hash_slot(111, 0);
    let blob: Vec<u8> = [7.0f32.to_le_bytes(), 3i32.to_le_bytes()].concat();
    let tail = build_one_part_tail(
        &[0],
        TailECtrType::BinarizedTargetMeanValue,
        0,
        0,
        &ihr,
        &blob,
    );
    match decode_ctr_model_parts(&tail) {
        Err(ModelError::Deserialize(_)) => {}
        other => panic!("mean ctr_type must be Deserialize error (v1, MAJOR-2), got {other:?}"),
    }
}

#[test]
fn decode_ctr_parts_duplicate_table_key_is_typed_error() {
    // Two parts with the IDENTICAL (ctr_type, projection) key.
    let ihr = hash_slot(111, 0);
    let blob: Vec<u8> = [10i32].iter().flat_map(|v| v.to_le_bytes()).collect();
    let part_a = {
        let t = build_one_part_tail(&[0], TailECtrType::Counter, 0, 5, &ihr, &blob);
        // strip the count+size framing to re-splice two parts manually below.
        t[8..].to_vec()
    };
    let part_b = part_a.clone();
    let mut tail = Vec::new();
    tail.extend_from_slice(&2u32.to_le_bytes());
    tail.extend_from_slice(&(part_a.len() as u32).to_le_bytes());
    tail.extend_from_slice(&part_a);
    tail.extend_from_slice(&(part_b.len() as u32).to_le_bytes());
    tail.extend_from_slice(&part_b);
    match decode_ctr_model_parts(&tail) {
        Err(ModelError::Deserialize(_)) => {}
        other => panic!("duplicate (ctr_type,projection) key must be Deserialize error, got {other:?}"),
    }
}

#[test]
fn one_hot_split_index_is_typed_error() {
    // A tree with ONE split referencing the one-hot range (float bins occupy
    // [0], one-hot occupies [1]) must be rejected — no `ModelSplit::OneHot`
    // variant exists (CTR-05).
    let buf = build_trees_buf(
        &[vec![0.5]],
        &[vec![10, 20]],
        &[],
        &[1], // TreeSplits: global index 1 -> the one-hot range.
        &[1], // TreeSizes: depth 1.
        &[0.0, 0.0],
    );
    let trees = flatbuffers::root::<TModelTrees>(&buf).expect("verifying root parse");
    match super::reconstruct_model(&trees, Vec::new(), &[]) {
        Err(ModelError::Deserialize(_)) => {}
        other => panic!("one-hot split index must be Deserialize error, got {other:?}"),
    }
}

#[test]
fn ctr_features_present_with_empty_tail_is_typed_error() {
    let cf = CtrFeatureSpec {
        cat_features: vec![0],
        ctr_type: CoreECtrType::Borders,
        target_border_idx: 0,
        prior_num: 0.0,
        prior_denom: 1.0,
        shift: 0.0,
        scale: 1.0,
        borders: vec![1.0],
    };
    let buf = build_trees_buf(
        &[],
        &[],
        std::slice::from_ref(&cf),
        &[0], // TreeSplits: global index 0 -> the (only) ctr bin.
        &[1],
        &[0.0, 0.0],
    );
    let trees = flatbuffers::root::<TModelTrees>(&buf).expect("verifying root parse");
    // Empty tail while `CtrFeatures` is non-empty: `decode_ctr_model_parts`
    // must reject it (never silently `ctr_data: None`).
    match super::reconstruct_model(&trees, Vec::new(), &[]) {
        Err(ModelError::Deserialize(_)) => {}
        other => panic!("CtrFeatures-present + empty tail must be Deserialize error, got {other:?}"),
    }
}

#[test]
fn numeric_only_model_still_decodes_with_ctr_data_none() {
    // Regression lock: no CtrFeatures / OneHotFeatures at all -> `ctr_data` is
    // still `None` and the split decodes as `ModelSplit::Float` (byte-identical
    // to the pre-CTR-load path).
    let buf = build_trees_buf(
        &[vec![0.5, 1.5]],
        &[],
        &[],
        &[0], // global index 0 -> Float{feature:0, border:0.5}
        &[1],
        &[0.0, 0.0],
    );
    let trees = flatbuffers::root::<TModelTrees>(&buf).expect("verifying root parse");
    let model = super::reconstruct_model(&trees, Vec::new(), &[]).expect("must decode");
    assert!(model.ctr_data.is_none(), "numeric model must keep ctr_data: None");
    assert_eq!(model.oblivious_trees.len(), 1);
    match &model.oblivious_trees[0].splits[0] {
        ModelSplit::Float(s) => {
            assert_eq!(s.feature, 0);
            assert!((s.border - 0.5).abs() < 1e-6);
        }
        other => panic!("expected a Float split, got {other:?}"),
    }
}

// ── SAVE T6 — save-side robustness / typed rejection ─────────────────────────

#[test]
fn save_ctr_split_missing_table_is_typed_error() {
    // A model carrying a CTR split but NO `ctr_data` (its apply-time table would
    // miss) must be rejected — never emit a CtrFeature whose lookup would fail.
    let model = model_with_ctr_splits(vec![save_ctr_split(&[0], 1.0, 15.0)]);
    assert!(model.ctr_data.is_none());
    match super::build_core_blob(&model) {
        Err(ModelError::Serialize(_)) => {}
        other => panic!("CTR split with missing ctr_data must be Serialize error, got {other:?}"),
    }
}

#[test]
fn save_non_symmetric_ctr_split_is_typed_error() {
    // A non-symmetric tree carrying a CTR split at an INTERIOR node: v1 supports
    // CTR splits on oblivious trees only — the save arm must error, not write `0`.
    let tree = crate::NonSymmetricTree {
        tree_splits: vec![crate::model::ModelSplit::Ctr(save_ctr_split(&[0], 1.0, 15.0))],
        step_nodes: vec![(1, 0)], // NOT (0,0) -> an interior (non-leaf) node.
        node_id_to_leaf_id: vec![0],
        leaf_values: vec![0.0],
        leaf_weights: vec![0.0],
    };
    let model = crate::Model {
        oblivious_trees: Vec::new(),
        non_symmetric_trees: vec![tree],
        region_trees: Vec::new(),
        bias: 0.0,
        float_feature_borders: Vec::new(),
        ctr_data: None,
        approx_dimension: 1,
        class_to_label: Vec::new(),
    };
    match super::build_core_blob(&model) {
        Err(ModelError::Serialize(_)) => {}
        other => panic!("non-symmetric CTR split must be Serialize error, got {other:?}"),
    }
}

#[test]
fn save_numeric_model_emits_no_tail() {
    // Regression lock: a numeric-only model (no CTR splits, `ctr_data: None`)
    // saves with NO model-parts tail — the file is exactly the 8-byte frame plus
    // the FlatBuffers core, byte-for-byte the pre-slice layout.
    let model = crate::Model {
        oblivious_trees: vec![crate::ObliviousTree {
            splits: vec![crate::model::ModelSplit::Float(crate::Split { feature: 0, border: 0.5 })],
            leaf_values: vec![0.1, -0.2],
            leaf_weights: vec![1.0, 1.0],
        }],
        non_symmetric_trees: Vec::new(),
        region_trees: Vec::new(),
        bias: 0.0,
        float_feature_borders: vec![vec![0.5]],
        ctr_data: None,
        approx_dimension: 1,
        class_to_label: Vec::new(),
    };
    let path = std::env::temp_dir().join(format!(
        "cb_model_no_tail_{}_{}.cbm",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    crate::save_cbm(&model, &path).expect("save_cbm numeric must succeed");
    let bytes = std::fs::read(&path).expect("read back");
    let _ = std::fs::remove_file(&path);

    let declared = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    assert_eq!(
        bytes.len(),
        8 + declared,
        "numeric model must have NO tail beyond frame + core"
    );
    // And it decodes back with ctr_data: None (never a phantom tail).
    let reloaded = super::decode_cbm(&bytes).expect("decode numeric");
    assert!(reloaded.ctr_data.is_none());
}

#[test]
fn non_symmetric_model_with_ctr_features_is_typed_error() {
    // A NonSymmetricStepNodes-carrying model with non-empty CtrFeatures must be
    // rejected (v1 scope: CTR splits are oblivious-tree only).
    let cf = CtrFeatureSpec {
        cat_features: vec![0],
        ctr_type: CoreECtrType::Borders,
        target_border_idx: 0,
        prior_num: 0.0,
        prior_denom: 1.0,
        shift: 0.0,
        scale: 1.0,
        borders: vec![1.0],
    };
    let mut fbb = FlatBufferBuilder::new();
    let cat_vec = fbb.create_vector(&cf.cat_features);
    let combination = TFeatureCombination::create(
        &mut fbb,
        &TFeatureCombinationArgs {
            CatFeatures: Some(cat_vec),
            ..TFeatureCombinationArgs::default()
        },
    );
    let base = TModelCtrBase::create(
        &mut fbb,
        &TModelCtrBaseArgs {
            FeatureCombination: Some(combination),
            CtrType: cf.ctr_type,
            TargetBorderClassifierIdx: 0,
        },
    );
    let ctr = TModelCtr::create(
        &mut fbb,
        &TModelCtrArgs {
            Base: Some(base),
            TargetBorderIdx: cf.target_border_idx,
            PriorNum: cf.prior_num,
            PriorDenom: cf.prior_denom,
            Shift: cf.shift,
            Scale: cf.scale,
        },
    );
    let borders_vec = fbb.create_vector(&cf.borders);
    let ctr_feature = TCtrFeature::create(
        &mut fbb,
        &TCtrFeatureArgs {
            Ctr: Some(ctr),
            Borders: Some(borders_vec),
        },
    );
    let ctr_features = fbb.create_vector(&[ctr_feature]);

    let step_node = crate::model_generated::ncat_boost_fbs::TNonSymmetricTreeStepNode::new(0, 0);
    let step_nodes_vec = fbb.create_vector(&[step_node]);
    let node_ids_vec = fbb.create_vector(&[0u32]);
    let splits_vec = fbb.create_vector(&[0i32]);
    let sizes_vec = fbb.create_vector(&[1i32]);
    let leaf_values_vec = fbb.create_vector(&[0.0f64]);

    let trees = TModelTrees::create(
        &mut fbb,
        &TModelTreesArgs {
            CtrFeatures: Some(ctr_features),
            NonSymmetricStepNodes: Some(step_nodes_vec),
            NonSymmetricNodeIdToLeafId: Some(node_ids_vec),
            TreeSplits: Some(splits_vec),
            TreeSizes: Some(sizes_vec),
            LeafValues: Some(leaf_values_vec),
            Scale: 1.0,
            ..TModelTreesArgs::default()
        },
    );
    fbb.finish(trees, None);
    let buf = fbb.finished_data().to_vec();
    let trees = flatbuffers::root::<TModelTrees>(&buf).expect("verifying root parse");

    match super::reconstruct_model(&trees, Vec::new(), &[]) {
        Err(ModelError::Deserialize(_)) => {}
        other => panic!("non-symmetric + CtrFeatures must be Deserialize error, got {other:?}"),
    }
}
