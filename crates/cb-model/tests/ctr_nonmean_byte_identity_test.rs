//! E00 / SPEC-CTRT-16 (acceptance A9) — the NON-MEAN `.cbm` CTR byte baseline.
//!
//! # Why this exists, and why it is captured FIRST
//!
//! W4 (E19/E20) lifts the `.cbm` codec's mean-CTR restriction: today
//! `build_tctr_value_table` refuses to serialize a mean-type table and
//! `decode_one_ctr_value_table` refuses to read one. SPEC-CTRT-16 requires that
//! lifting the restriction leaves the **non-mean** encoding byte-for-byte
//! unchanged. That claim is only meaningful against bytes frozen BEFORE the codec
//! changes — capturing after would degenerate into a self-comparison.
//!
//! # Why the model is HAND-CONSTRUCTED, not trained
//!
//! W1–W3 deliberately change the trainer's chosen candidate set (SPEC-CTRT-11
//! changes tie-breaks outright). A *trained* baseline would drift with those
//! changes and this gate would silently become vacuous. A hand-constructed model
//! isolates the **serializer**, which is exactly what SPEC-CTRT-16 gates.
//!
//! # The fixture is FROZEN
//!
//! No later task may regenerate `ctr_nonmean_byte_identity/`. The capture fn is
//! `#[ignore]`d precisely so a routine `cargo test` can never rewrite the bytes
//! the gate compares against.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::collections::BTreeMap;
use std::path::PathBuf;

use cb_model::{
    ctr_base_key, save_cbm, CtrData, CtrSplit, CtrValueTable, ECtrType, Model as CbModel,
    ModelSplit, ObliviousTree, Prior,
};

/// The frozen baseline fixture root.
fn baseline_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cb-oracle")
        .join("fixtures")
        .join("ctr_nonmean_byte_identity")
}

/// The three pinned bucket hashes. None may equal `EMPTY_HASH_MARKER`
/// (`0xFFFF_FFFF_FFFF_FFFF`), which the encoder reserves for empty dense-hash
/// slots — a bucket carrying it would be silently unreachable at inference.
const HASHES: [u64; 3] = [
    0x1111_1111_1111_1111,
    0x2222_2222_2222_2222,
    0x3333_3333_3333_3333,
];

const EMPTY_HASH_MARKER: u64 = 0xFFFF_FFFF_FFFF_FFFF;

/// The hand-constructed non-mean CTR model under gate.
///
/// Fully deterministic: no RNG, no trainer, no fixture inputs. Two depth-1
/// oblivious trees, each carrying exactly one `ModelSplit::Ctr`, over a `CtrData`
/// with one `Borders` table and one `Counter` table — the two non-mean shapes the
/// `.cbm` codec encodes differently (`Borders` carries per-class counts and
/// `counter_denominator == 0`; `Counter` carries a single count per bucket and a
/// non-zero `counter_denominator`).
fn hand_constructed_model() -> CbModel {
    for h in HASHES {
        assert_ne!(
            h, EMPTY_HASH_MARKER,
            "a bucket hash must never collide with the empty-slot marker"
        );
    }

    let borders_table = CtrValueTable {
        ctr_type: ECtrType::Borders,
        target_classes_count: 2,
        hashes: HASHES.to_vec(),
        int_counts: vec![vec![3, 7], vec![11, 2], vec![0, 5]],
        mean: Vec::new(),
        counter_denominator: 0,
    };

    let counter_table = CtrValueTable {
        ctr_type: ECtrType::Counter,
        target_classes_count: 0,
        hashes: HASHES.to_vec(),
        int_counts: vec![vec![10], vec![13], vec![5]],
        mean: Vec::new(),
        counter_denominator: 13,
    };

    let mut tables = BTreeMap::new();
    tables.insert(ctr_base_key(ECtrType::Borders, &[0]), borders_table);
    tables.insert(ctr_base_key(ECtrType::Counter, &[0]), counter_table);

    let projection = cb_train::TProjection::from_features(&[0]);

    let tree_for = |ctr_type: ECtrType, border: f64, leaves: [f64; 2]| ObliviousTree {
        splits: vec![ModelSplit::Ctr(CtrSplit {
            projection: projection.clone(),
            ctr_type,
            prior: Prior::unit(0.5),
            target_border_idx: 0,
            border,
            shift: 0.0,
            scale: 1.0,
        })],
        leaf_values: leaves.to_vec(),
        leaf_weights: vec![2.0, 3.0],
    };

    CbModel::new(
        vec![
                tree_for(ECtrType::Borders, 0.25, [-0.1, 0.2]),
                tree_for(ECtrType::Counter, 0.5, [0.05, -0.15]),
            ],
        0.125,
        Vec::new(),
    )
    .with_ctr_data(CtrData { tables })
}

/// Serialize the hand-constructed model through the production encoder and return
/// its bytes.
fn serialize_baseline_model() -> Vec<u8> {
    let model = hand_constructed_model();
    let tmp = std::env::temp_dir().join(format!(
        "ctr_nonmean_byte_identity_{}.cbm",
        std::process::id()
    ));
    save_cbm(&model, &tmp).expect("save_cbm must succeed on a non-mean CTR model");
    let bytes = std::fs::read(&tmp).expect("read back the serialized model");
    let _ = std::fs::remove_file(&tmp);
    bytes
}

/// CAPTURE ONLY — freezes the baseline. Run ONCE, BEFORE any W4 codec change,
/// with `-- --ignored`. `#[ignore]`d so no routine test run can silently rewrite
/// the very bytes SPEC-CTRT-16 compares against.
#[test]
#[ignore = "capture-only: run once, before the W4 .cbm mean codec change, to freeze the fixture"]
fn capture_ctr_nonmean_baseline() {
    let dir = baseline_dir();
    std::fs::create_dir_all(&dir).expect("create fixture dir");

    let bytes = serialize_baseline_model();
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
            "# non-mean `.cbm` CTR byte-identity baseline (SPEC-CTRT-16 / E00 / A9)\n\
             \n\
             **CAPTURED_AT_SHA: `{sha}`**\n\
             \n\
             Captured on top of the UNCOMMITTED one-hot device wave present in the\n\
             working tree at capture time.\n\
             \n\
             Written by `ctr_nonmean_byte_identity_test::capture_ctr_nonmean_baseline`\n\
             (`#[ignore]`d; run with `-- --ignored`).\n\
             \n\
             ## What is frozen\n\
             \n\
             The `save_cbm` bytes of a HAND-CONSTRUCTED `cb_model::Model` — no\n\
             trainer, no RNG, no fixture inputs — carrying:\n\
             \n\
             - two depth-1 oblivious trees, each with exactly one\n\
               `ModelSplit::Ctr` (borders `0.25` / `0.5`, prior `0.5/1`,\n\
               `shift 0.0`, `scale 1.0`, `target_border_idx 0`), leaf values\n\
               `[-0.1, 0.2]` / `[0.05, -0.15]`, leaf weights `[2.0, 3.0]`;\n\
             - `bias = 0.125`, `approx_dimension = 1`, no float features, no\n\
               class-to-label map;\n\
             - a `CtrData` with TWO tables over projection `[0]`, keyed by\n\
               `ctr_base_key`:\n\
               1. `Borders`, `target_classes_count = 2`,\n\
                  `int_counts = [[3,7],[11,2],[0,5]]`, `counter_denominator = 0`;\n\
               2. `Counter`, `target_classes_count = 0`,\n\
                  `int_counts = [[10],[13],[5]]`, `counter_denominator = 13`;\n\
               both over bucket hashes\n\
               `[0x1111111111111111, 0x2222222222222222, 0x3333333333333333]`\n\
               (none equal to the `0xFFFFFFFFFFFFFFFF` empty-slot marker).\n\
             \n\
             ## Why hand-constructed and not trained\n\
             \n\
             W1-W3 change the trainer's chosen CTR candidate set on purpose\n\
             (SPEC-CTRT-11 changes tie-breaks). A trained baseline would drift\n\
             with those changes and this gate would silently become vacuous. A\n\
             hand-constructed model isolates the SERIALIZER, which is what\n\
             SPEC-CTRT-16 gates.\n\
             \n\
             ## Do not regenerate\n\
             \n\
             These bytes exist to prove that W4's mean-CTR codec lift\n\
             (E19 decode, E20 encode) leaves the NON-MEAN encoding untouched.\n\
             Regenerating them after that change turns SPEC-CTRT-16 into a\n\
             self-comparison that proves nothing. Still frozen from here on.\n"
        ),
    )
    .expect("write README.md");
}

/// The gate: today's encoder output must equal the frozen bytes.
///
/// A W4 change that perturbs the non-mean encoding — a reordered FlatBuffers
/// section, a changed `CounterDenominator`, a widened blob stride applied to the
/// non-mean path — shows up here as a byte difference.
#[test]
fn nonmean_ctr_cbm_bytes_match_the_frozen_baseline() {
    let baseline_path = baseline_dir().join("baseline.cbm");
    let expected = std::fs::read(&baseline_path).unwrap_or_else(|e| {
        panic!(
            "frozen baseline missing at {} ({e}). Capture it FIRST, BEFORE any W4 \
             codec change: cargo test -p cb-model --test ctr_nonmean_byte_identity_test \
             -- --ignored",
            baseline_path.display()
        )
    });

    let actual = serialize_baseline_model();

    assert_eq!(
        actual.len(),
        expected.len(),
        "non-mean CTR .cbm LENGTH changed ({} -> {}): the non-mean encoding is not \
         byte-identical to the pre-W4 baseline (SPEC-CTRT-16 / A9)",
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
            "non-mean CTR .cbm BYTES changed at offset {first_diff} (len {}): the \
             mean-CTR codec lift has leaked into the NON-MEAN encoding \
             (SPEC-CTRT-16 / A9)",
            expected.len()
        );
    }
}

/// The fixture must carry its provenance — without the capture SHA there is no
/// way to tell a genuine pre-change baseline from one quietly regenerated after.
#[test]
fn frozen_baseline_records_its_capture_sha() {
    let readme = std::fs::read_to_string(baseline_dir().join("README.md"))
        .expect("the frozen baseline must carry a README.md");
    assert!(
        readme.contains("CAPTURED_AT_SHA"),
        "README.md must record the SHA the bytes were captured at"
    );
    assert!(
        readme.contains("Do not regenerate"),
        "README.md must state that regeneration voids SPEC-CTRT-16"
    );
}
