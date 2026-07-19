"""Generate the CoreML-export oracle fixtures (Phase 17, EXPORT-02 / CM-04).

FROZEN GENERATOR. The committed `model.cbm` / `reference.mlmodel` /
`structure.json` / `golden.mlmodel` / `config.json` under this directory are the
GROUND TRUTH the Rust oracle (`crates/cb-model/tests/coreml_export_test.rs`) is
tested against. CI does NOT run this script (no `catboost` / `coremltools`
install in CI) and never regenerates these fixtures.

# What this produces

  - `model.cbm`        — a float-only oblivious scalar CatBoostRegressor (the
                         export INPUT `cb_model::load_cbm` reads).
  - `reference.mlmodel`— CatBoost's OWN CoreML output for `model.cbm`
                         (`m.save_model(path, format="coreml")`).
  - `structure.json`   — the reference's tree structure decoded with
                         `coremltools` (per-tree: for each nodeId its behavior,
                         branchFeatureIndex, branchFeatureValue,
                         true/falseChildNodeId, and leaf evaluationValue; plus
                         basePredictionValue / numPredictionDimensions). This is
                         the ORACLE TARGET: the Rust exporter's emitted structure
                         must match it EXACTLY (node ids/children/behaviors) with
                         branch thresholds + leaf values within 1e-5.
  - `golden.mlmodel`   — the Rust exporter's OWN frozen output for `model.cbm`
                         (a byte-pin regression artifact). Produced by running
                         `crates/cb-model/tests/coreml_export_test.rs` ONCE with
                         `CB_COREML_REGEN_GOLDEN=1` after `model.cbm` was frozen;
                         NOT written by this Python script (it needs the Rust
                         encoder). See that test's header.
  - `config.json`      — versions, seed, n_trees, depth, artifact list.

# Reproducibility caveat (load-bearing — do NOT "fix" by re-running)

CatBoost's border/quantization step is run-to-run nondeterministic independent
of the exposed `random_seed` (observed even at `thread_count=1`; documented for
the CTR fixtures too). This script documents the pinned RECIPE for provenance
only — it is NOT expected to byte-reproduce the committed `model.cbm`, and must
never be used to regenerate the fixtures for CI. The checked-in files ARE the
fixtures.

# Pinned recipe

  - `catboost==1.2.10`, `coremltools==9.0`.
  - Reuse `crates/cb-oracle/fixtures/inputs/numeric_tiny/X.npy` (50 rows,
    4 float features); target is a fixed deterministic linear-ish function of
    the columns (float-only, no categorical => oblivious float regressor).
  - `thread_count=1`, `bootstrap_type="No"`, `random_seed=0`, `depth=3`,
    `iterations=5`, `learning_rate=0.3`, `loss_function="RMSE"`.
"""

import json
import os
import pathlib

import numpy as np
from catboost import CatBoostRegressor
from coremltools.proto import Model_pb2

HERE = pathlib.Path(__file__).resolve().parent
INPUTS = HERE.parent / "inputs" / "numeric_tiny"

CATBOOST_VERSION = "1.2.10"
COREMLTOOLS_VERSION = "9.0"
SEED = 0
DEPTH = 3
ITERATIONS = 5
LEARNING_RATE = 0.3


def train_model():
    x = np.load(INPUTS / "X.npy").astype(np.float64)
    # Deterministic float-only regression target (no categorical signal).
    rng = np.random.RandomState(SEED)
    coeffs = rng.uniform(-1.0, 1.0, size=x.shape[1])
    y = x @ coeffs + 0.5 * x[:, 0] - 0.25

    model = CatBoostRegressor(
        iterations=ITERATIONS,
        depth=DEPTH,
        learning_rate=LEARNING_RATE,
        loss_function="RMSE",
        bootstrap_type="No",
        thread_count=1,
        random_seed=SEED,
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(x, y)
    return model


def decode_structure(mlmodel_bytes):
    spec = Model_pb2.Model()
    spec.ParseFromString(mlmodel_bytes)  # sanity: coremltools parses the reference
    reg = spec.treeEnsembleRegressor
    params = reg.treeEnsemble

    trees = {}
    for node in params.nodes:
        tree = trees.setdefault(str(node.treeId), {})
        entry = {
            "behavior": int(node.nodeBehavior),
            "branch_feature_index": int(node.branchFeatureIndex),
            "branch_feature_value": float(node.branchFeatureValue),
            "true_child_node_id": int(node.trueChildNodeId),
            "false_child_node_id": int(node.falseChildNodeId),
        }
        # Leaf value (LeafNode behavior == 6): single evaluationInfo entry.
        if len(node.evaluationInfo) > 0:
            entry["leaf_value"] = float(node.evaluationInfo[0].evaluationValue)
            entry["evaluation_index"] = int(node.evaluationInfo[0].evaluationIndex)
        trees[str(node.treeId)][str(node.nodeId)] = entry

    return {
        "spec_type": spec.WhichOneof("Type"),
        "num_prediction_dimensions": int(params.numPredictionDimensions),
        "base_prediction_value": [float(v) for v in params.basePredictionValue],
        "trees": trees,
    }


def main():
    model = train_model()

    cbm_path = HERE / "model.cbm"
    mlmodel_path = HERE / "reference.mlmodel"
    model.save_model(str(cbm_path), format="cbm")
    model.save_model(str(mlmodel_path), format="coreml")

    ref_bytes = mlmodel_path.read_bytes()
    structure = decode_structure(ref_bytes)
    assert structure["spec_type"] == "treeEnsembleRegressor", structure["spec_type"]

    (HERE / "structure.json").write_text(json.dumps(structure, indent=2, sort_keys=True))

    n_trees = len(structure["trees"])
    config = {
        "catboost_version": CATBOOST_VERSION,
        "coremltools_version": COREMLTOOLS_VERSION,
        "seed": SEED,
        "depth": DEPTH,
        "iterations": ITERATIONS,
        "learning_rate": LEARNING_RATE,
        "n_trees": n_trees,
        "loss_function": "RMSE",
        "bootstrap_type": "No",
        "thread_count": 1,
        "input_dataset": "inputs/numeric_tiny/X.npy",
        "artifacts": [
            "model.cbm",
            "reference.mlmodel",
            "structure.json",
            "golden.mlmodel",
            "config.json",
        ],
        "golden_provenance": (
            "golden.mlmodel is the Rust exporter's own output for model.cbm, "
            "produced by running `cargo test -p cb-model --test coreml_export_test` "
            "once with CB_COREML_REGEN_GOLDEN=1 (NOT by this Python script). "
            "Frozen thereafter; regenerate deliberately only on a schema change."
        ),
        "note": (
            "FROZEN. CatBoost quantization is run-to-run nondeterministic; never "
            "regenerate at test time. structure.json is the CM-04 oracle target."
        ),
    }
    (HERE / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    print(f"wrote model.cbm, reference.mlmodel, structure.json, config.json")
    print(f"n_trees={n_trees}, base_prediction_value={structure['base_prediction_value']}")


if __name__ == "__main__":
    main()
