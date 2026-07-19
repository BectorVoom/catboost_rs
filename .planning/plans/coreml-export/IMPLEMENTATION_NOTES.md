# EXPORT-02 CoreML — Authoritative Implementation Notes (schema + layout + verification)

> Addendum to SPEC.md / PLAN.md, written after the "strengthen verification"
> decision. Resolves SPEC R2 (schema was UNVERIFIED) and R1 (no oracle) with
> facts sourced LIVE on this host from `coremltools==9.0` and CatBoost 1.2.10's
> own `save_model(format="coreml")` output. Every field tag / enum value / node
> layout rule below was extracted from the compiled CoreML descriptors and
> confirmed by re-parsing a hand-built minimal model through coremltools.

## 1. The authoritative CoreML protobuf subset (RESOLVES R2)

CatBoost emits a `CoreML.Specification.Model` whose `Type` oneof is
`treeEnsembleRegressor` (field 302). The minimal subset the exporter needs
(field numbers are LOAD-BEARING — verified against coremltools proto descriptors):

```
message Model {
  int32 specificationVersion = 1;          // CatBoost emits 1
  ModelDescription description = 2;
  bool isUpdatable = 10;                    // omit (false)
  oneof Type { TreeEnsembleRegressor treeEnsembleRegressor = 302; }
}
message ModelDescription {
  repeated FeatureDescription input = 1;
  repeated FeatureDescription output = 10;
  string predictedFeatureName = 11;        // "prediction"
  // Metadata metadata = 100;  // OPTIONAL — CatBoost sets shortDescription
                               //  "Catboost model"; we OMIT it (not needed for
                               //  a valid, coremltools-parseable model).
}
message FeatureDescription {
  string name = 1;                         // "feature_0", "feature_1", ...
  string shortDescription = 2;             // omit
  FeatureType type = 3;
}
message FeatureType {
  oneof Type {
    DoubleFeatureType doubleType = 2;      // each input feature
    ArrayFeatureType  multiArrayType = 5;  // the single output
  }
}
message DoubleFeatureType {}               // empty
message ArrayFeatureType {
  repeated int64 shape = 1;                // [1]
  ArrayDataType dataType = 2;              // DOUBLE = 65600
  enum ArrayDataType { INVALID_ARRAY_DATA_TYPE=0; FLOAT32=65568; DOUBLE=65600;
                       INT32=131104; FLOAT16=65552; }
}
message TreeEnsembleRegressor {
  TreeEnsembleParameters treeEnsemble = 1;
  TreeEnsemblePostEvaluationTransform postEvaluationTransform = 2; // NoTransform=0
}
enum TreeEnsemblePostEvaluationTransform { NoTransform=0; Classification_SoftMax=1;
  Regression_Logistic=2; Classification_SoftMaxWithZeroClassReference=3; }
message TreeEnsembleParameters {
  repeated TreeNode nodes = 1;
  uint64 numPredictionDimensions = 2;      // 1
  repeated double basePredictionValue = 3; // [bias]  (see §3 conditional)
  message TreeNode {
    uint64 treeId = 1;
    uint64 nodeId = 2;
    TreeNodeBehavior nodeBehavior = 3;     // LeafNode=6 / BranchOnValueGreaterThan=3
    uint64 branchFeatureIndex = 10;
    double branchFeatureValue = 11;
    uint64 trueChildNodeId = 12;
    uint64 falseChildNodeId = 13;
    bool   missingValueTracksTrueChild = 14; // false (CatBoost sets false)
    repeated EvaluationInfo evaluationInfo = 20;  // leaves only
    double relativeHitRate = 30;           // omit
    enum TreeNodeBehavior { BranchOnValueLessThanEqual=0; BranchOnValueLessThan=1;
      BranchOnValueGreaterThanEqual=2; BranchOnValueGreaterThan=3;
      BranchOnValueEqual=4; BranchOnValueNotEqual=5; LeafNode=6; }
    message EvaluationInfo { uint64 evaluationIndex = 1; double evaluationValue = 2; }
  }
}
```

The committed prost module `crates/cb-model/src/generated/coreml_generated.rs`
(hand-authored, mirroring the `generated/` convention) implements EXACTLY this
subset and has been round-trip-verified: bytes encoded by the Rust prost structs
re-parse through `coremltools`'s `Model_pb2` with every field intact (see the
`gen_fixtures.py` cross-check step).

## 2. CatBoost's node layout — replicate it EXACTLY (so the oracle is a clean diff)

For a scalar oblivious tree with `k` splits (`2^k` leaves, `2^k - 1` internal
nodes), CatBoost's `.mlmodel` numbers nodes **leaves-first, internal bottom-up,
root LAST** — confirmed on depth-2 and depth-3 reference models:

- **Leaf nodes**: `nodeId = 0 .. 2^k - 1`. Leaf `nodeId` carries
  `evaluationInfo = [{evaluationIndex:0, evaluationValue: leaf_values[nodeId]}]`.
  This is the canonical FORWARD-BIT leaf index — cb-model's
  `ObliviousTree::leaf_values[nodeId]` maps DIRECTLY, no permutation (same
  invariant the ONNX exporter relies on).
- **Internal nodes**: numbered by DECREASING depth. Level `L` (with `L = 0` the
  root) has `2^L` nodes. Emit level `k-1` (deepest internal, just above leaves)
  first, then `k-2`, …, then level `0` (the root, a single node with the highest
  `nodeId = 2^(k+1) - 2`).
- **Split at a level**: level `L` tests `splits[k-1-L]` — i.e. the root tests
  `splits[k-1]` (the MSB / highest-index split), the deepest internal level tests
  `splits[0]` (bit 0). `branchFeatureIndex = splits[k-1-L].feature`,
  `branchFeatureValue = splits[k-1-L].border`, `nodeBehavior =
  BranchOnValueGreaterThan`. (Same reversed-split-order mapping as ONNX.)
- **Children** of the node at level `L`, position `p` (`0 .. 2^L - 1` left→right):
  - if `L == k-1` (children are leaves): `falseChildNodeId = 2p`,
    `trueChildNodeId = 2p + 1`.
  - else (children are level-`L+1` nodes): `falseChildNodeId = firstId(L+1) + 2p`,
    `trueChildNodeId = firstId(L+1) + 2p + 1`, where `firstId(level)` is the first
    nodeId of that level's block.
  - `missingValueTracksTrueChild = false`.

`firstId(level)` layout for a single tree (offsets within the tree; multiply
nothing — each tree restarts its own `nodeId` at 0 since `treeId` disambiguates):
`firstId(k-1) = 2^k`, and `firstId(L) = firstId(L+1) + 2^(L+1)` for `L < k-1`.

Worked check (depth-2, `k=2`, leaves 0..3, internal 4,5,6):
- level 1 block starts at `2^2 = 4`: n4 (p0) F→leaf0 T→leaf1; n5 (p1) F→leaf2 T→leaf3; both test `splits[0]`.
- level 0 (root) starts at `4 + 2^2 = ` … = nodeId 6: n6 tests `splits[1]`, F→n4, T→n5.

Worked check (depth-3, `k=3`, leaves 0..7, internal 8..14):
- level 2 (deepest) block @ `2^3 = 8`: n8..n11 test `splits[0]`, F→leaf(2p) T→leaf(2p+1).
- level 1 block @ `8 + 2^2 = 12`: n12,n13 test `splits[1]`, children n8..n11.
- level 0 (root) @ `12 + 2^1 = 14`: n14 tests `splits[2]`, F→n12, T→n13.

Both match CatBoost's emitted `.mlmodel` byte-for-structure.

`treeId = tree index` in boosting order over `model.oblivious_trees`. A
**depth-0 tree** (`k == 0`, a single leaf, no splits) emits ONE leaf node
(`nodeId 0`, `evaluationValue = leaf_values[0]`, no internal nodes) — handle the
`k == 0` edge (don't underflow `k-1`).

## 3. bias / basePredictionValue

`treeEnsemble.basePredictionValue` carries `[model.bias]`. CatBoost ALWAYS emits
it (even at bias 0.0), so — UNLIKE the ONNX exporter's `bias != 0.0` conditional
— emit `basePredictionValue = [model.bias]` unconditionally for a clean diff
against the CatBoost reference. `numPredictionDimensions = 1`. (The model has no
separate scale field; scale is assumed 1.0, same as ONNX — do NOT add a scale
guard, per SPEC §Note-on-scale.)

## 4. Output / input feature descriptors

- One `input` FeatureDescription per float feature: `name = "feature_{i}"`
  (`i` in `0..n_float`), `type.doubleType = {}`.
- One `output` FeatureDescription: `name = "prediction"`,
  `type.multiArrayType = { shape:[1], dataType: DOUBLE(65600) }`.
- `predictedFeatureName = "prediction"`, `specificationVersion = 1`.

`n_float = model.float_feature_borders.len()`.

## 5. Strengthened verification (RESOLVES R1) — three layers

1. **Round-trip (CM-02, unit)** — encode a known small model with the Rust
   exporter, decode the bytes back with the SAME `coreml_generated` prost structs,
   assert: tree count == `oblivious_trees.len()`; per tree the branch
   thresholds/features per level == the reversed-split-order borders; leaf
   `evaluationValue`s == `leaf_values`; `basePredictionValue == [bias]`.
2. **coremltools + CatBoost-reference oracle (CM-04, integration)** — the NEW,
   host-verifiable oracle replacing the plan's "golden bytes with unverified
   schema". `gen_fixtures.py` trains a float-only regressor, saves BOTH
   `model.cbm` (loaded by cb-model) AND `reference.mlmodel` (CatBoost's own CoreML
   output), and decodes the reference with coremltools into a `structure.json`
   (per-tree: nodeId→(behavior, feature, threshold, children, leafValue)). The
   Rust integration test loads `model.cbm`, exports to `.mlmodel`, decodes it with
   the prost structs, and asserts its structure matches `structure.json` (borders
   within 1e-5 to absorb any f32/f64 rounding; node ids/children EXACT because §2
   replicates CatBoost's numbering). This proves the emitted schema is the REAL
   CoreML schema AND the tree semantics match CatBoost — the true parity the plan
   said was unreachable.
3. **Golden bytes (CM-04, regression pin)** — additionally pin the Rust-emitted
   bytes for one frozen tiny `.cbm` to a committed `golden.mlmodel` so encoding
   drift is caught deterministically. Regenerate deliberately, never silently.

`gen_fixtures.py` MUST also cross-check its own reference by re-parsing the
Rust-independent path: assert `coremltools` can `Model_pb2.ParseFromString` the
CatBoost reference (sanity) before freezing.

## 6. Fixture files (under `crates/cb-oracle/fixtures/coreml_export/`)

- `gen_fixtures.py` — pinned seed, `catboost==1.2.10`, `thread_count=1`,
  `bootstrap_type="No"`, float-only `CatBoostRegressor`, reuse
  `inputs/numeric_tiny/X.npy`.
- `model.cbm` — frozen float-only oblivious regressor (the export INPUT).
- `reference.mlmodel` — CatBoost's own CoreML output for `model.cbm`.
- `structure.json` — decoded reference tree structure (the oracle target).
- `golden.mlmodel` — the Rust exporter's frozen output (byte-pin, CM-04).
- `config.json` — versions, seed, n_trees, depth, the artifact list.

Do NOT regenerate at test time (CatBoost quantization is run-to-run
nondeterministic — freeze all artifacts once).

## 7. Environment

`coremltools==9.0` is installed in `.venv` (via `uv pip install coremltools`;
network was reachable). CatBoost `save_model(format="coreml")` works on this
Linux host (no Apple runtime needed for STRUCTURAL export/decode; only *running*
predictions would need one, which is still out of scope — but structure is now
authoritatively verifiable, unlike the plan's original assumption).
