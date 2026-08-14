# float-only `.cbm` byte-identity baseline (SPEC-OH-31 / T00)

**CAPTURED_AT_SHA: `21dbd13142c617cf3c0af393093f1ec1b82ffbe2`** (plus the uncommitted working tree at
capture time — see the re-baseline note below).

Written by `float_only_byte_identity_test::capture_float_only_baseline`
(`#[ignore]`d; run with `-- --ignored`).

## RE-BASELINED once, deliberately (learning_rate f32)

The ORIGINAL capture, taken at plan-base
`9bf734512d7fccb25a9e8304b34183375ae3e7f5` before any one-hot
production change, is preserved verbatim as `baseline_pre_lr_f32.cbm`
(sha256 `37b373e7e16499fdba8abbefda956bbb355ca3908b510b131f23f3990dce6b21`).

`baseline.cbm` was regenerated ONCE, for a single deliberate reason:
upstream stores `learning_rate` as an **f32**, so every leaf value is
scaled by `f32(lr)` rather than the `f64` the caller supplied. Fixing
that (`cb-train/src/boosting.rs`) moves every leaf by the constant
factor `f32(0.3) / 0.3` at this fixture's pinned rate — TOWARD
upstream, verified to one ulp against real catboost 1.2.10.

**The SPEC-OH-31 guarantee was NOT simply discarded.**
`baseline_only_differs_from_pre_lr_baseline_by_the_learning_rate_factor`
proves mechanically that the two baselines have IDENTICAL tree
structure (same split count, features and borders) and that every
leaf differs by exactly that one factor. So `no one-hot change leaked
into the float-only path` remains PROVEN, not merely asserted.

## RE-BASELINED a second time, deliberately (greedy border tie-break)

The capture taken before the `feature_border_type` wave is preserved
verbatim as `baseline_pre_border_tiebreak.cbm` (sha256
`876310517504e8d9cbf55e85cf46aeb9f96d30a011179ce5537bb890fe9fd629`).

`baseline.cbm` was regenerated for one deliberate reason: the greedy
binarizer resolved TIED split scores with a libstdc++ heap emulation
that does not match catboost 1.2.10. This fixture quantizes 512
UNIQUE values into 32 borders, so its budget BINDS — precisely the
regime where the tie-break decides where borders land — and the frozen
bytes therefore encoded the wrong border set.

**This re-baseline is a move TOWARD upstream, and that is PROVEN, not
asserted.** `cb-data`'s `border_types` oracle now includes the cell
`float_only_byte_identity.bc32.GreedyLogSum` — catboost's own
`Pool.quantize(...).save_quantization_borders()` output for THIS
corpus at THIS budget — and
`baseline_moved_its_borders_onto_the_catboost_oracle` checks that the
new `.cbm` stores exactly those borders while the pre-fix `.cbm` does
not. The bytes changed because the quantization got CORRECT.

## Still frozen from here on

Do not regenerate again without the same treatment: preserve the prior
bytes and prove the delta. An unexplained regeneration turns
SPEC-OH-31 into a self-comparison that proves nothing.

## Contents

- `baseline.cbm` — the pinned float-only fit, serialized.
- `baseline_pre_lr_f32.cbm` — the ORIGINAL plan-base capture (frozen).
- `baseline_pre_border_tiebreak.cbm` — the capture taken before the
greedy border tie-break fix (frozen).
- `inputs/X.npy` — `512 x 4` float64 features.
- `inputs/y.npy` — `512` float64 RMSE target.

## Pinned configuration

RMSE, `iterations=3`, `depth=3`, `learning_rate=0.3`,
`l2_leaf_reg=3.0`, `random_strength=0`, `boost_from_average=false`,
`bootstrap_type=No`, `random_seed=0`, `score_function=L2`,
`grow_policy=SymmetricTree`, `border_count=32`.

`bootstrap_type=No` + `random_strength=0` keep the RNG stream out of
the comparison entirely, so any byte difference is attributable to the
tree-construction or serialization path this plan touches.
