# float-only `.cbm` byte-identity baseline (SPEC-OH-31 / T00)

**PLAN_BASE_SHA: `9bf734512d7fccb25a9e8304b34183375ae3e7f5`**

Captured BEFORE any one-hot production change, by
`float_only_byte_identity_test::capture_float_only_baseline`
(`#[ignore]`d; run with `-- --ignored`).

## THIS FIXTURE IS FROZEN

No later task may regenerate it. Regenerating it after a production
change turns SPEC-OH-31 into a self-comparison that proves nothing —
the whole point is that these bytes predate the change.

## Contents

- `baseline.cbm` — the pinned float-only fit, serialized.
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
