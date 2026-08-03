# Device float-only identity baseline (SPEC-OH-31 / T00)

## THIS FIXTURE IS FROZEN

Captured at the plan-base SHA recorded in the parent directory's
`README.md`, BEFORE any one-hot production change. No later task may
regenerate it — a baseline regenerated after a change proves nothing.

## Contents

- `packed_cindex.json` — `(words, offsets, shifts, masks)` for the pinned
float-only quantized input (4 features x 64 objects, 32 buckets each).
Pure HOST bit-packing: capturable and comparable with no GPU.

## Why it matters

The one-hot work adds a separate real-cardinality array alongside
`TCFeature.folds` and threads `feature_lo`/`feature_hi` through the
scorer. Any of that leaking into the float-only packing would change
these words and break every existing device oracle. This fixture makes
that failure loud and immediate instead of subtle.

## Not captured here

`scorer_winners.json` and `device_baseline.cbm` require a live GPU
session (`score_partition_over_binsums` needs a client and real bin
sums). They are captured on a GPU-enabled run; their absence is why
T29b's scorer assertion is gated on the artifact being present.
