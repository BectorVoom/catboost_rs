# `learning_rate` is an f32 upstream — a global parity gap

## Verdict

```
RULE: the rate that scales every leaf value is f64::from(learning_rate as f32)
STATUS: VERIFIED (exact, to one ulp)
SCOPE: every model this library trains — NOT one-hot-specific
APPLIED: YES (2026-08-01) — SPEC-OH-31 baseline re-captured, delta PROVEN
```

Upstream's `TBoostingOptions::LearningRate` is a **float**. We hold it as `f64`
and multiply leaf values by the f64 the user typed, so for the ubiquitous
`learning_rate = 0.1` every leaf of every tree is off by a CONSTANT relative

```
f32(0.1) / 0.1 - 1  =  1.4901161193847656e-08
```

(`f32(0.1) = 0.10000000149011612`).

## Evidence

Measured against the committed, frozen upstream fixture
`crates/cb-oracle/fixtures/one_hot_train/multi` (real catboost 1.2.10,
`learning_rate = 0.1`). Tree 0's eight leaf values, upstream vs ours:

| upstream | ours | upstream/ours − 1 |
|---|---|---|
| 0.0034090909598903222 | 0.0034090909090909089 | 1.4901161193847656e-08 |
| 0.04642857212041105 | 0.04642857142857143 | 1.4901160971803051e-08 |
| −0.043750000651925802 | −0.043750000000000004 | 1.4901161193847656e-08 |
| 0.033823529915774572 | 0.033823529411764711 | 1.4901161193847656e-08 |
| −0.044565218055377831 | −0.044565217391304347 | 1.4901161193847656e-08 |
| 0.040540541144641673 | 0.040540540540540543 | 1.4901161193847656e-08 |
| −0.043750000651925802 | −0.043750000000000004 | 1.4901161193847656e-08 |
| −0.020689655480713678 | −0.020689655172413793 | 1.4901161193847656e-08 |

The ratio is CONSTANT across all eight leaves and equals `f32(0.1)/0.1` exactly.
Multiplying our values by that factor reproduces upstream's to **6.9e-18** (one
ulp) on every leaf.

## End-to-end effect (measured before vs after applying the fix)

| gate | without the fix | with the fix |
|---|---|---|
| `one_hot_train/default_binary`, full production train→predict | `max\|diff\| = 1.998e-9` | **`2.776e-17`** |
| `one_hot_train/multi`, trees 0 and 1 leaf values | ~1e-8 relative | **6.9e-18** (one ulp) |
| every existing ≤1e-5 upstream oracle | pass | pass |
| SPEC-OH-31 frozen float-only `.cbm` byte-identity | pass | bytes move — baseline re-captured, delta PROVEN (below) |

Why it has been invisible: `1.49e-8` relative is four orders of magnitude under
the project's ≤1e-5 oracle bar, so no fixture could ever have caught it. It
surfaced only because it compounds through the boosting residuals and was large
enough to flip a NEAR-TIED one-hot split at `multi`'s third tree — turning an
`1e-8` arithmetic error into a `4.6e-2` prediction difference.

## The fix

`crates/cb-train/src/boosting.rs`, immediately after the auto-LR resolution:

```rust
let learning_rate = f64::from(learning_rate as f32);
```

One line, applied immediately after the auto-LR resolution, with the
measurements above transcribed into the comment at that site.

## How the SPEC-OH-31 baseline was handled

Applying the fix changes the float-only `.cbm` bytes, so the frozen SPEC-OH-31
baseline (`crates/cb-oracle/fixtures/float_only_byte_identity/`) had to be
re-captured. Regenerating a baseline normally destroys its entire value — a
baseline compared against itself proves nothing, and that fixture was the only
evidence that the one-hot work did not disturb the float path (acceptance A2).

**It was NOT simply discarded.** The procedure used:

1. The ORIGINAL plan-base bytes were preserved verbatim as
   `baseline_pre_lr_f32.cbm` (sha256
   `37b373e7e16499fdba8abbefda956bbb355ca3908b510b131f23f3990dce6b21`,
   captured at `9bf734512d7fccb25a9e8304b34183375ae3e7f5`).
2. The one-line fix was applied and `baseline.cbm` re-captured.
3. A NEW test,
   `baseline_only_differs_from_pre_lr_baseline_by_the_learning_rate_factor`,
   proves the delta mechanically:
   * identical tree count, identical splits `(feature, border)` in identical
     order, identical leaf weights, identical bias — i.e. the SEARCH is
     untouched, which is what SPEC-OH-31 actually asserts;
   * tree 0's leaves scaled by EXACTLY `f32(0.3)/0.3` (to 4 ulp);
   * later trees' leaves within `1e-6` relative of that scaling — exact only to
     first order, because tree 0's rescaled leaves shift the running approx and
     therefore tree 1's gradients. The measured residual is ~1e-8, the same
     order as the rate change itself; a structural change would move leaves by
     percents AND trip the split assertion.
4. `float_only_byte_identity/README.md` records the re-baseline, its reason, and
   the pointer to the preserved bytes. Its provenance test now REQUIRES that
   disclosure and requires the preserved file to still exist.

So "no one-hot change leaked into the float-only path" remains PROVEN.

## Possible follow-up

Add a companion gate pinned at an f32-EXACT learning rate (e.g. `0.25`), for
which this rounding is a mathematical no-op. Such a gate is invariant to this
class of change and would never need re-baselining.

## Not to be confused with

The separate, still-OPEN `multi` divergence: upstream's one-hot split SCORE
differs from ours for at least one candidate. That is independent of this gap —
it persists with the learning-rate fix applied. See the doc comment on
`one_hot_train_matches_upstream_within_1e5_multi` in
`crates/cb-train/tests/one_hot_oracle_test.rs`.
