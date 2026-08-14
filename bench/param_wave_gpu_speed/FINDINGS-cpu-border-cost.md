# `feature_border_type` build cost — local CPU measurement

**Date:** 2026-08-15 · **Box:** local dev machine (CPU path only; no GPU involved)
**Corpus:** `n = 200_000`, `f = 20`, `border_count = 254`, depth 6, `bootstrap_type=No`,
`random_strength=0` — median of 3 repeats per cell, everything but the border algorithm
held fixed, so the DIFFERENCE between cells is the border-build cost.

This is a **CPU** result and needs no GPU. It is recorded separately from the GPU grid
because it stands on its own: `feature_border_type` chooses a host-side quantization
algorithm, and that choice costs the same wall clock whether the fit later runs on the CPU
or commits to a device.

## The number

| `feature_border_type` | median | vs `GreedyLogSum` | |
|---|---|---|---|
| `GreedyLogSum` (catboost default) | 0.55 s | 1.00× | greedy heap |
| `UniformAndQuantiles` | 0.55 s | 1.00× | *within noise* |
| `Median` | 0.56 s | 1.02× | *within noise* |
| `Uniform` | 0.59 s | 1.08× | |
| `GreedyMinEntropy` | 0.60 s | 1.10× | greedy heap |
| **`MaxLogSum`** | **3.95 s** | **7.23×** | **exact DP** |
| **`MinEntropy`** | **4.30 s** | **7.86×** | **exact DP** |

The two exact dynamic-programming types cost ~8× the whole fit at this iteration count.
Their min/max ranges do not overlap `GreedyLogSum`'s (`MinEntropy` 4.27–4.32 s against
0.54–0.56 s), so this is not noise. The other four are at or near free.

## It is one-time PREP, not per-iteration work

An 8× headline is misleading on its own, so the cost was attributed rather than assumed.
If the extra time were in the grow loop it would scale with iterations; if it is border
building it is paid once and the ratio must collapse as iterations grow:

| iterations | `GreedyLogSum` | `MinEntropy` | delta | ratio |
|---|---|---|---|---|
| 10 | 0.57 s | 4.41 s | **3.85 s** | 7.79× |
| 40 | 2.27 s | 6.10 s | **3.82 s** | 2.68× |
| 160 | 8.36 s | 11.94 s | **3.58 s** | 1.43× |

The delta is flat (3.85 → 3.82 → 3.58 s) while the ratio collapses. So the honest statement
is **"`MinEntropy` adds ~3.8 s of one-time border-build cost on this corpus"**, NOT
"`MinEntropy` is 8× slower" — the 8× is an artifact of a short 10-iteration fit, and a
real 1000-iteration training run would barely notice it.

An independent re-run through the committed script gives 3.57 / 3.65 / 4.37 s (ratio
7.02 → 2.65 → 1.52), so treat the delta as **~3.5–4.4 s** rather than a precise 3.8. The
run-to-run spread does not touch the conclusion: the delta stays flat within noise across a
16× change in iteration count while the ratio falls monotonically, which is the signature
of fixed prep and is incompatible with per-iteration work.

## Is the ~3.8 s ours, or the algorithm's?

The exact DP is asymptotically dearer than the greedy heap, so *some* penalty is expected
and the raw number cannot answer this. See `FINDINGS-cpu-border-vs-upstream.md` for the
comparison against official CatBoost's own `MinEntropy`-vs-`GreedyLogSum` penalty on the
same axis, each engine measured against its OWN `GreedyLogSum` so absolute engine speed
cancels out.

## Reproducing

The measurement is three fits per cell through the public Python surface with a single
kwarg varied; there is nothing to configure. `border_count = 254` is deliberately at the
upstream maximum — the DP's cost grows with the border count, so this is the pessimistic
end of the axis, not a typical one.
