# Failing tests — `worktree-gpu-full-parameter-parity` @ `d24f25b`

Scope: **only the tests that failed.** Measured 2026-08-07 on the local rig (AMD gfx1151,
ROCm) for the device lane and default features for the CPU lane.

**Two tests fail. Neither is caused by this phase's work, and neither indicates a defect in
the code under test.** One is a load-sensitive performance threshold; the other asserts
behaviour the product deliberately changed. Both were verified against a baseline that
excludes every change made here.

| # | test | crate / lane | verdict | cause |
|---|---|---|---|---|
| F1 | `kernels::poisson_bootstrap_speed_test::poisson_parallel_draw_outpaces_the_serial_stream_draw` | cb-backend, `--features rocm` | pre-existing, **environmental** | GPU contention when the full suite shares the device; passes 10.5–10.7× in isolation |
| F2 | `monotone_non_symmetric_and_region_are_typed_errors` | cb-train, default features | pre-existing, **stale assertion** | asserts `grow_policy=Region` is rejected; GPUT-18 deliberately LIFTED that rejection |

Everything else is green: cb-backend rocm 259 passed, cb-train CPU 57 suites ok, and the
whole rocm device set (ordered / bias / ctr / nonsym / weighted / seam / gate_composition /
fpp_composition / exact_leaf).

---

## F1 — `poisson_parallel_draw_outpaces_the_serial_stream_draw`

**File:** `crates/cb-backend/src/kernels/poisson_bootstrap_speed_test.rs:119`
**Asserts:** `speedup >= MIN_SPEEDUP` where `MIN_SPEEDUP = 5.0`, `N = 2_000_000`.
It times the parallel grid-stride Poisson draw against a serial single-thread stream draw and
fails if the ratio drops below 5×, on the theory that a collapse means "the grid-stride
transcription of upstream's `PoissonBootstrapImpl` has regressed to a serial loop".

### It is not a regression — it is contention

The parallel arm is fine. Run alone, it is **twice as fast as the bar requires**:

| run | parallel | serial | ratio | result |
|---|---|---|---|---|
| isolated 1 | 32.7 ms | 347.4 ms | **10.6×** | PASS |
| isolated 2 | 33.0 ms | 346.4 ms | **10.5×** | PASS |
| isolated 3 | 33.2 ms | 355.8 ms | **10.7×** | PASS |

Under the full `--lib` suite, where other tests are hitting the same GPU, it collapses:

| run | parallel | serial | ratio | result |
|---|---|---|---|---|
| full suite, **baseline** (no changes) 1 | 218.4 ms | 723.3 ms | 3.3× | FAIL |
| full suite, **baseline** 2 | 186.3 ms | 703.4 ms | 3.8× | FAIL |
| full suite, **baseline** 3 | 171.4 ms | 627.2 ms | 3.7× | FAIL |
| full suite, with this phase's changes 1 | 293.3 ms | 352.4 ms | 1.2× | FAIL |
| full suite, with this phase's changes 2 | 218.1 ms | 456.8 ms | 2.1× | FAIL |
| full suite, with this phase's changes 3 | 217.1 ms | 626.1 ms | 2.9× | FAIL |

The mechanism is visible in the numbers. Under load the parallel (GPU-bound) arm degrades
~6× (33 → ~200 ms) while the serial arm degrades only ~2× (350 → ~700 ms). The ratio is a
quotient of two quantities that do not degrade equally, so it collapses even though the
kernel is untouched. Nothing here measures the property the assertion claims to protect.

### Pre-existing, and the wider spread is noise not signal

**Baseline fails 3/3.** The comparison was run by stashing every change and re-running the
identical full suite. Baseline passes 255 tests and fails this one; with the phase's changes
it passes 259 and fails this one. The change is `+4 passing, +0 failing`.

The runs carrying this phase's changes show a *lower and wider* ratio (1.2–2.9× vs 3.3–3.8×).
That is not evidence of a slowdown introduced here: this phase adds 4 tests that themselves
use the GPU, which increases contention, and the individual measurements are extremely noisy
(the serial arm alone ranges 352–723 ms across runs of identical code). The parallel arm's
isolated time is unchanged.

### Recommendation

The test is measuring the wrong thing in a shared-device suite. Options, best first:

1. **Serialize it** against other GPU tests (a suite-wide device mutex), so it measures the
   kernel rather than the scheduler.
2. **Compare against a fixed budget** — e.g. assert the parallel draw is under ~60 ms at
   N=2 000 000 — instead of a ratio against a co-scheduled serial arm.
3. Mark `#[ignore]` and run it in a dedicated single-test lane.

Lowering `MIN_SPEEDUP` is the wrong fix: it would keep a metric whose value under load is
governed by co-scheduling, not by the code.

---

## F2 — `monotone_non_symmetric_and_region_are_typed_errors`

**File:** `crates/cb-train/tests/monotone_oracle_test.rs:286`
**Failing assertion:** sub-assertion (2) —

```rust
let mut params = isolating_params(vec![]);      // EMPTY monotone_constraints
params.grow_policy = EGrowPolicy::Region;
let result = train(&CpuBackend, ...);
assert!(result.is_err(),
    "grow_policy=Region must be rejected with a typed error (D-6.6-04 \"Region OUT\")");
```

Sub-assertion (1) — `monotone_constraints` × {Lossguide, Depthwise} must be rejected — still
**passes**. Only the bare-Region case fails, and it fails because `train` now SUCCEEDS.

### The product deliberately changed; the test did not

`validate_grow_policy` (`crates/cb-train/src/boosting.rs`) says so in as many words:

```rust
// GPUT-18 / D-03a: the "Region OUT" rejection is LIFTED — Region now grows on the
// CPU as a `TRegionModel`-style path (`region_grower`). The monotone guard below
// STILL rejects Region + monotone_constraints ...
let non_symmetric_policy = grow_policy.is_non_symmetric() || grow_policy == EGrowPolicy::Region;
if non_symmetric_policy && !monotone_constraints.is_empty() { ... }
```

Region is implemented — `region_grower` is dispatched at `boosting.rs:5943`, `TrainedModel`
carries `region_trees`, and `device_region_fit_test` passes. So a Region fit with EMPTY
constraints is *supposed* to train. The test still pins the pre-GPUT-18 contract
(D-6.6-04 "Region OUT") and was never updated when that gap was closed.

**This is a false failure: the assertion is obsolete, not the code.**

### A stale doc comment travels with it

The doc block directly above `validate_grow_policy` still lists, as item 1:

> `EGrowPolicy::Region` — UNIMPLEMENTED on the CPU path ("Region OUT", D-6.6-04). There is no
> Region grower arm; selecting it errors rather than silently falling back…

which contradicts the inline comment a few lines below it and the shipped `region_grower`.
Anyone reading the doc would reach the same wrong conclusion the test encodes.

### Pre-existing — verified, not assumed

`crates/cb-train/src/boosting.rs` was reverted to `ce9d604` (the commit before this phase
touched cb-train) and the test re-run: **still FAILED**, same assertion. This phase's cb-train
changes (populating `DeviceOrderedConfig`, removing the `ordered_learning_perm.is_none()`
device-eligibility clause) are unrelated to Region validation, and the CPU lane does not
consult `device_host_eligible` at all.

### Recommendation

1. **Fix the test:** drop sub-assertion (2), or invert it to assert a bare-Region fit now
   trains and produces `region_trees`. Keep sub-assertion (1) — the monotone × non-symmetric
   rejection is live behaviour and still correct.
2. **Fix the stale doc** on `validate_grow_policy` so item 1 reflects that Region is
   implemented and only Region × monotone is rejected.
3. Rename the test — `..._and_region_are_typed_errors` will be wrong once (1) is the only
   surviving claim.

---

## Not covered by this report

These lanes were **not run** in this session, so this report says nothing about them. Prior
notes record pre-existing failures in at least the first two:

- `catboost-rs-py` (Python binding link/test lane)
- cb-backend under the default `cpu` feature (CubeCL MLIR)
- the full workspace build (`cargo test --workspace`)

The DoD's full command list should be run before any release claim; this report covers only
what was executed here.
