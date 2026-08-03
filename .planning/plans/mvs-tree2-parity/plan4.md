---
plan: 4
task_id: TASK-04
phase: mvs-tree2-parity
status: pending
order: 4
wave: serial (v2 — Wave B dissolved, PLAN-CHECK CRITICAL-3)
hardware: none (CPU only)
depends_on: [TASK-01]
blocks: [TASK-05, TASK-06]
specifications: [MVS-S4]
parallelizable: false
parallel_with: []
revision_note: >
  v2: must run AFTER TASK-02 and TASK-03 have captured their pre-fix Reds (MAJOR-1 — v1's
  "provably cannot invalidate" claim is withdrawn); the "clippy clean" / "gate scripts
  pass" criteria become differential (CRITICAL-1, CRITICAL-2); the exact per-call-site
  f32-vs-f64 delta table is added, since it is what makes TASK-06's mirror test necessary
  (MAJOR-3).
---

# Task 4: The block threshold target reproduces upstream's `float` expression

## Objective

After this task the per-block MVS threshold-search target is
`f64::from((sample_rate as f32) * block_size as f32)` — a faithful transcription of
upstream's `float SampleRate * ui32 blockSize` (`mvs.cpp:202`, `mvs.h:47`) — instead
of the `f64` product it is today.

**Observable completion condition:** at `(sample_rate, block_size) = (0.8, 1500)`
the target is **exactly `1200.0`**, not `1200.0000178813934` (a `+1.788e-5`
absolute error today), while `(0.8, 8192)` stays exactly `6553.60009765625`
(unchanged — a power-of-two scaling was already exact).

## Specification references

- `MVS-S4` — the block threshold target reproduces upstream's `float` expression.
  Principal failure reason: *the threshold-search target is computed in `f64` where
  upstream computes `float SampleRate * ui32 blockSize` in `f32`.*

## Prerequisites and blocking

- Prerequisite: **TASK-01** — same file (`crates/cb-train/src/bootstrap.rs`), and
  the oracle baseline must be at the post-fix state before a numerics change is
  layered on.
- **Additional prerequisite (new in v2, MAJOR-1): TASK-02 and TASK-03 must have CAPTURED
  their pre-fix Reds first.** v1 permitted this task to land in parallel with them on the
  claim that its numerics "provably cannot invalidate" theirs. That claim is **withdrawn**:
  research §8.5/§8.6 spiked only the POST-fix combinations (`A + B`, `A + B + C`), so it
  supports only "TASK-04 cannot change their post-fix pass/fail verdict". The
  **defect-present + f32-target** state — which is what their Reds measure — was never
  spiked, and the `+1.788e-5` target shift at `block_size = 1500` is precisely the
  perturbation that can move a near-tied argmax, so the recorded pre-fix `StageDiverged`
  values and failing set could legitimately differ with this change present.
- Blocks TASK-05 (same file, serialised) and TASK-06 (the mirror).
- **NOT parallelisable.** The phase is serial (PLAN.md §1).

## Context and evidence

- **The site, verbatim at HEAD** `[VERIFIED: read crates/cb-train/src/bootstrap.rs:306-313]`:

  ```
  306	        // thresholdCandidates[idx] = sqrt(lambda + der^2) over the block.
  307	        let mut candidates: Vec<f64> = block.iter().map(|&d| (lambda + d * d).sqrt()).collect();
  308	        let threshold = calculate_threshold(
  309	            &mut candidates,
  310	            0.0,
  311	            0.0,
  312	            sample_rate * block_size as f64,
  313	        );
  ```
- **Upstream.** `TMvsSampler::SampleRate` is a `float` (`mvs.h:47`); `mvs.cpp:202`
  calls `CalculateThreshold(begin, end, 0, 0, SampleRate * blockSize)` where
  `blockSize` is a `ui32`, so the product is evaluated in **`float`** and only then
  widened into the `double sampleSize` parameter
  `[VERIFIED: research.md §1.4, §2.1, §2.6 N1]`.
- **The quantified delta** `[VERIFIED: research.md §8.4, numpy]`:

  | `blockSize` | upstream `float(0.8f*bs)` | ours `f64(0.8f)*bs` | ours − upstream |
  |---|---|---|---|
  | 1500 | `1200.0` | `1200.0000178813934` | `+1.788e-05` |
  | 8192 | `6553.60009765625` | `6553.60009765625` | `0` |
  | 3616 | `2892.800048828125` | `2892.800043106079` | `-5.722e-06` |
  | 20000 | `16000.0` | `16000.00023841858` | `+2.384e-04` |

  Only partial blocks and small `n` diverge; a FULL 8192 block is exact because the
  scaling is a power of two.
- **The precondition making the transcription exact.** `block_size ≤ MVS_BLOCK_SIZE = 8192
  < 2^24` (`bootstrap.rs:64`, `:301-304`), so `block_size as f32` is exact and the
  expression is a faithful `float * ui32`
  `[VERIFIED: read; research.md §5.3]`.
- **`sample_rate` is ALREADY f32-narrowed** one line earlier, at `bootstrap.rs:294`
  (`let sample_rate = f64::from(sample_rate as f32);`), mirroring the `float`
  `SampleRate` member. The `MVS-S4` expression therefore applies a **second,
  idempotent** `as f32` — deliberate, and it must be documented as such so a future
  reader does not "simplify" it away (risk `P4` in PLAN.md §6).
- **`calculate_threshold`'s algorithm is upstream-faithful and must NOT change**
  (`bootstrap.rs:209-273`; SPEC `MVS-S4` non-goal, research §5.5). Only the fourth
  ARGUMENT changes.
- **Spiked safe.** `A + B` (TASK-01's deletion + this f32 target): all 5
  `bootstrap_oracle_*` ok, `bootstrap_dev` ok at 3 trees, the multi-seed probe
  **5/5 and 5/5**, and every printed residual **byte-identical** to `A` alone
  `[VERIFIED: research.md §8.5]`. Reason MVS weights are this insensitive: they reach
  the model ONLY through the discrete split argmax — leaf values are estimated on the
  un-sampled averaging fold — so a ≤1e-7 relative perturbation is invisible unless it
  flips a near-tie `[VERIFIED: research.md §2.6]`.
- **`f32` arithmetic check for the acceptance values** (independently re-derived):
  `0.8_f32 = 0.800000011920928955078125`; `× 1500 = 1200.0000178813934…`, whose
  nearest `f32` is `1200.0` (the `f32` spacing at 1200 is `2^-13 ≈ 1.22e-4`), so the
  `f32` product is exactly `1200.0`. `× 8192` is an exact exponent shift ⇒
  `6553.60009765625`. Both match the research table.
- **The mirror obligation.** `crates/cb-backend/src/kernels/mvs_device_test.rs:130-133`
  carries the SAME expression (`sample_rate * block.len() as f64` at `:132`)
  `[VERIFIED: read]`. It is a deliberate inline copy (no `cb-train` dep). TASK-06
  mirrors it — **do not** edit `cb-backend` from this task.
- **Why the mirror needs its OWN test (feeds TASK-06 / MAJOR-3).** Measured across every
  `cpu_block_threshold` call site in the device self-oracle, the f32-vs-f64 target delta is
  `[VERIFIED: RUN — numpy]`:

  | `(rate, n)` | site | f32 target | f64 target | rel. delta |
  |---|---|---|---|---|
  | `(0.5, 48)` | `mvs_device_test.rs:204` | `24.0` | `24.0` | **0 (bit-identical)** |
  | `(0.7, 64)` | `:204` | `44.79999923706055` | same | **0** |
  | `(0.3, 200)` | `:204` | `60.000003814697266` | `60.00000238418579` | **2.384e-8** |
  | `(0.6, 96)` | `:248` | `57.60000228881836` | same | **0** |
  | `(0.5, 8192)` / `(0.5, 24)` | `:284` | `4096.0` / `12.0` | same | **0** |

  So 4 of the 5 call sites cannot detect this change at all, and the 5th moves 4 orders
  under the self-oracle's `TOL = 1e-4`. A device-only criterion for the mirror is
  therefore satisfiable while the two transcriptions silently disagree — which is why
  TASK-06 must add a non-device-gated arithmetic test.

## Files

- Modify: `crates/cb-train/src/bootstrap.rs`
  - add a private helper `mvs_block_sample_size(sample_rate: f64, block_size: usize) -> f64`
    near `mvs_sample_weights` (i.e. between `calculate_threshold` at `:273` and
    `mvs_sample_weights` at `:281`);
  - replace the `sample_rate * block_size as f64` argument at `:312` with a call to
    it.
- Modify: `crates/cb-train/src/bootstrap_test.rs` — add ONE test.
- Do NOT touch: `calculate_threshold`'s body, `:294`, `:323` (TASK-05), the module
  docs (TASK-07), `crates/cb-backend/**` (TASK-06).

## TDD sequence

### 1. Red

The value under test is currently an inline expression inside a private function, so
it has no testable seam. Create the seam FIRST, behaviour-preserving, then write the
value-Red. Both sub-steps are mandatory and must be run in this order.

**1a. Create the seam (behaviour-identical).** Add

```
fn mvs_block_sample_size(sample_rate: f64, block_size: usize) -> f64
```

whose body is, for now, the CURRENT in-situ expression
`f64::from(sample_rate as f32) * block_size as f64`, and call it at `:312`. Then run
`cargo test -p cb-train --test bootstrap_oracle_test --test bootstrap_dev_oracle_test`
and `cargo test -p cb-train --lib bootstrap`. **All must be green and the printed
residuals identical** — this step changes nothing numerically. If anything moves, the
extraction was wrong; fix it before continuing.

> **CONSTANT TRAP (plan-check pass 3, `[C3-1]`) — read before writing step 1a.**
> The seam body is **not** the bare `sample_rate * block_size as f64` that appears at
> `:312`. By the time control reaches `:312` the rate has ALREADY been narrowed at
> `crates/cb-train/src/bootstrap.rs:294` (`let sample_rate = f64::from(sample_rate as f32);`),
> so the in-situ value is `f64::from(0.8_f32) * 1500.0 = 1200.0000178813934`. The unit
> test below calls the helper with a RAW `0.8` literal, and `0.8_f64 * 1500.0` is
> **exactly `1200.0`** — so a helper carrying the bare expression makes assertion 1 PASS
> on its first run, the documented Red unreachable, and the `MVS-S4` half of the mirror
> ship untested. The helper must therefore do its own f32 narrowing, which is idempotent
> when the caller has already narrowed (`f32(f64(f32(x))) == f32(x)`), so `:312` is
> unaffected. Verified at HEAD:
> `(0.8,1500)` → bare `1200.0`, narrowed `1200.0000178813934`, target `1200.0`;
> `(0.8,3616)` → bare `2892.8`, narrowed `2892.800043106079`, target `2892.800048828125`.

**1b. The failing test.** In `crates/cb-train/src/bootstrap_test.rs` (a child module
of `bootstrap`, so it can call the private helper — the existing tests already reach
private items through `use crate::bootstrap::{…}` at `:17-19`; extend that import):

- **Test name:** `mvs_block_sample_size_reproduces_upstream_float_expression`
- **Assertions, in this order** (exact equality with `assert_eq!`, NOT an epsilon —
  the whole point is bit-level transcription):
  1. `assert_eq!(mvs_block_sample_size(0.8, 1500), 1200.0);`
  2. `assert_eq!(mvs_block_sample_size(0.8, 8192), 6553.60009765625);`
  3. `assert_eq!(mvs_block_sample_size(0.8, 3616), 2892.800048828125);`
- **Expected initial failure** (from assertion 1):

  ```
  assertion `left == right` failed
    left: 1200.0000178813934
   right: 1200.0
  ```

  Assertion 2 must PASS both before and after (the power-of-two case is already
  exact) — it is the no-regression leg, and if it ever fails the helper is wrong in a
  new way.
- Run: `cargo test -p cb-train --lib mvs_block_sample_size`

### 2. Green

Change the helper's body to the upstream expression:

```
f64::from((sample_rate as f32) * block_size as f32)
```

Document, on the helper:

- upstream is `CalculateThreshold(…, SampleRate * blockSize)` (`mvs.cpp:202`) with
  `float SampleRate` (`mvs.h:47`) and `ui32 blockSize`, so the product is a **`float`**
  expression widened into the `double sampleSize` parameter;
- `block_size ≤ MVS_BLOCK_SIZE = 8192 < 2^24` makes `block_size as f32` exact, so
  this is a faithful transcription, not an approximation;
- the `sample_rate as f32` is **deliberately redundant** with the narrowing at
  `:294` (idempotent) — it keeps the helper correct in isolation and must not be
  "simplified" away;
- the measured effect at the fixture (`block_size = 1500`): `1200.0` instead of
  `1200.0000178813934`, i.e. the old code was `+1.788e-5` off.

Do NOT change `calculate_threshold`, the candidate construction at `:307`, or the
reweight loop at `:315-328`.

- Run: `cargo test -p cb-train --lib bootstrap`
- Run: `cargo test -p cb-train --test bootstrap_oracle_test` (5 frozen — must stay
  green)
- Run: `cargo test -p cb-train --test bootstrap_dev_oracle_test` (green at 3/3 after
  TASK-02; at 2 trees if TASK-02 has not landed yet — either is acceptable, it must
  simply not regress)

### 3. Refactor

- Keep the helper private (`fn`, not `pub fn`) — `MVS-S4` makes no public-surface
  change and `bootstrap`'s signature is unchanged (SPEC §4).
- If clippy objects to the cast (e.g. `cast_precision_loss`), add a NARROW
  `#[allow(...)]` on the helper with a one-line rationale citing the upstream `float`
  expression — the same pattern `fast_log2f` uses at `bootstrap.rs:117` for its
  deliberate transcription allowances. Do not add a crate-level allow.
- Run: `cargo test -p cb-train --lib bootstrap`
- **Lint gates — DIFFERENTIAL** (CRITICAL-1, CRITICAL-2; `PLAN.md` §4.11 baselines,
  §4.12 form). Never `cargo clippy -p cb-train --all-targets` (it aborts on the
  `cb-oracle` dev-dependency), never "the gate scripts pass" (two exit 1 at HEAD):

  ```bash
  # ERROR-attributed (C2-1): a `-->` grep also catches the pre-existing
  # `bootstrap.rs:134` excessive_precision WARNING and would be red before any work.
  clippy_error_files -p cb-train --all-targets | grep -E "src/bootstrap\.rs|bootstrap_test\.rs"  # must be EMPTY
  bash scripts/check-source-test-separation.sh                              # ABSOLUTE: exit 0
  bash scripts/check-no-raw-float-sum.sh 2>&1 | grep -E "src/bootstrap\.rs|bootstrap_test\.rs"  # must be EMPTY
  bash scripts/check-no-anyhow.sh        2>&1 | grep -E "src/bootstrap\.rs|bootstrap_test\.rs"  # must be EMPTY
  ```

  All measured EMPTY at HEAD `[VERIFIED: RUN]`. **Doc-text constraint (C2-6):** the helper's
  doc comment must not contain the literal `.sum()` or `.fold(0.0` — the D-08 script greps
  comments in non-test source and `bootstrap.rs` is currently clean of that pattern.

### 4. Verify

- Run: `cargo test -p cb-train --lib bootstrap` → 9 bootstrap unit tests
  (7 baseline + TASK-01's + this one).
- Run: `cargo test -p cb-train --test bootstrap_oracle_test` → 5 passed, values
  unchanged.
- Run: `cargo test -p cb-train --test bootstrap_dev_oracle_test -- --nocapture` →
  green; the printed residuals must be **byte-identical** to the TASK-01/TASK-02
  run (research §8.5 measured exactly this). Record them; a change means the
  extraction or the cast is wrong.
- Run: `cargo test -p cb-train --test mvs_seeds_oracle_test -- --nocapture`
  (if TASK-03 has landed) → still 10/10, residuals unchanged.
- Run: `cargo test -p cb-train --test regularization_oracle_test` → green.
- Run: `git status --short crates/cb-oracle/fixtures/` → EMPTY (no fixture moved).
- Run: `grep -n "sample_rate \* block_size" crates/cb-train/src/bootstrap.rs` → no
  output (the old expression is gone).
- Confirm: `git diff crates/cb-train/src/bootstrap.rs` shows changes ONLY at the new
  helper and the `:312` call site.

## Completion criteria

- [ ] The seam extraction (1a) was proven behaviour-identical before the value
      change.
- [ ] The Red failed with `left: 1200.0000178813934, right: 1200.0`.
- [ ] All three acceptance values hold exactly: `1200.0`, `6553.60009765625`,
      `2892.800048828125`.
- [ ] `calculate_threshold`'s body is byte-unchanged.
- [ ] The helper's doc records the upstream citation, the `< 2^24` exactness
      argument, and the deliberate double narrowing.
- [ ] `bootstrap_oracle_test` 5/5, `bootstrap_dev_oracle_test` green with
      byte-identical residuals, `mvs_seeds_oracle_test` 10/10 (it MUST already exist —
      TASK-03 precedes this task in v2).
- [ ] **Differential lint gates**: the diff-scoped greps are EMPTY and
      `check-source-test-separation.sh` exits 0. Do NOT assert "clippy clean" or "gate
      scripts pass" (`PLAN.md` §4.11 `B3`/`B4`/`B8`/`B9`).
- [ ] Fixtures byte-unchanged; `crates/cb-backend/**` untouched.

## Completion evidence to record in `progress.md`

- The Red failure text and the three post-fix exact values.
- The `bootstrap_dev` residuals before and after this task (must match).
- Confirmation that `crates/cb-backend/**` was not touched (deferred to TASK-06).

## Risks and guardrails

- **SPEC `MVS-S4` non-goal — reimplementing `calculate_threshold`.** Forbidden. Only
  the fourth argument changes. In particular do NOT "match the device kernel" by
  converting the quickselect to bisection: the CPU function is the upstream-faithful
  one and the device kernel deliberately matches only the SEMANTICS
  (`mvs_device.rs:16-22`).
- **`R2` / `MVS-S6` — the unmirrored device copy.** After this task
  `mvs_device_test.rs:132` disagrees with the CPU. That is expected and is TASK-06's
  job; it is invisible on the default `cpu` CI, which is exactly why TASK-06 is a
  blocking successor. Do not "fix it while you are here" — the two-file split keeps
  the rocm-only failure attributable.
- **A hairline oracle pass.** If any oracle residual MOVES after this change, stop:
  research measured them byte-identical. A movement means the extraction changed
  more than the target (e.g. `block_size` vs `block.len()` mismatch).
- **Clippy cast lints.** Handle with a narrow, documented `#[allow]`, never by
  widening the cast or by a crate-level allow.
