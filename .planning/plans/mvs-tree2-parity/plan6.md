---
plan: 6
task_id: TASK-06
phase: mvs-tree2-parity
status: pending
order: 6
wave: serial (v2)
hardware: default `cpu` feature for the NEW mirror test + local ROCm gfx1151 for the device self-oracle
depends_on: [TASK-04, TASK-05]
blocks: [TASK-08, TASK-09]
specifications: [MVS-S6]
parallelizable: false
revision_note: >
  v2, three corrections. (1) MAJOR-3: v1's Red was a static grep and 4 of 5 device call
  sites are bit-identical under the f32 change, so the mirror would have shipped with NO
  test; this task now ADDS a non-device-gated host-arithmetic test that runs on the default
  `cpu` feature. (2) MAJOR-4: v1's Verify said `max_div` "should DROP" — backwards, and a
  route to the forbidden kernel edit; it will RISE from ~1e-13 to ~1e-7. (3) MINOR-1: a
  doc-only note at `mvs_device.rs:145-146` is now REQUIRED and the gate becomes "no
  executable change" rather than "byte-unchanged".
  v3 (pass-2 C2-4): step **1a is now MANDATORY** — extract `cpu_block_sample_size` with the
  CURRENT `f64` body and prove it inert BEFORE writing the Red, exactly as `plan4.md` does.
  Without it the natural implementation authors the helper post-fix, assertion 1 passes on
  first run, and the `MVS-S4` half of the mirror ships untested. Clippy checks switched to
  the error-attributed form (C2-1) because `mvs_device.rs:80` warns pre-existingly.
---

# Task 6: Mirror the two f32 transcription fixes into `cb-backend`'s inline CPU copies

## Objective

After this task `crates/cb-backend/src/kernels/mvs_device_test.rs`'s deliberate,
dependency-free inline copies of the CPU sampler compute the SAME `sampleSize` and
store the SAME `f32`-narrowed weight as the post-TASK-04/05 CPU sampler, so the
device MVS self-oracle keeps holding at its `TOL = 1e-4` bar on rocm/cuda.

**Observable completion condition — two parts (revised, MAJOR-3):**

1. `cargo test -p cb-backend --lib mvs` on the **default `cpu` feature** passes a NEW
   non-device-gated test that fails if either mirrored expression regresses. This is the
   part v1 was missing entirely, and it is what closes the "invisible on default CI" hole
   `MVS-S6` itself names.
2. `cargo test -p cb-backend --no-default-features --features rocm --lib mvs -- --test-threads 1`
   is green (4 tests: the 3 existing device oracles + the new host one), and
   `grep -n "sample_rate \* block.len() as f64" crates/cb-backend/src/kernels/mvs_device_test.rs`
   returns nothing.

## Specification references

- `MVS-S6` — the `cb-backend` inline CPU transcription stays consistent.
  Principal failure reason: *`MVS-S4`/`MVS-S5` change the CPU sampler's numerics
  while `cb-backend`'s deliberately duplicated CPU reference still models the old
  ones, so the device MVS self-oracle diverges — and only on rocm/cuda, invisibly to
  the default CPU CI.*

## Prerequisites and blocking

- Prerequisites: **TASK-04** and **TASK-05** — this task mirrors both.
- Blocks TASK-08 (the rocm sign-off must run against the mirrored reference) and
  TASK-09.
- **Not parallelisable.** It is the only writer of `mvs_device_test.rs`, but it must
  follow both source changes or the mirror would be written twice.

## Context and evidence

### Why these are copies and not imports

`crates/cb-backend/src/kernels/mvs_device_test.rs:1-24` states it explicitly: the CPU
reference transcribes `cb-train/src/bootstrap.rs`'s `single_probability` /
`calculate_threshold` / `mvs_sample_weights` **INLINE**, with "NO `cb-train` dep even
in the test — the feature-unification landmine", held against the independently
validated `cb_core` RNG + `sum_f64` so the oracle is non-tautological
`[VERIFIED: read]`. `cb-backend` must never depend on `cb-train`. The duplication is
therefore deliberate and permanent; `MVS-S6`'s non-goal is explicit that eliminating
it is out of scope.

### The two lines to mirror — verbatim at HEAD `[VERIFIED: read]`

```
130	fn cpu_block_threshold(block: &[f64], lambda: f64, sample_rate: f64) -> f64 {
131	    let mut candidates: Vec<f64> = block.iter().map(|&d| (lambda + d * d).sqrt()).collect();
132	    calculate_threshold(&mut candidates, 0.0, 0.0, sample_rate * block.len() as f64)
133	}
```

```
157	        for (offset, &der) in block.iter().enumerate() {
158	            let grad2 = der * der;
159	            let probability = single_probability((grad2 + lambda).sqrt(), threshold);
160	            let idx = begin + offset;
161	            if probability > f64::EPSILON {
162	                let weight = 1.0 / probability;
163	                let r = block_rng.gen_rand_real1();
164	                if let Some(slot) = weights.get_mut(idx) {
165	                    *slot = weight * f64::from(r < probability);
166	                }
167	            } else if let Some(slot) = weights.get_mut(idx) {
168	                *slot = 0.0;
169	            }
170	        }
```

`:132` is the `MVS-S4` mirror (note it uses `block.len()`, not a `block_size`
binding); `:165` is the `MVS-S5` mirror. Everything else in the file —
`single_probability` (`:42-50`), `calculate_threshold` (`:55-115`),
`mvs_lambda_iter0` (`:119-127`), the per-block reseed in `cpu_mvs_sample`
(`:150-156`), the conditional draw (`:161`, `:163`) — already matches the CPU and
must stay byte-identical.

### The three tests this protects `[VERIFIED: read mvs_device_test.rs:197-300]`

| test | line | what it asserts |
|---|---|---|
| `mvs_weights_match_frozen_cpu_sample_within_epsilon` | `:197-234` | `max_divergence(device, cpu) ≤ TOL` (`1e-4`, `:27`) over `(seed, rate, n)` ∈ `{(17,0.5,48),(42,0.7,64),(2024,0.3,200)}`, **plus** an exact `assert_eq!(kept_dev, kept_cpu)` keep-count equality (`:224-227`) |
| `mvs_per_block_threshold_matches_and_reweight_is_consistent` | `:236-274` | `cpu_block_threshold` finite/positive; `max_div ≤ TOL`; for each kept un-capped object `|w − threshold/cand| ≤ max(TOL, |expect|·1e-6)` |
| `mvs_multi_block_reseed_is_deterministic_and_finite` | `:276-300` | determinism (`assert_eq!(a, b)`) and finiteness over `n = MVS_BLOCK_SIZE + 24 = 8216` |

All three early-return with an `eprintln!` when
`device_backend_active()` (`:33-35`) is false, i.e. off rocm/cuda — so a `cpu` run
proves nothing here and **this task genuinely requires the GPU**.

### Why a device-only criterion is NOT sufficient (MAJOR-3 — measured)

`[VERIFIED: RUN — numpy f32-vs-f64 target table over every call site]`

| `(rate, n)` | site | f32 target | f64 target | rel. delta |
|---|---|---|---|---|
| `(0.5, 48)` | `:204` | `24.0` | `24.0` | **0 — bit-identical** |
| `(0.7, 64)` | `:204` | `44.79999923706055` | same | **0** |
| `(0.3, 200)` | `:204` | `60.000003814697266` | `60.00000238418579` | **2.384e-8** (1.431e-6 abs) |
| `(0.6, 96)` | `:248` | `57.60000228881836` | same | **0** |
| `(0.5, 8192)` / `(0.5, 24)` | `:284` | `4096.0` / `12.0` | same | **0** |

So the `MVS-S4` half of the mirror is **undetectable in 4 of the 5 call sites**, and in the
fifth it perturbs weights ~2.4e-8 relative against a `TOL = 1e-4` bar — four orders of
slack. The `MVS-S5` half is ~6e-8 relative everywhere, likewise invisible. **After a
device-only mirror, no test anywhere in the repo would constrain the two transcriptions to
agree**, and the next phase that edits `mvs_block_sample_size` or the weight store (e.g.
adding `mvs_reg`, or the Design B′ wiring) would silently re-open risk `R2` with every CPU
and rocm test still green. That is why step 2 below adds a real test rather than a grep.

### The residual this task deliberately leaves (planner finding `P1`)

The **kernel** `mvs_sample_kernel` computes its target in `f64`:

```
crates/cb-backend/src/kernels/mvs_device.rs:145-146
            // sample_size = sample_rate · blockSize (rate is f32-rounded on the host).
            let sample_size = rate * f64::cast_from(u64::cast_from(bs));
```

`[VERIFIED: read]` — and `launch_mvs_weights_resident` already f32-rounds only the
RATE on the host (`mvs_device.rs:302`). `MVS-S6`'s non-goal forbids changing
`mvs_sample_kernel`, so after this task the test's CPU reference uses an `f32` target
while the kernel bisects against an `f64` one. Consequences, sized:

- the threshold root shifts by ~1.5e-8 **relative**, so each `p = cand/μ` and each
  weight `1/p` shifts by ~1.5e-8 relative — four orders inside `TOL = 1e-4`;
- the only sharp edge is the EXACT `assert_eq!(kept_dev, kept_cpu)` at `:224-227`: a
  keep flip needs some object's pinned `NextUniformF` draw to fall inside a ~1.5e-8
  window around `p`. Across the covered `n` values (48, 64, 200, 8216) that is a
  ~1e-4-scale coincidence, not an expectation;
- blast radius if it ever flips: **the self-oracle only**.
  `launch_mvs_weights_resident` / `mvs_sample_kernel` are dead on the live path —
  `cb-train` never sets `config.mvs_lambda`, so `MvsState` is never built
  (`gpu_runtime/session.rs` builds it only when `config.mvs_lambda.is_some()`)
  `[VERIFIED: research.md §6.2, §7 R7]`. No real fit is affected.

Escalation rule (below) covers the flip case.

### Build-line convention

Device runs are ALWAYS `--no-default-features --features rocm`. `cb-backend --lib`
under rocm DOES build as a whole (unlike a blanket `cb-train` test build)
`[PROJECT: .planning/plans/device-bootstrap-parity/plan3.md:167; research.md §8.8]`.
Local rig: gfx1151 / AMD Radeon 860M `[VERIFIED: RUN rocminfo]`.

## Files

- Modify: `crates/cb-backend/src/kernels/mvs_device_test.rs` — two expressions
  (`:132`, `:165`) plus their doc comments, **plus ONE new non-device-gated test**
  (MAJOR-3).
- Modify (**comment lines ONLY**): `crates/cb-backend/src/kernels/mvs_device.rs:145-146`
  — the doc-only deviation note required by MINOR-1. The gate for this file changes from
  "byte-unchanged" to "**no executable change**", proven by a `git diff` containing only
  comment lines.
- Do NOT touch: `mvs_sample_kernel`'s body or `launch_mvs_weights_resident` (executable
  code), `TOL` (`:27`), `calculate_threshold` / `single_probability` / `mvs_lambda_iter0` /
  `make_derivatives` in the same test file, `crates/cb-backend/src/kernels.rs`'s mount
  (`:2955-2963`), or any `#[cube]` body anywhere.

## TDD sequence

### 1. Red — a REAL failing test, no GPU needed (revised, MAJOR-3)

v1 made the Red a `grep`, conceding that "a silent pass with a drifted reference" was an
acceptable outcome. It is not: with 4 of 5 call sites bit-identical (table above), a grep
is the *only* thing that would have failed, and greps do not run in CI. Write the test
first, watch it fail, then mirror.

#### 1a. MANDATORY: create the target seam FIRST, behaviour-identical (C2-4)

v2 offered this only parenthetically ("or by calling `cpu_block_threshold`'s target
sub-expression directly **if it is factored out**"), without saying who factors it out or
when. That is not enough: at HEAD `cpu_block_threshold` (`mvs_device_test.rs:130-133`)
computes the target **inline** and returns a *threshold*, so there is no observable target
seam and the threshold at `(0.8, 1500)` is not a pinnable constant. The natural
implementation therefore writes a fresh helper already carrying the **new** `f32`
expression, assertion 1 passes on its first run, the pinned Red text is unreachable, and
the `MVS-S4` half of the mirror ships with a test that could never have failed — reopening
`R2` on exactly the half where 4 of 5 device call sites are bit-identical.

Mirror `plan4.md`'s step 1a discipline exactly:

1. Extract the target sub-expression out of `cpu_block_threshold` into a named test-local
   helper, e.g. `fn cpu_block_sample_size(sample_rate: f64, block_len: usize) -> f64`,
   whose body is **the CURRENT in-situ expression**
   `f64::from(sample_rate as f32) * block_len as f64`, and call it from
   `cpu_block_threshold`. **NOT** the bare `sample_rate * block_len as f64`: see the
   CONSTANT TRAP in `plan4.md` — `mvs_device_test.rs:145` narrows the rate before the
   multiply exactly as `bootstrap.rs:294` does, and a bare body makes assertion 1 pass on
   its first run (`0.8_f64 * 1500.0 == 1200.0` exactly), which is precisely the vacuous
   Red this task exists to prevent.
2. Prove it inert: run
   `cargo test -p cb-backend --lib mvs` (default `cpu`, the three device oracles compile
   and skip) and, on the ROCm rig,
   `cargo test -p cb-backend --no-default-features --features rocm --lib mvs -- --nocapture --test-threads 1`
   → all three device oracles green with **unchanged** printed `max_div` /
   `kept dev/cpu` / `cpu_threshold`. If anything moves, the extraction was not
   behaviour-preserving — fix that before writing the test.
3. **Only then** write the Red (step 1b). **If assertion 1 passes on its first run,
   step 1a was authored wrongly** — either with the post-fix expression, or with the bare
   `sample_rate * block_len as f64` (the `[C3-1]` trap, which yields exactly `1200.0`).
   Restore the in-situ form `f64::from(sample_rate as f32) * block_len as f64`, confirm it
   reproduces `1200.0000178813934`, and re-run; do not proceed with a green assertion 1.

#### 1b. The failing test

Add to `crates/cb-backend/src/kernels/mvs_device_test.rs` (which already carries
`use cb_core::{sum_f64, TFastRng64};` at `:22` and the `MVS_BLOCK_SIZE` const at `:30`):

- **Test name:** `cpu_reference_mirrors_cb_train_mvs_transcription`
- **NOT gated on `device_backend_active()`** — it is pure host `f64`/`f32` arithmetic and
  MUST run under the default `cpu` feature. Do not add the `:199-202` early-return guard.
  Add a short doc comment saying exactly that, and why (the `MVS-S6` "invisible on default
  CI" hole).
- **Assertions, in this order:**
  1. **The mirrored target expression**, calling the step-1a helper by name (NOT a fresh
     inline expression — that is what makes the Red real):
     `assert_eq!(cpu_block_sample_size(0.8, 1500), 1200.0);`
     `assert_eq!(cpu_block_sample_size(0.8, 8192), 6553.60009765625);`
     `assert_eq!(cpu_block_sample_size(0.8, 3616), 2892.800048828125);`
     — exact equality, not an epsilon. These are the identical pins as
     `crates/cb-train/src/bootstrap_test.rs`'s
     `mvs_block_sample_size_reproduces_upstream_float_expression`, which is what makes the
     two transcriptions provably the same function.
  2. **The mirrored weight store**: run `cpu_mvs_sample(seed, &der, lambda, 0.5)` on
     `make_derivatives(200)` with `mvs_lambda_iter0(&der)`, then assert (a) anti-vacuous —
     the returned vector has both zero and nonzero entries — and (b) every weight
     satisfies `w == f64::from(w as f32)`.
- **Expected initial failure** (assertion 1, before the mirror):

  ```
  assertion `left == right` failed
    left: 1200.0000178813934
   right: 1200.0
  ```

  If instead assertion 2 fails first, the assertion order above was not followed. **If
  assertion 1 PASSES, step 1a was skipped** — see 1a.3.
- Run: `cargo test -p cb-backend --lib cpu_reference_mirrors` **(default `cpu` feature —
  no GPU)**. The mount is unconditional (`crates/cb-backend/src/kernels.rs:2962-2963`
  `#[cfg(test)] mod mvs_device_test;`, no feature predicate) `[VERIFIED: grep]`, so this
  test really does run under `--features cpu`.

Then, on the ROCm rig, capture the pre-mirror device baseline for comparison (this is
evidence, not the Red):

- Run: `cargo test -p cb-backend --no-default-features --features rocm --lib mvs -- --nocapture --test-threads 1`
- Record the printed `max_div` / `kept dev=… cpu=…` / `cpu_threshold=…` lines from
  `:215-218` and `:255`. Expect the three device oracles to **PASS** here — pre-mirror the
  reference and the kernel agree to bisection precision (`MVS_BISECTION_ITERS = 100`,
  `mvs_device.rs:67`), so `max_div` should be ~1e-13. That agreement is exactly what the
  mirror will (correctly) break; see the Verify step.

### 2. Green

Two minimal value edits (the seam already exists from step 1a):

1. Change the **step-1a helper's body** — `cpu_block_sample_size` — from
   `sample_rate * block_len as f64` to
   `f64::from((sample_rate as f32) * block_len as f32)`, the same expression TASK-04
   installed in `bootstrap.rs`. `cpu_block_threshold` (`:130-133`) keeps calling the
   helper, so its `:132` call site no longer carries the arithmetic itself. Add to
   `cpu_block_threshold`'s / the helper's doc (`:128-129`) that this transcribes upstream's
   **`float`** `SampleRate * blockSize` (`mvs.cpp:202`, `mvs.h:47`), that
   `block.len() ≤ MVS_BLOCK_SIZE = 8192 < 2^24` makes the cast exact, and that it is
   a MIRROR of `cb-train/src/bootstrap.rs`'s `mvs_block_sample_size` which must be kept in
   sync (naming the reason there is no import: the `cb-backend → cb-train` dependency ban).
2. `:165` → `*slot = f64::from((weight * f64::from(r < probability)) as f32);`
   — the same store TASK-05 installed. Add to `cpu_mvs_sample`'s doc (`:135-137`) that
   upstream's `SampleWeights` is a `TVector<float>` at
   **`catboost/private/libs/algo/fold.h:217`** (`SPEC.md` §10.1 — there is no
   `algo_helpers/fold.h`) and that this mirrors the CPU narrowing.
3. Add ONE sentence to the file's module doc (`:9-14`, which already explains the
   deliberate duplication) recording the `P1` residual: the KERNEL keeps an `f64`
   target (`mvs_device.rs:146`) by design (`MVS-S6` non-goal), so a ~2.4e-8-relative
   threshold difference between this reference and the kernel is expected on the
   `(rate 0.3, n 200)` fixture (zero on the other four) and is absorbed by `TOL = 1e-4`.
4. **Doc-only note in the KERNEL file (MINOR-1 — required, not optional).** Add a comment
   at `crates/cb-backend/src/kernels/mvs_device.rs:145-146`, immediately above
   `let sample_size = rate * f64::cast_from(u64::cast_from(bs));`, recording that this
   `f64` target is a **known deviation** from upstream's `float SampleRate * ui32 blockSize`
   (`mvs.h:47`, `mvs.cpp:202`); that the host CPU sampler and this file's sibling test
   reference both use the `f32` expression
   (`cb-train/src/bootstrap.rs`'s `mvs_block_sample_size`); that the difference is
   ≤2.4e-8 relative on the covered shapes and inert while the kernel is unreachable from
   `cb-train` (`config.mvs_lambda` is never `Some`); and that **a Design B′ wiring of
   `launch_mvs_weights_resident` onto the live path must revisit it**. A future implementer
   reads the kernel, not the sibling test file — which is why v1's test-file-only note was
   insufficient. **Comment lines only**: no executable change.

Do NOT change `TOL`. Do NOT change any executable line in the kernel. Do NOT convert the
copies into imports.

- Run: `cargo test -p cb-backend --lib cpu_reference_mirrors` (default `cpu`) → now passes
- Run: `cargo test -p cb-backend --no-default-features --features rocm --lib mvs -- --nocapture --test-threads 1`

### 3. Refactor

- None beyond doc wording. Resist any urge to factor the two transcriptions into a
  shared helper crate: the duplication is a deliberate architectural choice
  (`MVS-S6` non-goal) and a shared crate would re-introduce the feature-unification
  hazard the file's header warns about.
- **Clippy — ERROR-attributed, differential, `--keep-going` mandatory** (CRITICAL-1 /
  C2-1). The default-`cpu` `--lib` baseline is exactly **4 errors across 3 files**
  (`cpu_runtime.rs` ×2 at `:696:13`/`:1025:29`, `kernels/bootstrap_device.rs:230:28`,
  `kernels/exact_quantile.rs:178:8`) `[VERIFIED: RUN twice, identical]`, plus 2 in the
  lib-test target (`kernels/gradient.rs:18`, `kernels/score_split.rs:374`).

  **A `-->`-line grep would be RED here before any work**: `mvs_device.rs:80` carries a
  pre-existing `clippy::manual_rotate` **warning** `[VERIFIED: RUN]`. Use the
  severity-filtered helper from `PLAN.md` §4.12:

  ```bash
  clippy_error_files -p cb-backend --lib | grep "mvs_device"                       # must be EMPTY
  clippy_error_files -p cb-backend --no-default-features --features rocm --all-targets \
    | grep "mvs_device"                                                            # must be EMPTY
  ```

  The first was measured EMPTY at HEAD `[VERIFIED: RUN]`. Under rocm the overall set may
  differ from the `cpu` baseline; record whatever it is as this feature combination's
  baseline and compare before/after **within this task**. Never assert "clippy clean".
  Note the phase's comment-only edit to `mvs_device.rs` cannot change its warning status.
- Run: `bash scripts/check-source-test-separation.sh` → exit 0 (ABSOLUTE gate).
- Run: `bash scripts/check-no-raw-float-sum.sh 2>&1 | grep mvs_device` → EMPTY. Do NOT
  require the script itself to pass: it exits 1 at HEAD (15 files / 36 lines)
  `[VERIFIED: RUN]`. This file is a `*_test.rs` and is exempt from the ban by the script's
  own exclusion list, so it should never appear.

### 4. Verify

- Run: `cargo test -p cb-backend --lib cpu_reference_mirrors` (default `cpu` feature) →
  passes. **This is `AC-6`'s new, GPU-free half** and the only permanent protection the
  mirror has.
- Run: `cargo test -p cb-backend --no-default-features --features rocm --lib mvs -- --nocapture --test-threads 1`
  → 4 tests passed (3 device oracles + the new host test).

  **`max_div` will RISE, and that is CORRECT (revised, MAJOR-4).** v1 said it "should
  DROP", which is backwards and contradicted this file's own `P1` analysis. The mirror
  moves the CPU **reference** *away* from the kernel, not toward it: the kernel keeps its
  `f64` target (`mvs_device.rs:146`) and stores un-narrowed `f64` weights, while the
  reference gains an `f32` target and an `f32`-narrowed store. Pre-mirror the two agreed to
  bisection precision (`MVS_BISECTION_ITERS = 100` resolves the root to full `f64`), so
  `max_div ≈ 1e-13`; post-mirror it is floored at roughly `|w| · 6e-8 ≈ 1e-7`.

  Expect **~1e-13 → ~1e-7, a rise of ~6 orders, still ≥3 orders under `TOL = 1e-4`.**
  The **load-bearing assertions are `kept_dev == kept_cpu` (exact) and
  `max_div ≤ 1e-4`** — nothing else. A rise is pre-authorised; do NOT read it as a broken
  mirror, and above all do NOT "fix" it by narrowing `mvs_sample_kernel`'s target, which
  `MVS-S6`'s non-goal forbids and which would require a CubeCL kernel change this phase has
  no mandate for.
- Run: `cargo test -p cb-backend --no-default-features --features rocm --lib -- --test-threads 1`
  → the whole `cb-backend` lib under rocm; record pass/fail counts and confirm no
  NEW failure relative to a pre-task run of the same command.
- Run: `cargo test -p cb-backend --lib mvs` (default `cpu` feature) → **4 tests**: the
  three device oracles COMPILE and take the `device_backend_active() == false` early
  return printing their `skipped` messages, and `cpu_reference_mirrors_cb_train_mvs_transcription`
  **actually runs and passes**. If the new test also skips, it was wrongly gated — remove
  the guard.
- Run: `git diff --stat crates/cb-backend/` → only
  `src/kernels/mvs_device_test.rs` and `src/kernels/mvs_device.rs` changed.
- Run: `git diff crates/cb-backend/src/kernels/mvs_device.rs` → **comment lines only**
  (MINOR-1). Verify with
  `git diff -U0 crates/cb-backend/src/kernels/mvs_device.rs | grep -E "^[+-]" | grep -v "^[+-][+-]" | grep -vE "^[+-]\s*(//|///|//!)"`
  → must be EMPTY, i.e. every added/removed line is a comment.
- Run: `grep -n "sample_rate \* block.len() as f64" crates/cb-backend/src/kernels/mvs_device_test.rs`
  → no output.
- Run: `grep -n "as f32" crates/cb-backend/src/kernels/mvs_device_test.rs` → shows
  both new narrowings.
- Confirm: `diff <(sed -n '130,133p' crates/cb-backend/src/kernels/mvs_device_test.rs) …`
  — do a manual side-by-side of the two mirrored expressions against
  `crates/cb-train/src/bootstrap.rs`'s versions and state in `progress.md` that they
  agree.

## Completion criteria

- [ ] **Step 1a was performed FIRST** (C2-4): `cpu_block_sample_size` was extracted with
      the **current `f64`** body, proven inert on both the default-`cpu` and the ROCm runs
      (unchanged `max_div` / `kept dev/cpu` / `cpu_threshold`), and only then was the test
      written. A first-run PASS of assertion 1 means 1a was skipped and must be redone.
- [ ] **`cpu_reference_mirrors_cb_train_mvs_transcription` exists, is NOT gated on
      `device_backend_active()`, calls the step-1a helper by name, FAILED before the mirror
      with `left: 1200.0000178813934, right: 1200.0`, and passes after** — on the default
      `cpu` feature (MAJOR-3). This is the criterion v1 lacked entirely.
- [ ] Its three pinned target values are byte-identical to TASK-04's unit test's, so the
      two transcriptions are provably the same function.
- [ ] The pre-mirror rocm device baseline was captured (`max_div ≈ 1e-13` expected).
- [ ] `cpu_block_threshold` uses `f64::from((sample_rate as f32) * block.len() as f32)`.
- [ ] `cpu_mvs_sample` stores `f64::from((weight * f64::from(r < probability)) as f32)`.
- [ ] Both mirrored expressions are textually equivalent to the `cb-train` versions
      (manually confirmed and recorded).
- [ ] The `P1` residual is documented in BOTH places: the test file's module doc AND —
      **comment-only** — at `mvs_device.rs:145-146` (MINOR-1).
- [ ] `TOL` is still `1e-4`; `mvs_device.rs` has **no executable change** (the
      comment-only `git diff` check passes); no `#[cube]` body touched.
- [ ] rocm `--lib mvs` **4/4** green with `kept_dev == kept_cpu` exactly, and `max_div`
      recorded as having RISEN to ~1e-7 (expected, MAJOR-4) while staying ≤1e-4; rocm
      `--lib` shows no new failure; the default-`cpu` run shows 3 skips + 1 real pass.
- [ ] The **error-attributed** rocm clippy check for `mvs_device` is EMPTY (not "clippy
      clean", and not a `-->` grep — `mvs_device.rs:80` warns pre-existingly).

## Completion evidence to record in `progress.md`

- The new test's Red failure text and its post-mirror pass.
- Pre- and post-mirror `max_div` (expect ~1e-13 → ~1e-7), `kept dev/cpu`,
  `cpu_threshold` values.
- The rocm `--lib` pass/fail tally before and after.
- The rocm clippy error set (this feature combination's baseline) and the diff-scoped grep
  result.
- The comment-only `git diff` proof for `mvs_device.rs`.

## Risks and guardrails

- **`P1` — the keep-count flip.** If `assert_eq!(kept_dev, kept_cpu)` fails after the
  mirror: do NOT loosen `TOL`, do NOT delete the assertion, and do NOT edit
  `mvs_sample_kernel` (SPEC `MVS-S6` non-goal). **STOP and escalate** with (i) the
  failing `(seed, rate, n)`, (ii) the object index that flipped, (iii) its `p` and its
  drawn `r`. The decision "narrow the kernel's target too" is a scope change requiring
  the user, and the blast radius is the self-oracle only because
  `launch_mvs_weights_resident` is dead on the live path.
- **SPEC `R3`/`R2` — the whole reason this task exists.** Editing the CPU sampler
  without mirroring here fails ONLY on rocm/cuda, invisibly to the default CI. Guard:
  this task is a blocking prerequisite of TASK-08 and TASK-09.
- **CubeCL.** No kernel change is needed or permitted. If a CubeCL build error
  appears anyway, STOP and read
  `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/cubecl_error_guideline.md`
  BEFORE attempting any fix (`CLAUDE.md`). Blind fixes are prohibited.
- **Wrong build line.** Bare `--features rocm` unifies `cpu` in and will produce
  misleading results. Always `--no-default-features --features rocm`.
- **A `cpu`-feature run is not evidence *for the device oracles*.** The three
  pre-existing tests early-return off rocm/cuda, so their green on the default feature
  proves only that the file compiles. **The new mirror test is the exception** — it must
  really run there, and gating it would reintroduce exactly the MAJOR-3 hole.
- **Misreading the `max_div` rise (MAJOR-4).** `max_div` going from ~1e-13 to ~1e-7 is the
  CORRECT, pre-authorised outcome of moving the reference away from the kernel. v1's plan
  said it "should DROP", which could have driven an implementer to revert the mirror or to
  narrow `mvs_sample_kernel`'s target — the one edit `MVS-S6` forbids. Only
  `kept_dev == kept_cpu` and `max_div ≤ 1e-4` are load-bearing.
