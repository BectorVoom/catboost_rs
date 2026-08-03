---
plan: 5
task_id: TASK-05
phase: mvs-tree2-parity
status: pending
order: 5
wave: serial (v2 — the phase is fully serial)
hardware: none (CPU only)
depends_on: [TASK-04]
blocks: [TASK-06, TASK-07]
specifications: [MVS-S5]
parallelizable: false
revision_note: >
  v2: the upstream citation this task writes into production is corrected —
  `catboost/private/libs/algo/fold.h:217` (v1 said `algo_helpers/fold.h`, which does not
  exist) and the read-back is at `tensor_search_helpers.cpp:456`, not `:457` (MINOR-5);
  the "clippy clean" / "gate scripts pass" criteria become differential (CRITICAL-1,
  CRITICAL-2).
---

# Task 5: Stored MVS sample weights are narrowed through `f32`

## Objective

After this task every weight the MVS arm stores into
`BootstrapResult.sample_weights` is a value representable in `f32` — mirroring
upstream's `TVector<float> SampleWeights` (`fold.h:217`) — while the container type
stays `Vec<f64>` so no caller, re-export or Python parameter changes.

**Observable completion condition:** for an MVS `bootstrap` call, every returned
weight `w` satisfies `w == f64::from(w as f32)`, and the zero-probability arm still
stores exactly `0.0`.

## Specification references

- `MVS-S5` — stored MVS sample weights are narrowed through `f32`.
  Principal failure reason: *the stored weight keeps `f64` precision where upstream
  stores `TVector<float>`.*

## Prerequisites and blocking

- Prerequisite: **TASK-04** — same file and same function
  (`crates/cb-train/src/bootstrap.rs`, `mvs_sample_weights`); serialised to keep the
  two f32 transcription changes independently attributable.
- Blocks TASK-06 (the `cb-backend` mirror needs both changes) and TASK-07 (the
  documentation should describe the final numerics).
- **Not parallelisable** with TASK-04 or TASK-06.

## Context and evidence

- **The site, verbatim at HEAD** `[VERIFIED: read crates/cb-train/src/bootstrap.rs:315-328]`:

  ```
  315	        for (offset, &der) in block.iter().enumerate() {
  316	            let grad2 = der * der;
  317	            let probability = single_probability((grad2 + lambda).sqrt(), threshold);
  318	            let idx = begin + offset;
  319	            if probability > f64::EPSILON {
  320	                let weight = 1.0 / probability;
  321	                let r = block_rng.gen_rand_real1();
  322	                if let Some(slot) = weights.get_mut(idx) {
  323	                    *slot = weight * f64::from(r < probability);
  324	                }
  325	            } else if let Some(slot) = weights.get_mut(idx) {
  326	                *slot = 0.0;
  327	            }
  328	        }
  ```

  Line `:323` is the store to change. Lines `:319` and `:321` — the CONDITIONAL draw
  (`p ≤ ε` consumes NO draw, matching `mvs.cpp:210-212`) — must stay exactly as they
  are: they are the per-block stream-phase contract
  `[VERIFIED: research.md §1.4, §2.4]`.
- **Upstream — CORRECTED PATHS (MINOR-5; these go into a production doc comment, so they
  must be right).** `fold->SampleWeights` is a `TVector<float>` at
  **`catboost/private/libs/algo/fold.h:217`** — v1 (and `SPEC.md` §10 before its v2
  revision) said `algo_helpers/fold.h`, and **that file does not exist**
  `[VERIFIED: RUN — ls on both paths; line 217 is exactly
  `TVector<float> SampleWeights; // Resulting bootstrapped weights of documents.`]`.
  `mvs.cpp:213` narrows `weight * (r < probability)` into it, and `CalcWeightedData`
  reads it back through a `const float*` at **`tensor_search_helpers.cpp:456`** (not
  `:457`), using it at `:462` and `:472` `[VERIFIED: RUN — read `:442-485`]`. Our
  container is `Vec<f64>` (`bootstrap.rs:90`), so the VALUE must be narrowed even though
  the storage is not. Use `SPEC.md` §10.1's verified citation set verbatim.
- **The target expression** (SPEC §4):
  `*slot = f64::from(((weight * f64::from(r < probability)) as f32));`
  Note the narrowing is applied to the PRODUCT, so the dropped-object case
  (`r >= probability` ⇒ factor `0.0`) still stores exactly `0.0`
  (`0.0_f32` round-trips exactly).
- **Magnitude of the change:** ~6e-8 relative per non-unit weight
  `[VERIFIED: research.md §2.6 N2]`.
- **Spiked safe.** `A + B + C` (TASK-01's deletion + TASK-04's target + this
  narrowing): identical results to `A` alone — **every printed residual unchanged**,
  all 5 `bootstrap_oracle_*` ok, `bootstrap_dev` ok at 3 trees, the multi-seed probe
  5/5 and 5/5 `[VERIFIED: research.md §8.6]`. This confirms MVS weights influence the
  model only through the discrete split argmax.
- **The `control` mask is derived from the stored weight** at `bootstrap.rs:426-429`
  (`w > f64::from(f32::EPSILON)`), which is upstream's `SetControlNoZeroWeighted`
  (`calc_score_cache.cpp:1196-1203`) `[VERIFIED: read]`. Narrowing a weight cannot
  cross that threshold in a meaningful way (the nonzero weights are `1/p ≥ 1`), and
  `MVS-S5`'s postcondition requires the mask to remain correct — assert it.
- **`BootstrapResult` and `bootstrap`'s signature do not change** (SPEC §4). The
  struct is re-exported from `crates/cb-train/src/lib.rs`; `EBootstrapType` is parsed
  from Python at `crates/catboost-rs-py/src/params.rs:395`
  `[VERIFIED: research.md §6.1]`. Nothing here touches either.
- **The mirror obligation.** `crates/cb-backend/src/kernels/mvs_device_test.rs:157-170`
  carries the SAME full-`f64` store (`:165`) `[VERIFIED: read]`. TASK-06 mirrors it —
  do NOT edit `cb-backend` here.
- **The existing unit test that must keep passing**:
  `mvs_full_subsample_is_identity_and_real_subsample_is_importance_weighted`
  (`bootstrap_test.rs:157-181`) asserts, for the nonzero weights,
  `w >= 1.0 - 1e-9` and `c == (w > f64::from(f32::EPSILON))`
  `[VERIFIED: read]`. Both survive an f32 narrowing (its `1e-9` slack is 10× the
  ~6e-8 relative shift only for `w ≈ 1`, so re-check this test's output rather than
  assuming).

## Files

- Modify: `crates/cb-train/src/bootstrap.rs` — the store at `:323` only.
- Modify: `crates/cb-train/src/bootstrap_test.rs` — add ONE test.
- Do NOT touch: `:319`/`:321` (the conditional draw), `:325-326` (the zero arm),
  `:426-429` (the control mask), `single_probability`, `calculate_threshold`,
  `BootstrapResult`, `crates/cb-backend/**` (TASK-06), the module docs (TASK-07).

## TDD sequence

### 1. Red

In `crates/cb-train/src/bootstrap_test.rs`:

- **Test name:** `mvs_sample_weights_are_f32_representable`
- **Setup:** `let n = 2000;` (single MVS block); the varied derivative pattern
  `(0..n).map(|i| (i as f64 % 13.0) - 6.0)` already used at `:160`;
  `let mut rng = TFastRng64::from_seed(0);`
- **Input:** `bootstrap(EBootstrapType::Mvs, &ders, 0.5, 0.0, None, &mut rng)`
  (`subsample = 0.5` so a healthy mix of kept and dropped objects appears — the
  existing test at `:169` uses the same rate).
- **Assertions, in this order:**
  1. **anti-vacuous first**: count `nonzero = weights.iter().filter(|&&w| w > 0.0).count()`
     and assert `nonzero > 0 && nonzero < n`. A test over an all-zero (or all-`1.0`)
     weight vector would pass trivially, so this leg must come first and must be
     part of the same test.
  2. the round-trip: for every `w` in `res.sample_weights`,
     `assert_eq!(w, f64::from(w as f32), "…");`
  3. the zero arm: assert that at least one weight is exactly `0.0` and that
     `0.0_f64.to_bits() == weights[that idx].to_bits()` (an exact zero, not a
     denormal).
  4. the mask invariant (`MVS-S5` postcondition): for each pair,
     `assert_eq!(c, w > f64::from(f32::EPSILON));`
- **Expected initial failure** (assertion 2), e.g.:

  ```
  assertion `left == right` failed: MVS weight must round-trip through f32 (upstream stores TVector<float>)
    left: 1.0938247319843172
   right: 1.0938247
  ```

  The exact digits depend on the derivative pattern and seed — the load-bearing part
  is that `left` carries more than `f32` precision. Capture the ACTUAL first failing
  pair and record it.
- If assertion 2 passes before the production change, the chosen inputs happened to
  produce only `f32`-exact weights (e.g. every `p == 1.0` ⇒ `w == 1.0`). Then
  assertion 1 has not done its job: widen the derivative spread or lower `subsample`
  until some `p < 1` object is KEPT, and re-run. Do not weaken the assertion.
- Run: `cargo test -p cb-train --lib mvs_sample_weights_are_f32`

### 2. Green

Replace the store at `:323` with the narrowed form:

```
*slot = f64::from((weight * f64::from(r < probability)) as f32);
```

Document, at the store or on `mvs_sample_weights`:

- upstream stores into `TVector<float> SampleWeights`
  (**`catboost/private/libs/algo/fold.h:217`**), narrowed at `mvs.cpp:213` and read back
  as `const float*` (`tensor_search_helpers.cpp:456`, consumed at `:462`/`:472`);
- our container stays `Vec<f64>` deliberately (`BootstrapResult.sample_weights`,
  `bootstrap.rs:90`) — only the VALUE is narrowed, so no caller, re-export or Python
  parameter changes;
- the narrowing wraps the PRODUCT so the dropped case still stores exactly `0.0`;
- the measured effect is ~6e-8 relative and provably unobservable through the model
  (MVS weights reach it only through the discrete split argmax) — recorded so a
  future reader does not mistake it for a live numeric fix.

Do NOT narrow the Bayesian or Bernoulli arms' weights (`MVS-S5` non-goal). Do NOT
change `BootstrapResult`'s type.

- Run: `cargo test -p cb-train --lib bootstrap`
- Run: `cargo test -p cb-train --test bootstrap_oracle_test` (5 frozen)

### 3. Refactor

- If the expression reads poorly inline, bind it (`let stored = (weight * kept) as f32;`)
  but keep the arithmetic order identical: narrow AFTER the multiply, never before.
- Confirm no `unwrap`/`expect`/`panic`/raw index was introduced; the `if let Some(slot)`
  guards at `:322`/`:325` stay.
- Run: `cargo test -p cb-train --lib bootstrap`
- **Lint gates — DIFFERENTIAL** (CRITICAL-1, CRITICAL-2; `PLAN.md` §4.11/§4.12):

  ```bash
  # ERROR-attributed (C2-1): a `-->` grep also catches the pre-existing
  # `bootstrap.rs:134` excessive_precision WARNING and would be red before any work.
  clippy_error_files -p cb-train --all-targets | grep -E "src/bootstrap\.rs|bootstrap_test\.rs"  # must be EMPTY
  bash scripts/check-source-test-separation.sh                              # ABSOLUTE: exit 0
  bash scripts/check-no-raw-float-sum.sh 2>&1 | grep -E "src/bootstrap\.rs|bootstrap_test\.rs"  # must be EMPTY
  bash scripts/check-no-anyhow.sh        2>&1 | grep -E "src/bootstrap\.rs|bootstrap_test\.rs"  # must be EMPTY
  ```

  All measured EMPTY at HEAD `[VERIFIED: RUN]`. **Doc-text constraint (C2-6):** the store's
  doc comment must not contain the literal `.sum()` or `.fold(0.0`.

  If a `cast_possible_truncation`-style lint fires on the `as f32`, handle it with a
  NARROW documented `#[allow]` on the function — not a crate-level allow, and never by
  widening the cast. Note that the deny list is only
  `unwrap_used`/`expect_used`/`panic`/`indexing_slicing`, so a cast lint is a warning
  unless promoted.

### 4. Verify

- Run: `cargo test -p cb-train --lib bootstrap` → 10 bootstrap unit tests
  (7 baseline + TASK-01 + TASK-04 + this).
- Run: `cargo test -p cb-train --lib mvs_full_subsample` → the pre-existing
  importance-weighting test still passes (its `w >= 1.0 - 1e-9` leg is the one most
  exposed to the narrowing; if it fails, the narrowing was applied to `weight`
  BEFORE the multiply or to `probability`).
- Run: `cargo test -p cb-train --test bootstrap_oracle_test` → 5 passed, values
  unchanged.
- Run: `cargo test -p cb-train --test bootstrap_dev_oracle_test -- --nocapture` →
  green; residuals **byte-identical** to the TASK-04 run (research §8.6 measured
  exactly that). Record them.
- Run: `cargo test -p cb-train --test mvs_seeds_oracle_test -- --nocapture`
  (if TASK-03 landed) → 10/10, residuals unchanged.
- Run: `cargo test -p cb-train --test multidim_sampling_regression_test` → green;
  `multiclass_onevsall_mvs_trains_per_object` (`:133-145`) is the MVS multi-dim smoke
  test with no numeric expectation `[VERIFIED: read]`.
- Run: `git status --short crates/cb-oracle/fixtures/` → EMPTY.
- Confirm: `git diff crates/cb-train/src/bootstrap.rs` touches only the store line
  (+ its doc) relative to TASK-04's state.

## Completion criteria

- [ ] The Red failed on the round-trip assertion with a recorded non-`f32` weight.
- [ ] The anti-vacuous leg proves a mixed kept/dropped weight vector was exercised.
- [ ] Every returned MVS weight satisfies `w == f64::from(w as f32)`.
- [ ] The zero arm stores a bit-exact `0.0`.
- [ ] The `control` mask still equals `w > f64::from(f32::EPSILON)` elementwise.
- [ ] The conditional draw (`if probability > f64::EPSILON` … `gen_rand_real1`) is
      byte-unchanged.
- [ ] `BootstrapResult`'s type and `bootstrap`'s signature are unchanged.
- [ ] `bootstrap_oracle_test` 5/5; `bootstrap_dev` and `mvs_seeds` residuals
      byte-identical to TASK-04's.
- [ ] **Differential lint gates**: the diff-scoped greps are EMPTY and
      `check-source-test-separation.sh` exits 0 (do NOT assert "clippy clean" / "gate
      scripts pass" — `PLAN.md` §4.11); fixtures byte-unchanged; `crates/cb-backend/**`
      untouched.

## Completion evidence to record in `progress.md`

- The Red failure text (the first non-`f32` weight observed).
- Post-change kept/dropped counts and the max `|w − f64::from(w as f32)|` (must be
  exactly 0).
- The `bootstrap_dev` / `mvs_seeds` residuals, confirmed identical to TASK-04.

## Risks and guardrails

- **Narrowing in the wrong place.** `((weight as f32) * kept) as f64` is NOT the same
  as `(weight * kept) as f32 as f64` for edge values, and narrowing `probability`
  would change the DRAW comparison `r < probability` and hence the keep-mask — a real
  behaviour change outside this spec. Guard: the pre-existing
  `mvs_full_subsample…` test plus the byte-identical-residual requirement.
- **`R2` / `MVS-S6` — the unmirrored device copy.** After this task
  `mvs_device_test.rs:165` disagrees with the CPU. Expected; TASK-06 owns it. Do not
  fix it here.
- **Perceived pointlessness.** This change has provably zero measurable effect on any
  current oracle (research §8.6). It is nonetheless a user-approved transcription
  fix; the correct response to "no effect" is to record that in the doc comment
  (done in Green), not to skip the task.
- **SPEC `R6` — an all-zero-gradient block.** `p = 0` ⇒ every object dropped is a
  KNOWN divergence from upstream's `inf`/`NaN` UB and is explicitly out of scope. Do
  not "harmonise" it while editing this loop.
