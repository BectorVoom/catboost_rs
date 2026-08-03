---
plan: 1
task_id: TASK-01
phase: mvs-tree2-parity
status: pending
order: 1
wave: serial (v2 — the phase is fully serial, PLAN-CHECK CRITICAL-3)
hardware: none (CPU only)
depends_on: []
blocks: [TASK-02, TASK-03, TASK-04]
specifications: [MVS-S1]
parallelizable: false
revision_note: >
  v2: the "clippy clean" / "gate scripts pass" criteria are replaced by the differential,
  diff-scoped form (CRITICAL-1, CRITICAL-2) — both were unsatisfiable at HEAD; the
  upstream citation set for the replacement doc comment is pinned to SPEC.md §10.1
  (MINOR-5 — `algo_helpers/fold.h` does not exist).
---

# Task 1: The MVS arm consumes exactly ONE main-stream draw

## Objective

After this task, one `bootstrap(EBootstrapType::Mvs, …, subsample < 1.0, …)` call
advances the persistent training RNG by **exactly one** `gen_rand()` — the
`rand_seed` at `bootstrap.rs:295` — and by nothing else. The two fabricated
"compensation" draws at `crates/cb-train/src/bootstrap.rs:413-423` are gone, and a
unit test pins the contract at a level no oracle can express.

**Observable completion condition:** on a fresh `TFastRng64::from_seed(s)`, after
an MVS `bootstrap` call with `subsample = 0.8`, `rng.call_count() == 1` and
`rng.raw_state()` equals that of a probe built with `TFastRng64::from_seed(s)`
followed by one `gen_rand()`.

This is the whole bug. It is a **deletion** — no algorithmic change, no signature
change.

## Specification references

- `MVS-S1` — an MVS `bootstrap()` call consumes exactly ONE main-stream draw.
  Principal failure reason: *the `Mvs` arm advances the persistent RNG by a number
  of `gen_rand` calls other than one.*

## Prerequisites and blocking

- Prerequisites: none. This is the phase root and the sole Wave-A task.
- Blocks: TASK-02 (its 3/3-tree claim is false without this), TASK-03 (7 of its 10
  scenarios fail without this), TASK-04 (same file, serialised).
- **Not parallelisable**: it is the only writer of `bootstrap.rs` in Wave A and
  every later task's baseline.

## Context and evidence

- **The defect, verbatim at HEAD** `[VERIFIED: read crates/cb-train/src/bootstrap.rs:410-434]`:

  ```
  410	        EBootstrapType::Mvs => {
  411	            let lambda = mvs_lambda(derivatives, prev_leaf_mean_l2);
  412	            let sample_weights = mvs_sample_weights(derivatives, lambda, subsample, rng);
  413	            if subsample < 1.0 {
  414	                // MVS uses `performRandomChoice=false` (calc_score_cache.cpp:752),
  ...     	        //  … "consumes two additional `GenRand()` draws" …
  421	                rng.gen_rand();
  422	                rng.gen_rand();
  423	            }
  424	            // performRandomChoice = false -> control = weight > eps
  426	            let control: Vec<bool> = sample_weights
  427	                .iter()
  428	                .map(|&w| w > f64::from(f32::EPSILON))
  429	                .collect();
  ```
- **Lines 424-429 are CORRECT and must NOT be touched** — `SetControlNoZeroWeighted`
  is `Control[i] = sampleWeights[i] > numeric_limits<float>::epsilon()`
  (`calc_score_cache.cpp:1196-1204`, mask at `:1202`) `[VERIFIED: research.md §1.7; SPEC.md §10.1]`.
- **Lines 413-423 are fabricated.** With `performRandomChoice == false`,
  `TCalcScoreFold::Sample` (`calc_score_cache.cpp:730-748`) takes the `else` branch:
  it sets `BernoulliSampleRate = 0.0f` and calls `SetControlNoZeroWeighted`, and
  **never touches `rand`**. `CalcWeightedData` (`tensor_search_helpers.cpp:442-485`)
  is draw-free `[VERIFIED: research.md §1.7, §2.5]`.
- **The one legitimate draw** is `bootstrap.rs:295 let rand_seed = rng.gen_rand();`
  inside `mvs_sample_weights`, mirroring `mvs.cpp:174
  const ui64 randSeed = rand->GenRand();`. Every other MVS draw is on the per-block
  child stream `TFastRng64::from_seed(rand_seed + block_idx).advance(10)`
  (`bootstrap.rs:299-300`), which never touches the main stream
  `[VERIFIED: read bootstrap.rs:281-331]`.
- **Trace arithmetic pinning the count to 1**: `tree_rng_pre_gts.cc = 2` → level-0
  `cc_start = 7` = 1 bootstrap draw + 4 RSM `GenRandReal1`s
  (`SelectCandidatesAndCleanupStatsFromPrevTree`, `greedy_tensor_search.cpp:329,
  343, 352`, one per candidate sublist even at `Rsm = 1.0`)
  `[VERIFIED: research.md §3, from the committed
  .planning/plans/bayesian-rng-draw-accounting/instrumented-ground-truth/mvs.jsonl]`.
- **The RNG primitives exist and are the right instruments**
  `[VERIFIED: read crates/cb-core/src/rng.rs]`: `from_seed` `:171`, `gen_rand`
  `:183` (increments `call_count`), `advance` `:192`, `call_count` `:204`,
  `raw_state` `:221` (`[r1.x, r1.c, r2.x, r2.c]`, read-only, does NOT advance),
  `gen_rand_real1` `:232` (routes through `gen_rand`, so it also counts).
- **Blast radius (CodeGraph + grep).** `bootstrap()` (`bootstrap.rs:383`, `pub`)
  has exactly three callers: `boosting.rs:3262` (device branch),
  `boosting.rs:3833` (CPU branch), and `bootstrap_test.rs`
  `[VERIFIED: grep "bootstrap(" crates/cb-train/src/boosting.rs → 3262, 3833;
  research.md §6.1]`. Because the device branch calls the SAME function, this fix
  moves device and CPU numbers identically — the desired outcome, re-verified in
  TASK-08.
- **No signature or re-export change.** `bootstrap`, `EBootstrapType`,
  `BootstrapResult`, `BAYESIAN_BLOCK_SIZE`, `MVS_BLOCK_SIZE` are re-exported from
  `crates/cb-train/src/lib.rs`; `EBootstrapType` is parsed from Python at
  `crates/catboost-rs-py/src/params.rs:395` `[VERIFIED: research.md §6.1]`. Nothing
  in this task touches any of them.
- **Test file and mount.** `crates/cb-train/src/bootstrap_test.rs` is mounted from
  `bootstrap.rs:55-57` as `#[cfg(test)] #[path = "bootstrap_test.rs"] mod tests;`
  `[VERIFIED: read]`. It already carries the file-level lint opt-out at `:10` and
  seven tests (`cargo test -p cb-train --lib bootstrap -- --list` → 7 tests, all
  under `bootstrap::tests::`) `[VERIFIED: RUN]`. **There is no MVS draw-count test
  today** — Bayesian has one (`:60-83`), Bernoulli one (`:107-137`), and
  `No`/MVS-`subsample=1` have zero-draw probes (`:141-151`, `:157-166`)
  `[VERIFIED: read]`.
- **Spike proof this is sufficient** `[VERIFIED: research.md §0, §8.2, §8.3]`:
  patching `:413` to `if false {` turned bias=true 3/5 → 5/5, bias=false 0/5 → 5/5,
  kept the frozen `bootstrap/mvs` oracle green, and left `cb-train` at 503/1.
- **Baseline confirmed green THIS session**
  `[VERIFIED: RUN cargo test -p cb-train --test bootstrap_oracle_test --test bootstrap_dev_oracle_test]`:
  `bootstrap_dev_cpu_matches_upstream ok`; all 5 `bootstrap_oracle_*` ok.

## Files

- Modify: `crates/cb-train/src/bootstrap_test.rs` — add ONE test (below).
- Modify: `crates/cb-train/src/bootstrap.rs`
  - delete the block at `:413-423` (the `if subsample < 1.0 { … rng.gen_rand(); rng.gen_rand(); }`
    and its whole justification comment);
  - add the trace-verified contract as a doc/comment at the draw site — either on
    `mvs_sample_weights`'s doc block (`:275-280`, which already names
    `GenSampleWeights`) or immediately above `:295`. Do **not** leave the deletion
    unexplained.
- Do NOT touch: `:424-429` (the control mask), `:294` (the `sample_rate` f32
  narrowing), `:312` (TASK-04), `:323` (TASK-05), `:35-39` (module bullet —
  TASK-07), `calculate_threshold`, `mvs_lambda`, `mean_grad_value`,
  `last_iter_mean_leaf_value`, or anything in `boosting.rs` / `tree.rs` /
  `device_draw_replay.rs`.

## TDD sequence

### 1. Red

Add to `crates/cb-train/src/bootstrap_test.rs` (the file already has the lint
opt-out; import `TFastRng64` is already in scope at `:15`):

- **Test name:** `mvs_bootstrap_consumes_exactly_one_main_stream_draw`
- **Setup:** `let n = 1500;` (one MVS block, `< 8192`, and the same object count as
  the fixture); `let ders: Vec<f64> = (0..n).map(|i| (i as f64 % 13.0) - 6.0).collect();`
  (the varied-magnitude pattern already used at `bootstrap_test.rs:160`, so the
  threshold is non-degenerate); `let seed = 0_u64;`
- **Input:** `bootstrap(EBootstrapType::Mvs, &ders, 0.8, 0.0, None, &mut rng)` on
  `let mut rng = TFastRng64::from_seed(seed);`
- **Assertion order matters** — the first assertion must be the one whose failure
  names the defect:
  1. `assert_eq!(rng.call_count(), 1, "…");`
  2. `assert_eq!(rng.raw_state(), probe.raw_state(), "…");` where
     `let mut probe = TFastRng64::from_seed(seed); let _ = probe.gen_rand();`
  3. the zero-draw regression leg: a second `TFastRng64::from_seed(seed)` through
     `bootstrap(Mvs, &ders, 1.0, …)` leaves `call_count() == 0`.
  4. the accumulation leg: three consecutive `bootstrap(Mvs, …, 0.8, …)` calls on
     ONE stream leave `call_count() == 3`.
  All four legs assert the SAME single behaviour (the per-call draw count), so this
  stays one focused test with one principal failure reason.
- **Expected initial failure** (before the production change), from
  `assertion 1`:

  ```
  assertion `left == right` failed: …
    left: 3
   right: 1
  ```

  `3` because the arm takes 1 real `rand_seed` (`:295`) plus the 2 fabricated draws
  (`:421-422`). If it is instead `raw_state` that fails first, the assertion order
  above was not followed — fix the order, do not reinterpret the result.
- Run: `cargo test -p cb-train --lib bootstrap`

### 2. Green

Delete `crates/cb-train/src/bootstrap.rs:413-423` in full — the `if subsample < 1.0`
guard, the two `rng.gen_rand();` calls, and the eight-line justification comment
(which is wrong on its own source). Nothing replaces the control flow: the arm
becomes `mvs_lambda` → `mvs_sample_weights` → the control map.

Then record the verified contract at the draw site. The doc must state, in this
substance:

> MVS consumes **exactly one** main-stream draw per `Bootstrap()` call
> (`mvs.cpp:174` `randSeed = rand->GenRand()`). `performRandomChoice = false`
> sends `TCalcScoreFold::Sample` down the `SetControlNoZeroWeighted` branch
> (`calc_score_cache.cpp:742-748`), which never touches `rand`; `CalcWeightedData`
> (`tensor_search_helpers.cpp:442-485`) is draw-free. Verified against the
> instrumented 1.2.10 trace
> (`.planning/plans/bayesian-rng-draw-accounting/instrumented-ground-truth/mvs.jsonl`):
> `tree_rng_pre_gts.cc = 2` → level-0 `cc_start = 7` = 1 bootstrap draw + 4 RSM
> draws.

**Use `SPEC.md` §10.1's verified citation set verbatim** (MINOR-5). Every line number
there was re-read in the upstream tree; the deleted comment's own citations
(`calc_score_cache.cpp:752`, `:1203-1211`) are **both wrong** and must not be carried
forward. In particular `CalcWeightedData` is `:442-485` (not `:442-486`) and
`SetControlNoZeroWeighted` is `:1196-1204` with the mask at `:1202`.

Do NOT add a compensating advance anywhere else. Do NOT touch `PRE_TREE_DRAWS`,
`POST_TREE_EXTRA_DRAWS`, or `replay_grow_draws` — SPEC `R1`/`MVS-S9`.

- Run: `cargo test -p cb-train --lib bootstrap`
- Run: `cargo test -p cb-train --test bootstrap_oracle_test`
  (the FROZEN family — 5 tests, must stay green; `bootstrap_oracle_mvs` is one of
  the 3-of-5 configurations that passed *despite* the defect and must keep passing)

### 3. Refactor

- Behaviour-preserving only: keep the `Mvs` arm's remaining three statements as
  they are; if the arm now reads awkwardly, collapse only whitespace/comment
  layout. Do not extract a helper (TASK-04 introduces the one seam this phase
  needs).
- Confirm no `unwrap`/`expect`/`panic`/raw index was introduced.
- Run: `cargo test -p cb-train --lib bootstrap`
- **Lint gates — DIFFERENTIAL, not absolute** (PLAN-CHECK CRITICAL-1/CRITICAL-2; v1 wrote
  "clippy clean" and "the gate scripts pass" here, and **both are unsatisfiable at
  HEAD**). Measured baselines are in `PLAN.md` §4.11; the required form is `PLAN.md`
  §4.12's diff-scoped grep:

  ```bash
  # Clippy MUST select ERRORS. Two traps (both measured at HEAD):
  #  (a) `cargo clippy -p cb-train --all-targets` ABORTS on the cb-oracle
  #      dev-dependency (model_json.rs:161) before reaching cb-train's targets;
  #  (b) grepping `-->` lines also catches WARNINGS — and `bootstrap.rs:134` carries a
  #      pre-existing `clippy::excessive_precision` warning that CANNOT be fixed
  #      (bootstrap.rs:110-116: trimming the literal breaks Bayesian parity).
  # Use PLAN.md §4.12's helper:
  clippy_error_files -p cb-train --all-targets | grep -E "src/bootstrap\.rs|bootstrap_test\.rs"   # must be EMPTY

  bash scripts/check-source-test-separation.sh                                # ABSOLUTE: exit 0
  bash scripts/check-no-raw-float-sum.sh 2>&1 | grep -E "src/bootstrap\.rs|bootstrap_test\.rs"   # must be EMPTY
  bash scripts/check-no-anyhow.sh        2>&1 | grep -E "src/bootstrap\.rs|bootstrap_test\.rs"   # must be EMPTY
  ```

  All three were measured EMPTY at HEAD `[VERIFIED: RUN]`; the clippy one stays EMPTY even
  after `touch crates/cb-train/src/bootstrap.rs`, proving it is not empty from caching.
  `check-no-raw-float-sum.sh` exits 1 at HEAD (15 files / 36 lines) and
  `check-no-anyhow.sh` exits 1 (12 files / 25 lines) `[VERIFIED: RUN]`. Making them pass
  would require editing ~25 files across five crates — including
  `crates/cb-train/src/boosting.rs:1649`, which `MVS-S9` requires byte-unchanged. Do NOT
  attempt it. Only `check-source-test-separation.sh` is an absolute gate (exit 0 at HEAD).

  **Doc-text constraint (C2-6):** the D-08 script greps **comments** in non-test source,
  and `bootstrap.rs` is currently clean of its `SUM_PATTERN` `[VERIFIED: RUN — grep exit
  1]`. So the contract doc this task writes must NOT contain the literal `.sum()` or
  `.fold(0.0`; say `sum_f64` or "a raw iterator summation" instead. Re-run the D-08
  diff-scoped grep after the doc edit.

### 4. Verify

- Run: `cargo test -p cb-train --lib bootstrap` → **8** bootstrap unit tests pass
  (7 pre-existing + the new one) `[baseline 7 VERIFIED: RUN --list]`.
- Run: `cargo test -p cb-train --test bootstrap_oracle_test` → 5 passed.
- Run: `cargo test -p cb-train --test bootstrap_dev_oracle_test` → 1 passed (still
  at `MVS_GATED_TREES = 2`; TASK-02 raises it).
- Run: `cargo test -p cb-train --test regularization_oracle_test` → green
  (the Bayesian / `random_strength` draw path must not move; SPEC `MVS-S9`).
- Run: `cargo test -p cb-train --test yetirank_pairwise_tree_rng_oracle_test`
  → green (the other RNG-phase-sensitive oracle; SPEC `R1`).
- Run: `git diff --stat crates/cb-train/src/bootstrap.rs` → a deletion of ~11 lines
  plus the doc, and **nothing else** in the file.
- Run: `git status --short crates/cb-oracle/fixtures/` → EMPTY.
- Confirm: `grep -n "gen_rand" crates/cb-train/src/bootstrap.rs` shows main-stream
  `gen_rand`/`gen_rand_real1` calls only at `generate_random_weights` (the Bayesian
  `rand_seed`), `set_sampled_control` (Bernoulli), and the ONE MVS `rand_seed` —
  no other MVS main-stream draw.

## Completion criteria

- [ ] The Red test failed with `left: 3, right: 1` on the FIRST assertion, before
      any production edit.
- [ ] `bootstrap.rs:413-423` is deleted; the file has no `if subsample < 1.0`
      block in the `Mvs` arm.
- [ ] The trace-verified one-draw contract is documented at the draw site with the
      `mvs.cpp:174` / `calc_score_cache.cpp:742-748` /
      `tensor_search_helpers.cpp:442-485` / `mvs.jsonl` citations.
- [ ] The control-mask lines (`w > f64::from(f32::EPSILON)`) are byte-unchanged.
- [ ] `cargo test -p cb-train --lib bootstrap` → 8 passed.
- [ ] `bootstrap_oracle_test` (5), `bootstrap_dev_oracle_test` (1),
      `regularization_oracle_test`, `yetirank_pairwise_tree_rng_oracle_test` all
      green.
- [ ] **Differential lint gates**: the three diff-scoped greps above are EMPTY, and
      `check-source-test-separation.sh` exits 0. **Do not** assert "clippy clean" or "the
      gate scripts pass" — clippy is red at HEAD and two of the three scripts exit 1
      (`PLAN.md` §4.11 `B3`, `B4`, `B8`, `B9`).
- [ ] `git diff` touches ONLY `crates/cb-train/src/bootstrap.rs` and
      `crates/cb-train/src/bootstrap_test.rs`.

## Completion evidence to record in `progress.md`

- The exact Red failure text.
- `call_count()` after one / three MVS calls, post-fix.
- The `bootstrap_oracle_test` and `bootstrap_dev_oracle_test` results.
- The `git diff --stat` line count for `bootstrap.rs`.

## Risks and guardrails

- **SPEC R1 — "fixing" the count by re-tuning the shared accounting.** Forbidden.
  `PRE_TREE_DRAWS` (`boosting.rs:59`), `POST_TREE_EXTRA_DRAWS` (`:69`),
  `replay_grow_draws` (`device_draw_replay.rs:64-85`) and
  `select_level_perturbed`'s draw shape are each independently verified by the
  instrumented trace and by the value-sensitive 3-tree `bootstrap_oracle_bayesian`.
  The MVS arm is the ONLY place with a wrong count. Guard: the Verify step runs
  both RNG-phase oracles.
- **SPEC R4 — regenerating a fixture to make a test pass.** Forbidden. Guard:
  `git status --short crates/cb-oracle/fixtures/` must be empty.
- **A vacuous Red.** If the new test passes before the deletion, either the
  assertion order was wrong or `subsample` was ≥ 1.0 (the zero-draw
  short-circuit at `bootstrap.rs:288`). Re-check the input, do not weaken the test.
- **Device numbers move.** Expected and desired: the device branch calls this same
  `bootstrap()` at `boosting.rs:3262`. Do not attempt to "hold the device stable" —
  TASK-08 re-verifies device-vs-CPU parity after the fix.
- **Pre-existing red.** `monotone_non_symmetric_and_region_are_typed_errors` is a
  known baseline failure. Never chase it here.
