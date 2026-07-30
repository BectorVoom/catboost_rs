# Plan Check Result

**Verdict:** ISSUES_FOUND
**Goal:** Root-cause and fix the CPU MVS sampler's divergence from upstream CatBoost 1.2.10 so MVS matches on ALL trees at ≤1e-5 for BOTH `boost_from_average` settings, remove the `MVS_GATED_TREES = 2` carve-out, land the two user-approved f32 transcription fixes (mirrored into `cb-backend`'s inline CPU copies), and add a committed multi-seed × bias MVS fixture family.
**Plan:** `.planning/plans/mvs-tree2-parity/PLAN.md` + `plan1.md` … `plan9.md` (spec: `SPEC.md`, research: `research.md`), base `2c14d7f`, branch `fix/bootstrap-rng-draw-accounting`
**Checked at:** HEAD `2c14d7f`, working tree clean apart from `.planning/plans/mvs-tree2-parity/`

---

## Summary

- The **diagnosis is correct and the fix is right**. I independently re-verified the defect, the upstream contract, and every load-bearing line number and numeric value. `crates/cb-train/src/bootstrap.rs:413-423` does fabricate two `rng.gen_rand()` draws in the `Mvs` arm; upstream takes exactly one (`mvs.cpp:174`), and `performRandomChoice == false` provably routes `TCalcScoreFold::Sample` down the draw-free `SetControlNoZeroWeighted` branch (`calc_score_cache.cpp:742,747` — verified in the local upstream tree, and the *existing* in-repo comment's citations `:752` / `:1203-1211` are the wrong ones). The f32 target values (`1200.0`, `6553.60009765625`, `2892.800048828125`) reproduce exactly. The baseline `503 passed / 1 failed` reproduces exactly, with the single failure being the recorded `monotone_non_symmetric_and_region_are_typed_errors` at `monotone_oracle_test.rs:286`.
- **Specification and AC coverage is complete**: all ten specs map to tasks, all ten ACs map to a task and an evidence command.
- **It nevertheless cannot be executed as written.** Three CRITICAL problems:
  1. `cargo clippy -p cb-train --all-targets` — a "clean" gate in six task completion criteria and the phase Definition of Done — is **RED at HEAD** with 15 pre-existing errors, and the plan does not record them.
  2. Two of the three mandated gate scripts (`check-no-raw-float-sum.sh`, `check-no-anyhow.sh`) **exit 1 at HEAD**; the DoD requires all three to pass.
  3. Wave B is **not parallel-safe**: TASK-02's and TASK-03's Red procedures both mutate `crates/cb-train/src/bootstrap.rs` — the file TASK-04 owns in the same wave — in one shared working tree (`planning/settings.json` → `"use_worktree": false`).
- Five MAJOR issues follow, the most consequential being that **TASK-06's mirror ends up with no test at all** (I measured that 4 of the 5 mirrored call sites are bit-identical under the f32 change, so risk R2 can silently recur), and that **AC-8's "device arm reports 3/3 for MVS" is asserted with zero evidence** while TASK-02 makes it binding five tasks earlier.

---

## Specification Coverage

| Spec / AC | Verdict | Evidence |
|---|---|---|
| `MVS-S1` / `AC-1` | [x] covered | TASK-01 deletes `bootstrap.rs:413-423` (verified present verbatim) and adds `mvs_bootstrap_consumes_exactly_one_main_stream_draw`. Instruments verified: `TFastRng64::{from_seed:171, gen_rand:183, advance:192, call_count:204, raw_state:221, gen_rand_real1:232}` in `crates/cb-core/src/rng.rs`; `call_count` starts at 0 and `gen_rand` increments it; `raw_state` does not advance. Baseline count 3 is real (1 at `:295` + 2 at `:421-422`). |
| `MVS-S2` / `AC-2` | [x] covered | TASK-02. Every cited site verified in `crates/cb-train/tests/bootstrap_dev_oracle_test.rs` (383 lines): `SCENARIOS:112-116`, `MVS_SCENARIO:119`, doc+const `:121-155`, `gated_trees:168`, truncation `:174-186`, printout `:216-219`, chains `:237`/`:347`, call sites `:253`/`:380`. Repo-wide grep confirms the carve-out is referenced only in that file. Full-slice comparison is length-consistent (3 trees × depth 2 = 6 splits, × 4 leaves = 12, × 1500 rows = 4500). |
| `MVS-S3` / `AC-3` | [x] covered, but see [MAJOR-6] | TASK-03. `gen_fixtures.py` is 3385 lines; `BOOTSTRAP_DEV:66`, `BOOTSTRAP_INPUT:67`, `ISOLATING_PARAMS:151-164` (does pin `random_seed: SEED` and does omit `boost_from_average` — risk P3 is real), `gen_bootstrap:710`, `gen_bootstrap_dev:858`, `gen_bootstrap_dev_only:951`, `--bootstrap-dev-only:3374`. `cb-oracle`/`ndarray`/`ndarray-npy` already in `cb-train` `[dev-dependencies]` → no manifest change ✔. |
| `MVS-S4` / `AC-4` | [x] covered | TASK-04. Site verified at `bootstrap.rs:306-313` (`sample_rate * block_size as f64` at `:312`). Upstream verified: `mvs.h:47 float SampleRate`, `mvs.h:48 const ui32 BlockSize = 8192`, `mvs.cpp:197-202 CalculateThreshold(..., SampleRate * blockSize)`. Numbers re-derived independently: `(0.8,1500)`→`1200.0` vs `1200.0000178813934` (+1.788e-5); `(0.8,8192)`→`6553.60009765625` both; `(0.8,3616)`→`2892.800048828125` vs `2892.800043106079`. `8192` and `3616` are exact in f32 ✔. |
| `MVS-S5` / `AC-5` | [x] covered | TASK-05. Site verified at `bootstrap.rs:315-328` (store at `:323`, conditional draw at `:319`/`:321`). Upstream verified: `catboost/private/libs/algo/fold.h:217 TVector<float> SampleWeights;`, narrowed at `mvs.cpp:213`, read as `const float*` at `tensor_search_helpers.cpp:456`. |
| `MVS-S6` / `AC-6` | [ ] **partially covered — no test** | TASK-06's edits are correct and the two sites verified (`mvs_device_test.rs:132`, `:165`), but see **[MAJOR-3]**: after the mirror nothing tests it. |
| `MVS-S7` / `AC-7` | [x] covered, wording defect | TASK-07. All nine cited locations verified, including `device-bootstrap-parity/SPEC.md:746`, `:793`, `progress.md:67`, `:69`, `:137`. See **[MINOR-2]** on the "anywhere in the repo" postcondition. |
| `MVS-S8` / `AC-8` | [ ] **covered by intent, unevidenced** | See **[MAJOR-4]**. |
| `MVS-S9` / `AC-9` | [x] covered | TASK-09. `PRE_TREE_DRAWS: usize = 2` at `boosting.rs:59`, `POST_TREE_EXTRA_DRAWS: usize = 2` at `:69`, `replay_grow_draws` at `device_draw_replay.rs:64`. Baseline `503 passed / 1 failed / 4 ignored` reproduced; 503 → 507 arithmetic (3 lib + 1 integration) is correct. |
| `MVS-S10` / `AC-10` | [x] covered | TASK-07. Deviation (a) target `mean_grad_value` doc at `bootstrap.rs:333-335` verified; `CB_THREAD_LIMIT = 128` verified at `options/restrictions.h:59`; `TMaybe<float> Lambda` at `mvs.h:49`; learn-weight multiply at `tensor_search_helpers.cpp:479-482`. |

---

## CodeGraph / source evidence

- `bootstrap` — `crates/cb-train/src/bootstrap.rs:383`, `pub`. CodeGraph blast radius: exactly three call sites — `boosting.rs:3262` (device branch), `boosting.rs:3833` (CPU branch), `bootstrap_test.rs`. Both `boosting.rs` sites confirmed by grep. **Impact: the fix moves device and CPU numbers identically by construction** — the device branch's own comment at `boosting.rs:3222` states "The device branch keeps the ENTIRE sampler on the host".
- `EBootstrapType` — `bootstrap.rs:69`. 15 dependents across `cb-train/src/lib.rs:38`, `boosting.rs`, `catboost-rs/src/builder.rs`, `catboost-rs-py/src/params.rs:395`. **No signature change needed.** Builder default is `EBootstrapType::No` (`builder.rs:110`), so **no default Python or Rust fit changes**; only explicit `bootstrap_type="MVS"` users are affected. No Python test references MVS (`grep bootstrap_type crates/catboost-rs-py/tests/` → only `"No"` and `"Bayesian"`). SPEC §7/§8's impact and compatibility claims are accurate.
- `BootstrapResult` — `bootstrap.rs:88`, 3 dependents; `sample_weights: Vec<f64>` at `:90`. Container unchanged by MVS-S5 ✔.
- `mvs_sample_weights` / `calculate_threshold` / `single_probability` / `mvs_lambda` / `mean_grad_value` — all private, all single-caller inside `bootstrap.rs`. CodeGraph flags "no covering tests found" for all five, which is exactly why AC-1/AC-4/AC-5's unit tests matter.
- `launch_mvs_weights_resident` — `crates/cb-backend/src/kernels/mvs_device.rs:281`; `draw_mvs_weights_host` — `:327` (calls the former at `:341`). **Dead on the live path confirmed**: `cb-compute/src/runtime.rs:1131` defaults `mvs_lambda: None`, `grep -rn mvs_lambda crates/cb-train/` shows no `Some(..)` assignment, and `boosting.rs:3173-3177` says so explicitly. Only `mvs_device_test.rs` exercises the kernel. The plan's blast-radius model is correct.
- `mvs_sample_kernel` — `mvs_device.rs:106-...`; `let sample_size = rate * f64::cast_from(u64::cast_from(bs));` at **`:146` confirmed**; host f32-rounds only the RATE at `:302`; `MVS_BISECTION_ITERS = 100` at `:67`.
- `replay_grow_draws` — `device_draw_replay.rs:64`, 9 references repo-wide. Untouched by every task ✔.

---

## Issues

### [CRITICAL-1] `cargo clippy -p cb-train --all-targets` is RED at HEAD; the plan gates six tasks and the DoD on it being "clean"

- **Plan location:** `PLAN.md:353` (Definition of Done); `plan1.md:210,247`, `plan2.md:185`, `plan3.md:264,304`, `plan4.md:199,234`, `plan5.md:189,229`, `plan7.md:223`, `plan9.md:171,238`.
- **Requirement:** the phase's lint discipline gate (guardrail 2, `CLAUDE.md` no-`unwrap`/`panic`/raw-index rule).
- **Evidence (RUN at `2c14d7f`):**
  - `cargo clippy -p cb-train --all-targets --no-deps` → **15 errors** in four pre-existing integration test files:
    `tests/ordered_boost_oracle_test.rs` (3 × `expect()` on Result, 3 × `panic`, 2 × indexing), `tests/permutation_oracle_test.rs` (3 × `panic`), `tests/s_order_ctr_bins_oracle_test.rs` (2 × indexing), `tests/learn_set_shuffle_oracle_test.rs` (2 × indexing). Those files are missing the file-level `#![allow(...)]` that `bootstrap_test.rs:10` and `bootstrap_dev_oracle_test.rs:24` carry.
  - **Without** `--no-deps` (i.e. exactly the command the plan writes) it additionally fails on `crates/cb-oracle/src/model_json.rs:161:17` ("indexing may panic") and on `cb-backend`'s 4 lib errors, because clippy lints path dependencies too. `cargo clippy -p cb-train --lib` also fails for the same reason.
- **Failure scenario:** TASK-01's Refactor step runs the command, sees 15–20 errors, and the implementer either (a) declares the task blocked, (b) "fixes" four unrelated oracle test files plus `cb-oracle` — direct scope leakage into a frozen parity-harness crate, contradicting SPEC `R5` — or (c) silently drops the gate, which removes the only lint check on the phase's own new code (exactly where the `as f32` casts and new test files land).
- **Impact:** every task's Refactor/Verify step and the phase Definition of Done are unachievable as written; the phase cannot be signed off.
- **Required revision:** (1) Change every invocation to `cargo clippy -p cb-train --all-targets --no-deps`. (2) Add to the "Known pre-existing reds" list in `PLAN.md` §4.11 and `progress.md`: the 15 `cb-train` test-target errors by file and count, plus `cb-oracle/src/model_json.rs:161` and the `neg_cmp_op_on_partial_ord` warning at `cb-oracle/src/compare.rs:58`. (3) Restate the gate as "no NEW clippy error name/location relative to the recorded baseline", not "clean".

### [CRITICAL-2] Two of the three mandated gate scripts exit 1 at HEAD; the DoD requires all three to pass

- **Plan location:** `PLAN.md:224,356` ("The three source/test/sum/anyhow gate scripts pass"); `plan1.md:212-213,247`, `plan4.md:200-201,234`, `plan5.md:192-193,229`, `plan6.md:217-219`, `plan9.md:174-176,240`.
- **Evidence (RUN at `2c14d7f`):**
  - `bash scripts/check-source-test-separation.sh` → exit 0, `OK: no inline #[cfg(test)] module bodies in production source` ✔ (this one is fine, and the plan's description of its brace-vs-semicolon behaviour is accurate).
  - `bash scripts/check-no-raw-float-sum.sh` → **exit 1**, 15 reported violations across `cb-compute/src/{leaf.rs:664,score.rs:223,243,269}`, `cb-backend/src/kernels/{grow_loop.rs,pairwise_hist.rs,pointwise_hist.rs,reduce.rs,score_split.rs,update_part_props.rs:164,184}`, `cb-train/src/{ctr/final_ctr.rs:123,ctr/online.rs:77,overfit.rs:521,boosting.rs:1649}`, `cb-model/src/{shap.rs:45,apply.rs:30,partial_dependence.rs:181}` — most are the script matching `.sum()` inside DOC COMMENTS that *describe the ban*, plus genuine integer sums (`sizes.iter().sum()` on `Vec<usize>`).
  - `bash scripts/check-no-anyhow.sh` → **exit 1**, 12 reported violations, every one a doc comment reading "no `anyhow`".
- **Failure scenario:** TASK-01 Refactor runs both scripts, both fail, and the only way to make the DoD's "the three gate scripts pass" true is to edit ~25 files across five crates — an enormous unrelated change inside a bug-fix phase, and one that would touch `boosting.rs` (which `MVS-S9`/TASK-09 requires to be byte-unchanged, creating a direct contradiction between two plan requirements).
- **Impact:** the DoD is self-contradictory and unreachable; tasks 01/04/05/06/09 all carry a gate that fails for reasons outside the phase.
- **Required revision:** record both scripts' exact HEAD-baseline output (counts + file:line list) as pre-existing reds, and change the gate wording to "produces the SAME violation set as the recorded HEAD baseline — no new file or line". Better still, make it a diff-scoped check, since the phase touches only three source files (`bootstrap.rs`, `bootstrap_test.rs`, `kernels/mvs_device_test.rs`) — none of which introduces a raw float sum or `anyhow`.

### [CRITICAL-3] Wave B is not parallel-safe: TASK-02 and TASK-03 both temporarily rewrite `bootstrap.rs`, the file TASK-04 owns in the same wave

- **Plan location:** `PLAN.md:80-83` (Wave B), `PLAN.md:123-150` (§1.1 write-conflict check), `plan2.md:117-141` (Red step 1.2), `plan3.md:223-236` (Red step 1c), frontmatter `parallelizable: true` in `plan2.md`, `plan3.md`, `plan4.md`.
- **Evidence:**
  - `PLAN.md` §1.1's ownership table lists TASK-02 as writing only `bootstrap_dev_oracle_test.rs` and TASK-03 as writing only `gen_fixtures.py` + a new fixture root + a new test file, and concludes "No shared file."
  - But `plan2.md:126-129` instructs: "`git stash push -- crates/cb-train/src/bootstrap.rs` … or, if TASK-01 is already committed … re-insert the two draws behind a one-line `if subsample < 1.0 { rng.gen_rand(); rng.gen_rand(); }` in a scratch edit", and `plan3.md:224-226` gives the identical instruction. Both therefore WRITE `crates/cb-train/src/bootstrap.rs`.
  - `planning/settings.json` (v2) — **verified** — `"implementation": {"use_worktree": false}`. All tasks edit one shared working tree in place, as `PLAN.md:28-31` itself states.
- **Failure scenario:** TASK-04 has uncommitted edits to `bootstrap.rs` (the new `mvs_block_sample_size` seam). TASK-02 runs `git stash push -- crates/cb-train/src/bootstrap.rs`; TASK-04's work vanishes mid-task and its in-flight `cargo test` runs against a *defect-restored* sampler, producing residuals that look like a broken extraction. Meanwhile TASK-03 issues its own `git stash push`/`git stash pop` on the same path; the two pops interleave and land the file in an indeterminate state — in the worst case with the two fabricated draws still present, silently un-fixing the phase's whole purpose while every CPU oracle still passes (they passed *with* the defect for 3 of 5 bias-true seeds).
- **Impact:** lost work in the single most important production file; three tasks' Red/Green evidence misattributed; a plausible path to shipping the phase with the defect partially reintroduced.
- **Required revision:** either (a) declare Wave B **serial** (TASK-04 → TASK-02 → TASK-03, or any fixed order) and drop `parallelizable: true` from all three frontmatters; or (b) require TASK-02 and TASK-03 to produce their controlled-revert Red in an **isolated throwaway `git worktree add` at `2c14d7f`** with only their own test/fixture change applied, and explicitly FORBID `git stash` on `crates/cb-train/src/bootstrap.rs` in the main tree. In either case correct `PLAN.md` §1.1 to list `crates/cb-train/src/bootstrap.rs` as a file TASK-02 and TASK-03 *temporarily write*.

### [MAJOR-1] The "TASK-04's numerics provably cannot invalidate TASK-03" claim only covers the post-fix state, yet plan2/plan3 bind exact pre-fix values

- **Plan location:** `PLAN.md:143-146`, `plan2.md:46-48`, `plan3.md:50-52`; the bound values are at `plan2.md:131-134,208`, `plan3.md:233-235,299-300`, `progress.md:155,159`.
- **Evidence:** the cited proof is research `§8.5`/`§8.6`, which spiked `A + B` (deletion + f32 target) and `A + B + C` — i.e. only the **post-fix** state — reporting 5/5 + 5/5 and byte-identical residuals. The **defect-present + f32 target** combination (`B` without `A`) was never measured; research §8.2's failing set `{3, 4, 4, 4, 5, 5, 4}` and the `StageDiverged { index: 5, expected: -0.025514747947454453, actual: -0.2692405581474304 }` value were measured with `bootstrap.rs:312` still in `f64`.
- **Failure scenario:** TASK-04 lands before TASK-02/TASK-03 capture their Reds (permitted by the wave). The reverted-state run now yields a different `actual` value at split 5 (the threshold target moves by +1.788e-5 at `block_size = 1500`, which is precisely the perturbation that can move a near-tied argmax), and plan2's completion criterion "reproduced `StageDiverged { … expected: -0.0255…, actual: -0.2692… }`" becomes unmeetable.
- **Impact:** TASK-02/TASK-03 blocked, or their Red criteria silently relaxed — which is exactly the "vacuous green" the plan elsewhere works hard to prevent.
- **Required revision:** state that both Reds MUST be captured against a state with the f32 target ABSENT (naturally satisfied by the isolated-worktree fix in CRITICAL-3), and rewrite the parallel-safety claim as: "TASK-04 cannot change the post-fix pass/fail verdict (research §8.5) but MAY change the pre-fix diagnostic values, so the Reds must be captured before or independently of TASK-04."

### [MAJOR-2] The device arm's new 3/3-tree MVS claim is unmeasured, is made binding by TASK-02 five tasks earlier, and has no defined fallback

- **Plan location:** `plan2.md:164-167` (edits BOTH `.chain(...)` sites), `plan8.md:121-132`, `SPEC.md` `MVS-S8`, `AC-8`; `progress.md` unresolved item 2.
- **Requirement:** `AC-8` — "the device arm reports 3/3 trees".
- **Evidence:** `bootstrap_dev_oracle_test.rs:347` is inside `bootstrap_dev_device_matches_upstream` (`#[cfg(any(feature = "rocm", feature = "cuda"))]`, `:259-382`), which compares the **device** fit to **upstream**. That comparison at 3 trees for MVS has never been executed: research §12 MEDIUM #4 says "Not run — no GPU run was performed in this session". Separately, the completed sibling phase recorded `split_mismatched_trees = 4/20` (`device-bootstrap-parity/progress.md:162`, verified) — the device *does* pick a different split from the CPU on near-ties; `device_bootstrap_parity_test.rs:411-417` tolerates that only because it bounds the *contribution*, whereas `compare_stage(Stage::Splits, …)` compares raw border values and a different split is a hard failure, not a small delta. `MVS-S8`'s "Principal failure reason" names only device-side draw accounting; it does not name device/CPU split-argmax disagreement on the MVS sample.
- **Failure scenario:** TASK-02 lands; the rocm test target is now red; nobody notices because five subsequent tasks are CPU-only; TASK-08 finally runs it and `[device] bootstrap_dev/mvs: splits diverged from upstream` fires at tree 2 because the device broke a near-tie the other way. AC-8 and AC-9 are then unachievable and the plan's only instruction is "escalate", with no defined acceptable outcome.
- **Impact:** the repo carries a red device suite across five tasks; the phase's headline device claim may be unreachable with no planned remedy short of re-opening the spec.
- **Required revision:** (1) add device/CPU split-argmax disagreement to `MVS-S8`'s failure modes; (2) move a cheap device probe of `bootstrap_dev_oracle_test`'s device arm into TASK-02's Verify step — it does **not** depend on TASK-06, because the `cb-backend` mirror is a change to a *test* file only (`kernels/mvs_device_test.rs`) and cannot affect `bootstrap_dev_oracle_test`; measuring it while the carve-out removal is still isolated makes any failure attributable; (3) pre-define the escalation outcome (a **device-only** documented residual with its own spec entry — never a tolerance loosening, never a `replay_grow_draws` or `POST_TREE_EXTRA_DRAWS` change).

### [MAJOR-3] After TASK-06 the `cb-backend` mirror has no test — `MVS-S6` has no falsifiable Red and risk R2 can silently recur

- **Plan location:** `plan6.md:152-177` (Red), `plan6.md:247-261` (completion criteria); `SPEC.md` `MVS-S6` acceptance examples.
- **Requirement:** `MVS-S6` / `AC-6` and SPEC risk `R2` ("Editing the sampler without mirroring `mvs_device_test.rs:55-172` → device self-oracle fails on rocm only — invisible on default CI").
- **Evidence:** `plan6.md:166-173` itself concedes outcome (b), "a silent pass with a drifted reference", is an acceptable Red, with the falsifying evidence being a `grep` rather than a test. I quantified why it is not merely possible but *likely*: `TOL = 1e-4` (`mvs_device_test.rs:27`), and across the five `cpu_block_threshold` call sites the f32-vs-f64 target delta is
  - `(rate 0.5, n 48)` → **0** (bit-identical)
  - `(rate 0.7, n 64)` → **0**
  - `(rate 0.6, n 96)` → **0**
  - `(rate 0.5, n 8192)` and `(0.5, 24)` → **0**
  - `(rate 0.3, n 200)` → 1.431e-6 absolute, **2.384e-8 relative** — the ONLY case with any shift at all.

  So the `MVS-S4` half of the mirror is undetectable in 4 of 5 call sites and 4 orders under `TOL` in the fifth; the `MVS-S5` half is ~6e-8 relative everywhere. After TASK-06 completes, **no test anywhere in the repo constrains the two transcriptions to agree.**
- **Failure scenario:** a future phase edits `mvs_block_sample_size` or the weight store in `bootstrap.rs` (e.g. to add `mvs_reg`, or during the Design B′ device-resident wiring). The `cb-backend` copy is not updated. Every CPU and rocm test still passes. R2 recurs in exactly the form this phase set out to close, and the next reader has only a prose "keep in sync" note.
- **Impact:** `MVS-S6` becomes a one-shot manual sync with no regression protection; `AC-6` is trivially satisfiable and proves nothing.
- **Required revision:** TASK-06 must add ONE host-arithmetic test to `crates/cb-backend/src/kernels/mvs_device_test.rs` that does **not** gate on `device_backend_active()` (it needs no GPU, so it runs under the default `cpu` feature — closing the "invisible on default CI" hole `MVS-S6` itself names), asserting (i) the mirrored target expression exactly, using the same three values TASK-04's test pins (`1200.0`, `6553.60009765625`, `2892.800048828125`), and (ii) that every weight `cpu_mvs_sample` returns satisfies `w == f64::from(w as f32)`. That converts TASK-06's Red from static-grep into a real failing test and gives the mirror permanent protection.

### [MAJOR-4] `plan6`'s Verify expectation "the `max_div` should DROP" is backwards and contradicts its own P1 analysis

- **Plan location:** `plan6.md:224-226`; `progress.md:169` measurement row.
- **Evidence:** the mirror moves the CPU **reference** away from the kernel, not toward it. `mvs_sample_kernel` keeps its `f64` target (`mvs_device.rs:146`, verified) and stores un-narrowed `f64` weights; the reference gains an f32 target and an f32-narrowed store. Pre-mirror the two agree to bisection precision (`MVS_BISECTION_ITERS = 100`, `mvs_device.rs:67` — 100 halvings resolve the root to full `f64`), so `max_div` is ~1e-13. Post-mirror `max_div` is floored at ≈ |w| · 6e-8 ≈ 1e-7. `plan6.md`'s own §"The residual this task deliberately leaves (planner finding `P1`)" says exactly this ("the threshold root shifts by ~1.5e-8 relative"), so `:224-226` contradicts `:117-129` of the same file.
- **Failure scenario:** the implementer sees `max_div` jump ~6 orders, concludes the mirror is wrong, and either reverts it or "fixes" it by narrowing `mvs_sample_kernel`'s target — the precise edit `MVS-S6`'s non-goal and `plan6.md:144-148` forbid, and one that would require a CubeCL kernel change this phase has no mandate for.
- **Impact:** a wrong acceptance expectation on the only device-gated assertion of TASK-06; a plausible route to a forbidden production kernel edit.
- **Required revision:** replace with: "`max_div` will RISE from ~1e-13 to ~1e-7 — still ≥3 orders under `TOL = 1e-4`. The load-bearing assertions are `kept_dev == kept_cpu` (exact) and `max_div ≤ 1e-4`." Record the predicted post-mirror order in `progress.md:169` so the rise is pre-authorised rather than discovered.

### [MAJOR-5] `plan3`'s Red criterion over-constrains the failing set against the plan's own fixture-nondeterminism warning

- **Plan location:** `plan3.md:230-241`, completion criterion `plan3.md:299-300`; contradicted by `plan3.md:277-283` and `progress.md` unresolved item 3.
- **Evidence:** the criterion is "**≥ 7 of 10** scenarios fail, matching the recorded failing set (`bias=true` seeds 1 and 4 plus `bias=false` seeds 0–4) and first-bad-split indices drawn from `{3, 4, 5}`". Those numbers were measured against research §8.7's **throwaway** scratch fixtures. The same plan simultaneously warns (`:277-283`) that "CatBoost quantization is known to be run-to-run nondeterministic in some configurations" and instructs freezing the first generation if a second differs. Different quantization borders produce a different fit, hence a possibly different failing set — and the whole family is compared against **its own** borders (SPEC `R3`).
- **Failure scenario:** the newly committed family yields 6 of 10 failing pre-fix. The family plainly discriminates, but TASK-03's binding criterion fails, and the implementer's only options are to stall or silently weaken the Red.
- **Impact:** TASK-03 can be blocked by a benign fixture regeneration; or its Red is weakened ad hoc, which is the failure mode `MVS-S3` exists to prevent.
- **Required revision:** make the binding criterion "**≥5 of 10** fail pre-fix, including **≥1 with `bias=true`** and **≥1 with `bias=false`** (so both bias settings are demonstrably gated)". Keep 7/10 and `{3,4,4,4,5,5,4}` as the *expected* value to record, and require any deviation to be recorded together with the observed borders — not treated as a blocker.

### [MINOR-1] The `P1` residual is recorded only in a test file, and the plan's own gate forbids a doc-only note on the kernel

- **Plan location:** `plan6.md:193-197` (note goes in `mvs_device_test.rs`'s module doc), `plan6.md:143-148` + `:257` + `plan9.md:224` (require `git diff crates/cb-backend/src/kernels/mvs_device.rs` to be EMPTY).
- **Evidence:** after this phase, `mvs_device.rs:145-146` carries a target that is a known deviation from upstream's `float` expression, and nothing at that site says so. A future implementer wiring Design B′ (`config.mvs_lambda = Some(..)` — currently never set, verified) reads the kernel, not the sibling test file.
- **Required revision:** allow — and require — a DOC-ONLY addition at `mvs_device.rs:145-146` naming the deviation and pointing at `bootstrap.rs`'s `mvs_block_sample_size`; change the criterion from "byte-unchanged" to "no executable change (`git diff` shows comment lines only)".

### [MINOR-2] `MVS-S7`'s postcondition and `AC-7` are literally unsatisfiable; the plan relaxes them without amending the spec

- **Plan location:** `SPEC.md` `MVS-S7` postcondition ("no occurrence … remains anywhere in the repo") and `AC-7`; relaxed at `plan7.md:236-241` and `plan9.md:216`.
- **Evidence (grep at HEAD):** the banned phrasings survive inside this phase's own artifacts — `mvs-tree2-parity/research.md:442,878`, `progress.md:33`, `plan7.md:26,62,80,84-86,145-146,232-233`, `plan9.md:214-215` — where they legitimately appear as explicitly refuted quotations.
- **Required revision:** amend `MVS-S7`'s postcondition and `AC-7` to read "anywhere in the repo **outside `.planning/plans/mvs-tree2-parity/`**, where the claims appear only as explicitly-refuted quotations", so the spec and the executable gate agree.

### [MINOR-3] Wave E's "disjoint by construction" ignores that TASK-08 compiles the crate TASK-07 edits

- **Plan location:** `PLAN.md:152-157`.
- **Evidence:** TASK-07 writes doc comments in `crates/cb-train/src/bootstrap.rs`; TASK-08 runs seven `cargo test -p cb-train --no-default-features --features rocm --test …` invocations that compile that crate, in the same working tree, alternating with `cpu`-feature builds. A mid-edit save forces a rebuild and the two cargo processes contend for the same target-dir lock.
- **Required revision:** either serialize TASK-07 after TASK-08, or state explicitly that TASK-07 must hold its `bootstrap.rs` write until TASK-08's runs complete. (Correctness risk is nil — a doc-only edit cannot move numerics — so say that too, rather than claiming disjointness.)

### [MINOR-4] The controlled-revert Reds have no byte-identity restore gate

- **Plan location:** `plan2.md:135-136`, `plan3.md:236`.
- **Evidence:** restoration is "`git stash pop`, or revert the scratch edit … and confirm the suite is green again". A green suite does not prove the file is byte-identical — a stray blank line, a half-reverted comment, or a leftover `if false {` would pass.
- **Required revision:** after each controlled revert, require `git diff crates/cb-train/src/bootstrap.rs` against the post-TASK-01 state to be empty AND `git stash list` to be empty, recorded in `progress.md`.

### [MINOR-5] Upstream citation path errors that will be transcribed into production doc comments

- **Evidence (verified in `/home/user/cb_instrumented_build/catboost-src`):**
  - `SPEC.md` §10 cites "`catboost/private/libs/algo_helpers/fold.h`". That file does not exist; the correct path is **`catboost/private/libs/algo/fold.h:217`** (`TVector<float> SampleWeights;`). `plan5.md`/`plan6.md` mandate writing this citation into `bootstrap.rs` and `mvs_device_test.rs` doc comments.
  - The `const float*` read of `SampleWeights` is at `tensor_search_helpers.cpp:456`, not `:457`; the learn-weight multiply is `:479-482`, not `:481-485`.
  - Confirmed correct as written: `mvs.h:47`, `mvs.h:48`, `mvs.h:49`, `mvs.cpp:174`, `mvs.cpp:202`, `mvs.cpp:210`, `mvs.cpp:212`, `mvs.cpp:213`, `mvs.cpp:17/21/37/67/81/120`, `calc_score_cache.cpp:742` and `:747`, `SetControlNoZeroWeighted` at `:1196`, `restrictions.h:59`.
  - Worth noting: the *existing* in-repo comment being deleted cites `calc_score_cache.cpp:752` and `:1203-1211`, both wrong. The plan's replacement citations are the correct ones.
- **Required revision:** fix the `fold.h` path in `SPEC.md` §10 (and the two off-by-ones) before TASK-05/TASK-06 transcribe them.

### [MINOR-6] The 507 target omits the ignored count; state it to avoid a false alarm

- **Evidence (RUN):** the HEAD tally is `503 passed / 1 failed / **4 ignored**`. `plan9.md:26,231` and `PLAN.md:349` state the target as "507 passed / 1 failed" with no mention of the 4 ignored tests.
- **Required revision:** state the target as `507 passed / 1 failed / 4 ignored` so a matching run is not misread as a discrepancy.

### [MINOR-7] "ALL trees" is proven only to 3 boosting iterations

- **Evidence:** every in-scope fixture pins `iterations = 3` (`gen_fixtures.py:899` for `bootstrap_dev`; `bootstrap_oracle_test.rs:66`; `plan3.md:161`). The original requirement says "matches on ALL trees".
- **Assessment:** acceptable, because `AC-1`'s unit contract pins the per-call draw count directly (including a 3-consecutive-call accumulation leg), which is what generalizes beyond 3 trees. But it should be stated as an explicit scope limit in `SPEC.md` §2 rather than left implicit under the words "ALL trees".

---

## Implementation Order Review

The dependency DAG (`PLAN.md:99-121`) is acyclic and `plan1…plan9` is a valid topological order. Prerequisites are individually correct: the deletion (01) before every claim that depends on it; the seam (04) before the store (05) since both edit `mvs_sample_weights`; the mirror (06) after both f32 changes; the doc pass (07) after the final numerics; the sign-off (09) last. Two ordering defects:

1. **Wave B must be serialized or worktree-isolated** — see CRITICAL-3 and MAJOR-1. Recommended corrected order:
   `TASK-01` → `TASK-02` (Red in an isolated worktree at `2c14d7f`) → `TASK-03` (Red in the same isolated worktree) → `TASK-04` → `TASK-05` → `TASK-06` → `TASK-08` → `TASK-07` → `TASK-09`.
   Rationale: 02 and 03 must capture their pre-fix Reds *before* TASK-04 perturbs the threshold target; putting 08 before 07 removes the Wave-E build contention (MINOR-3) at zero cost, since 08 writes nothing.
2. **Add a device probe to TASK-02** — see MAJOR-2. TASK-02 changes a rocm-gated code path (`bootstrap_dev_oracle_test.rs:347`) whose new claim nothing measures until TASK-08. That probe has no dependency on TASK-06 (the mirror is a test-file-only change in a different crate), so it can and should run inside TASK-02 while the change is still isolated.

Everything else about sequencing is sound. The mandatory rocm `--no-run` compile in TASK-02 (`plan2.md:200-204`) is a genuinely good catch: only a rocm build type-checks the second `.chain(...)` site.

---

## Potential Bugs (assessed, with verdicts)

- **All-zero-gradient block (`p = 0` ⇒ every object dropped) — correctly scoped out, no new hazard.** I traced it: with all `der = 0` and `lambda = 0`, `calculate_threshold` returns `0.0` (pivot `0.0` → `threshold != 0.0` false → `INFINITY > sample_size` → `large` empty → `(0+0+0)/sample_size = 0.0`); `single_probability(0.0, 0.0)` takes the `else` arm and returns `0.0`; every object gets weight `0` and consumes **no** child-stream draw. Crucially the skipped draws are on the per-block child RNG (`bootstrap.rs:299-300`), which is discarded, so there is **no main-stream phase effect** — the fix cannot interact with this case. The downstream consequence (an all-false `control` → empty split histogram) is pre-existing and unchanged. `plan5.md:253-255`'s "do not harmonise to upstream's `inf`/`NaN` UB" is the right guardrail. **No issue.**
- **f32 weight narrowing vs the `control` threshold `w > f64::from(f32::EPSILON)` — provably cannot interact.** Nonzero MVS weights are `1/p` with `p ∈ (0, 1]` (`single_probability` returns `1.0` above threshold, `der/threshold ≤ 1` below), so `w ≥ 1` — nine orders above `f32::EPSILON = 1.1920929e-7`. Dropped objects store the product `weight * 0.0 = 0.0`, and `0.0_f32` round-trips exactly, so the mask's `false` side is bit-preserved. The upper end is also safe: the `probability > f64::EPSILON` guard at `:319` bounds `w < 4.5e15 ≪ f32::MAX`, so no overflow-to-`inf`. The narrowing is applied to the PRODUCT, not to `probability` or `r`, so the **keep decision is untouched**. `plan5.md`'s assertion 4 and its "narrowing in the wrong place" guardrail correctly identify the one way to get this wrong. **No issue; the plan's claim is sound and I confirm it.**
- **Fixture-family discriminating power — guaranteed by construction, but over-constrained.** `plan3.md` step 1c makes "≥7 of 10 fail with the fix reverted" an executable gate and explicitly fails the task on a 10/10 pre-fix pass, so the power is *tested*, not assumed — a genuine strength. The `P3` trap (`ISOLATING_PARAMS:151-164` pinning `random_seed: SEED` and omitting `boost_from_average`) is real and correctly called out. See MAJOR-5 for the over-constraint.
- **Keep-count flip risk in the device self-oracle (`assert_eq!(kept_dev, kept_cpu)`, `mvs_device_test.rs:224-227`).** I sized it: only the `(rate 0.3, n 200)` fixture has a non-zero target shift (2.384e-8 relative). A flip needs an object's pinned `r` inside a 2.4e-8·p window around `p`; over 200 objects the probability is ~5e-6. The `MVS-S5` narrowing does not affect the keep decision at all. So `P1`'s escalation rule **does hold** and does **not** mask a real defect on the live path — `launch_mvs_weights_resident` is verifiably dead (`runtime.rs:1131 mvs_lambda: None`; no `Some` in `cb-train`). It *does* leave a real documentation gap (MINOR-1) and, more importantly, an unprotected mirror (MAJOR-3).
- **`f64::from(bool)`** — used by both the existing and the new store expression; valid (stabilized `impl From<bool> for f64`), and the current code compiles, so no hazard.
- **Private-helper access from `bootstrap_test.rs`** — `bootstrap_test.rs` is mounted as `bootstrap::tests` (`bootstrap.rs:55-57`), a descendant module, so `use crate::bootstrap::mvs_block_sample_size` on a private `fn` resolves. TASK-04's plan is correct even though no existing test reaches a private item.
- **`block_size as f32` exactness** — `MVS_BLOCK_SIZE = 8192 < 2^24` caps every block; I confirmed `8192` and `3616` are exact in f32. The double narrowing of `sample_rate` (`:294` then inside the helper) is idempotent; documenting it (risk `P4`) is the right call.
- **No existing test breaks on the deletion.** `mvs_full_subsample_is_identity_and_real_subsample_is_importance_weighted` (`bootstrap_test.rs:157-181`) only probes the RNG on the `subsample = 1.0` leg (`:166`), never after the `0.5` call — which is precisely why the defect survived and why `AC-1`'s test is necessary.

---

## Required Plan Revisions

1. **[CRITICAL-1]** Change every `cargo clippy -p cb-train --all-targets` to `--all-targets --no-deps`; record the 15 pre-existing `cb-train` test-target clippy errors (by file and count), plus `cb-oracle/src/model_json.rs:161` and `cb-oracle/src/compare.rs:58`, as baseline reds; restate the gate as "no NEW clippy error", not "clean".
2. **[CRITICAL-2]** Record the exact HEAD output of `check-no-raw-float-sum.sh` (exit 1, 15 reported violations) and `check-no-anyhow.sh` (exit 1, 12 reported violations) as pre-existing reds; change the gate from "the three gate scripts pass" to "the same violation set as the recorded HEAD baseline"; note that `check-source-test-separation.sh` does pass and remains a real gate.
3. **[CRITICAL-3]** Serialize Wave B, or require TASK-02/TASK-03 to capture their controlled-revert Reds in an isolated `git worktree` at `2c14d7f`; forbid `git stash` on `crates/cb-train/src/bootstrap.rs`; correct `PLAN.md` §1.1's ownership table.
4. **[MAJOR-1]** Restate the parallel-safety claim to cover only the post-fix verdict, and require both Reds to be captured with the f32 target absent.
5. **[MAJOR-2]** Add device/CPU split-argmax disagreement to `MVS-S8`'s failure modes; move a device probe of `bootstrap_dev_oracle_test`'s device arm into TASK-02; pre-define the escalation outcome for a device-only 3/3 failure.
6. **[MAJOR-3]** Add one non-device-gated host-arithmetic test to `mvs_device_test.rs` pinning the mirrored target values and the f32-representability of the reference weights.
7. **[MAJOR-4]** Correct `plan6.md`'s `max_div` expectation to "will RISE from ~1e-13 to ~1e-7, still ≥3 orders under `TOL`", and name `kept_dev == kept_cpu` + `max_div ≤ 1e-4` as the load-bearing assertions.
8. **[MAJOR-5]** Relax TASK-03's binding Red to "≥5 of 10, including ≥1 per bias setting", keeping 7/10 and `{3,4,4,4,5,5,4}` as expected-and-recorded.
9. **[MINOR-1]** Permit and require a doc-only note at `mvs_device.rs:145-146`; change "byte-unchanged" to "no executable change".
10. **[MINOR-2]** Amend `MVS-S7`'s postcondition and `AC-7` to exclude this phase's own artifacts.
11. **[MINOR-3]** Serialize TASK-07 after TASK-08 (or forbid the concurrent `bootstrap.rs` write) and drop the "disjoint by construction" claim.
12. **[MINOR-4]** Add a byte-identity restore gate after each controlled revert.
13. **[MINOR-5]** Fix `SPEC.md` §10's `fold.h` path to `catboost/private/libs/algo/fold.h:217` and the two `tensor_search_helpers.cpp` off-by-ones before they are transcribed into production docs.
14. **[MINOR-6]** State the target tally as `507 passed / 1 failed / 4 ignored`.
15. **[MINOR-7]** Record "3 boosting iterations" as the explicit scope limit behind the phrase "ALL trees" in `SPEC.md` §2.

---

## Unverified Items

- **ROCm device outcomes (AC-6, AC-8).** No GPU run was performed by this review, matching research §12 MEDIUM #4. `rocminfo` is present at `/home/user/rocm/opt/rocm/bin/rocminfo`, and the `cb-backend --lib` rocm build convention is corroborated by `device-bootstrap-parity/plan3.md:167`. The `bootstrap_dev` device arm's MVS 3/3 claim remains the phase's single largest unmeasured risk (MAJOR-2).
- **Fixture byte-reproducibility of the new `mvs_seeds` family.** Not testable before generation; `plan3.md:277-283` handles it correctly by freezing the first generation.
- **The exact pre-fix `StageDiverged` values and failing set after the new fixtures are generated.** Depends on the regenerated borders; see MAJOR-5.
- **Post-mirror `max_div` on real hardware.** My ~1e-7 estimate is analytic (from the measured f32 target deltas and the ~6e-8 weight narrowing) and assumes the kernel's 100-iteration bisection resolves the root to `f64` precision, as `mvs_device.rs:62-67` claims; not executed on gfx1151.

---
---

# Plan Check Result — PASS 2 (revised artifacts, v2)

**Verdict:** ISSUES_FOUND
**Goal:** unchanged from pass 1 (MVS ≤1e-5 on all trees for both `boost_from_average` settings, carve-out removed, both f32 transcription fixes mirrored, committed multi-seed × bias fixture family)
**Plan:** `PLAN.md` v2 + `plan1.md`…`plan9.md` (all revised) + `SPEC.md` `spec_version: 2` + `progress.md` v2
**Checked at:** HEAD `2c14d7f1e008ed37e36c6d1fe2d0e5298aec7f1f`, branch `fix/bootstrap-rng-draw-accounting`, working tree clean apart from `.planning/plans/mvs-tree2-parity/` `[VERIFIED: RUN git rev-parse/status]`
**Method:** every pass-1 finding re-checked against the revised text; every load-bearing citation re-verified from scratch (not trusted from pass 1) by reading the file at HEAD, by CodeGraph, or by executing the command.

## Summary

- **12 of the 15 pass-1 findings are genuinely resolved.** MAJOR-1, MAJOR-2, MAJOR-4, CRITICAL-2, MINOR-1/2/3/4/5/6/7 are all properly actioned, and the spec-level corrections are real: I re-verified **every** entry of the new `SPEC.md` §10.1 citation set against `/home/user/cb_instrumented_build/catboost-src` and all fifteen are now correct (`fold.h:217`, `mvs.h:47/48/49`, `mvs.cpp:174/202/210-212/213`, `calc_score_cache.cpp:742-748` + `1196-1204`/mask `:1202`, `tensor_search_helpers.cpp:442-485`/`:456`/`:482`/`:487`, `restrictions.h:59`). `check-no-raw-float-sum.sh` (exit 1, 36 lines, **15** `D-08 violation:` headers) and `check-no-anyhow.sh` (exit 1, 25 lines, **12** `D-14` headers) reproduce the recorded `B8`/`B9` counts exactly, `check-source-test-separation.sh` exits 0, and the D-08/D-14 diff-scoped greps for this phase's files are genuinely EMPTY at HEAD.
- **The remedy for CRITICAL-1 reproduces the very defect it replaces.** The new diff-scoped clippy gate — `… --no-deps --keep-going | grep -E "^\s+--> " | grep -E "src/bootstrap\.rs|…"`, required in **five** task files plus the phase DoD — is **NOT EMPTY at HEAD**: it matches the pre-existing warning `crates/cb-train/src/bootstrap.rs:134:5` ("float has excessive precision"), and the `cb-backend` variant matches `crates/cb-backend/src/kernels/mvs_device.rs:80:15` ("no need to manually implement bit rotation"). Separately, the recorded `B4` baseline is wrong (**10** files / 100 errors, not 14; its own table sums to 104) and none of `plan9.md`'s four set-equality commands produce the recorded numbers (`^error` counts are 110 and 5, not 100 and 4). So the phase still has no executable lint gate.
- **The new `git worktree` mechanism is the right design but is pointed at a 16 GB RAM-backed tmpfs.** `W=/tmp/mvs-red-task02` means cargo builds into `$W/target` on `tmpfs` (16 G total, RAM-backed); a sibling worktree of this same repo carries a **25 GB** target dir. Both controlled-revert Reds — the discriminating-power evidence for `AC-2` and `AC-3` — risk ENOSPC / RAM exhaustion mid-run.
- Two more MAJORs: `plan6.md`'s new mirror test has **no mandated seam**, so its pinned Red (`left: 1200.0000178813934`) is probably unachievable and the `MVS-S4` half of the Red can be vacuous (the exact hole MAJOR-3 was raised to close); and `plan3.md`'s **Observable completion condition still binds ≥7 of 10** while the rest of the file binds ≥5 — MAJOR-5 applied everywhere except the headline.
- Everything else I re-checked held: `bootstrap.rs:413-423` verbatim, `:294/295/306-313/312/315-328/323`, `mvs_sample_weights:281`, `calculate_threshold:209-273`, `bootstrap:383`, `BootstrapResult:88-93`; `bootstrap_dev_oracle_test.rs` `:108-116/118-119/121-155/163-169/174-186/190-191/210-219/225-228/234-237/253/344-347/380`; `mvs_device_test.rs` `:22/27/30/33-35/130-133/138-173/197-234/224-227/236-274/276-300` and the five `cpu_block_threshold` shapes `(0.5,48)/(0.7,64)/(0.3,200)/(0.6,96)/(0.5,8192)+(0.5,24)`; `mvs_device.rs:145-146/302`, `MVS_BISECTION_ITERS:67`; `boosting.rs:59/69/1649/3222/3262/3833`; `device_draw_replay.rs:64`; `runtime.rs:1131 mvs_lambda: None`; `gen_fixtures.py` `:66/67/151-164/710/858/951/3374`, `:898-899`; `cb-train` dev-deps; `kernels.rs:2962-2963` `#[cfg(test)] mod mvs_device_test;` (**confirms the new mirror test really does run on the default `cpu` feature**); 7 baseline `bootstrap::tests::*`; sibling-phase `progress.md:67/69/137/162/166/167` and `SPEC.md:746/793`.

## Per-finding resolution table

| pass-1 finding | verdict | evidence |
|---|---|---|
| **CRITICAL-1** clippy gate red at HEAD | **PARTIALLY RESOLVED — new problems** | `B3`/`B4` recorded and every invocation switched to `--no-deps --keep-going` ✔, but the replacement diff-scoped gate is itself red at HEAD and `B4`'s baseline is wrong → **[C2-1]**, **[C2-2]** |
| **CRITICAL-2** two gate scripts exit 1 | **RESOLVED** | `B8` 15 files/36 lines and `B9` 12 files/12 headers reproduce exactly; gates differential; the `boosting.rs:1649` ↔ `MVS-S9` contradiction named in `SPEC.md` `R5`, `PLAN.md` §4.12, `plan9.md`. Residual trap → **[C2-6]** |
| **CRITICAL-3** Wave B not parallel-safe | **RESOLVED in design; new operational risk** | phase is fully serial (no `parallelizable: true` anywhere ✔), `git stash` on `bootstrap.rs` forbidden in plan2/plan3/PLAN.md ✔, §1.3 now lists the temporary writes ✔ — but the worktree location is unsafe → **[C2-3]** |
| **MAJOR-1** over-broad "cannot invalidate" | **RESOLVED** | withdrawn verbatim in `PLAN.md:189-199`, `plan2.md:56-62`, `plan3.md:56-63`, `plan4.md:47-55`; enforced by the total order (02→03→04) and by the `2c14d7f` worktree where the f32 target is absent by construction |
| **MAJOR-2** device 3/3 unmeasured/binding early | **RESOLVED** | `MVS-S8` gains the split-argmax mode + a pre-defined device-only-residual escalation; `AC-8` marked projected; `plan2.md` §4b probe is mandatory. I re-verified the probe's independence from TASK-06: `bootstrap_dev_oracle_test` has no path to `mvs_device_test.rs` |
| **MAJOR-3** mirror had no test | **PARTIALLY RESOLVED** | the test is specified, is explicitly not gated on `device_backend_active()`, and the mount at `kernels.rs:2962` confirms it compiles+runs under the default `cpu` feature ✔ — but no seam is mandated, so its Red is unreliable → **[C2-4]** |
| **MAJOR-4** `max_div` "should DROP" | **RESOLVED** | corrected in `plan6.md:316-329`, `plan6.md:407-411`, `progress.md:229`; `kept_dev == kept_cpu` + `max_div ≤ 1e-4` named load-bearing |
| **MAJOR-5** Red over-constrained | **PARTIALLY RESOLVED** | `SPEC.md` `MVS-S3`/`AC-3`, `plan3.md` step 1c and completion criteria all now say ≥5 incl. ≥1 per bias ✔ — but `plan3.md:37-38` still says ≥7 → **[C2-5]** |
| **MINOR-1** `P1` recorded only in a test file | **RESOLVED** | comment-only note at `mvs_device.rs:145-146` now REQUIRED (`plan6.md` step 2.4, `PLAN.md` §4.10); gate changed to "no executable change" with a comment-only `git diff` check in `plan6.md` and `plan9.md` |
| **MINOR-2** `MVS-S7`/`AC-7` unsatisfiable | **RESOLVED** | scoped to exclude `.planning/plans/mvs-tree2-parity/` in `SPEC.md`, `AC-7`, `plan7.md`, `plan9.md`; I ran the exclusion grep — it filters correctly. Counts drift slightly (28 hits today vs the recorded 26) and one grep is line-break-blind → **[C2-9]**, **[C2-10]** |
| **MINOR-3** Wave E build contention | **RESOLVED** | TASK-08 `order: 7` / TASK-07 `order: 8`, `depends_on` updated both ways, rationale restated as target-dir lock contention with nil correctness risk |
| **MINOR-4** no byte-identity restore gate | **RESOLVED** | worktree teardown + `git worktree list` + `git stash list` + `git diff --stat` in plan2/plan3/plan9. Wording defect only → **[C2-7]** |
| **MINOR-5** wrong upstream citations | **RESOLVED in `SPEC.md`** | new §10.1 re-verified entry by entry against the upstream tree — all 15 correct. Two stale `:442-486` strings survive in the plans that transcribe them → **[C2-8]** |
| **MINOR-6** ignored count omitted | **RESOLVED** | `507 passed / 1 failed / 4 ignored` in `SPEC.md` `AC-9`, `PLAN.md:445`, `plan9.md:31-38`, `progress.md` |
| **MINOR-7** "ALL trees" unbounded | **RESOLVED** | `SPEC.md` §2 "Scope limit behind the phrase ALL trees" — 3 boosting iterations, with `MVS-S1` named as what generalises |

## CodeGraph / source evidence re-established this pass

- `bootstrap` — `crates/cb-train/src/bootstrap.rs:383`, `pub`; blast radius unchanged: `boosting.rs:3262` (device), `:3833` (CPU), `bootstrap_test.rs`. `boosting.rs:3222` still reads "The device branch keeps the ENTIRE sampler on the host". Impact: the fix moves device and CPU identically — TASK-08's premise holds.
- `mvs_sample_weights:281` / `calculate_threshold:209` / `single_probability:193` — each exactly one caller, all inside `bootstrap.rs`; CodeGraph reports "no covering tests" for all three, so AC-1/AC-4/AC-5's unit tests remain necessary.
- `crates/cb-backend/src/kernels.rs:2962-2963` — `#[cfg(test)] mod mvs_device_test;` with no feature predicate. **This is what makes MAJOR-3's remedy structurally viable**: a non-device-gated test in that file does run under `--features cpu`.
- `cpu_block_threshold` (`mvs_device_test.rs:130`) has **two** in-file callers — `:156` (inside `cpu_mvs_sample`) and `:249` — reached from the three test bodies at `:204`, `:248`, `:284`. The plan's "5 call sites" table is a table of *shapes*, and the shapes are right.
- `mvs_sample_kernel`'s `f64` target at `mvs_device.rs:146` and the host rate-only f32 rounding at `:302` confirmed; `MVS_BISECTION_ITERS = 100` at `:67`, with the file's own comment asserting full-`f64` root agreement — so `plan6.md`'s ~1e-13 → ~1e-7 prediction is sound.
- `crates/cb-compute/src/runtime.rs:1131 mvs_lambda: None` confirmed ⇒ `launch_mvs_weights_resident` remains dead on the live path; `P1`'s blast-radius claim holds.

## Issues

### [C2-1] [CRITICAL] The replacement diff-scoped clippy gate is RED at HEAD for the two files the phase edits most

- **Plan location:** `PLAN.md` §4.12 (`:356-363`) and DoD (`:451-462`); `plan1.md:229-230, 276-279`; `plan4.md:235-236, 273-275`; `plan5.md:204-205, 250-252`; `plan6.md:295-296, 373`; `plan7.md:260-261, 316-318`; `plan9.md:207-212, 305-310`.
- **Requirement:** the pass-1 CRITICAL-1 remedy — an executable lint gate on the phase's own code.
- **Evidence (RUN at `2c14d7f`):**
  - `cargo clippy -p cb-train --all-targets --no-deps --keep-going 2>&1 | grep -E "^\s+--> " | grep -E "src/bootstrap\.rs|bootstrap_test\.rs"` → **`--> crates/cb-train/src/bootstrap.rs:134:5`**. The preceding line is `warning: float has excessive precision`; `bootstrap.rs:131` carries `#[allow(clippy::approx_constant)]` **only**, whereas `:117` carries both `excessive_precision` and `approx_constant` — so `fast_logf`'s literal warns and always has.
  - `cargo clippy -p cb-backend --lib --no-deps --keep-going 2>&1 | grep -E "^\s+--> " | grep "mvs_device"` → **`--> crates/cb-backend/src/kernels/mvs_device.rs:80:15`**, preceded by `warning: there is no need to manually implement bit rotation`.
  - Root cause: the grep filters on `-->` *location* lines, which clippy emits for **warnings as well as errors** (77 such lines for `cb-backend --lib`, of which only 4 belong to errors).
  - By contrast the same-shaped D-08/D-14 greps ARE empty at HEAD (verified: no `bootstrap`/`mvs_device` entry in either script's output), and `plan2.md`'s `bootstrap_dev_oracle_test` grep and `plan3.md`'s `mvs_seeds_oracle_test` grep are also empty. Only the clippy form is broken.
- **Failure scenario:** TASK-01's Refactor runs the gate, sees a hit naming `src/bootstrap.rs`, and the implementer either declares the task blocked, or "fixes" `bootstrap.rs:134` by trimming the upstream-verbatim `0.693_147_18_f32` literal — which `bootstrap.rs:110-116` explicitly says "would change the exact f32 bit pattern and break Bayesian parity at the ~1e-5 oracle bound" — or silently drops the gate, which is exactly the outcome pass-1 CRITICAL-1 was raised to prevent. The same happens at TASK-04, TASK-05, TASK-06, TASK-07 and TASK-09.
- **Impact:** the phase again has no working lint check on its own new code (`as f32` casts, four new test files); five task Refactor steps and the DoD are unachievable as written.
- **Required revision:** make the gate error-attributed rather than location-line-based, e.g.
  `awk '/^error/ {if ($0 !~ /could not compile/) {want=1; next}} want && /^[[:space:]]+--> / {print; want=0}'`
  piped into the path grep; **and** record the two pre-existing warning entries (`crates/cb-train/src/bootstrap.rs:134:5` = `excessive_precision`, `crates/cb-backend/src/kernels/mvs_device.rs:80:15` = `manual_rotate`) in `PLAN.md` §4.11 so that, if the warning form is kept, the criterion reads "no entry other than these two".

### [C2-2] [CRITICAL] `B4`'s recorded clippy baseline is not what the command emits, and none of `plan9.md`'s four set-equality checks can match their targets

- **Plan location:** `PLAN.md` §4.11 rows `B4`/`B5` (`:325-327`) and DoD (`:455-460`); `plan9.md:214-219`, `:305-310`; `progress.md:58-59, 246-249, 320-328`.
- **Requirement:** the "fallback — set-equality" half of §4.12, and `AC-9`'s "no NEW entry relative to its recorded HEAD baseline". This is the only check that would catch a lint regression in a file the phase did **not** plan to touch.
- **Evidence (RUN at `2c14d7f`):**
  - Error-attributed distribution for `cargo clippy -p cb-train --all-targets --no-deps --keep-going`: **100 errors across exactly 10 files** — `tensor_ctr_oracle_test.rs` 31, `device_seam_test.rs` 22, `yetirank_pairwise_tree_rng_oracle_test.rs` 11, `ordered_ctr_oracle_test.rs` 11, `plain_ctr_oracle_test.rs` 8, `ordered_boost_oracle_test.rs` 8, `permutation_oracle_test.rs` 3, `structure_fold_cycle_oracle_test.rs` 2, `s_order_ctr_bins_oracle_test.rs` 2, `learn_set_shuffle_oracle_test.rs` 2 (sums to 100). `B4` says **14 files**, and its own per-file list sums to **104**; the four "+1 each" files (`tensor_ctr_e2e_oracle_test.rs`, `multilabel_oracle_test.rs`, `multiclass_oracle_test.rs`, `ctr_split_scoring_test.rs`) contribute **warnings**, not errors.
  - `… -p cb-train … | grep -cE "^error"` → **110** (100 errors + 10 `error: could not compile … (test "X")` aggregate lines).
  - `plan9.md:217` `cargo clippy -p cb-backend --lib --no-deps --keep-going 2>&1 | grep -cE "^error"  # 4` → actual **5** (4 errors + `error: could not compile 'cb-backend' (lib) due to 4 previous errors`).
  - `plan9.md:215-216` `… | grep -E "^\s+--> " | cut -d: -f1 | sort | uniq -c    # same 14 files, same counts` → actual output is **~40 files** including `crates/cb-train/src/tree.rs` (14), `src/boosting.rs` (3), `src/metrics.rs` (2), `src/device_draw_replay.rs` (2) and even `crates/cb-backend/src/kernels.rs` (2), because warnings are included.
  - `plan9.md:218-219`'s two script counts (`grep -c "^D-08 violation:"` → 15, `"^D-14 violation:"` → 12) **do** reproduce exactly.
- **Failure scenario:** TASK-09 runs the four set-equality commands, gets 110/5/40-files instead of 100/4/14-files, and cannot tell a real regression from a mis-recorded baseline. The DoD line "shows the same 14-file / 100-error set as baseline `B4`" can never be satisfied. The likely outcome is that the implementer rewrites the baseline to whatever the run printed — destroying the differential property.
- **Impact:** `AC-9`'s lint half is unverifiable; scope leakage into an unrelated crate would not be caught.
- **Required revision:** replace `B4` with the measured **10-file / 100-error** error-attributed set; state the `^error` totals as 110 (`cb-train`, incl. 10 aggregate lines) and 5 (`cb-backend --lib`, incl. 1) or use `grep -cE "^error" | ` with `grep -v "could not compile"`; replace `plan9.md:215-216` with the error-attributed extraction from **[C2-1]**. Also add the invocation CI actually runs — `cargo clippy --workspace --lib -- -D warnings` (`.github/workflows/ci.yml:47`) — to §4.11, since that is the command that will judge the phase's new **lib** code and it is already red at HEAD.

### [C2-3] [MAJOR] The throwaway worktree is pointed at `/tmp`, a 16 GB RAM-backed tmpfs; cargo's target dir goes with it

- **Plan location:** `plan2.md:147` (`W=/tmp/mvs-red-task02`), `plan2.md:158` (`cargo test --manifest-path "$W/Cargo.toml" …`); `plan3.md:250` (`W=/tmp/mvs-red-task03`), `:255`.
- **Requirement:** CRITICAL-3's remedy — both controlled-revert Reds, i.e. the discriminating-power evidence behind `AC-2` and `AC-3`.
- **Evidence (RUN at `2c14d7f`):**
  - `df -h /tmp` → `tmpfs 16G` total, 16 G available. tmpfs is **RAM-backed**; `free -h` shows 30 Gi total / 3.3 Gi free / 8 Gi swap with 3.2 Gi already used.
  - No `CARGO_TARGET_DIR` is set anywhere (`env`, `/home/user/.cargo/config.toml`, no repo `.cargo/`), so `cargo test --manifest-path "$W/Cargo.toml"` writes to **`$W/target`** — inside the tmpfs.
  - The build is cold: `/target` is gitignored, so a fresh worktree has none. Real-world size for this repo: the existing sibling worktree `/home/user/Documents/workspace/catboost_rs-worktrees/23-ctr-model-loading/target` is **25 GB**; the main tree's `target/debug` is 588 GB.
  - The checkout itself is cheap (`git ls-files` = 1220 files, `.git` = 14 MB), so the worktree files are not the problem — only the build products are.
- **Failure scenario:** mid-Red, `rustc` fails with ENOSPC on the tmpfs, or tmpfs pages push the machine into swap thrash. The implementer then falls back to the thing the plan forbids (`git stash` on `bootstrap.rs`) or skips the Red and records the pass-1-recorded values as if observed. Pass 1's own warning applies: a Red that cannot run is worse than the stash it replaced.
- **Impact:** the falsifying evidence for `MVS-S2` and `MVS-S3` — the two claims that justify the phase — may never be produced; and the fallback is the exact hazard CRITICAL-3 closed.
- **Required revision:** put the worktree on the disk-backed filesystem, following the repo's existing convention (`/home/user/Documents/workspace/catboost_rs-worktrees/mvs-red-task0N`, on `/home` with 209 GB free), and set an explicit disk-backed `CARGO_TARGET_DIR` for the worktree runs. State the expected cold-build cost so it is not mistaken for a hang, and add a `df` pre-check before `git worktree add`.

### [C2-4] [MAJOR] `plan6.md`'s new mirror test has no mandated seam, so its pinned Red is probably unachievable and the `MVS-S4` half of it can be vacuous

- **Plan location:** `plan6.md:204-214` (assertion 1, "via a tiny local helper mirroring `mvs_block_sample_size` (or by calling `cpu_block_threshold`'s target sub-expression directly **if it is factored out**)"), `:219-227` (expected Red `left: 1200.0000178813934 / right: 1200.0`), completion criterion `:355-358`.
- **Requirement:** `MVS-S6` / `AC-6` and the MAJOR-3 remedy — a falsifiable Red for the mirror.
- **Evidence:** at HEAD `cpu_block_threshold` (`mvs_device_test.rs:130-133`) computes the target inline and returns a *threshold*; there is no observable target seam, and the threshold at `(0.8, 1500)` is not a pinnable constant. So assertion 1 can only fail if a helper containing the **current** `sample_rate * block.len() as f64` expression already exists when the test is written. `plan4.md` mandates exactly that discipline for the CPU side ("**1a. Create the seam (behaviour-identical)** … **1b. The failing test**", `plan4.md:154-166`); `plan6.md` has no equivalent sub-step, and its parenthetical is conditional ("if it is factored out") without saying who factors it out or when.
- **Failure scenario:** the implementer writes a fresh local `fn target(rate, n) -> f64` in the test — naturally with the new `f32` expression, since that is what the task is about — and assertion 1 passes on first run. Assertion 2 (the weight round-trip) still fails, so the Red is not wholly vacuous, but the completion criterion "FAILED before the mirror with `left: 1200.0000178813934, right: 1200.0`" becomes unmeetable and the criterion is either recorded falsely or the `MVS-S4` half of the mirror ships with a test that never could have failed — reopening `R2` for the target expression, which is the half where **4 of 5 device call sites are bit-identical** (re-verified: only `(0.3, 200)` moves, 2.384e-8 relative, against `TOL = 1e-4` at `mvs_device_test.rs:27`).
- **Impact:** MAJOR-3 is only half-closed; the permanent protection the mirror gains covers `MVS-S5` but not reliably `MVS-S4`.
- **Required revision:** add a mandatory `plan6.md` step 1a mirroring `plan4.md`'s: extract the target sub-expression out of `cpu_block_threshold` into a named test-local helper whose body is **the current `f64` expression**, prove the three device oracles and the default-`cpu` run are unchanged, and only then write the Red. State that the helper must be introduced before the test, and that a first-run pass of assertion 1 means step 1a was skipped.

### [C2-5] [MAJOR] `plan3.md`'s Observable completion condition still binds ≥7 of 10 — MAJOR-5 applied everywhere except the headline

- **Plan location:** `plan3.md:35-38` — "**Observable completion condition:** … and with the TASK-01 deletion reverted, **≥ 7 of the 10** scenarios fail."
- **Evidence:** the same file's step 1c (`:268-270`) states "**BINDING criterion (MAJOR-5):** **≥ 5 of the 10** scenarios diverge, including ≥1 per bias setting", its completion criteria (`:360-364`) say ≥5, its risk section (`:388-391`) says "The exact 7/10 set is NOT a gate. If 6 of 10 fail … the task passes", and `SPEC.md` `MVS-S3`/`AC-3` + `PLAN.md` §3 + `progress.md:217` all say ≥5. Only the Observable completion condition — the first thing an implementer reads — still says ≥7.
- **Failure scenario:** the newly committed family (quantized by its own CatBoost run, which `progress.md:287-291` warns may not be reproducible) yields 6 of 10 failing pre-fix. The implementer checks the Observable completion condition, concludes the task failed, and either stalls or regenerates the fixtures to chase 7 — the `R4` failure mode the phase exists to prevent.
- **Impact:** the contradiction re-creates exactly the blocking/ad-hoc-weakening dilemma MAJOR-5 identified.
- **Required revision:** change `plan3.md:37-38` to "**≥ 5 of the 10** scenarios fail, including ≥1 per `boost_from_average` setting (7/10 expected and recorded, not binding)".

### [C2-6] [MINOR] The phase's own D-08 diff-scoped gate breaks if any added `bootstrap.rs` doc text contains `.sum()`

- **Plan location:** `plan7.md:215-219` (deviation (a) doc on `mean_grad_value`, about upstream's blocked reduction vs "a flat ordered `sum_f64`"); the gate at `plan1.md:233`, `plan4.md:238`, `plan5.md:207`, `plan9.md:211`.
- **Evidence (RUN):** `scripts/check-no-raw-float-sum.sh`'s `SUM_PATTERN` is `\.sum\(\)|\.fold\(0\.0|\.fold\(0_f|\.fold\(0f`, applied with `grep -RIlE` to every non-`*_test.rs` `.rs` file in `crates/cb-train` — **comments included**. That is precisely why 12 of the 15 baseline violations are doc comments describing the ban, including `crates/cb-train/src/boosting.rs:1649` ("the sanctioned `cb_core::sum_f64` primitive (D-08 — never a raw `iter().sum()`"). `crates/cb-train/src/bootstrap.rs` is currently clean of the pattern (verified), which is the only reason its diff-scoped D-08 grep is empty.
- **Failure scenario:** TASK-07 writes the natural sentence — "ours is a flat ordered `sum_f64`, never a naive `.sum()`" — and `check-no-raw-float-sum.sh` newly names `crates/cb-train/src/bootstrap.rs`, turning the phase's own diff-scoped D-08 gate red. The implementer then either weakens the documentation `MVS-S10` requires or accepts a red gate.
- **Required revision:** add an explicit instruction to `plan1.md`/`plan4.md`/`plan5.md`/`plan7.md`: doc text added to `crates/cb-train/src/bootstrap.rs` must not contain the literal `.sum()` or `.fold(0.0` — refer to the reduction as `sum_f64` / "a raw iterator summation" — because the D-08 backstop greps comments in non-test source, and `bootstrap.rs` is currently clean.

### [C2-7] [MINOR] "`git worktree list` has no leftover entry" is literally false at HEAD

- **Plan location:** `plan2.md:170, 294-296`; `plan3.md:281-284, 365-367`; `plan9.md:288-289`; `PLAN.md:475-476`.
- **Evidence (RUN):** `git worktree list` already shows **five** other worktrees — `catboost_rs-worktrees/23-ctr-model-loading` plus four `.claude/worktrees/agent-*`.
- **Required revision:** phrase the criterion as `PLAN.md` does — "no leftover **throwaway** worktree (no entry under the path used in TASK-02/TASK-03)" — in plan2 and plan3, and record the 5 pre-existing entries as the baseline so the check is differential like every other gate in this phase.

### [C2-8] [MINOR] The stale `tensor_search_helpers.cpp:442-486` survives in exactly the two places that transcribe it into production doc comments

- **Plan location:** `plan1.md:270` (completion criterion) and `plan7.md:203` (the doc text to write in the `bootstrap.rs` module bullet).
- **Evidence (RUN):** `CalcWeightedData` spans `:442-485`; `:486` is blank and `void Bootstrap(` begins at `:487`. `SPEC.md` §10.1 and `plan1.md:203`, `plan7.md:140`, `plan7.md:307` all correctly say `:442-485`; the two locations above still say `:442-486`.
- **Required revision:** change both to `:442-485` before TASK-01/TASK-07 write them into `bootstrap.rs`.

### [C2-9] [MINOR] Two of the three `MVS-S7` greps are line-break-blind, so `plan7.md`'s incompleteness detector for TASK-02 is vacuous

- **Plan location:** `plan7.md:176-187` (the Red and its expectations).
- **Evidence (RUN):** in `crates/cb-train/tests/bootstrap_dev_oracle_test.rs` the second claim is wrapped across `:143-144` ("the divergence enters when tree" / "2's sample is drawn from that λ") and the third across `:153-154` ("Raise this to 3 once the MVS tree-2" / "sampling gap is fixed"), so `grep -rn "divergence enters when tree 2"` and `grep -rn "MVS tree-2 sampling gap"` return **0 hits in `crates/` already at HEAD**. Measured totals: 15 / 7 / 6 hits for the three phrases, with only `never trees 0 or 1` matching in `crates/` (1 hit, `:141`) and each phrase matching once in `device-bootstrap-parity/progress.md` (`:67`, `:69`).
- **Impact:** `plan7.md`'s "If the third still hits, TASK-02 is incomplete" can never fire; phrase 1 and the `MVS_GATED_TREES` grep in TASK-02 are the real detectors.
- **Required revision:** either normalise whitespace in the gate (`tr -s '[:space:]' ' '` per file, or grep the distinctive fragments `"divergence enters when tree"` and `"MVS tree-2"`), or drop the claim that phrase 3 detects TASK-02 incompleteness and rely on `grep -rn "MVS_GATED_TREES\|MVS_SCENARIO\|gated_trees" crates/`.

### [C2-10] [MINOR] Cosmetic and bookkeeping residue

- `plan5.md` frontmatter still reads `wave: C` while every sibling reads `wave: serial (v2 …)`.
- `SPEC.md` `MVS-S5` scope says the weight store is "currently `:321-326`"; the store is `:323` (`:321` is the conditional draw). `plan5.md:51-68` has it right.
- `SPEC.md` `MVS-S7`/`plan7.md` record "23 of the 26 repo-wide hits"; measured today it is 25 of 28 (the counts move as the artifacts themselves quote the phrases). Immaterial, but it will not reproduce.
- **Filename/rank mismatch (coordinator question 4):** acceptable. `plan7.md order: 8` / `plan8.md order: 7`, `depends_on`/`blocks` are consistent in both directions, and the inversion is signposted in `PLAN.md:133-139`, `progress.md:100-103` and both files' `plan_file_note`. The only consequence of an agent executing in filename order is `plan7.md`'s last completion criterion ("This task ran AFTER TASK-08") turning false; the correctness risk is nil, since TASK-07 is doc-only and TASK-08 writes nothing.

## Implementation Order Review

The revised order is a total order and is correct:

```
TASK-01 → TASK-02 → TASK-03 → TASK-04 → TASK-05 → TASK-06 → TASK-08 → TASK-07 → TASK-09
```

- Prerequisites re-checked individually: 02/03 before 04 (MAJOR-1 — their Reds bind pre-f32 values; the `2c14d7f` worktree makes this belt-and-braces); 04 before 05 (same function); 06 after both f32 changes; 08 after 06 (the rocm run must see the mirrored reference); 07 after 08 (doc-only, removes target-dir contention); 09 last. `PLAN.md` §1.3's ownership table now correctly lists `crates/cb-train/src/bootstrap.rs` as a **temporary** write of TASK-02 and TASK-03, in the worktree — which was the substantive CRITICAL-3 defect.
- TASK-08's Verify runs `--test mvs_seeds_oracle_test` under rocm, which requires TASK-03 (rank 3 < 7) ✔. TASK-02's §4b probe is legitimately independent of TASK-06 ✔ (no dependency path from `bootstrap_dev_oracle_test` to `mvs_device_test.rs`).
- No intermediate state leaves the repo unbuildable: TASK-01 is a deletion; TASK-04's step 1a is a behaviour-preserving extraction; TASK-06's rocm-only exposure between TASK-04/05 and TASK-06 is deliberate and documented.
- Only ordering-adjacent defect is operational, not topological: **[C2-3]**.

## Potential Bugs

- **Pre-existing clippy warnings inside the phase's own edit surface** — trigger: running the §4.12 gate at all; failure mode: gate red for a reason the phase cannot fix without breaking Bayesian parity; mitigation: **[C2-1]**.
- **tmpfs exhaustion during the worktree Red** — trigger: cold `cargo test` in `/tmp`; failure mode: ENOSPC or swap thrash mid-Red; mitigation: **[C2-3]**.
- **Vacuous half-Red on the mirrored target** — trigger: authoring the helper post-fix; failure mode: `MVS-S4`'s mirror ships untested; mitigation: **[C2-4]**.
- **D-08 doc-comment self-trip** — trigger: writing `.sum()` in a new `bootstrap.rs` doc comment; mitigation: **[C2-6]**.
- Re-confirmed sound and unchanged from pass 1 (no new hazard found): the all-zero-gradient block cannot interact with the fix (the skipped draws are on the discarded per-block child stream); the f32 narrowing cannot cross the `w > f64::from(f32::EPSILON)` mask (nonzero weights are `1/p ≥ 1`; the dropped case stores a bit-exact `0.0`; `bootstrap_test.rs:174-179` asserts both and survives); `bootstrap_test.rs` is mounted as `bootstrap::tests` (`bootstrap.rs:55-57`) so TASK-04's private-helper test resolves, and its import block at `:17-19` is the one to extend; `8192` and `3616` are exact in `f32`; the `:294` double narrowing is idempotent; the 507 = 503 + 4 arithmetic is right (7 baseline `bootstrap::tests::*` re-verified by `--list`; TASK-06's mirror test correctly excluded from the `cb-train` tally and added to `cb-backend`'s 173 → 174).

## Required Plan Revisions (pass 2)

1. **[C2-1]** Make the diff-scoped clippy gate error-attributed (pair `^error` with its following `-->`, excluding `could not compile`), and/or record `crates/cb-train/src/bootstrap.rs:134:5` and `crates/cb-backend/src/kernels/mvs_device.rs:80:15` as the pre-existing warning baseline. Update all six locations plus the DoD.
2. **[C2-2]** Correct `B4` to **100 errors across 10 files** with the measured per-file counts; correct the `^error` expectations (110 / 5, or filter the aggregate lines); replace `plan9.md:215-216`'s `uniq -c` command with the error-attributed extraction; add `cargo clippy --workspace --lib -- -D warnings` (ci.yml:47) to §4.11.
3. **[C2-3]** Move both throwaway worktrees to the disk-backed `catboost_rs-worktrees/` convention with an explicit `CARGO_TARGET_DIR`, add a `df` pre-check, and state the cold-build cost.
4. **[C2-4]** Add a mandatory `plan6.md` step 1a: extract the target sub-expression with the **current** `f64` body, prove it behaviour-identical, then write the Red.
5. **[C2-5]** Fix `plan3.md:37-38` to ≥5 of 10 including ≥1 per bias setting.
6. **[C2-6]** Forbid the literal `.sum()` / `.fold(0.0` in any doc text added to `crates/cb-train/src/bootstrap.rs`.
7. **[C2-7]** Restate the worktree-cleanup criterion as "no entry under the TASK-02/TASK-03 worktree path" and record the 5 pre-existing worktrees.
8. **[C2-8]** Fix `plan1.md:270` and `plan7.md:203` to `tensor_search_helpers.cpp:442-485`.
9. **[C2-9]** Make the `MVS-S7` greps whitespace-insensitive, or drop phrase 3's role as a TASK-02 incompleteness detector.
10. **[C2-10]** `plan5.md` `wave:` field; `SPEC.md` `MVS-S5` scope line numbers; the 26-hit count.

## Unverified Items (pass 2)

- **ROCm device outcomes (`AC-6`, `AC-8`).** Still unmeasured; no GPU run was performed by this review. The plan now handles it correctly (projected + early probe + pre-defined escalation), so this is a residual risk, not a plan defect.
- **Fixture byte-reproducibility of the new `mvs_seeds` family**, and the pre-fix failing set it yields. Not testable before generation; handled by freezing the first generation and by the (now-relaxed) ≥5/10 threshold.
- **Post-mirror `max_div` on real hardware.** The ~1e-13 → ~1e-7 prediction remains analytic; `mvs_device.rs:62-67`'s own claim of full-`f64` root agreement supports it.
- **Whether a cold `cargo test -p cb-train --test <one>` build fits in 16 GB.** Not executed. The 25 GB sibling-worktree target dir is strong circumstantial evidence that it does not, which is why **[C2-3]** asks for a disk-backed path rather than a measurement.
- **The `503 passed / 1 failed / 4 ignored` full-suite tally.** Not re-run this pass (pass 1 measured it and the planner re-measured it); the 7-test `bootstrap` lib baseline WAS re-verified by `--list`.

---
---

# Plan Check Result — PASS 3 (FINAL GATE, v3 artifacts)

**Verdict:** ISSUES_FOUND
**Goal:** unchanged — MVS matches upstream on all trees at ≤1e-5 for BOTH `boost_from_average`
settings, `MVS_GATED_TREES` removed, both f32 transcription fixes mirrored into `cb-backend`,
committed multi-seed × bias fixture family.
**Plan:** `PLAN.md` v3 + `plan1.md`…`plan9.md` (all revised) + `SPEC.md` v3 + `progress.md` v3
**Checked at:** HEAD `2c14d7f`, branch `fix/bootstrap-rng-draw-accounting`, working tree clean
apart from `.planning/plans/mvs-tree2-parity/` `[VERIFIED: RUN]`
**Method:** every pass-2 finding re-checked against the revised text; **every** baseline
re-executed from scratch (nothing carried over from passes 1–2); the worktree Red physically
reproduced; the phase's core numeric claim (does the fix actually reach 3/3?) executed
end-to-end in a throwaway worktree; every pinned literal recomputed in Rust.

## Summary

- **9 of the 10 pass-2 findings are genuinely resolved**, and I re-measured every one of
  them rather than trusting the planner's report. `B4` = **100 errors / 10 files** exactly,
  `B5` = **4 errors / 3 files** (`cpu_runtime.rs` ×2), `B7` exit 0, `B8` = 15 headers / 36
  lines, `B9` = 12 headers / 25 lines, `B11` = exit **101** aborting at `cb-data` (and, with
  `--keep-going --message-format=json`, **5 errors / 100 warnings** with **exactly one**
  naming a phase file: `mvs_device.rs:80 clippy::manual_rotate`), `B12` reproduced (×2), `B1`
  = **503 passed / 1 failed / 4 ignored** with the failure in `monotone_oracle_test`. **All
  four §4.12 differential gates are EMPTY at HEAD**, they stay EMPTY after
  `touch crates/cb-train/src/bootstrap.rs` (while the same run emits the two
  `bootstrap.rs:134` warnings, proving the file is linted and not cache-skipped), and the
  `awk` fallback is **byte-identical** to the `jq` form. The `MVS-S7` normalising script
  returns `TOTAL_HITS=5`, exit 1, with all three phrases hitting.
- **I physically reproduced the worktree Red**, byte-for-byte:
  `StageDiverged { stage: Splits, index: 5, expected: -0.025514747947454453, actual: -0.2692405581474304, diff: 0.24372581019997597 }`,
  MVS-only, the other three scenarios at 3/3. Cost **64 s cold / 6.6 GB** into a disk-backed
  `CARGO_TARGET_DIR` on `/home` (208 G free vs the plan's `>= 15G` pre-check). C2-3 is fully
  closed; nothing outside a 4 KB patch file still points at `/tmp`.
- **I independently verified the phase's headline goal is reachable.** In the same worktree I
  applied TASK-01 + TASK-04 + TASK-05 and ran the oracles:
  `[cpu] bootstrap_dev/mvs: splits + leaf values + staged within 1e-5 of upstream over 3/3 trees`
  and `bootstrap_oracle_test` **5/5 green** (the `boost_from_average=true` family). The
  deletion alone (TASK-01, no f32 changes) also reaches 3/3 with the 5 frozen oracles green,
  so B and C are transcription-faithfulness work, not load-bearing for the fix. **The
  diagnosis, the fix and the goal are sound.**
- **It nevertheless cannot be executed as written, because the C2-4 remedy is built on a
  numeric value that is wrong.** The step-1a seam the plan mandates (`f64` body
  `sample_rate * block_size as f64`, called from the test with the literal `0.8`) returns
  **exactly `1200.0`** — so `assert_eq!(mvs_block_sample_size(0.8, 1500), 1200.0)` **passes
  on its first run**, the documented Red text `left: 1200.0000178813934` is unreachable, and
  the assertion the plan says "must PASS both before and after" is the one that actually
  fails. This breaks TASK-04 and TASK-06, and TASK-06's new C2-4 completion criterion
  ("assertion 1 passing ⇒ step 1a was skipped ⇒ redo it") turns it into an unsatisfiable
  loop. See **[C3-1]** — one CRITICAL, execution-blocking, with a two-line fix.
- Four MINORs follow, all safely handleable during implementation.

## Per-finding resolution table (pass-2 findings)

| pass-2 finding | verdict | evidence re-measured this pass |
|---|---|---|
| **[C2-1]** clippy gate red at HEAD | **RESOLVED** | `clippy_error_files` (jq severity filter) used at all 6 task locations + DoD (`plan1:234`, `plan2:266`, `plan3:338`, `plan4:237`, `plan5:206`, `plan6:339-340`, `plan7:319`, `plan8:218`, `plan9:226-227`, `PLAN.md:440-443`). All four gates **EMPTY** (grep exit 1); EMPTY again after `touch bootstrap.rs` while the same run emits `clippy::excessive_precision crates/cb-train/src/bootstrap.rs:134` ×2 → the file IS linted. No `-->`-line form survives anywhere. |
| **[C2-2]** `B4`/`B5` mis-recorded, set-equality unmatched | **RESOLVED** | `B4` → 100 errors / 10 files, per-file counts reproduce **exactly** as tabulated and sum to 100. `B5` → 4 errors / 3 files, `cpu_runtime.rs:696` + `:1025`, `bootstrap_device.rs:230`, `exact_quantile.rs:178`. `awk` fallback → identical 100/10 and identical EMPTY filter. Set-equality withdrawn ✔. `B11` added and reproduced: plain form exit **101** at `cb-data`; json/keep-going form **5 errors + 100 warnings**, one phase file. |
| **[C2-3]** worktree on tmpfs | **RESOLVED** | `plan2:149-152`, `plan3:250-253` use `catboost_rs-worktrees/mvs-red-task0N` + `TD=…/.target-mvs-red` + `df -BG … /home` needing ≥15 G. Measured: `/home` btrfs **208 G** avail, `/tmp` tmpfs 16 G. I created the worktree, ran the Red, measured **6.6 G / 64 s**, tore it down: `git worktree list \| grep -c mvs-red` → **0**, `git status --short` → only the planning dir, `git stash list` → empty. Only `/tmp` use left is a 4 KB `git diff` patch file (`plan2:155-156`) — harmless. |
| **[C2-4]** mirror Red had no mandated seam | **RESOLVED IN FORM — NEW CRITICAL DEFECT** | `plan6.md:201-228` now mandates step 1a (extract `cpu_block_sample_size` with the current `f64` body, prove inert on cpu + rocm, then write the Red), the test calls the helper by name, and `plan6.md:399-402` makes the "first-run pass ⇒ 1a skipped" rule a completion criterion. **But the pinned value is wrong** → **[C3-1]**. `progress.md:438` confirms the planner's C2-4 verification was **"Static:"** only — the literal was never recomputed. |
| **[C2-5]** `plan3` headline still ≥7 | **RESOLVED** | `plan3.md:35-41` now reads "**≥ 5 of the 10** … including ≥1 with `boost_from_average = true` and ≥1 with `false`. (7/10 with `{3,4,4,4,5,5,4}` … **not** binding)". `grep "≥ *7"` across `plan3/SPEC/PLAN/progress` finds only the explicitly-non-binding mentions. |
| **[C2-6]** D-08 doc-comment self-trip | **RESOLVED** | `PLAN.md` §4.13 (`:480-491`) + `plan7.md:260-272` HARD CONSTRAINT block + notes in plan1/4/5, with a mandatory post-edit re-check. Re-verified: `bootstrap.rs` is clean of `SUM_PATTERN` (grep exit 1), which is the only reason its `B8` gate is empty. |
| **[C2-7]** worktree-cleanup criterion false | **RESOLVED** | Rescoped to `git worktree list \| grep -c 'mvs-red'` → 0 in `plan2:198`, `plan3`, `plan9`, `PLAN.md:594`. Measured: **6** pre-existing entries (main + `23-ctr-model-loading` + 4 `.claude/worktrees/agent-*`), `grep -c mvs-red` → **0**. |
| **[C2-8]** stale `:442-486` | **RESOLVED** | `grep -rn "442-486"` → 0 hits in the plan files (only `research.md`, which is unchanged by design and is not transcribed). Re-verified in the upstream tree: `CalcWeightedData` spans `:442-485`, `:486` blank, `void Bootstrap(` at `:487`, `const float* sampleWeightsData` at `:456`, learn-weight multiply `:479-484` statement `:482`. |
| **[C2-9]** line-break-blind `MVS-S7` greps | **RESOLVED** | I ran `plan7.md:190-203`'s script verbatim: **`TOTAL_HITS=5`, exit 1**, hits exactly as documented — all three phrases detective, phrase 3 only in `bootstrap_dev_oracle_test.rs`. |
| **[C2-10]** cosmetic residue | **RESOLVED** | All nine `wave:` fields read `serial`; `SPEC.md:369-371` scopes `MVS-S5` to the single line `:323` with `:321`/`:325-326` named out of scope; `B11` recorded with `.github/workflows/ci.yml:47` verified as `cargo clippy --workspace --lib -- -D warnings`. |

## Evidence re-established this pass (executed, not cited)

| gate / claim | command | measured at `2c14d7f` |
|---|---|---|
| `B1` | `cargo test -p cb-train --no-fail-fast` | **503 passed / 1 failed / 4 ignored**; failing target `--test monotone_oracle_test` ✔ |
| `B4` | `clippy_error_files -p cb-train --all-targets` | **100 errors / 10 files**, per-file counts exact ✔ |
| `B5` | `clippy_error_files -p cb-backend --lib` | **4 errors / 3 files** (`cpu_runtime.rs` ×2) ✔ |
| `B7` | `bash scripts/check-source-test-separation.sh` | exit **0** ✔ |
| `B8` | `bash scripts/check-no-raw-float-sum.sh` | exit **1**, 15 headers / 36 lines ✔ |
| `B9` | `bash scripts/check-no-anyhow.sh` | exit **1**, 12 headers / 25 lines ✔ |
| `B11` | `cargo clippy --workspace --lib -- -D warnings` | exit **101** at `cb-data`; json form 5 errors / 100 warnings / 1 phase file ✔ |
| `B12` | same run, warning level | `bootstrap.rs:134 excessive_precision` ×2, `mvs_device.rs:80 manual_rotate` ✔ |
| §4.12 gate ×4 | jq + `$PHASE` grep | **all EMPTY**, and EMPTY after `touch bootstrap.rs` ✔ |
| awk fallback | `plan9.md:218` / `PLAN.md:457-461` | byte-identical to jq (100 errors / 10 files / EMPTY) ✔ |
| worktree Red | `git worktree add --detach … 2c14d7f`, `MVS_GATED_TREES=3` | `StageDiverged { … index: 5, expected: -0.025514747947454453, actual: -0.2692405581474304, diff: 0.24372581019997597 }`, MVS-only, **64 s / 6.6 G** ✔ |
| **goal reachability** | same worktree + TASK-01/04/05 applied | `bootstrap_dev/mvs … 3/3 trees`; `bootstrap_oracle_test` **5/5** ✔ |
| rocm clippy gate | `clippy_error_files -p cb-backend --features rocm --all-targets \| grep mvs_device` | **EMPTY** (baseline 4 errors / 3 files, none in `mvs_device*`) ✔ |
| rocm build feasibility | `cargo test -p cb-train --features rocm --test bootstrap_dev_oracle_test --no-run` | **exit 0**, 12 m 45 s cold — TASK-02's mandatory probe is buildable ✔ |
| ROCm rig | `rocminfo` | gfx1151 / Radeon 860M, `/dev/kfd` present ✔ |
| upstream citations | `sed` over `/home/user/cb_instrumented_build/catboost-src` | `mvs.h:47/48/49`, `mvs.cpp:174/202/210-213`, `fold.h:217`, `tensor_search_helpers.cpp:442-485/456/479-484/487`, `calc_score_cache.cpp:742/747/1196`, `restrictions.h:59` — **all correct** ✔ |

## Issues

### [C3-1] [CRITICAL — EXECUTION-BLOCKING] The step-1a seam returns `1200.0`, so the pinned Red of TASK-04 **and** TASK-06 is unreachable and the "no-regression leg" is the one that fails

- **Plan location:** `plan4.md:154-160` (step 1a body), `:174-176` (assertions), `:177-183`
  (expected failure), `:185-187` ("assertion 2 must PASS both before and after … if it ever
  fails the helper is wrong in a new way"), `:270` (completion criterion);
  `plan6.md:215-218` (step 1a body), `:226-228` ("If assertion 1 passes on its first run,
  step 1a was skipped … revert it to the `f64` form and re-run"), `:243-245` (assertions),
  `:254-263` (expected failure), `:399-402` + `:404-408` (completion criteria);
  `progress.md:261`, `:268`; `SPEC.md:625` (`AC-4`).
- **Requirement:** `MVS-S4` / `AC-4`, and `MVS-S6` / `AC-6` via the C2-4 remedy — a
  falsifiable Red for both halves of the transcription.
- **Evidence (RUN — recomputed in Rust at `2c14d7f`, not inferred):**

  | call | step-1a body as written<br>`sample_rate * block_size as f64` | body **with** the in-situ narrowing<br>`f64::from(sample_rate as f32) * block_size as f64` | green body<br>`f64::from((sample_rate as f32) * block_size as f32)` |
  |---|---|---|---|
  | `(0.8, 1500)` | **`1200.0`** | `1200.0000178813934` | `1200.0` |
  | `(0.8, 8192)` | `6553.6` | `6553.60009765625` | `6553.60009765625` |
  | `(0.8, 3616)` | `2892.8` | `2892.800043106079` | `2892.800048828125` |

  `0.8_f64 * 1500.0 == 1200.0` **exactly** in IEEE-754 binary64. The documented "before"
  values (`1200.0000178813934`, `2892.800043106079`) are the **in-situ** production values —
  correct for `bootstrap.rs:312`, because `:294` narrows `sample_rate` *before* the
  multiplication (`let sample_rate = f64::from(sample_rate as f32);` — verified at HEAD; the
  same pre-narrowing exists at `mvs_device_test.rs:145`). The extracted helper as specified
  does **not** narrow, and the tests call it with a raw `0.8_f64` literal, so it is only
  behaviour-preserving *at the call site*, never at the test seam.
- **Failure scenario:** the executor performs step 1a exactly as instructed, writes the three
  assertions, and runs. Assertion 1 **passes**. Assertion 2 **fails** with
  `left: 6553.6, right: 6553.60009765625`. In TASK-04 the plan says that assertion is the
  no-regression leg and "if it ever fails the helper is wrong in a new way", so the executor
  hunts a non-existent extraction bug. In TASK-06 the plan says a first-run pass of assertion
  1 *proves* step 1a was skipped and instructs "revert it to the `f64` form and re-run" —
  the helper **is** in the `f64` form, so the instruction is a no-op and the task cannot be
  completed. The likely escapes are all bad: (a) stall/declare blocked; (b) rewrite the pin
  to `6553.6`, destroying `AC-4`'s acceptance value; (c) decide the criterion is wrong and
  transcribe the plan's pre-written `left: 1200.0000178813934` into `progress.md` as if
  observed — a **silent wrong "done"**, and exactly the fabricated-evidence path the phase's
  Red discipline exists to prevent.
- **Second-order impact:** `AC-4` as stated ("the block threshold target is exactly `1200.0`
  at `(0.8, 1500)`") is satisfied by the **unfixed** helper when tested this way, so the
  phase's headline acceptance value for `MVS-S4` is non-discriminating at the point it is
  checked.
- **Impact:** TASK-04 and TASK-06 blocked or falsely signed off; the `MVS-S4` half of the
  `cb-backend` mirror — the half where **4 of 5** device call sites are bit-identical — can
  still ship with a test that never could have failed, reopening risk `R2`, which is the
  entire purpose of the C2-4 remedy.
- **Required revision (two lines, no re-planning needed):**
  1. `plan4.md:160` and `plan6.md:215-218` — the step-1a body must be
     **`f64::from(sample_rate as f32) * block_size as f64`** (narrow first, multiply in
     `f64`). This is bit-identical in situ (the caller's rate is already f32-narrowed at
     `bootstrap.rs:294` / `mvs_device_test.rs:145`, so the extra `as f32` is idempotent —
     `PLAN.md` risk `P4` already says so), and it makes **every** documented value hold
     exactly: assertion 1 fails with `left: 1200.0000178813934 / right: 1200.0`, assertion 2
     passes before and after (`6553.60009765625`), assertion 3 fails with
     `2892.800043106079`. Add one sentence saying the pre-narrowing is what the call site
     already does, so the seam reproduces the *observable* current behaviour rather than the
     literal source text.
     *(Equivalent alternative if the literal-text extraction is preferred: change the three
     test calls to pass `f64::from(0.8_f32)`. Do NOT do both.)*
  2. `SPEC.md:625` (`AC-4`) — state the discriminating case, e.g. "the target is
     `2892.800048828125` at `(0.8, 3616)` and `1200.0` at `(0.8, 1500)`", or note that the
     `(0.8, 1500)` pin is only discriminating against the **f32-narrowed** input. `plan4.md:82-84`'s
     before/after table is already correct and needs no change.

### [C3-2] [MINOR] `plan6.md`'s completion criterion contradicts its own Green step about where the arithmetic lives

- **Plan location:** `plan6.md:409` — "`cpu_block_threshold` uses
  `f64::from((sample_rate as f32) * block.len() as f32)`" — versus `plan6.md:283-287`, which
  says the arithmetic moves into `cpu_block_sample_size` and "`cpu_block_threshold` keeps
  calling the helper, so its `:132` call site no longer carries the arithmetic itself".
- **Failure scenario:** the executor satisfies the criterion literally by inlining the
  expression back into `cpu_block_threshold`, which removes the step-1a seam the test calls
  by name and breaks compilation of the new test — or, worse, duplicates it.
- **Required revision:** restate as "`cpu_block_sample_size` uses
  `f64::from((sample_rate as f32) * block_len as f32)` and `cpu_block_threshold` calls it".
- **Handling:** safe to fix during implementation.

### [C3-3] [MINOR] `plan3.md`'s "warm 4 s" cost is false, because `plan2.md` deletes the shared target dir

- **Plan location:** `plan3.md:253` ("reuse the target dir TASK-02 already warmed") and
  `:271-273` ("**Cost:** … a **warm** build — measured at **4 s**") versus `plan2.md:197`
  and `plan2.md:330` (`git worktree remove --force "$WT" && rm -rf "$TD"`, made a completion
  criterion).
- **Evidence (RUN):** a cold build of that exact target in a fresh worktree with an empty
  `CARGO_TARGET_DIR` took **64 s / 6.6 GB** this pass.
- **Failure scenario:** benign — the executor waits ~60 s instead of 4 s. Only risk is
  mistaking it for a hang, which `plan2.md:172` already warns against but `plan3.md` does not.
- **Required revision:** either drop `rm -rf "$TD"` from TASK-02's teardown (keep it in
  TASK-03's) or restate TASK-03's cost as "cold, ~60 s / 6.6 GB — not a hang".
- **Handling:** safe to fix during implementation.

### [C3-4] [MINOR] No gate in the phase can catch a NEW clippy **warning** in a phase file

- **Evidence (RUN):** §4.12's helper selects `.level=="error"` only — correct for C2-1's
  purpose. But `B11` (`cargo clippy --workspace --lib`, the invocation CI runs with
  `-D warnings`) **never reaches `cb-train` at all**: `cb-backend`'s lib fails with its 4
  denied-lint errors, and `cb-train` depends on it, so no `cb-train` artifact is produced
  (verified: the `--workspace --lib` json contains diagnostics only for `cb-backend`,
  `cb-compute`, `cb-data`, `cb-oracle`, and zero `cb_train` compiler-artifact records).
  `PLAN.md:398`'s statement "`crates/cb-train/src/bootstrap.rs` does **NOT** appear in this
  gate's set (measured before and after `touch`)" is therefore literally true but for the
  wrong reason, and reads as reassurance that it is not.
- **Failure scenario:** TASK-04/05/07 add `as f32` casts and doc text to `bootstrap.rs` that
  trip, say, `clippy::cast_precision_loss` or `doc_markdown`. No phase gate reports it, and
  CI cannot report it until someone unrelated fixes `cb-backend`/`cb-data`. The debt surfaces
  later, attributed to the wrong phase.
- **Required revision:** add one line to §4.12 — a `clippy_warn_files` twin (`.level=="warning"`)
  over `-p cb-train --all-targets` and `-p cb-backend --lib`, filtered by `$PHASE`, whose
  accepted output is exactly `B12`'s two entries. Also correct `B11`'s note to say cb-train is
  never reached rather than implying it is clean.
- **Handling:** safe to add during implementation; note it in `progress.md` if skipped.

### [C3-5] [MINOR] Operational residue

- `git worktree list | grep -c 'mvs-red'` prints `0` **and exits 1**; under `set -e` (or as
  the last command of a task script) that reads as failure. Phrase it as
  `test "$(git worktree list | grep -c 'mvs-red')" -eq 0`.
- The cold ROCm build cost is unstated anywhere in the plan. Measured this pass:
  `cargo test -p cb-train --no-default-features --features rocm --test bootstrap_dev_oracle_test --no-run`
  → exit 0 in **12 m 45 s** cold, and it invalidates the default-`cpu` artifacts in the
  shared `target/` (feature thrash), so TASK-02's §4b probe and TASK-08 each cost a
  double rebuild. Worth one line in `PLAN.md` §5 so a 13-minute wait is not read as a hang.
- `plan8.md`'s `depends_on: [TASK-06]` omits TASK-03, whose fixtures its
  `--test mvs_seeds_oracle_test` rocm run reads. Harmless under the total order
  (TASK-03 is rank 3 < 7), noted for completeness.

## Implementation Order Review

The total order is unchanged and remains correct:

```
TASK-01 → TASK-02 → TASK-03 → TASK-04 → TASK-05 → TASK-06 → TASK-08 → TASK-07 → TASK-09
```

- Re-checked against the DAG in the frontmatter: `depends_on`/`blocks` are mutually
  consistent for all nine files, `order:` is a valid linear extension, and no
  `parallelizable: true` survives.
- 02/03 before 04 is still required (their Reds bind pre-f32 diagnostic values); the
  `2c14d7f` worktree makes it belt-and-braces, and I confirmed the recorded `StageDiverged`
  values are exactly what that state produces.
- 04 before 05 (same function), 06 after both f32 changes, 08 after 06, 07 after 08
  (doc-only, removes target-dir contention), 09 last — all sound.
- No intermediate state leaves the repo unbuildable. I verified the two riskiest
  intermediate states by execution: TASK-01 alone (deletion only) keeps
  `bootstrap_oracle_test` 5/5 **and** already reaches `bootstrap_dev/mvs` 3/3; TASK-01+04+05
  likewise. So the tree is green at every commit boundary from TASK-01 onward.
- **[C3-1] is not an ordering defect** — it is a wrong constant inside two tasks' Red steps.

## Potential Bugs

- **Vacuous / unsatisfiable Red on the mirrored target** — trigger: performing step 1a
  exactly as written; failure mode: assertion 1 passes, assertion 2 fails against an explicit
  "must pass" claim, TASK-06's completion criterion becomes a no-op loop; impact: blocked
  tasks or a fabricated `progress.md` row; mitigation: **[C3-1]**.
- **Re-confirmed sound, no new hazard** (each re-derived this pass, not carried over): the
  all-zero-gradient block cannot interact with the fix (skipped draws are on the discarded
  per-block child stream — `bootstrap.rs:299-300`); the f32 weight narrowing cannot cross the
  `w > f64::from(f32::EPSILON)` mask (nonzero MVS weights are `1/p ≥ 1`; the dropped case
  stores a bit-exact `0.0`); `bootstrap_test.rs` is mounted as `bootstrap::tests`
  (`bootstrap.rs:55-57`) so the private-helper import resolves; `bootstrap()`'s signature is
  `(EBootstrapType, &[f64], f64, f32, Option<f64>, &mut TFastRng64)`, matching plan1's and
  plan5's call texts exactly; `TFastRng64::{from_seed:171, gen_rand:183, advance:192,
  call_count:204, raw_state:221, gen_rand_real1:232}` all exist with the assumed shapes;
  `8192` and `3616` are exact in `f32`; the `:294` double narrowing is idempotent; 503 + 4 =
  507 with the four new `cb-train` tests named (`mvs_bootstrap_consumes_exactly_one_main_stream_draw`,
  `mvs_seeds_cpu_matches_upstream_across_seeds_and_bias`,
  `mvs_block_sample_size_reproduces_upstream_float_expression`,
  `mvs_sample_weights_are_f32_representable`) and TASK-06's mirror test correctly excluded
  (it lives in `cb-backend`).
- **`MVS-S5`'s Red is robust** — `plan5.md:172-185` pins no literal, leads with an
  anti-vacuity count, and pre-defines the "weights happened to be f32-exact" escape. No issue.

## Fencing of the four explicitly-unverified items

| item | fenced? | assessment |
|---|---|---|
| ROCm `AC-8` (device arm 3/3) | **yes** | `AC-8` marked *projected*; `MVS-S8` names device/CPU split-argmax disagreement as a second, distinguishable failure mode; a **pre-defined device-only documented residual** is the accepted outcome, explicitly never a tolerance loosening / `replay_grow_draws` / carve-out reinstatement; probed early in TASK-02 §4b while the change is isolated. I confirmed the rocm target builds (exit 0) and the rig is present (gfx1151, `/dev/kfd`), so the probe is executable. Cannot silently produce a wrong "done". |
| ROCm `AC-6` (device self-oracle) | **mostly** | `P1`'s escalation is "STOP and escalate with the observed object index; never loosen `TOL`, never edit the kernel". No pre-defined acceptable outcome, unlike `AC-8` — but the event is a ~5e-6 keep-flip on a code path that is provably dead on the live path (`runtime.rs:1131 mvs_lambda: None`), so an open escalation is the right call, not a gap. |
| fixture byte-reproducibility (`AC-3`) | **yes** | first generation frozen; binding threshold relaxed to ≥5/10 with ≥1 per bias, consistently in all six places now; a 10/10 pre-fix pass explicitly FAILS the task. Cannot silently pass. |
| post-mirror `max_div` | **yes** | MAJOR-4's correction is in place (`~1e-13 → ~1e-7`, load-bearing assertions named as `kept_dev == kept_cpu` and `max_div ≤ 1e-4`), and the rise is pre-authorised in `progress.md`. |
| *(fifth, and the one that fails)* TASK-06 step-1a inertness | **NO — misfires** | the fence ("assertion 1 passing ⇒ 1a was skipped") fires on a correctly-performed step 1a and cannot fire on an incorrectly-performed one. **[C3-1]**. |

## Does the plan achieve the original goal?

**Yes — verified by execution, not argument.** In a throwaway worktree at `2c14d7f` I applied
TASK-01 (delete `bootstrap.rs:413-423`), TASK-04 (f32 target) and TASK-05 (f32 weight store)
and measured:

- `[cpu] bootstrap_dev/mvs: splits + leaf values + staged within 1e-5 of upstream over 3/3 trees`
  (the `boost_from_average = false` family, previously the 0/5 case) — with `no`, `bayesian`
  and `bernoulli` unchanged at 3/3;
- `bootstrap_oracle_test` **5/5 green** including `bootstrap_oracle_mvs` (the
  `boost_from_average = true` family);
- the same result with TASK-01 **alone**, so the two f32 changes are transcription fidelity,
  not load-bearing — which further de-risks the phase.

So `MVS_GATED_TREES` can be removed and both bias settings hold at ≤1e-5 over the fixtures'
3 boosting iterations (the scope limit `SPEC.md` §2 now states explicitly, with `AC-1`'s
per-call draw contract as what generalises beyond 3). What remains unproven by this review is
the 10-scenario committed family (TASK-03 must generate it) and the device arm (TASK-02 §4b /
TASK-08), both properly fenced.

## Required Plan Revisions (pass 3)

**Execution-blocking — must be fixed before implementation starts:**

1. **[C3-1]** Change the step-1a helper body in `plan4.md:160` and `plan6.md:215-218` to
   `f64::from(sample_rate as f32) * block_size as f64`, and add one sentence explaining that
   the seam must reproduce the call site's *observable* behaviour (whose rate is already
   f32-narrowed at `bootstrap.rs:294` / `mvs_device_test.rs:145`), not the literal source
   text. Then every pinned value in `plan4.md:174-187`, `plan4.md:270`, `plan6.md:243-263`,
   `plan6.md:404-408` and `progress.md:261,268` holds exactly as written. Also tighten
   `SPEC.md:625` (`AC-4`) so its acceptance value is discriminating.

**Safe to handle during implementation (record in `progress.md` either way):**

2. **[C3-2]** Restate `plan6.md:409` in terms of `cpu_block_sample_size`.
3. **[C3-3]** Reconcile TASK-02's `rm -rf "$TD"` with TASK-03's "warm 4 s" claim.
4. **[C3-4]** Add a warning-level twin of the `clippy_error_files` gate whose accepted output
   is exactly `B12`; correct `B11`'s note to say `cb-train` is never reached.
5. **[C3-5]** `grep -c` exit-status phrasing; record the 12 m 45 s cold ROCm build cost in
   `PLAN.md` §5; add TASK-03 to `plan8.md`'s `depends_on`.

## Unverified Items (pass 3)

- **ROCm runtime outcomes (`AC-6`, `AC-8`).** I verified the rig exists (gfx1151, `/dev/kfd`),
  that the rocm `cb-train` test target **builds** (exit 0, 12 m 45 s), and that the rocm
  `cb-backend` clippy gate for `mvs_device` is EMPTY at HEAD — but I did not execute a GPU
  test run. Residual risk, correctly fenced by `MVS-S8`'s pre-defined escalation.
- **The 10-scenario `mvs_seeds` family's pre-fix failing set.** Not generatable before
  TASK-03. Fenced by the ≥5/10 + ≥1-per-bias threshold and by freezing the first generation.
- **Post-mirror `max_div` on real hardware.** The `~1e-13 → ~1e-7` prediction remains
  analytic.
- **`B2`** (`cb-backend --lib` 173 passed / 60 failed) was not re-run this pass; it is a
  pre-existing CubeCL-CPU MLIR baseline unrelated to the phase's edit surface, and `B1`,
  `B4`, `B5`, `B7`, `B8`, `B9`, `B11`, `B12` were all re-measured.
