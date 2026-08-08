---
title: Device CTR Coverage P1 — CTR type and projection completeness — TDD Implementation Plan
spec: ./SPEC.md
research: ./research.md
prior_art: ../gpu-device-coverage-expansion/PLAN.md
status: draft
plan_version: 2
updated_at: 2026-08-08T00:00:00Z
tasks: 25
waves: 8
check: ./PLAN-CHECK.md
check_verdict_pass1: ISSUES_FOUND (1 CRITICAL, 3 MAJOR, 13 MINOR) — all resolved in v2
worktree: /home/user/Documents/workspace/catboost_rs/.claude/worktrees/gpu-borders-shared-sample
branch: worktree-gpu-borders-shared-sample
base_commit: a0a67ec
---

# PLAN.md — Device CTR Coverage P1

This plan converts `SPEC.md` (DCTR-01 … DCTR-20) into 25 independently executable
Red→Green→Refactor tasks. It does **not** re-author the specification's WHAT/WHY —
`SPEC.md` is authoritative and where this plan disagrees with it, §1 says so explicitly
and the task applies the correction.

**v2 (post-check).** `plan-checker` returned ISSUES_FOUND on v1: 1 CRITICAL, 3 MAJOR,
13 MINOR (`./PLAN-CHECK.md`). Every item is resolved here; §8 is the item-by-item
disposition. Two structural consequences: **T00** is new and ordered before every other
gate task (CRITICAL-1 / R-21), and **T22** is new and covers DCTR-20 (MAJOR-1). Old T22
and T23 became T23 and T24.

Planning method: **goal-backward only**. Every task traces to one of `SPEC.md` §6's **ten**
acceptance scenarios via §5's coverage tables. **No GSD skill, command, workflow, agent
or asset was invoked, imported, or depended on in producing this plan.**

---

## 0. Reading order for the implementing agent

1. **§1 — verified corrections.** Six of them change task shape; two (C-1, C-2) would
   otherwise produce a test that cannot compile or cannot fail. **Read before any task.**
2. §2 — global rules (commands, lints, test conventions, CubeCL rules).
3. §3 — waves and the dependency graph.
4. §4 — the 25 tasks (T00 … T24).
5. §5 — coverage tables (DCTR spec → task; acceptance scenario → task).
6. §6 — blockers, unverified assumptions, planner decisions.
7. §7 — phase definition of done.
8. §8 — item-by-item disposition of the checker's 17 findings.

---

## 1. Verification results and corrections

### V-0 — CodeGraph index validity for this worktree (METHOD NOTE, load-bearing)

The CodeGraph MCP index is rooted at the **main checkout**, not this worktree, and warns
on every call. Verified this session:

```
git diff --stat main -- crates/   ⇒  crates/catboost-rs/src/builder.rs
                                     crates/cb-data/src/borders.rs
                                     crates/cb-data/src/borders_test.rs
                                     crates/cb-data/src/lib.rs
```

Exactly **four** files differ from `main`, and **none** of them is in `cb-train`,
`cb-backend`, `cb-compute` or `cb-oracle` — the only crates this phase touches. The
CodeGraph blast-radius and caller data below is therefore valid for this phase.
Independently, **every line number, symbol and signature cited in this plan was
re-verified with `Read`/`grep` against the worktree files**, not taken from CodeGraph.
`git status --short` at planning time: only the untracked `.planning/plans/device-ctr-full-coverage/`.

CodeGraph blast radius, quoted verbatim and re-verified on disk:

| symbol | CodeGraph | re-verified on disk (`grep`, v2) |
|---|---|---|
| `ctr_types_are_device_covered` | 1 caller; **⚠️ no covering tests found** — **FALSE, see V-0b** | `boosting.rs:2355-2366`; sole production call at `:4426`; **five test assertions** at `device_ctr_combo_config_test.rs:132, :144, :156, :163, :170`, imported at `:21` |
| `DeviceCtrColumn` | 8 callers; tests `cb-compute/src/runtime_ctr_averaging_test.rs`, `cb-backend/src/gpu_runtime/session_depth_gt1_test.rs` | struct at `cb-compute/src/runtime.rs:1105-1128`; **8 struct literals**, of which **3 use `..DeviceCtrColumn::default()`** ⇒ **5 need edits** (C-12) |
| `build_device_ctr_config` | 3 callers; tests `device_ctr_combo_config_test.rs` | `boosting.rs:2429-2545` |
| `ctr_covered` | 1 caller; ⚠️ no covering tests found | `session.rs:134-163`, called **twice** — `:1471` and `:1701` (C-14). No covering test **confirmed by grep**. |
| `build_ctr_cindex_columns` | 1 caller; ⚠️ no covering tests found | `session.rs:201-234`, called **twice** at `:1730`, `:1739`. Confirmed by grep. |
| `CtrSearchState` | 1 caller (`begin_inner`); ⚠️ no covering tests found | `session.rs:178-198`, constructed at `:1768`. Confirmed by grep. |

The three remaining ⚠️ markers were **each re-confirmed with a repo-wide `grep`** before
being relied on (R-22). Only the first was wrong — and it was the most load-bearing one.

### V-0b — CRITICAL-1: the gate has FIVE covering assertions; CodeGraph missed them

`[V]` `grep -rn "ctr_types_are_device_covered" crates/ --include=*.rs` and reading
`crates/cb-train/src/device_ctr_combo_config_test.rs:115-173`:

| site | assertion | conjunct pinned |
|---|---|---|
| `:21` | `use super::{build_device_ctr_config, ctr_types_are_device_covered};` | — |
| `:132` | a set containing a **combination** projection must **decline** | arity |
| `:144` | an all-simple **Borders** set must remain **covered** | — (D-04 positive pin) |
| `:156` | `ctr_type = Counter` must **decline** | type |
| `:163` | `prior_denom = 2.0` must **decline** | prior denominator |
| `:170` | `target_border_idx = 1` must **decline** | target border |

Those are **exactly the four conjuncts P1 removes**. The file states at `:124-126`:

> *"This test pins the CLOSED gate deliberately: re-opening it must be a conscious act
> accompanied by a passing `device_ctr_combo_fit_test` (currently `#[ignore]`d), not an
> accident."*

Mounted at `boosting.rs:7406-7408` ⇒ it runs under `cargo test -p cb-train --lib`.
v1 of this plan claimed the predicate had no covering tests, took that from CodeGraph as
verified fact, and built T23's completion criterion on it. **That claim is withdrawn.**

**Root cause — and the general lesson (R-22).** *CodeGraph's test attribution is
unreliable for `#[path]`-mounted in-crate sibling modules.* The module is not a `tests/`
integration target and not a `#[cfg(test)] mod` inside the production file, so the
attribution pass misses it — which is precisely the layout **this repository mandates**
by its source/test-separation rule, so the blind spot is systematic here, not incidental.
⇒ **A "⚠️ no covering tests found" result is never evidence of absence. Confirm every one
with `grep -rn "<symbol>" crates/ --include=*.rs` before using it to justify an edit.**
That rule now governs the three surviving ⚠️ rows above, all re-confirmed.

**Remedy**: **T00** (new, ordered before every gate-touching task) performs the conscious
migration; and `cargo test -p cb-train --lib device_ctr_combo_config_tests` is added to
the Validation block of **T01, T10, T12, T16, T19 and T23** — every task that mutates the
gate expression.

### V-0c — refinement the checker did not have: all four negatives flip at T19, not one each

`[V]` `covered_column(projection, bucket_count)` (`device_ctr_combo_config_test.rs:41-52`)
builds a Borders / `target_border_idx = 0` / `prior_denom = 1.0` column over the
**caller-supplied projection**, and all three sub-assertions of
`a_non_borders_column_still_declines` (`:156`, `:163`, `:170`) pass
`TProjection::from_features(&[0, 1])` — a **combination**. So each of them declines via
the **arity** conjunct as well as via its named conjunct:

| assertion | named conjunct removed by | still declines after that task? | actually flips at |
|---|---|---|---|
| `:163` `prior_denom = 2.0` | T01 | **yes** — arity still rejects `[0,1]` | **T19** |
| `:170` `target_border_idx = 1` | T10 | **yes** — arity | **T19** |
| `:156` `ctr_type = Counter` | T12 | **yes** — arity | **T19** |
| `:132` combination set | T19 | — | **T19** |
| `:144` all-simple Borders covered | never | n/a | never |

⇒ the checker's predicted failure schedule (T01 breaks `:163`, T10 breaks `:170`, T12
breaks `:156`) is **not** what happens: nothing breaks until **T19**, where all four
negatives flip **at once**. This refinement **does not** weaken CRITICAL-1 or excuse T00:
the assertions still silently stop discriminating their named conjunct the moment it is
deleted (a `:163` that passes only because of the arity conjunct is no longer a
prior-denominator pin — it is a **latent vacuous test**), and the `:124-126` contract
still demands a conscious act. T00 exists to convert the guardrail into one that stays
discriminating at every step; §8's disposition records both the finding and why the
remedy is unchanged. It also removes a real hazard the checker's schedule created: an
implementer expecting T01 to turn the binary red would have found it green and concluded
the migration was unnecessary.

### V-1 — `ctr_types_are_device_covered` is exactly as SPEC quotes it `[V]`

`crates/cb-train/src/boosting.rs:2355-2366`, verbatim on disk:

```rust
fn ctr_types_are_device_covered(cols: &[crate::ctr::CtrFeatureColumn]) -> bool {
    !cols.is_empty()
        && cols.iter().all(|col| {
            // ESCALATED (FPP-11): the projection-arity conjunct is RESTORED. …
            col.projection.is_simple()
                && col.ctr_type == crate::ctr::ECtrType::Borders.as_i8()
                && col.target_border_idx == 0
                && col.prior_denom == 1.0
        })
}
```

Sole consumer, `boosting.rs:4424-4426`:

```rust
|| (learning_folds_for_cycle == 1
    && one_hot_bins.is_empty()
    && ctr_types_are_device_covered(&materialized_ctr_features))
```

`CTR_PRIOR_DENOM: f64 = 1.0` at `boosting.rs:2257`, consumed at the single production
materialization call `boosting.rs:2237`. `ctr_splits_for_tree` (`:2278-2293`) also
hard-codes `prior_denom: 1.0` with the `ctr_helper.cpp:50` citation already in place.
⇒ **DCTR-02 is a provable no-op, exactly as specified.**

### C-1 — `train_cat`'s sixth argument is `weights`, NOT `eval_sets` (**CORRECTION**)

`[V]` `boosting.rs:2930-2939`:

```rust
pub fn train_cat<R: Runtime>(runtime, feature_values, feature_borders, cat_columns,
                             target, weights: &[f64], params, staged_out) -> …
```

Every existing device CTR test passes `&[]` in that slot — that is the **weights**
argument, not an empty eval-set list. Eval sets require a **different entry point**:

`[V]` `boosting.rs:2970-2981`:
```rust
pub fn train_cat_with_eval_sets<R: Runtime>(runtime, feature_values, feature_borders,
    cat_columns, target, weights, params, staged_out,
    eval_sets: &[EvalSet], history: Option<&mut EvalMetricHistory>) -> …
```
`[V]` `EvalSet<'a> { feature_values: &'a [Vec<f32>], target: &'a [f64], cat_columns: &'a [Vec<String>] }`
(`boosting.rs:2551-2563`).

**Consequence**: DCTR-11's negative test (T13) **must** use `train_cat_with_eval_sets`
with a genuinely non-empty `eval_sets` slice, or it asserts nothing. A test written
against `train_cat` would pass vacuously (eval sets are structurally empty there) — a
false pass of exactly the R-8 class this phase exists to eliminate. T13 encodes this.

### C-2 — DCTR-12's f32-width detector cannot be a binclf value comparison (**CORRECTION**)

`SPEC.md` §5 DCTR-12 requires "an f64 device sum must FAIL this test". Verified
arithmetic: `online_mean_prefix` (`cb-train/src/ctr/online.rs:298-360`) adds
`class as f32 / divisor` with `divisor = classes.saturating_sub(1).max(1)`, and
`SIMPLE_CLASSES_COUNT == 2` (`online.rs:52`) ⇒ at binclf the added value is exactly
`0.0f32` or `1.0f32`. Both are exact in f32 **and** f64, and a running sum of such
values is an integer ≤ n; f32 represents integers exactly up to 2²⁴. ⇒ **at binclf an
f32 and an f64 accumulator are bit-identical for every reachable n.** A value assertion
at binclf therefore cannot discriminate the width, and a test claiming to would be a
false pass.

**Corrected detector (T14 encodes both, and both are required):**

1. **Output-width pin** (reframed per checker MINOR-3): read the resident
   `ResidentCtrMean.sum` buffer back and assert its byte length is
   `n * size_of::<f32>()` and that it round-trips through
   `bytemuck::cast_slice::<u8, f32>`. **This pins the per-document OUTPUT buffer, not the
   per-bucket accumulator** — a `sums: Array<f64>` bucket history feeding an `f32` output
   would still satisfy it. It is a cheap shape regression guard, **not** the width proof.
   v1 claimed "an f64 accumulator produces `n * 8` bytes"; that rested on an unstated
   assumption that the two share a type. **Claim withdrawn.**
2. **Numeric, synthetic — this is the actual proof**: make the kernel take the divisor as
   a runtime scalar and
   drive the self-oracle with `classes = 4` ⇒ `divisor = 3` ⇒ added values ∈
   `{0, 1/3, 2/3, 1}`, which are **inexact in f32**. Compare against an inline CPU
   reference that accumulates in `f32`. An f64 device accumulator diverges on the
   third-or-later document of any bucket. Production always launches `divisor = 1`;
   the multi-class input is test-only and future-proofs the kernel.
   **The checker measured this detector: 22–41 mismatches per 96 documents at
   `divisor = 3`.** It is real and discriminating; T14's mutation check is what records it.

### C-3 — `cb-backend` must NOT reuse `ECtrType::from_i8` / `is_cpu_supported` (**CLARIFICATION**)

`SPEC.md` §4.7 and research's "Do Not Hand-Roll" say *"reuse `from_i8`/`is_cpu_supported`;
do not hand-roll a second type list."* `[V]` both live in `crates/cb-train/src/ctr/mod.rs`
(`from_i8` at `:96-108`, `is_cpu_supported` at `:155-160`). **`cb-backend` must never
depend on `cb-train`** (`ctr_device.rs:11-14`, `mod.rs:4381-4382`, T-10-04).

⇒ The reuse mandate applies to **`ctr_types_are_device_covered` in cb-train only**
(T22). Inside `cb-backend`, the CTR-type discriminants are **transcribed inline**
(Pattern B), exactly as `resident_cat_feature_weight` and `calc_hash` already are. Every
cb-backend transcription must carry a comment naming `cb_train::ctr::ECtrType`'s
discriminants (`0 Borders, 1 Buckets, 2 BinarizedTargetMeanValue, 3 FloatTargetMeanValue,
4 Counter, 5 FeatureFreq`, `[V]` `ctr/mod.rs:96-108`) and the upstream `restrictions.h`
citation, so the drift risk is documented rather than silent.

### C-4 — `target_border_idx` is `usize` in cb-train, `u32` on the seam (**CORRECTION**)

`[V]` `CtrFeatureColumn.target_border_idx: usize` (`ctr/ctr_feature.rs:80`);
`SPEC.md` §4.1 specifies `target_border_idx: u32` on `DeviceCtrColumn`. The conversion in
`build_device_ctr_config` must be `u32::try_from(col.target_border_idx).map_err(|_| CbError::OutOfRange(…))?`
— never `unwrap`, and **never** a silent `unwrap_or(0)` (which would map an out-of-range
Buckets selector onto the `b = 0` numerator: a silent wrong answer). The same rule
applies to `projection_members`, built from `col.projection.cat_features()`
(`[V]` `projection.rs:130-132`, returns a **sorted, deduped** `&[usize]` — sortedness is
guaranteed by `from_features` at `:121-126` and `single` at `:111-115`, so
`SPEC.md` §4.1's "SORTED" requirement holds by construction and needs no re-sort).

**Identifier-space note (checker MINOR-12b, must be documented in T02)**:
`projection_members` carries **absolute** cat-feature indices
(`col.projection.cat_features()`), while its sibling `member_bins` is resolved into
**CTR-eligible position** space via `eligible_absolute.iter().position(…)`
(`boosting.rs:2487-2500`). The two fields therefore live in **different index spaces on
the same struct**. That is *self-consistent* for T17's use — the eligibility predicate
only ever compares `projection_members` against other `projection_members`, never against
a bin index — but it is a genuine trap for any future consumer, and T02's doc comment must
say so explicitly.

### C-5 — tree-lifetime scope for D-1 is FREE, and the R-2 failure mode has an e2e detector `[V]`

`grow_oblivious_tree_resident` (`cb-backend/src/gpu_runtime/mod.rs`, pass C at
`:4890-4944`, bookkeeping at `:4980-5005`) is invoked **once per tree**. A
function-local `let mut chosen_ctr_projections: Vec<Vec<u32>> = Vec::new();` is therefore
**tree-lifetime by construction** and resets for free. The R-2 failure mode (conflating
it with the fit-lifetime `group_used`, which lives on `CtrSearchState`,
`session.rs:178-198`, and is passed in as `&'a mut Vec<bool>`) is only reachable if the
implementer deliberately hoists the list onto the session.

**Detector**: the `ctr_device_combo` e2e (T19). Under a fit-lifetime list, tree 0 chooses
simple `[0]` at level 0 and combination `[0,1]` at level 1, so at **tree 1 level 0** the
combination would become eligible — a candidate upstream structurally cannot have
(`AddTreeCtrs`'s `baseProj.IsEmpty()` skip). T19's Verify **mandates** this as an
executed mutation check.

### C-6 — the CubeCL error-guide path in CLAUDE.md/AGENTS.md is wrong (**CORRECTION**)

`[V]` `ls /home/user/Documents/workspace/cubecl_manual/manual/Cubecl/` — there is **no**
`cubecl_error_guideline.md`. The real error guidance is the directory
`/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/cubecl_error_solution_guide/`
containing `mismatched types.md` and
`calling a "normal" Rust function from inside a cube macro function fails in CubeCL.md`.

The project mandate ("on ANY CubeCL build error, load the error guideline before
attempting a fix") stands unchanged — only the path is corrected. Both are restated in
§2.4 and in every kernel task.

### C-7 — `ctr_covered`'s border-table shape check is UNCHANGED for all four types `[V]`

`session.rs:149-162` requires `!col.member_bins.is_empty() && col.borders.len() + 1 == n_bins`
for every structure **and** averaging column. Buckets and Counter both quantize through
`calc_ctr_online_bin` on the CPU (`ctr_feature.rs:327-340`, the `quantize_in_f32 == false`
arm), and BTMV's f32 quantizer was measured bit-identical to the f64 border table for
every prior in `[0,1]` (research spike Q2, 4,504,501 pairs/prior). ⇒ **No per-type
special case in the shape check.** What `ctr_covered` gains is a per-type *admission*
list (C-3's inline transcription), nothing else. This keeps T09/T12/T16 to one conjunct
each.

### C-8 — `run_device_tests.sh` currently lists 23 binaries; T23 owns the file `[V]`

`[V]` `run_device_tests.sh` `TESTS=(…)` holds exactly 23 names plus one `PERF_TESTS`
entry (`kernels::poisson_bootstrap_speed_test`). Four new binaries land in this phase
(`device_ctr_buckets_fit_test`, `device_ctr_counter_fit_test`, `device_ctr_btmv_fit_test`,
`device_ctr_type_gate_test`). To avoid four-way merge conflicts on one file, **T23 is the
sole owner of `run_device_tests.sh`**; every other task verifies its binary directly with
an explicit `--test` invocation.

### C-9 — `build_device_ctr_config` already groups on `(ctr_type, projection)` `[V]`

`boosting.rs:2517-2524`:
```rust
let key = (col.ctr_type, col.projection.clone());
let group = match group_keys.iter().position(|g| *g == key) { … };
```
`target_border_idx` is **not** in the key, and `bucket_count` comes straight from
`col.bucket_count` (the projection cardinality). ⇒ **DCTR-07's grouping requirement is
already satisfied structurally**; T09's Red is a *characterization* test proving it, and
its Green is the seam population only. If that test passes on first write, T09 must
execute the mutation check in §2.5.

### C-10 — pass C's `eligible_max` is one expression `[V]`

`mod.rs:4895`:
```rust
let eligible_max = cs.bucket_counts.iter().copied().max().unwrap_or(1).max(1);
```
with the comment *"simple columns are always eligible — the device gate admits only
simple projections"*. That comment becomes false at T19 and must be corrected in T18.
The CPU counterpart to mirror is `eligible_max_bucket_count` (`cb-train/src/tree.rs:2919-2933`),
which filters on `c.projection.is_simple() || combination_ctr_eligible(&c.projection, used_projections)`.

### C-11 — T09's test cannot compile as v1 specified (**CORRECTION**, checker MAJOR-2)

v1 said to mount `session_ctr_type_test.rs` "next to `session_depth_gt1_test.rs`'s
existing mount in `gpu_runtime/session.rs`". `[V]` that grep returns **nothing**: the
mounts are `crates/cb-backend/src/gpu_runtime/mod.rs:768` (`mod session_residency;`) and
`:775` (`mod session_depth_gt1_test;`). `[V]` `ctr_covered` is a **private** free function
inside `mod session` (`session.rs:134`, no `pub`), so a module mounted in `mod.rs` is a
**sibling** of `session` and cannot name it ⇒ **`E0603`**.

⇒ **Mount `session_ctr_type_test.rs` inside `session.rs`**, at the file's end, as
`#[cfg(test)] #[path = "session_ctr_type_test.rs"] mod session_ctr_type_test;` — the same
in-module-sibling shape `boosting.rs:7406-7408` uses. **Keep `ctr_covered` private**; do
**not** widen it to `pub(crate)` to satisfy a mis-placed test. `session_depth_gt1_test`'s
`mod.rs` mount is **not** the pattern to copy here — it works only because that file
touches public API.

### C-12 — `DeviceCtrColumn` literal count (**CORRECTION**, checker MINOR-4)

v1 said 9 literals. `[V]` `grep -rn "DeviceCtrColumn {" crates/ --include=*.rs` returns
**9 lines, one of which is the struct definition** (`cb-compute/src/runtime.rs:1105`) ⇒
**8 literals**. Of those, **3 use `..DeviceCtrColumn::default()`**
(`session_depth_gt1_test.rs:629, :698, :779`) and compile unchanged.
**Exactly 5 sites need edits**: `session_depth_gt1_test.rs:501`, `:553`, `:630`,
`cb-compute/src/runtime_ctr_averaging_test.rs:14`, `cb-train/src/boosting.rs:2519`.
The three `..Default::default()` sites yield `projection_members: vec![]` — harmless while
the gate admits only simple projections, but T02 must still add its
`!projection_members.is_empty()` production invariant, and those three literals must be
re-reviewed once T19 lands (T22's differential is what would catch a bad default).

### C-13 — line drift in v1 (**CORRECTIONS**, checker MINOR-5)

| v1 said | verified |
|---|---|
| pass C loop `mod.rs:4919` | **`mod.rs:4907`** (`for c in 0..cs.n_ctr {`) |
| `let eligible_max = …` `mod.rs:4895` | correct |
| — | `let max_bucket_count = eligible_max.max(phantom_max).max(1);` is **`mod.rs:4905`** |
| `#[ignore]` at `device_ctr_combo_fit_test.rs:180-181` | **`:179`** (its doc block is `:169-178`) |
| `ordered_ctr_prefix_kernel` fn at `ctr_device.rs:134-167` | `#[cube(launch)]` at `:134`, **`fn` at `:135`**, body to `:167` |

### C-14 — `ctr_covered` has TWO callers, and the second is a safety property (checker MINOR-1)

`[V]` `session.rs:1471` (the `None` grow-policy arm's coverage disjunction) and
`session.rs:1701` (the CTR augmentation arm). v1 named only `:1701`.

`:1471` is load-bearing for this plan's whole test strategy and v1 never stated it: if
`ctr_covered` rejects a config whose `config.ctr` is nonetheless `Some`, the
`if !config.is_covered_regime() && !exact_covered && !bootstrap_covered && !ctr_is_covered
&& !mvs_covered && !host_sample_covered { return Ok(None); }` block at **`session.rs:1505`**
declines the **entire fit**. ⇒ a cb-train-gate / cb-backend-list mismatch can **never**
silently drop CTR columns onto a device fit; it degrades loudly to `grown == 0`.
**That is precisely why `CountingGpu.grown.get() == params.iterations` is sufficient
evidence** for T10/T12/T16/T19, and why T09/T12/T16 may widen `ctr_covered` and the
cb-train gate in separate hunks without a torn intermediate state. Recorded in T09.

### C-15 — one smoke-test file per fixture (**CORRECTION**, checker MINOR-10)

v1 had T05/T06/T07 share `cb-oracle/tests/ctr_device_fixture_smoke_test.rs`, then told
each to "coordinate or rebase" — contradicting §3's claim that they are parallel.
`[V]` `ls crates/cb-oracle/tests/` shows the shipped convention is **one file per
fixture**: `ctr_device_combo_fixture_smoke_test.rs`, `ctr_device_mixed_fixture_smoke_test.rs`,
`weighted_device_fixture_smoke_test.rs`, `bias_device_fixture_smoke_test.rs`,
`exact_leaf_device_fixture_smoke_test.rs`. ⇒ T05/T06/T07 each create
`ctr_device_{buckets,counter,btmv}_fixture_smoke_test.rs`. The contradiction disappears
and the three are **genuinely parallel**.

### C-16 — T18 must NOT drop `phantom_max` (checker MINOR-12d)

`[V]` device `mod.rs:4905`: `let max_bucket_count = eligible_max.max(phantom_max).max(1);`
`[V]` CPU `tree.rs:3117-3125` calls `max_bucket_count_with_phantom(matrix, ctr_features,
chosen, n_objects, eligible_max, cat_eligible_buckets, &leaf_of)` with the **already
filtered** `eligible_max` (`tree.rs:3116`), and that function ends
`eligible_max.max(phantom_max).max(1)` (`tree.rs:3033`).

⇒ **The eligibility filter applies to `eligible_max` ONLY; the phantom mixed-projection
count is folded in OUTSIDE it.** T18 replaces the *inner* max and must leave
`.max(phantom_max).max(1)` untouched. Filtering the phantom too would diverge from the CPU
in the opposite direction. Stated as an explicit invariant in T18.

### C-17 — T04's `norm == 0.0` guard is dead code (checker MINOR-12c)

`[V]` `calc_normalization(p) = (-min(0,p), max(1,p) - min(0,p))` ⇒ `norm = right - left`
with `right >= 1` and `left <= 0` ⇒ **`norm >= 1` always**. `calc_ctr_online_bin`'s
`if norm == 0.0 { return 0.0; }` (`calc_ctr.rs:99-101`) is unreachable.
⇒ T04 may **either** omit the guard **or** mirror it with a comment saying it is
shape-mirroring and unreachable. It must **not** be presented as a live safety check.
(v1 said "guard `norm == 0.0` the same way" without qualifying it.)

### C-18 — a one-hot × CTR device decline test already exists (checker MINOR-7)

`[V]` `crates/cb-train/tests/device_fpp_composition_test.rs:255`
`pub fn one_hot_x_ctr_still_declines()`, driven by `fpp17_one_hot_x_ctr_still_declines`
at `:381-385`. ⇒ T21 **re-runs and records** it rather than authoring a third copy; its
own new work is the *remaining* boundary pins plus the DCTR-03 doc rationale.

---

## 2. Global rules for every task

### 2.1 Commands (exact; every deviation below is a documented trap)

| purpose | command |
|---|---|
| compute-crate types | `cargo check -p cb-compute` |
| backend build-verify (no GPU needed) | `cargo check -p cb-backend --no-default-features --features rocm` |
| **backend device self-oracles** | `cargo test -p cb-backend --no-default-features --features rocm` |
| backend self-oracle, filtered | `cargo test -p cb-backend --no-default-features --features rocm --lib kernels::ctr_device_test` |
| cb-train device test, build-verify | `cargo check -p cb-train --no-default-features --features rocm --test <name>` |
| **cb-train device test, real run** | `cargo test -p cb-train --no-default-features --features rocm --test <name>` |
| CPU oracle (default features) | `cargo test -p cb-train --test <name>` |
| workspace non-device regression | `cargo test --workspace` |
| **full device DoD suite** | `bash ./run_device_tests.sh` → 23/23 PASS + perf lane (baseline at `a0a67ec`) |

**Traps — never write a task command in any other form:**

- **R-9**: NEVER `--features rocm` without `--no-default-features`. `cb-backend`'s
  `default = ["cpu"]` stays on, `SelectedRuntime` resolves to `cubecl::cpu::CpuRuntime`
  (`cb-backend/src/lib.rs:63-64` wins the `cfg` chain over the rocm arm at `:82-83`), and
  "device" tests compare cpu-vs-cpu — a **false pass**.
- **R-10**: NEVER a package-wide `cargo test -p cb-train --no-default-features --features rocm`
  without `--test <name>`. ~52 test files `use cb_backend::CpuBackend;` at module scope
  and fail `E0432`, blocking the whole run (documented in `run_device_tests.sh:2-4`).
  Multiple `--test <name>` flags in one invocation are fine and are how the DoD suite runs.
- **R-13**: `kernels::poisson_bootstrap_speed_test` is a **known, do-not-chase** flake
  under concurrent GPU load (3.3–3.8× vs a 5× bar in-suite; 9.6–10.7× isolated). It lives
  in `run_device_tests.sh`'s isolated `PERF_TESTS` lane. Never chase it.

### 2.2 New-test conventions (mandatory; verified against `device_ctr_gate_test.rs`)

Every new `crates/cb-train/tests/device_*.rs` file MUST:

1. carry `#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing, clippy::float_cmp)]`;
2. put every device-touching item inside `#[cfg(any(feature = "rocm", feature = "cuda"))] mod device { … }`;
3. **never** `use cb_backend::CpuBackend` (not compiled under rocm/cuda; breaks `--no-default-features`);
4. mark shared helpers `#[cfg_attr(not(any(feature = "rocm", feature = "cuda")), allow(dead_code))]`;
5. **SKIP by printing and returning** on cpu/wgpu (`eprintln!("SKIP …: needs rocm/cuda")`)
   — never `panic!`, never `#[ignore]`;
6. **assert device commitment via `CountingGpu`** — `grown.get() == params.iterations`.
   Copy the wrapper **verbatim** from `crates/cb-train/tests/device_ctr_gate_test.rs:60-170`.
   `[V]` its `Runtime` impl overrides `compute_gradients`, `begin_device_training`,
   `grow_tree_on_device(&self, approx, target, sample, family: Option<&FamilyTreeArgs<'_>>)`
   and `end_device_training`, forwarding each to `self.inner: GpuBackend` and incrementing
   `grown: Cell<usize>` only when `grow_tree_on_device` returns `Some`.
   **`oblivious_trees.len() == iterations` is NOT a device-commit assertion** — the CPU
   oblivious grower satisfies it too. That is precisely what makes today's
   `device_ctr_combo_fit_test` a false pass (R-8).

Every new e2e test also keeps the structure-vs-averaging permutation divergence guard
(`[V]` `device_ctr_combo_fit_test.rs:112-121`):
```rust
assert_ne!(create_shuffled_indices(n, seed), averaging_ctr_permutation(n, 1, seed));
```
so a structure-only leaf gather cannot pass.

### 2.3 Source rules

- **Source/test separation is mandatory.** No `#[cfg(test)] mod tests` in any production
  file. Tests live in a dedicated `*_test.rs` mounted via `#[path = "…"] mod …;`
  (`[V]` the pattern at `boosting.rs:7407-7408`, `kernels.rs:3147-3149`,
  `ctr/mod.rs:44-53`) or in `crates/<pkg>/tests/*.rs`.
- No `unwrap` / `expect` / `panic` / raw indexing anywhere in `crates/cb-backend/src/**`
  (workspace lints + D-13). Use `.get(..).copied().unwrap_or(..)` / `let … else`.
  `unwrap_or` is permitted; a **silent** `unwrap_or` on a value whose fallback is
  semantically wrong (C-4) is not — return a typed `CbError` instead.
- Libraries use `thiserror` (`CbError`); no `anyhow` in `crates/cb-*`.
- Any host float reduction routes through `cb_core::sum_f64`.
- **`cb-backend` must never depend on `cb-train`**; **`cb-compute` must stay `cubecl`-free**.
- Preserve D-04 (byte-unchanged CPU path for anything not newly admitted), D-05, D-06
  (CTR values never touch the host — only the final `u32` bin column is read back),
  D-07, D-10-01, WR-02, SPEC-OH-26, SPEC-CTRB-01.

### 2.4 CubeCL rules for the three kernel tasks (T08, T11, T14)

Before writing any kernel code the implementer **must** read
`/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md`. On **any** CubeCL
build error the implementer **must** load the error guidance **before** attempting a fix
— blind fixes are prohibited. Per **C-6** the real path is the directory
`/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/cubecl_error_solution_guide/`
(`mismatched types.md`, `calling a "normal" Rust function from inside a cube macro function fails in CubeCL.md`);
CLAUDE.md/AGENTS.md's `cubecl_error_guideline.md` does not exist.

Carried forward from the research's own manual consultation:

- **`Cubecl_conditionals.md`** — use `if` / `else` **statements**, never `if`-expressions
  inside a `#[cube]` body. Declare the result with `let mut x = <default>;` and assign in
  each arm.
- **`Cubecl_loop_control.md`** — `while` with an **explicit counter**; no iterator
  adapters, no `for … in range` sugar beyond what the existing kernels already use.
- **generics-float** is mandatory for new kernels (`<F: Float>`), with one **documented**
  exception: any buffer whose **width is a parity contract** stays concrete. That is
  exactly BTMV's `Array<f32>` accumulator (`TCtrMeanHistory::Sum` is `float` upstream) —
  it must carry a comment saying so and citing `online.rs:294`'s
  `btmv_sum_is_accumulated_in_f32_not_f64`. The pre-existing concrete-`f64`
  `ordered_ctr_prefix_kernel` (`ctr_device.rs:134-167`) is a documented prior deviation;
  do not widen it, and do not "fix" it in this phase.
- No `-inf` literal in any `#[cube]` body (Pattern D).
- Every device index must derive from a **host-validated** bound, mirroring
  `launch_ordered_ctr_resident`'s existing bin/class guards (`ctr_device.rs:251-260`).

### 2.5 Mutation check for green-on-write tests (mandatory)

Several tasks add tests that **pass on first write** because they pin behaviour that is
already correct (C-9's grouping, the DCTR-03/DCTR-18 negative tests, the acceptance-9
boundary pins). A green-on-write test proves nothing until it is shown to be
discriminating. For every such test the implementer **must**:

1. temporarily invert the production behaviour it pins (delete the conjunct, flip the
   key, hoist the scope);
2. confirm the test **FAILS**, and record the observed failure message in the task's
   completion note;
3. revert the mutation and confirm the test is green again.

This is the executable form of R-8's prevention rule. A task whose completion note lacks
the recorded mutation failure for its green-on-write tests is **not** done.

**Isolation rule (checker MINOR-9).** A §2.5 mutation edits **production** code, so while
one is live the tree is deliberately broken. **No mutation may be live while any other
task runs a workspace-wide or suite-wide validation** (`cargo test --workspace`,
`bash ./run_device_tests.sh`, `cargo test -p cb-backend --no-default-features --features rocm`).
Mutate → run **only** the single focused test → record → revert → re-run the focused test
green, before handing off. This matters most for **T20**, whose mutation (`&& false` in
`ctr_types_are_device_covered`) targets the **same expression T01 edits** — which is why
T20 is now serialised **after** T01 rather than sharing Wave 0 with it.

**Mutation candidates in this plan (nine).** T03, T09-B, T13, T14-B, T15, T19 (×2), T20,
T22 (×2 — see its two mandatory mutations; this is why the isolation rule below scopes
over the `{T21, T22}` parallel set),
T21 (per pin), T23. T15 is listed explicitly per the checker's judgement request — see
that task for why its `≥2 distinct bins` non-vacuity guard is an accepted substitute for a
full mutation.

### 2.6 Frozen-fixture rule (R-12)

CatBoost quantization is run-to-run nondeterministic on categorical routing. Every
fixture in this phase is generated **once**, offline, against the installed
`catboost==1.2.10`, and **never regenerated in CI**. Every `config.json` carries the
`"note": "FROZEN…"` marker and the reproducibility caveat docstring. Regenerating any
fixture invalidates the ≤1e-5 gate for every downstream test.

---

## 3. Waves and dependency graph

Ownership reality: Tracks A, C and B all edit **`cb-backend/src/kernels/ctr_device.rs`**,
**`cb-backend/src/gpu_runtime/session.rs`** and **the same gate expression** in
`boosting.rs`. They are therefore **serial**, not parallel, whatever the track structure
suggests. The genuinely parallel work is the three fixtures, the Track E characterization,
and the `device_ctr_fit_test` hygiene fix.

```text
WAVE 0 — foundations and frozen artifacts (fully parallel: disjoint files)
  T00 (R-21 / DCTR-18a)  device_ctr_combo_config_test.rs — the conscious gate-pin migration
                         *** must precede EVERY gate-touching task ***
  T02 (DCTR-01)          cb-compute seam + build_device_ctr_config + 5 ctor sites
  T03 (DCTR-05)          cb-train ctr/calc_ctr_test.rs — Track E inertness proof
  T05, T06, T07          cb-oracle fixtures {buckets, counter, btmv} (one smoke file each, C-15)

WAVE 1 — gate head and Track E fix (parallel with each other)
  T01 (DCTR-02)  needs T00        boosting.rs gate — HEAD of the serial gate chain
  T04 (DCTR-04)  needs T03        the CPU BTMV normalization fix

WAVE 2 — Track A head + the hygiene fix (parallel: different files)
  T08 (DCTR-06)  needs T02        ctr_device.rs Buckets numerator
  T20 (DCTR-19)  needs T01        device_ctr_fit_test CountingGpu   [serialised after T01
                                  per MINOR-9: its mutation targets T01's expression]

WAVE 3 — Track A tail
  T09 (DCTR-07) -> T10 (DCTR-08)
     T09 needs T02,T08;  T10 needs T00,T01,T05,T09

WAVE 4 — Track C (Counter)
  T11 (DCTR-09) -> T12 (DCTR-10) -> T13 (DCTR-11)
     T11 needs T08 (file order);  T12 needs T06,T09,T10,T11;  T13 needs T12

WAVE 5 — Track B (BTMV)
  T14 (DCTR-12) -> T15 (DCTR-13) -> T16 (DCTR-14)
     T14 needs T04,T11;  T15 needs T14;  T16 needs T07,T12,T14

WAVE 6 — Track D (combination) — MUST BE LAST OF THE FOUR TRACKS
  T17 (DCTR-15) -> T18 (DCTR-16) -> T19 (DCTR-17)
     T17 needs T02,T10,T12,T16;  T18 needs T17;  T19 needs T16,T18

WAVE 7 — cross-product coverage and boundary pins (parallel: different files)
  T22 (DCTR-20)  needs T19        combination × {Buckets, Counter, BTMV} differential
                                  *** must precede T23 — SPEC DCTR-20 ***
  T21 (DCTR-03)  needs T19        surviving-clause negative tests

WAVE 8 — closure
  T23 (DCTR-18)  needs T00,T19,T22        final gate form + covering tests
  T24 (DoD)      needs T21,T23 and every task above
```

ASCII dependency edges (acyclic; verified by topological walk):

```text
T00 ─► T01 ─► T10 ─► T12 ─► T16 ─► T19 ─► T22 ─► T23 ─► T24
  └────────────────────────────────────────────────► T23
T01 ─► T20 ────────────────────────────────────────────────► T24
T02 ─► T08 ─► T09 ─► T10
T02 ──────────────────────────────► T17 ─► T18 ─► T19 ─► T21 ─► T24
T03 ─► T04 ─► T14 ─► T15
T05 ─► T10        T06 ─► T12        T07 ─► T16
T08 ─► T11 ─► T12          T11 ─► T14 ─► T16
T12 ─► T13 ────────────────────────────────────────────────► T24
T10, T12, T16 ─► T17
```

**Critical edges that must not be reordered** (from `SPEC.md` §5 and research's
"Planning Guidance"):

- **T00 before every gate-touching task** (T01, T10, T12, T16, T19, T23) — the
  `:124-126` conscious-act contract (R-21). Per **V-0c** the existing assertions do not
  actually turn red until T19; T00 is what stops them going *vacuous* at T01/T10/T12 in
  the meantime.
- **T22 before T23** — `SPEC.md` DCTR-20: the combination × non-Borders cross-product must
  have a passing detector **before** the final gate that admits it ships.
- **T20 after T01** — MINOR-9: its §2.5 mutation targets the expression T01 edits.
- **T02 before every kernel/gate task** — the seam carries `ctr_type` /
  `target_border_idx` (Tracks A/B/C) and `projection_members` (Track D).
- **T03 → T04 before T14** — Track E lands before Track B, so "device == CPU" is provable
  against the **corrected** CPU at every point, never against a known-wrong one.
- **T17/T18/T19 last of the tracks** — D-1's eligibility gate must cover the Buckets,
  Counter and BTMV columns added by Waves 2–4, not just Borders.
- **T18 with T17, never T17 alone** — they share one eligibility predicate (R-20), and
  `ctr_device_combo` does **not** discriminate T18.
- **Fixtures (T05/T06/T07) block their e2e tests only** — they are otherwise free.

---

