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

## 4. Tasks

---

### T00 — Consciously migrate the gate's five existing covering assertions

- **Specs**: DCTR-18 (the migration clause added by the checker pass), R-21
- **Depends on**: none — **WAVE 0**. **Blocks T01, T10, T12, T16, T19, T23.**
- **Parallel**: YES with T02/T03/T05/T06/T07. Owns
  `crates/cb-train/src/device_ctr_combo_config_test.rs` jointly with T02 (T02 adds a new
  function; T00 rewrites two existing ones) — **run T00 first**, then T02 rebases.

**Why this task exists.** v1 of this plan asserted, on CodeGraph's authority, that
`ctr_types_are_device_covered` had no covering tests. **It has five** (V-0b), and they pin
exactly the four conjuncts P1 removes, under an explicit contract at
`device_ctr_combo_config_test.rs:124-126`:

> *"This test pins the CLOSED gate deliberately: re-opening it must be a conscious act
> accompanied by a passing `device_ctr_combo_fit_test` (currently `#[ignore]`d), not an
> accident."*

P1 **is** that conscious act. This task is where the repository is told so, once, in
writing — instead of the guardrail being dissolved a conjunct at a time by six later tasks
that never mention it.

**Goal / observable completion**: the two flip-prone test functions are replaced by a
single **gate-state table** that pins the admitted set as it is *right now*, names the task
that flips each row, and stays discriminating at every intermediate wave. The unchanging
positive pin at `:144` survives untouched.

**Decision: rewrite, not retire.** The coordinator allowed either. Retiring hands all
coverage to T23 and leaves the gate **unpinned from T01 through T22** — precisely the
window in which five conjunct edits land. Rewriting keeps a pin live at every step and
makes the contract *stronger* than it is today, because after this task the gate's admitted
set is asserted in **every** wave rather than only in its closed state. The written
justification below is what `:124-126` asks for and must be committed with the change.

**Files / symbols (verified on disk)**
- Modify: `crates/cb-train/src/device_ctr_combo_config_test.rs:116-173` — replace
  `a_combination_column_set_is_NOT_device_covered_yet` (`:117-146`) and
  `a_non_borders_column_still_declines` (`:149-173`) with the single table-driven
  `gate_admits_exactly_the_current_wave` below. **Keep** the `:144` positive assertion
  (an all-simple Borders set stays covered) as a distinct, never-flipping function
  `an_all_simple_borders_set_stays_covered` — it is D-04's pin and no task in P1 may flip it.
- Keep: `covered_column` (`:41-52`) and every other function in the file unchanged.
- Production files: **none**. This task changes no production behaviour.

**Red**
- Test name: `gate_admits_exactly_the_current_wave`.
- Shape — a table whose **expected** column is edited by exactly one later task, named
  inline:
  ```rust
  // GATE STATE TABLE (P1 / R-21). Each row is (arity, ctr_type, target_border_idx,
  // prior_denom) -> expected coverage. The `flips_at` note names the ONE task allowed to
  // change that row, and that task MUST change it in the SAME commit as its conjunct
  // deletion. See the module doc for the :124-126 conscious-act justification.
  //   row 1  simple   Borders   0  1.0  => true    flips_at: never (D-04 pin)
  //   row 2  combo    Borders   0  1.0  => false   flips_at: T19 (arity)
  //   row 3  simple   Counter   0  1.0  => false   flips_at: T12 (type)
  //   row 4  simple   Buckets   1  1.0  => false   flips_at: T10 (type + target border)
  //   row 5  simple   BTMV      0  1.0  => false   flips_at: T16 (type)
  //   row 6  simple   Borders   0  2.0  => false   flips_at: T01 (prior denominator)
  //   row 7  simple   FloatTMV  0  1.0  => false   flips_at: never (no parity surface)
  //   row 8  simple   FeatureFreq 0 1.0 => false   flips_at: never
  ```
- **Crucial difference from the code being replaced (V-0c)**: rows 3–6 use a **simple**
  projection, whereas the originals at `:156/:163/:170` used
  `TProjection::from_features(&[0, 1])`. With a combination projection those assertions
  pass for the *wrong reason* the moment their named conjunct is deleted — the arity
  conjunct keeps them green until T19, so they go silently **vacuous** at T01/T10/T12
  rather than red. Simple projections let each row isolate the conjunct named in its
  `flips_at` column — with one honest exception: **row 4** (`simple / Buckets / b=1`)
  varies *two* attributes vs row 1 (type **and** target border), so it cannot distinguish
  "T10 removed the target-border conjunct globally" from "T10 made it conditional on
  Buckets". Add `simple/Buckets/b=0` and `simple/Borders/b=1` rows (both `flips_at: T10`)
  if that distinction is wanted; low impact either way, since T23 collapses the predicate
  and retires the whole table.
- Expected initial failure: **rows 3–6 fail as written on the unmodified tree** — no: they
  must be authored to match today's behaviour, so the table is green on first run. This is
  therefore a **§2.5 green-on-write test**. Mutation check: delete
  `&& col.prior_denom == 1.0` from `boosting.rs:2364` and confirm **row 6** fails with
  `row 6 (simple/Borders/b=0/denom=2.0): expected false, got true`; revert. That single
  mutation also proves the table is wired to the real predicate.
  **Note this is exactly T01's Green** — so the recorded mutation output doubles as T01's
  predicted Red, and T01 becomes a one-line edit plus a one-row table update.
- Run: `cargo test -p cb-train --lib device_ctr_combo_config_tests`

**Green**
- The rewrite plus a module-doc paragraph, which is the written justification `:124-126`
  demands. It must state: (a) that P1 is the conscious re-opening the original comment
  anticipated; (b) that `device_ctr_combo_fit_test` is un-ignored **in T19** with a
  `CountingGpu` device-commit assertion, satisfying the original's "accompanied by a
  passing `device_ctr_combo_fit_test`" condition; (c) the measured evidence
  (control `2.746e-2` → D-1 `2.082e-17`); and (d) that the table replaces per-conjunct
  negative tests **because** V-0c showed those go vacuous rather than red.
- Do **not** delete any conjunct here. T00 changes tests only.

**Refactor + regression scope**
- Also correct the stale `#[ignore]`-referencing sentence at `:121-126` (T19 removes that
  attribute).
- Regression scope: `cargo test -p cb-train --lib` only. No production change ⇒ no device
  behaviour can move.

**Validation commands**
```
cargo test -p cb-train --lib device_ctr_combo_config_tests
cargo test -p cb-train --lib
```

**Completion evidence**: the eight-row table green; the row-6 mutation failure recorded
verbatim; the justification paragraph present in the module doc; `git diff` limited to
`device_ctr_combo_config_test.rs`.

**Rollback**: restore the two original functions from `a0a67ec`. Nothing depends on the
rewrite except the later tasks' one-row updates.

**Risks / guardrails**: the table must assert on `ctr_types_are_device_covered` **directly**
(the file already imports it at `:21`) — not on a re-implementation. If a later task
changes a row **without** deleting the corresponding conjunct, or deletes a conjunct
without changing its row, the contract is broken; each gate task's Validation block now
runs this binary so that cannot pass unnoticed.

---

### T01 — Delete the provably-dead `prior_denom == 1.0` conjunct

- **Specs**: DCTR-02
- **Depends on**: **T00** — **WAVE 1**
- **Parallel**: YES with T04. Shares `boosting.rs` with T02 but a **different function**
  (`ctr_types_are_device_covered` at `:2355-2366` vs `build_device_ctr_config` at
  `:2429-2545`) — rebase-safe. It is the **head of the serial gate chain**:
  T01 → T10 → T12 → T16 → T19 → T23 all edit the same expression.
- **v2 note**: T00 has already recorded this task's exact Red by mutation, so T01 is a
  one-line deletion plus a one-row table flip. **T20 is now serialised after T01**
  (MINOR-9) because its own mutation targets this same expression.

**Goal / observable completion**: `ctr_types_are_device_covered` no longer mentions
`prior_denom`, and no fit's routing changes, proven by a characterization test that pins
`CTR_PRIOR_DENOM == 1.0` at the single production materialization site.

**Evidence**: `ctr_types_are_device_covered` — 1 production caller (`boosting.rs:4426`)
and, per **V-0b**, **five existing test assertions**, migrated by T00 into the gate-state
table. (v1 said "no covering tests found" here; withdrawn — CRITICAL-1.)

**Files / symbols (verified on disk)**
- Modify: `crates/cb-train/src/boosting.rs:2364` — delete `&& col.prior_denom == 1.0`.
- Modify: `crates/cb-train/src/boosting.rs` doc block above `:2355` — record that upstream
  `ctr_helper.cpp:50` (`CB_ENSURE(denom == 1.0, "Error: CPU could use only 1 as denom for
  ctrs currently")`, v1.2.10) forbids `denom != 1` on the CPU task type, so there is no
  parity surface to gain or lose.
- Create: `crates/cb-train/src/boosting_ctr_gate_test.rs` (new sibling test file), mounted
  at the end of `boosting.rs` as
  `#[cfg(test)] #[path = "boosting_ctr_gate_test.rs"] mod boosting_ctr_gate_tests;`
  next to the existing `#[path = "device_ctr_combo_config_test.rs"]` mount at `:7407-7408`.
  **This file is reused by T10/T12/T16/T19/T22** — it is the gate's covering-test home.

**Red** — two parts, both in-crate, no device:
1. **Flip T00's row 6** in `gate_admits_exactly_the_current_wave`
   (`device_ctr_combo_config_test.rs`) from `false` to `true`, and update its
   `flips_at: T01` note to `flipped by T01`. Expected initial failure is **exactly the
   message T00 recorded** under its mutation, now with the sign reversed:
   `row 6 (simple/Borders/b=0/denom=2.0): expected true, got false`.
2. Test name `ctr_prior_denom_is_structurally_unit`, in the **new**
   `crates/cb-train/src/boosting_ctr_gate_test.rs` — the characterization DCTR-02 requires:
   assert `super::CTR_PRIOR_DENOM == 1.0` (`[V]` the const at `boosting.rs:2257`) and that
   the only production materialization call passing it is `boosting.rs:2237`. This pins
   *why* the deletion is a no-op, which the routing table alone does not express.
   Green-on-write ⇒ §2.5 applies: mutate the const to `2.0` and confirm the assert fires.
- Run: `cargo test -p cb-train --lib device_ctr_combo_config_tests`
  then `cargo test -p cb-train --lib boosting_ctr_gate_tests`

**Green**
- Delete the single conjunct. Nothing else. Do **not** touch the type, arity or
  target-border conjuncts (T10/T12/T16/T19 own those). The row-6 flip must be in the
  **same commit** as the deletion — that is T00's contract.

**Refactor + regression scope**
- Rewrite the doc block's prior-denominator paragraph with the `ctr_helper.cpp:50`
  citation and a pointer to `CTR_PRIOR_DENOM` as the structural reason the removal is a
  no-op. No behavioural change.
- Regression scope: every CTR-routing test. `prior_denom` is `1.0` at the only
  production site, so no fit's routing can change.

**Validation commands**
```
cargo test -p cb-train --lib device_ctr_combo_config_tests     # CRITICAL-1: mandatory on every gate task
cargo test -p cb-train --lib boosting_ctr_gate_tests
cargo test -p cb-train --lib
cargo test -p cb-train --test ctr_feature_materialize_test
cargo test -p cb-train --no-default-features --features rocm --test device_ctr_fit_test
cargo test -p cb-train --no-default-features --features rocm --test device_ctr_gate_test
```

**Completion evidence**: `ctr_prior_denom_is_structurally_unit` green; the gate-state
table green with row 6 flipped; `cargo test -p cb-train --lib` green **in full** (the
CRITICAL-1 regression that v1 would have shipped red); the two device tests
unchanged-green; `git diff` limited to the one conjunct, the doc block, the row-6 flip,
the new test file and its mount.

**Rollback**: re-add `&& col.prior_denom == 1.0`. One line; no other task depends on the
removal for correctness (only T22's final collapse assumes it is gone).

---

### T02 — Widen the device CTR seam with per-column CTR identity

- **Specs**: DCTR-01
- **Depends on**: none — **WAVE 0**
- **Parallel**: YES with T01/T03/T05/T06/T07/T20. Shares `boosting.rs` with T01
  (different function) — rebase-safe.

**Goal / observable completion**: `DeviceCtrColumn` carries `ctr_type`,
`target_border_idx` and sorted `projection_members`, populated from the owning
`CtrFeatureColumn`; the workspace compiles; **device output on the existing
`ctr_device_mixed` path is byte-identical** (pure plumbing — nothing reads the new fields
yet).

**Evidence (corrected in v2 — C-12)**: CodeGraph reports 8 callers plus the two
constructing test files. `[V]` `grep -rn "DeviceCtrColumn {" crates/ --include=*.rs`
returns **9 lines: 1 struct definition + 8 literals**, and **3 of the 8 use
`..DeviceCtrColumn::default()`**:

| site | form | needs edit? |
|---|---|---|
| `cb-compute/src/runtime.rs:1105` | `pub struct DeviceCtrColumn {` — the **definition** | add the 3 fields |
| `session_depth_gt1_test.rs:501` | full literal | **yes** |
| `session_depth_gt1_test.rs:553` | full literal | **yes** |
| `session_depth_gt1_test.rs:629` | `..DeviceCtrColumn::default()` | no |
| `session_depth_gt1_test.rs:630` | full literal | **yes** |
| `session_depth_gt1_test.rs:698` | `..DeviceCtrColumn::default()` | no |
| `session_depth_gt1_test.rs:779` | `..DeviceCtrColumn::default()` | no |
| `cb-compute/src/runtime_ctr_averaging_test.rs:14` | full literal | **yes** |
| `cb-train/src/boosting.rs:2519` | full literal (production) | **yes** |

⇒ **5 edits**, not 9. A site not in that table means the blast radius was mis-measured:
stop and re-run the grep. The three `..Default::default()` sites will receive
`projection_members: vec![]`, which is inert while the gate admits only simple projections
— re-review them once T19 lands.

**Files / symbols (verified on disk)**
- Modify: `crates/cb-compute/src/runtime.rs:1105-1128` — add three plain host fields per
  `SPEC.md` §4.1:
  ```rust
  pub ctr_type: i8,               // ECtrType discriminant: 0 Borders, 1 Buckets, 2 BTMV, 4 Counter
  pub target_border_idx: u32,     // Buckets numerator selector; 0 for every other type
  pub projection_members: Vec<u32>, // SORTED cat-feature ids; len() >= 1
  ```
  No `cubecl` import (D-03 / T-10-04). `Debug, Clone, Default, PartialEq` derives must
  still hold (`Vec<u32>` and the scalars all satisfy them).
- Modify: `crates/cb-train/src/boosting.rs:2519-2527` (the `Ok(cb_compute::DeviceCtrColumn { … })`
  literal inside `build_columns`) — populate all three from `col`, using **C-4**'s
  `u32::try_from(…).map_err(|_| CbError::OutOfRange(…))?` for both the border index and
  each projection member. `col.projection.cat_features()` is already sorted+deduped
  (`projection.rs:121-132`) — **do not re-sort**, and add a comment saying why.
- Modify (compile-only, add the three fields — **5 sites per the C-12 table above**):
  `crates/cb-compute/src/runtime_ctr_averaging_test.rs:14`,
  `crates/cb-backend/src/gpu_runtime/session_depth_gt1_test.rs:501, :553, :630`.
  The three `..DeviceCtrColumn::default()` literals need **no** edit.
- Do **not** touch `ctr_covered`, `build_ctr_cindex_columns`, `CtrSearchState` or
  `ResidentCtrSearch` — nothing consumes the new fields until T09/T17.

**Red**
- Test name: `device_ctr_column_carries_type_border_and_sorted_members`.
- File: `crates/cb-train/src/device_ctr_combo_config_test.rs` (existing; `[V]` mounted at
  `boosting.rs:7407-7408`).
- Setup: materialize a two-member combination CTR column and a simple Buckets column
  through the same path the existing tests in that file use, call
  `build_device_ctr_config`, and assert for each emitted `DeviceCtrColumn`:
  `ctr_type == source.ctr_type`, `target_border_idx == source.target_border_idx as u32`,
  and `projection_members == source.projection.cat_features()` cast to `u32` **and**
  `projection_members.windows(2).all(|w| w[0] < w[1])` (strictly increasing ⇒ sorted +
  deduped).
- Expected initial failure: **compile error** `E0609: no field 'ctr_type' on type
  'DeviceCtrColumn'` (and the same for the other two fields) — the specified Red for a
  type-shape change, exactly as the prior phase's T02 used `E0050`/`E0061`.
- Run: `cargo test -p cb-train --lib device_ctr_combo_config_tests`

**Green**
- Add the three fields; populate them; mechanically extend the **5** struct literals that
  need edits (of **8** literal sites total — three use `..Default::default()`; see C-12's
  verified table above). Every
  test-side literal uses the neutral values `ctr_type: 0, target_border_idx: 0,
  projection_members: vec![…]` matching whatever `member_bins` that literal already
  declares, so no existing assertion changes meaning.

**Refactor + regression scope**
- Doc-comment each new field with its upstream anchor: `ctr_type` → `ECtrType`
  discriminants + `restrictions.h:20-32`; `target_border_idx` → `GetTargetBorderCount`
  (`ctr_helper.h:35-42`) and the note that only Buckets ever produces `> 0`;
  `projection_members` → `AddTreeCtrs`' `seenProj` subset rule (`greedy_tensor_search.cpp:491-551`)
  and the forward reference to T17.
- **Mandatory identifier-space warning (C-12b / checker MINOR-12b)**: the doc comment on
  `projection_members` must state that it carries **absolute** cat-feature indices while
  the sibling `member_bins` is in **CTR-eligible position** space
  (`boosting.rs:2487-2500`), that the two are therefore **not** index-compatible, and that
  this is safe only because the eligibility predicate compares `projection_members`
  against other `projection_members` and never against a bin index. Without this note the
  next reader will index one with the other.
- **Byte-identity invariant (DCTR-01)**: nothing reads the new fields, so
  `device_ctr_fit_test` and `device_ctr_gate_test` must be **numerically unchanged**.
  Record the printed `max |Δpred|` from `device_ctr_fit_test` before and after and assert
  they are equal in the completion note.

**Validation commands**
```
cargo check -p cb-compute
cargo check -p cb-backend --no-default-features --features rocm
cargo test -p cb-train --lib device_ctr_combo_config_tests
cargo test --workspace
cargo test -p cb-train --no-default-features --features rocm --test device_ctr_fit_test
cargo test -p cb-train --no-default-features --features rocm --test device_ctr_gate_test
bash ./run_device_tests.sh
```

**Completion evidence**: 23/23 device PASS + perf lane; `device_ctr_fit_test`'s printed
`max |Δpred|` identical to the pre-change value; `git diff --stat` limited to
`cb-compute/src/runtime.rs`, `cb-train/src/boosting.rs`, the three test files.

**Rollback**: revert the three fields and the 5 edited literals. No behaviour depends on them at
this point.

**Risks / guardrails**: a literal that silently defaults `projection_members` to `vec![]`
would make T17's subset test wrongly treat a combination as simple. **Guardrail**:
`build_device_ctr_config` already errors on an empty projection (`boosting.rs:2478-2482`);
T02 must add the mirroring invariant `!projection_members.is_empty()` as a typed
`CbError::Degenerate`, and the Red test asserts `len() >= 1`.

---

### T03 — Prove Track E is inert before changing anything

- **Specs**: DCTR-05
- **Depends on**: none — **WAVE 0**
- **Parallel**: YES with everything. Sole edit is `cb-train/src/ctr/calc_ctr_test.rs`.

**Goal / observable completion**: an in-repo characterization test proves
`calc_normalization(p) == (0.0, 1.0)` for every prior the repository can produce, so
T04's change is provably a no-op for every committed artifact and **no fixture is
re-baselined**.

**CodeGraph / disk evidence**: `[V]` `calc_normalization` (`cb-train/src/ctr/calc_ctr.rs:60-66`):
`left = min(0, p); right = max(1, p); shift = -left; norm = right - left` ⇒ `p ∈ [0,1] ⇒ (0.0, 1.0)`.
`[V]` every prior in the repository is in `[0,1]`: `{Prior=0, Prior=0.25, Prior=0.5, Prior=1}`
across all fixture `config.json`s and `{vec![0.25], vec![0.5]}` across the Rust tests
(research grep, re-checked against `ctr_device_combo/config.json` `"Borders:Prior=0.5"`
and `ctr_counter_simple/config.json` `"Counter:Prior=0.5"`).

**Files / symbols**
- Modify: `crates/cb-train/src/ctr/calc_ctr_test.rs` (existing; `[V]` mounted at
  `ctr/mod.rs:47`).
- Production files: **none**. This task changes no production code.

**Red**
- Test name: `calc_normalization_is_identity_for_every_repo_prior`.
- File: `crates/cb-train/src/ctr/calc_ctr_test.rs`.
- Setup: for `p` in `[0.0, 0.25, 0.5, 1.0]` (the full set of priors reachable in this
  repository) assert `calc_normalization(p) == (0.0, 1.0)` exactly (bit equality, not
  epsilon). Add the out-of-range companions as pinned values: `calc_normalization(2.0) == (0.0, 2.0)`
  and `calc_normalization(-0.5) == (0.5, 1.5)`.
- **This test is green-on-write.** Per §2.5 the implementer must execute the mutation
  check: change `let right = f64::max(1.0, prior);` to `f64::max(2.0, prior)` and confirm
  the test fails with the `(0.0, 2.0) != (0.0, 1.0)` mismatch on `p = 0.0`, then revert.
- Expected recorded failure (mutation): `assertion `left == right` failed: left: `(0.0, 2.0)`, right: `(0.0, 1.0)``.
- Run: `cargo test -p cb-train --lib ctr::calc_ctr_test`

**Green**
- Nothing. The task's Green is the recorded mutation-check evidence: the test is
  discriminating and the production behaviour it pins is already correct.

**Refactor + regression scope**
- Add a module comment naming this test as **DCTR-05's no-op proof for DCTR-04** and
  stating that no fixture re-baseline is required because every repository prior is in
  `[0,1]`.
- Regression scope: none — no production change.

**Validation commands**
```
cargo test -p cb-train --lib ctr::calc_ctr_test
cargo test -p cb-train --test ctr_btmv_simple_oracle_test
```
(the second establishes the **pre-change green baseline** T04 must preserve unchanged.)

**Completion evidence**: the characterization test green; the mutation failure message
recorded; `ctr_btmv_simple_oracle_test` green **before** T04, with its output captured for
byte comparison.

**Rollback**: delete the test. Trivially safe.

---

### T04 — Fix the CPU BTMV `(ctr + shift)/norm` normalization (Track E)

- **Specs**: DCTR-04
- **Depends on**: T03 (its inertness proof must exist first) — **WAVE 1**
- **Parallel**: NO with T03. YES with T05/T06/T07/T20. Must land **before T14**.

**Goal / observable completion**: the BTMV quantizer applies
`trunc(((ctr + shift)/norm) · bc)` with `(shift, norm) = calc_normalization(prior)`, all
in f32, matching upstream's all-`float` `CalcCTR`; **`ctr_btmv_simple_oracle_test` stays
green unchanged** (the no-op proof) and no fixture is regenerated.

**Disk evidence**: `[V]` `crates/cb-train/src/ctr/ctr_feature.rs:327-333`:

```rust
let bin_f = if quantize_in_f32 {
    let norm = (total as f32) + 1.0f32;                    // <- the CTR DENOMINATOR,
    let ctr  = (good as f32 + prior_scalar as f32) / norm; //    NOT calc_normalization's norm
    f64::from(ctr * ctr_border_count as f32)               // <- missing (ctr + shift)/norm
} else {
    calc_ctr_online_bin(good, total, prior_scalar, ctr_border_count)
};
```
The local `norm` **shadows the name** of `calc_normalization`'s `norm` — a readability
trap the fix must clear by renaming it `denom`. Upstream:
`CalcCTR(sum, count, priorX, shiftX, normX, bc) = (ctr + shift)/norm * borderCount`
(`online_ctr.h:128-131`, called from `CalcOnlineCTRMean` at `online_ctr.cpp:483-489`).

**Files / symbols**
- Modify: `crates/cb-train/src/ctr/ctr_feature.rs:327-333` only.
- Test: `crates/cb-train/src/ctr/ctr_feature_test.rs` — **new sibling file**, mounted in
  `crates/cb-train/src/ctr/mod.rs` next to the existing `mod online_test; mod calc_ctr_test;`
  block (`[V]` `ctr/mod.rs:44-53`) as
  `#[cfg(test)] #[path = "ctr_feature_test.rs"] mod ctr_feature_test;`.
  (If a `ctr_feature`-owned test module already exists at execution time, extend it
  instead of creating a second one — verify with `grep -n "mod .*test" crates/cb-train/src/ctr/mod.rs`.)

**Red**
- Test name: `btmv_bin_applies_shift_and_norm_at_out_of_range_prior`.
- File: `crates/cb-train/src/ctr/ctr_feature_test.rs`.
- Setup: drive `materialize_ctr_feature` with `ctr_type = ECtrType::BinarizedTargetMeanValue`,
  `prior_num = 2.0`, `prior_denom = 1.0`, `ctr_border_count = 15`, and a single-object
  column so the first document sees `good = 0.0, total = 0` (the read-before-increment
  prefix's initial state — this is the spike's first mismatch pair).
- Input → expected: `(shift, norm) = calc_normalization(2.0) = (0.0, 2.0)`;
  `ctr = (0 + 2)/(0 + 1) = 2.0`; corrected `bin_f = ((2.0 + 0.0)/2.0) · 15 = 15.0` ⇒
  `bin == 15`. Uncorrected: `2.0 · 15 = 30.0`, clamped to `ctr_border_count` = `15`…
  **⚠ the clamp at `ctr_feature.rs:341-348` masks this specific value.** Therefore the
  test must **also** assert an unclamped case: use `prior_num = 2.0` with a bucket state
  reached after one class-1 document (`good = 1.0f32, total = 1`) ⇒
  `ctr = (1 + 2)/(1 + 1) = 1.5`; corrected `((1.5)/2.0)·15 = 11.25 ⇒ bin 11`;
  uncorrected `1.5 · 15 = 22.5 ⇒ trunc 22 ⇒ clamped to 15`. `11 != 15` **is not masked**.
  Assert `bins[doc_with_prefix_1_1] == 11`.
  The implementer must verify by construction which document lands on that prefix state
  and assert on that index; if the two-document construction does not produce it, widen
  the column until it does — **do not** weaken the assertion to the clamped value.
- Expected initial failure:
  `assertion `left == right` failed: left: 15, right: 11` (the uncorrected form clamps 22 → 15).
- Run: `cargo test -p cb-train --lib ctr::ctr_feature_test`

**Green**
```rust
let bin_f = if quantize_in_f32 {
    // Upstream's CalcCTR is all-`float` for the mean types (online_ctr.h:128-131).
    let (shift, norm) = crate::ctr::calc_normalization(prior_scalar);
    let denom = (total as f32) + 1.0f32;            // the CTR denominator (hard +1)
    let ctr = (good as f32 + prior_scalar as f32) / denom;
    f64::from(((ctr + shift as f32) / norm as f32) * ctr_border_count as f32)
} else { … };
```
**On the `norm == 0.0` guard (C-17, corrected in v2)**: `calc_ctr_online_bin` has one
(`calc_ctr.rs:99-101`), but it is **unreachable** — `calc_normalization` returns
`norm = max(1,p) - min(0,p) >= 1` for every `p`. Either omit it, or mirror it with a
comment saying it is shape-mirroring and dead. Do **not** present it as a live safety
check. Change **nothing else** — not the `else` arm, not the clamp, not the dispatch.

**Refactor + regression scope**
- Rename the shadowing local to `denom` and comment that `calc_normalization`'s `norm` is
  a different quantity.
- **Regression scope is the whole CPU CTR oracle set.** DCTR-05 predicts zero change:
  run the full list and diff their printed values against the T03 baseline.

**Validation commands**
```
cargo test -p cb-train --lib ctr::ctr_feature_test
cargo test -p cb-train --lib ctr::online_test
cargo test -p cb-train --test ctr_btmv_simple_oracle_test
cargo test -p cb-train --test ctr_btmv_bake_upstream_table_test
cargo test -p cb-train --test ctr_buckets_simple_oracle_test
cargo test -p cb-train --test ctr_counter_simple_oracle_test
cargo test -p cb-train --test ctr_borders_multiprior_oracle_test
cargo test -p cb-train --test ctr_feature_materialize_test
cargo test --workspace
```

**Completion evidence**: the `prior = 2.0` test green; **`ctr_btmv_simple_oracle_test`
green with a byte-identical printed value to T03's captured baseline** (that is DCTR-05's
proof); zero fixture files changed (`git status` shows nothing under
`crates/cb-oracle/fixtures/`).

**Rollback**: restore the three-line uncorrected form. Because the change is a proven
no-op for every committed artifact, rollback is risk-free.

**Risks / guardrails**: if any oracle value moves, **stop** — that would falsify DCTR-05
and mean a prior outside `[0,1]` exists somewhere the research grep missed. Do not
re-baseline the fixture; escalate.

---

### T05 — Frozen fixture `ctr_device_buckets`

- **Specs**: supports DCTR-08
- **Depends on**: none — **WAVE 0**. Requires a local `catboost==1.2.10`; generation is
  **offline**, CI only reads the committed artifacts.
- **Parallel**: YES — touches only `crates/cb-oracle/fixtures/ctr_device_buckets/`.

**Goal / observable completion**: a frozen fixture directory exists whose upstream model
contains **both** `ctr_target_border_idx` values 0 and 1, at least one float split, and
whose quantized-pool fit is bit-identical to its raw-pool fit.

**Files (all new, under `crates/cb-oracle/fixtures/ctr_device_buckets/`)**
`gen_fixtures.py`, `config.json`, `X.npy`, `X_cat.npy`, `y.npy`, `borders.npy`,
`model.json`, `predictions.npy`.

**Pinned recipe — copy `crates/cb-oracle/fixtures/ctr_device_combo/gen_fixtures.py`
verbatim and change ONLY the marked lines** (`[V]` that generator read in full this
session):

```
SCENARIO = "ctr_device_buckets"
CARDS    = (6,)                       # STARTING point — see the escalation ladder below
PARAMS["simple_ctr"]          = ["Buckets:Prior=0.5"]
PARAMS["combinations_ctr"]    = []
PARAMS["max_ctr_complexity"]  = 1
```

**GATE-load-bearing — never change these** (each one, if changed, makes the fit
device-unreachable or breaks a device invariant):
`border_count=15` (R-11: `ctr_covered` needs `borders.len()+1 == n_bins`),
`one_hot_max_size=1` (routes cat columns to CTR, not one-hot),
`permutation_count=1`, `bootstrap_type="No"`, `random_strength=0`,
`boost_from_average=False`, `N_FLOAT >= 1` (cat-only pools can never reach the device),
`score_function="L2"`, `leaf_estimation_method="Gradient"`,
`leaf_estimation_iterations=1`, `thread_count=1`, and **no `task_type`** (CPU oracle).

**TUNABLE — this is the escalation ladder (checker MAJOR-4).** `CARDS`, `N_ROWS`,
`iterations` and `SEARCH_SEEDS` are **search parameters, not invariants**. v1 froze
`CARDS=(6,)`, `N_ROWS=64`, `iterations=5` and allowed escalation only by widening seeds or
cardinality — which risks a deadlock whose only escape is weakening the
`target_border_idx ∈ {0,1}` guard, and that guard is **the only thing in the whole phase
that exercises T08's `Buckets@1` numerator**.

`[V]` the sole in-repo precedent that achieves this guard, `ctr_buckets_simple/`, needed
**60 rows, TWO cat columns (cardinalities 6 and 5), 10 iterations, seed 0**
(`config.json`: `n_rows=60`, `cardinalities={cat0:6, cat1:5}`,
`observed_target_border_idxs=[0,1]`; `X_cat.npy` is `(60, 2) int32`). So the v1 shape is
*below* the only configuration known to work.

Apply **in this order**, stopping at the first success, and record which rung was used in
`config.json` as `"escalation_rung"`:

1. widen `SEARCH_SEEDS` from `range(24)` to `range(64)`;
2. raise the cardinality, e.g. `CARDS=(8,)`;
3. **add a second cat column** — `CARDS=(6, 5)`, the cardinalities `ctr_buckets_simple`
   uses. ⚠ **Not an exact match, and the difference cuts against us**: `[V]`
   `ctr_buckets_simple/config.json` is a **categorical-ONLY** pool (zero float columns, 60
   rows, 10 iterations), which T05's gate-load-bearing `N_FLOAT >= 1` forbids — cat-only
   pools can never reach the device. Two float columns compete for the same split slots,
   so both `target_border_idx` values are **strictly harder** to observe here than in the
   precedent. Expect to need rung 4 as well; do not read "the precedent worked" as a
   guarantee for this shape.
   This stays a *simple*-CTR fixture: `max_ctr_complexity=1` and `combinations_ctr=[]`
   mean no combination projection can form, so Track A's scope is unchanged;
4. raise `iterations` to `10` (also `ctr_buckets_simple`'s value — more trees means more
   chances for the `b=1` column to win a split).

**Never** weaken guard 1. If all four rungs fail, stop and escalate — a Buckets fixture
without both target borders cannot discharge DCTR-08.

**Data shape (load-bearing)**: `N_ROWS=64`, `N_FLOAT=2`. **Cat-only pools can never reach
the device** (`has_any_scorable_feature` needs `n_float > 0 || n_one_hot > 0`,
`boosting.rs:3284-3286`), so the float columns are mandatory. Keep the combo generator's
**per-object float ramp in the target** (R-14): adapt
`logit = 3.0*(cat_term - 0.5) + 2.0*(x[:,0] - x[:,1])` so a purely categorical ±1 target
cannot hide a structure-vs-averaging leaf swap.

**Anti-false-pass guards (replace the combo generator's `max_members >= 2` guard)**
1. **THE discriminating assertion**: both `target_border_idx` values appear. `[V]` the
   accessor is **top-level on each CTR descriptor** — verified by loading
   `crates/cb-oracle/fixtures/ctr_buckets_simple/model.json`, whose
   `features_info.ctrs[0]` keys are exactly
   `['borders','ctr_type','elements','identifier','prior_denomerator','prior_numerator','scale','shift','target_border_idx']`:
   ```python
   assert sorted({c["target_border_idx"] for c in model_json["features_info"]["ctrs"]}) == [0, 1]
   ```
   (v1 guessed `c["ctr"]["identifier"].get("ctr_target_border_idx", 0)` — **wrong**, and
   its `.get(…, 0)` default would have made the guard silently vacuous. Checker MINOR-2.)
   Also assert `c["ctr_type"] == "Buckets"` for every descriptor.
2. ≥1 float split (`any(len(f.get("borders", [])) > 0 for f in … float_features)`).
3. `predictions.std() > 1e-6`.
4. Seed search over `range(24)`, first passing seed frozen and recorded as `data_seed`.
   **If no seed passes, widen `SEARCH_SEEDS` or the cardinality — never lower a guard.**

**Border freezing (R-15)**: `Pool.quantize(border_count=15)` + `save_quantization_borders`
→ full per-feature border set → `borders.npy`; assert exactly 15 borders per float
feature; then assert `np.abs(quantized_fit_preds - raw_fit_preds).max() == 0.0`.
`model.json`'s float borders are the **pruned used subset** and must never be fed to the
trainer.

**Overflow margin (mandatory docstring + `config.json` field)**: Logloss der1 ∈ (−1, 1)
with unit weights at `n = 64` ⇒ `n·max(w)·max(|der1|) ≤ 64 ≪ 2³³ ≈ 8.6e9`, margin > 1.3e8×.

**Red**
- Test name: `ctr_device_buckets_fixture_loads_with_both_target_borders`.
- File: `crates/cb-oracle/tests/ctr_device_buckets_fixture_smoke_test.rs` (**new**).
  **One smoke file per fixture** — `[V]` the shipped convention
  (`ctr_device_combo_fixture_smoke_test.rs`, `ctr_device_mixed_fixture_smoke_test.rs`,
  `weighted_device_fixture_smoke_test.rs`, …). C-15: v1 shared one file across T05/T06/T07
  and then contradicted its own parallelism claim; with one file each they are genuinely
  parallel.
- Setup: load `X.npy` `[64,2] f32`, `X_cat.npy` `[64] int32`, `y.npy`, `borders.npy`
  `[2,15] f64`, `predictions.npy`, `config.json`; assert shapes, assert
  `config["observed_target_border_idxs"] == [0, 1]`, assert every border row has length 15.
- Expected initial failure: the directory does not exist ⇒ npy load error
  (`No such file or directory (os error 2)`).
- Run: `cargo test -p cb-oracle --test ctr_device_buckets_fixture_smoke_test`

**Green**: run `gen_fixtures.py` offline against the installed `catboost==1.2.10`; commit
the artifacts with `"note": "FROZEN…"` and `"requirement": "DCTR-08"`.

**Refactor + regression scope**: none — new files only. Keep the generator's corpus
contamination check (`git status --porcelain` over the fixtures dir, rejecting any path
outside `SCENARIO`) verbatim from the combo generator.

**Validation commands**
```
cargo test -p cb-oracle --test ctr_device_buckets_fixture_smoke_test
git status --porcelain crates/cb-oracle/fixtures   # only ctr_device_buckets/ paths
```

**Completion evidence**: directory committed; smoke test green; `config.json` records
`data_seed`, `observed_target_border_idxs: [0, 1]`, `overflow_margin`, and the FROZEN note.

**Rollback**: delete the directory. Nothing depends on it until T10.

---

### T06 — Frozen fixture `ctr_device_counter`

- **Specs**: supports DCTR-10
- **Depends on**: none — **WAVE 0**
- **Parallel**: YES, with no coordination — own fixture directory **and** its own smoke
  file `crates/cb-oracle/tests/ctr_device_counter_fixture_smoke_test.rs` (C-15).

**Goal / observable completion**: a frozen Counter fixture exists with the prior pinned
**explicitly on both sides**.

**Files**: `crates/cb-oracle/fixtures/ctr_device_counter/{gen_fixtures.py,config.json,X.npy,X_cat.npy,y.npy,borders.npy,model.json,predictions.npy}`.

**Recipe deltas from `ctr_device_combo/gen_fixtures.py`**
```
SCENARIO = "ctr_device_counter"
CARDS    = (6,)
PARAMS["simple_ctr"]         = ["Counter:Prior=0.5"]   # EXPLICIT — see the trap below
PARAMS["combinations_ctr"]   = []
PARAMS["max_ctr_complexity"] = 1
```
**THE COUNTER TRAP (R-11 sibling)**: upstream's default Counter prior is `0/1`, **not**
`0.5` (`[V]` `cb-train/src/ctr/mod.rs:191-206` `default_priors()`). The prior must be
pinned **on both sides** — `"Counter:Prior=0.5"` in the fixture params **and**
`simple_ctr_priors: vec![0.5]` in T12's `BoostParams`. A mismatch produces a silent,
plausible-looking divergence with no compile or shape error.
Everything else identical to T05, including the float columns, the target ramp, the
border freezing, the quantized-vs-raw bit-identity assertion and the overflow margin.
Anti-false-pass guard: ≥1 float split, ≥1 Counter CTR descriptor in `model.json`
(`ctr_type == "Counter"`), `predictions.std() > 1e-6`.

**Red**
- Test name: `ctr_device_counter_fixture_loads_with_pinned_prior`.
- File: `crates/cb-oracle/tests/ctr_device_counter_fixture_smoke_test.rs` (**new**).
- Setup: shapes as T05; assert `config["params"]["simple_ctr"] == ["Counter:Prior=0.5"]`
  (the explicit-prior pin) and that `model.json` contains ≥1 CTR descriptor of type
  `Counter`.
- Expected initial failure: directory missing ⇒ npy load error.
- Run: `cargo test -p cb-oracle --test ctr_device_counter_fixture_smoke_test`

**Green / Refactor / Validation / Completion / Rollback**: as T05, with
`"requirement": "DCTR-10"`.

---

### T07 — Frozen fixture `ctr_device_btmv`

- **Specs**: supports DCTR-14
- **Depends on**: none — **WAVE 0**
- **Parallel**: YES, with no coordination — own fixture directory **and** its own smoke
  file `crates/cb-oracle/tests/ctr_device_btmv_fixture_smoke_test.rs` (C-15).

**Goal / observable completion**: a frozen BTMV fixture exists.

**Files**: `crates/cb-oracle/fixtures/ctr_device_btmv/{…}` as T05.

**Recipe deltas**
```
SCENARIO = "ctr_device_btmv"
CARDS    = (6,)
PARAMS["simple_ctr"]         = ["BinarizedTargetMeanValue:Prior=0.5"]
PARAMS["combinations_ctr"]   = []
PARAMS["max_ctr_complexity"] = 1
```
`[V]` `ECtrType::target_border_count(BTMV) == 1` (`ctr/mod.rs:137-146`) ⇒ **exactly one
column per prior**; the generator must assert `len(ctrs) == 1` rather than expecting two.
**Prior must be in `[0,1]`** (R-3's residual hazard, closed by T04 but pinned here anyway).
Anti-false-pass guard: ≥1 float split, ≥1 `BinarizedTargetMeanValue` descriptor,
`predictions.std() > 1e-6`.

**Red**
- Test name: `ctr_device_btmv_fixture_loads_with_single_target_border`.
- File: `crates/cb-oracle/tests/ctr_device_btmv_fixture_smoke_test.rs` (**new**).
- Setup: shapes as T05; assert exactly one CTR descriptor of type
  `BinarizedTargetMeanValue` and that its prior is `0.5`.
- Expected initial failure: directory missing ⇒ npy load error.

**Green / Refactor / Validation / Completion / Rollback**: as T05, with
`"requirement": "DCTR-14"`.

---

### T08 — Device Buckets numerator mode in the ordered prefix kernel

- **Specs**: DCTR-06
- **Depends on**: T02 — **WAVE 2**
- **Parallel**: **NO** within the kernel chain. T08 → T11 → T14 all edit
  `crates/cb-backend/src/kernels/ctr_device.rs` and its self-oracle file; serialise them
  under one owner.

**Goal / observable completion**: for `(ctr_type, target_border_idx)` ∈
`{(Borders,0), (Buckets,0), (Buckets,1)}` the device's per-object `good`/`total` are
**integer-equal** to the `online_class_prefix` reference; `(Borders, 0)` is
**byte-unchanged** from today.

**Disk evidence**: `[V]` `ordered_ctr_prefix_kernel` (`#[cube(launch)]` at
`ctr_device.rs:134`, `fn` at **`:135`**, body to `:167` — C-13) already
keeps the full 2-class `[N0, N1]` bucket history in `counts[2*bucket + class]` and reads
it **before** incrementing; only the numerator selection is hard-coded
(`let g = n1;` at `:156`). `[V]` the CPU reference `online_class_prefix`
(`cb-train/src/ctr/online.rs:552-570`):
```rust
let num = if matches!(ctr_type, Buckets) { counts[b] }
          else { total - Σ_{c<=b} counts[c] };
```
At binclf (`SIMPLE_CLASSES_COUNT == 2`, `online.rs:52`): `Borders@0 → n1`,
`Buckets@0 → n0`, `Buckets@1 → n1`, and `total = n0 + n1` for every mode.

**CubeCL prerequisites**: read `INDEX.md` first; on any build error load
`cubecl_error_solution_guide/` (C-6). Use `if`/`else` **statements**
(`Cubecl_conditionals.md`), `while` + explicit counter (`Cubecl_loop_control.md`).

**Files / symbols**
- Modify: `crates/cb-backend/src/kernels/ctr_device.rs:135-167` — add one runtime scalar
  parameter `mode: &Array<u32>` (length 2: `[is_buckets, target_border_idx]`) and select
  `good` inside the existing serial `while` body. The kernel is serial on unit 0, so a
  per-iteration branch costs nothing and divergence is not a concern.
  ```rust
  // SIMPLE_CLASSES_COUNT == 2 (cb_train::ctr::online.rs:52) — the explicit 2-class form
  // is the faithful collapse of online_class_prefix at binclf.
  let mut g = 0u32;
  if is_buckets == 1u32 {
      if b == 0u32 { g = n0; } else { g = n1; }
  } else {
      if b == 0u32 { g = t - n0; } else { g = 0u32; }
  }
  ```
  (`Borders@1` is unreachable — `target_border_count(Borders, 2) == 1` — and is pinned to
  `0` with a comment rather than left undefined.)
- Modify: `crates/cb-backend/src/kernels/ctr_device.rs:231-310` `launch_ordered_ctr_resident`
  — add `ctr_type: i8, target_border_idx: u32` parameters, host-validate them
  (`target_border_idx <= 1` at binclf; anything else ⇒ `CbError::OutOfRange`, mirroring
  the existing bin/class guards at `:251-260`), upload the 2-element `mode` array, and
  thread it into the launch. **Defaulting to `(0, 0)` must reproduce today's behaviour
  byte-for-byte** (D-04).
- Modify: `crates/cb-backend/src/gpu_runtime/session.rs:222-229` — the single existing
  call site inside `build_ctr_cindex_columns`; pass `(0, 0)` **for now**. T09 makes it
  read the column's real values.
- Test: `crates/cb-backend/src/kernels/ctr_device_test.rs` (existing; `[V]` mounted at
  `kernels.rs:3147-3149`, 6 tests today).

**Red**
- Test name: `buckets_numerator_matches_cpu_reference`.
- File: `crates/cb-backend/src/kernels/ctr_device_test.rs`.
- Setup: reuse the file's `synth_fixture(n, cardinality, seed)` helper (`:39`) at
  `n = 128, cardinality = 5, seed = 7`. Extend the inline CPU transcription
  (`cpu_ordered_ctr`, `:55-84`) with a `(is_buckets, b)` parameter implementing
  `online_class_prefix`'s rule **inline** (cb-backend cannot `use cb_train` — C-3).
- Input → expected: for each of the three modes, read back the device `good`/`total`
  buffers and assert **exact integer equality** against the reference, element-wise.
- **Non-vacuity guard (mandatory)**: assert that the three modes' `good` vectors are not
  all equal — specifically `good(Buckets,0) != good(Borders,0)` somewhere. Without it a
  fixture where every bucket is singleton (all prefixes zero) would pass trivially.
- Expected initial failure: `(Buckets, 0)` — the device returns `n1` where the reference
  returns `n0`, so
  `assertion `left == right` failed: left: 1, right: 0` at the first document whose
  bucket has already seen a class-0 document.
- Run: `cargo test -p cb-backend --no-default-features --features rocm --lib kernels::ctr_device_test`

**Green**
- Implement exactly the numerator selection above and the two host parameters. Do **not**
  touch `binarize_ctr_kernel`, the border table, `ctr_covered`, the gate, or
  `build_ctr_cindex_columns`'s dispatch (T09 owns that).

**Refactor + regression scope**
- Doc-comment the `mode` array with the `online_class_prefix` / `UpdateGoodCount`
  (`online_ctr.cpp:115-121`) citation and the `SIMPLE_CLASSES_COUNT == 2` assumption.
- Regression scope: the 6 existing `ctr_device_test` oracles and the whole device suite —
  the `(0,0)` default must leave them byte-unchanged.

**Validation commands**
```
cargo check -p cb-backend --no-default-features --features rocm
cargo test -p cb-backend --no-default-features --features rocm --lib kernels::ctr_device_test
cargo test -p cb-backend --no-default-features --features rocm
cargo test -p cb-train --no-default-features --features rocm --test device_ctr_fit_test
bash ./run_device_tests.sh
```

**Completion evidence**: the new oracle green for all three modes with the non-vacuity
guard satisfied; all 6 pre-existing `ctr_device_test` oracles green; 23/23 device PASS.

**Rollback**: revert the `mode` parameter and restore `let g = n1;`. Self-contained in one
file plus one call site.

**Risks / guardrails**: adding a `#[cube]` parameter changes the launch signature; a
missed call site is a compile error, not a silent bug. `grep -n "ordered_ctr_prefix_kernel::launch"`
must show exactly one site (`ctr_device.rs:291`).

---

### T09 — Per-type dispatch, Buckets shape acceptance, and weight-group sharing

- **Specs**: DCTR-07
- **Depends on**: T02, T08 — **WAVE 3**
- **Parallel**: NO. Owns `cb-backend/src/gpu_runtime/session.rs`'s CTR helpers; T12, T16
  and T17 edit the same region afterwards.

**Goal / observable completion**: `ctr_covered` admits `{Borders, Buckets}` and declines
every other discriminant with the shape checks otherwise unchanged (**C-7**);
`build_ctr_cindex_columns` dispatches on `(ctr_type, target_border_idx)`; and the two
Buckets columns per `(projection, prior)` provably share one `weight_group` and carry the
same `bucket_count`.

**Evidence**: `ctr_covered`, `build_ctr_cindex_columns` and `CtrSearchState` all report
**⚠️ no covering tests found** — and unlike the gate (V-0b), **all three were re-confirmed
by repo-wide `grep` in v2** per R-22. This task adds the first covering test.

**Safety property to record in this task (C-14 / checker MINOR-1)**: `ctr_covered` has
**two** callers — `session.rs:1471` and `:1701`. The first feeds the coverage disjunction
whose failure path is `return Ok(None)` at **`session.rs:1505`**, declining the **whole
fit**. ⇒ a mismatch between the cb-train gate's admitted type list and this backend list
can never silently drop CTR columns onto a device fit; it degrades loudly to
`grown == 0`. That is why `CountingGpu.grown.get() == params.iterations` is sufficient
evidence in T10/T12/T16/T19, and why the two lists may be widened in separate hunks
without a torn intermediate state. Put this in `ctr_covered`'s doc comment.

**Files / symbols (verified on disk)**
- Modify: `crates/cb-backend/src/gpu_runtime/session.rs:134-163` `ctr_covered` — add
  **one** conjunct to each of the two `.all(|col| …)` closures:
  `&& matches!(col.ctr_type, 0 | 1)`. Per **C-3** this is an inline transcription with a
  comment naming `cb_train::ctr::ECtrType`'s discriminants and `restrictions.h:20-32`.
  Per **C-7** `col.borders.len() + 1 == n_bins` is **unchanged**.
- Modify: `crates/cb-backend/src/gpu_runtime/session.rs:201-234` `build_ctr_cindex_columns`
  — replace the unconditional `launch_ordered_ctr_resident(…)` at `:222-229` with a
  `match col.ctr_type` whose `0 | 1` arm passes `(col.ctr_type, col.target_border_idx)`
  through and whose `_` arm returns
  `CbError::Unsupported(format!("device CTR type {} is not implemented", col.ctr_type))`.
  T12 and T16 add their arms here.
- Test A (backend): `crates/cb-backend/src/gpu_runtime/session_ctr_type_test.rs` — **new**
  sibling, mounted **inside `session.rs`** (at the file's end) as
  `#[cfg(test)] #[path = "session_ctr_type_test.rs"] mod session_ctr_type_test;`.
  **C-11 / checker MAJOR-2 — v1's hint was wrong and this test could not have compiled.**
  `[V]` `grep -n "session_depth_gt1_test" crates/cb-backend/src/gpu_runtime/session.rs`
  returns **nothing**; that module is mounted at `gpu_runtime/mod.rs:775` (and
  `session_residency` at `:768`). `[V]` `ctr_covered` is **private** to `mod session`
  (`session.rs:134`), so a module mounted in `mod.rs` is a *sibling* of `session` and
  cannot name it ⇒ **`E0603`**. **Keep `ctr_covered` private** — do not widen it to
  `pub(crate)` to make a mis-placed test compile. The `mod.rs` mount works for
  `session_depth_gt1_test` only because that file uses public API; it is **not** the
  pattern to copy here. The pattern to copy is `boosting.rs:7406-7408`.
- Test B (cb-train grouping): extend `crates/cb-train/src/device_ctr_combo_config_test.rs`.

**Red**
- Test A name: `ctr_covered_declines_unimplemented_ctr_types`.
  Setup: build a `DeviceTrainConfig` with a CTR config whose single column has
  `ctr_type = 4` (Counter) and otherwise valid shapes (mirror the existing
  `session_depth_gt1_test.rs` construction). Assert `ctr_covered(&config, n, n_bins) == false`,
  and `== true` for `ctr_type ∈ {0, 1}`.
  Expected initial failure: `ctr_type = 4` returns `true` ⇒
  `assertion failed: !ctr_covered(&cfg_counter, n, n_bins)`.
- Test B name: `buckets_columns_share_one_weight_group_and_bucket_count`.
  Setup: materialize Buckets columns for one `(projection, prior)` at binclf
  (`target_border_count(Buckets, 2) == 2` ⇒ two columns), call `build_device_ctr_config`,
  assert `columns.len() == 2`, `columns[0].weight_group == columns[1].weight_group`,
  `columns[0].bucket_count == columns[1].bucket_count`, and
  `{columns[0].target_border_idx, columns[1].target_border_idx} == {0, 1}`.
  **This test is green-on-write** (C-9: `build_device_ctr_config:2517` already keys
  `(ctr_type, projection)` with `target_border_idx` deliberately excluded). §2.5 mutation
  check: temporarily add `col.target_border_idx` to the `key` tuple and confirm the test
  fails with `left: 0, right: 1` on the group comparison; revert.
- Run:
  `cargo test -p cb-backend --no-default-features --features rocm --lib gpu_runtime::session::session_ctr_type_test`
  and `cargo test -p cb-train --lib device_ctr_combo_config_tests`

**Green**
- The one conjunct in `ctr_covered` and the dispatch `match`. Nothing else. Do **not**
  relax the cb-train gate (T10 owns it).

**Refactor + regression scope**
- Correct `ctr_covered`'s doc comment (`session.rs:125-133`), which currently describes
  only the single-permutation Borders regime.
- Regression scope: every device CTR path. `Borders` fits must be byte-unchanged.

**Validation commands**
```
cargo check -p cb-backend --no-default-features --features rocm
cargo test -p cb-backend --no-default-features --features rocm --lib gpu_runtime::session::session_ctr_type_test
cargo test -p cb-backend --no-default-features --features rocm
cargo test -p cb-train --lib device_ctr_combo_config_tests
cargo test -p cb-train --no-default-features --features rocm --test device_ctr_fit_test
bash ./run_device_tests.sh
```

**Completion evidence**: both tests green; the mutation failure for Test B recorded;
23/23 device PASS; `device_ctr_fit_test`'s printed `max |Δpred|` unchanged.

**Rollback**: revert the conjunct and the `match` (restoring the unconditional call).

---

### T10 — Buckets end-to-end oracle with device commitment

- **Specs**: DCTR-08
- **Depends on**: **T00**, T01 (gate chain head), T05 (fixture), T09 — **WAVE 3**
- **Parallel**: NO — second link in the serial gate chain
  (T01 → **T10** → T12 → T16 → T19 → T23).

**Open question this task must settle (checker MINOR-15).** It is **not established**
which leaf-value path a device Buckets fit takes. `[V]` the CPU/host gather
`assign_leaf_over_ctr_columns` (`boosting.rs:2001`, called at `:5547`, `:6565`, `:6868`)
matches an averaging column to a split on the **full identity** —
`projection && ctr_type && target_border_idx && prior_num.to_bits() && prior_denom.to_bits()`
(`boosting.rs:2055-2062`) — whereas the SPD-03 device path consumes
`DeviceGrownTree.leaf_of` with the averaging columns indexed by **CTR column position**
(`session.rs:2723-2784`, `ctr_averaging_bins`). **Buckets is the first type that emits TWO
columns for one `(projection, prior)`**, differing only in `target_border_idx`. T10 must
determine which path actually runs for this fit and confirm it is correct for that layout;
a position-indexed gather that assumed one column per projection would silently pair a
`b = 0` split with the `b = 1` averaging column. If the answer is "the device path, and it
is position-indexed and correct", record the evidence; if it is wrong, **stop and escalate**
rather than patching the gather inside an e2e task.

**Goal / observable completion**: a `simple_ctr = Buckets:Prior=0.5` fit **commits to the
device** (`grown.get() == params.iterations`) and matches upstream `catboost==1.2.10` at
**≤1e-5**.

**Files / symbols**
- Modify: `crates/cb-train/src/boosting.rs:2362-2363` — widen the type conjunct to
  `matches!(crate::ctr::ECtrType::from_i8(col.ctr_type), Some(ECtrType::Borders | ECtrType::Buckets))`
  and **delete** `&& col.target_border_idx == 0`. (`from_i8` is in-crate here, so the
  Do-Not-Hand-Roll rule applies and must be honoured — C-3.)
- Modify: `crates/cb-train/src/device_ctr_combo_config_test.rs` — flip **row 4** of T00's
  gate-state table (`simple / Buckets / b=1 / denom=1.0` ⇒ now `true`, which is why that
  row deliberately varies **both** the type and the target border: this task deletes both
  conjuncts) **in the same commit** as the conjunct edit. That is T00's contract.
- Create: `crates/cb-train/tests/device_ctr_buckets_fit_test.rs`, modelled on
  `device_ctr_fit_test.rs` **plus** the `CountingGpu` wrapper copied verbatim from
  `device_ctr_gate_test.rs:60-170`.

**Red**
- Test name: `device_ctr_buckets_fit_commits_and_matches_upstream`.
- File: `crates/cb-train/tests/device_ctr_buckets_fit_test.rs`.
- Setup: load the `ctr_device_buckets` fixture (`X.npy`, `X_cat.npy` `[N]` int32,
  `y.npy`, `borders.npy`, `predictions.npy`). `BoostParams` mirroring the fixture
  `config.json` exactly — the `ctr_params()` block in `device_ctr_gate_test.rs:24-66` is
  the template; change **only** `simple_ctr: cb_train::ECtrType::Buckets` and keep
  `simple_ctr_priors: vec![0.5]`, `max_ctr_complexity: 1`,
  `combinations_ctr: cb_train::combinations_ctr_default()`, `permutation_count: 1`,
  `one_hot_max_size: 1`, `bootstrap_type: EBootstrapType::No`, `random_strength: 0.0`,
  `boost_from_average: false`, `leaf_method: LeafMethod::Gradient`,
  `score_function: EScoreFunction::L2`.
- Assertions, **all three required**:
  1. `gpu.grown.get() == params.iterations` (device commitment — DCTR-08's "both");
  2. `max |actual − expected| <= 1e-5` per object, via
     `CbModel::from_trained(...).with_ctr_data(CtrData::from_baked(&baked))` +
     `predict_raw_cat`;
  3. the structure-vs-averaging divergence guard (§2.2).
  Plus a vacuity guard: `Σ tree.ctr_splits.len() >= 1`.
- Expected initial failure: **`assertion `left == right` failed: left: 0, right: 5`** on
  `gpu.grown.get()` — the gate still declines Buckets, so the fit runs on the CPU grower
  and `grow_tree_on_device` is never called.
- Run: `cargo test -p cb-train --no-default-features --features rocm --test device_ctr_buckets_fit_test`

**Green**
- The two-part gate edit above. Nothing else. Do **not** add Counter or BTMV (T12/T16),
  do **not** touch the arity conjunct (T19).

**Refactor + regression scope**
- Update `ctr_types_are_device_covered`'s doc block: Buckets and multi-target-border are
  now covered; cite the kernel numerator contract (`SPEC.md` §4.2) and T08's oracle.
- Regression scope: the full device suite plus `ctr_buckets_simple_oracle_test` (CPU).

**Validation commands**
```
cargo test -p cb-train --lib device_ctr_combo_config_tests     # CRITICAL-1: mandatory on every gate task
cargo check -p cb-train --no-default-features --features rocm --test device_ctr_buckets_fit_test
cargo test -p cb-train --no-default-features --features rocm --test device_ctr_buckets_fit_test
cargo test -p cb-train --test ctr_buckets_simple_oracle_test
cargo test -p cb-train --no-default-features --features rocm --test device_ctr_fit_test
cargo test -p cb-train --no-default-features --features rocm --test device_ctr_gate_test
bash ./run_device_tests.sh
```

**Completion evidence**: `grown == 5` and the printed `max |Δpred|` recorded (expected
well below 1e-5); 23/23 device PASS; the CPU Buckets oracle green.

**Rollback**: restore `col.ctr_type == ECtrType::Borders.as_i8() && col.target_border_idx == 0`.
Per `SPEC.md` §8 each track's gate widening is independently revertible.

**Risks / guardrails**: if `grown == 5` but the ≤1e-5 bar fails, the defect is in T08's
numerator or T09's dispatch, **not** the gate — do not widen the tolerance. If `grown == 0`
after the gate edit, the fit is being declined elsewhere: check `ctr_covered`'s
`borders.len() + 1 == n_bins` (R-11 — the fixture must have `border_count = 15`).

---

### T11 — Device Counter statistic (whole-set tally, constant denominator)

- **Specs**: DCTR-09
- **Depends on**: T08 (same file) — **WAVE 4**
- **Parallel**: NO — kernel chain T08 → **T11** → T14.

**Goal / observable completion**: a device Counter column reproduces
`online_counter_column` exactly — whole-set per-bucket totals, a **constant** max
denominator for every object, and **permutation independence**.

**Disk evidence**: `[V]` `online_counter_column` (`cb-train/src/ctr/online.rs:493-521`)
takes **no permutation parameter at all**: `totals[b] = #{obj : bin[obj] == b}`,
`denominator = max_b totals[b]`, output column `totals[bin[obj]]`. `[V]`
`ctr_feature.rs:296-309` then sets `denoms = vec![denominator; n]` and
`value = calc_ctr_online(num, denominator, prior) = (num + prior)/(denominator + 1)`,
quantized by `calc_ctr_online_bin` — the **same f64 path** as Borders/Buckets, so the
existing border table and `binarize_ctr_kernel` apply unchanged (C-7).

**Design decision (from `SPEC.md` §9)**: **serial two-pass on unit 0**, not parallel
atomics. Rationale: it preserves D-06 residency, adds no atomics, and Counter columns are
built **once per fit** inside `begin` (`session.rs:1725-1789`), never per tree — so it
cannot be a hot path. Revisit only on profiling evidence.

**CubeCL prerequisites**: `INDEX.md` first; `cubecl_error_solution_guide/` on any build
error; `if`/`else` statements; `while` + explicit counter.

**Files / symbols**
- Create: a new `#[cube(launch)] fn counter_ctr_kernel` in
  `crates/cb-backend/src/kernels/ctr_device.rs`, alongside `ordered_ctr_prefix_kernel`
  (do **not** branch the prefix kernel — Counter has no permutation and no
  read-before-increment, and polluting the prefix loop with a dead mode is worse than a
  sibling). Serial, unit 0, three passes:
  ```
  pass 1  tally:  while i < n { counts[bins[i]] += 1; i += 1; }
  pass 2  max:    while j < k { if counts[j] > m { m = counts[j]; } j += 1; }
  pass 3  map:    while d < n { let t = counts[bins[d]];
                                good[d] = t; total[d] = m;
                                value[d] = (f64::cast_from(t) + pr) / (f64::cast_from(m) + 1.0);
                                d += 1; }
  ```
  Generic over `F: Float` for the `value` array where the manual allows it; the counts are
  `Array<u32>` (exact integer tally). `total[d] = m` is deliberate — it mirrors
  `denoms = vec![denominator; n]`.
- Create: `pub(crate) fn launch_counter_ctr_resident(client, bins, prior, bucket_count, n) -> CbResult<ResidentCtr>`
  — **no `perm`, no `class` parameter** (structural permutation independence). Reuse the
  same host-side bin-range guard as `launch_ordered_ctr_resident:251-255`, the same wgpu
  rejection (`wgpu_reject()`, `:198-204`), and the same `n == 0` short-circuit.
- Test: `crates/cb-backend/src/kernels/ctr_device_test.rs`.

**Red**
- Test name A: `counter_ctr_matches_cpu_reference`.
  Setup: `synth_fixture(96, 5, 11)`; inline CPU transcription of `online_counter_column` +
  `calc_ctr_online`. Assert per-object `good` equals `totals[bin]` exactly, `total`
  equals the constant `max` for **every** object, and `value` matches within ε = 1e-4
  (D-07) — with an additional bit-exact assertion on the binarized `u32` bin column via
  `binarize_ctr_column_resident` + the border table (the
  `ctr_binarized_cindex_column_bit_exact` template at `:209`).
- Test name B: `counter_ctr_is_permutation_independent`.
  Setup: build the same column twice under two **different** permutations (identity and a
  reversal) and assert the emitted bin columns are **bit-identical**. This pins
  `IsPermutationDependentCtrType(Counter) == false` (`ctr_type.cpp:43-56`).
- Expected initial failure: **compile error** `cannot find function 'launch_counter_ctr_resident'
  in this scope` — the specified Red for a new device path.
- Run: `cargo test -p cb-backend --no-default-features --features rocm --lib kernels::ctr_device_test`

**Green**
- The kernel and its launch wrapper. Do **not** wire it into `build_ctr_cindex_columns`,
  `ctr_covered` or the gate (T12).

**Refactor + regression scope**
- Doc-comment with the upstream anchors: `CalcOnlineCTRCounter` (`online_ctr.cpp:503-568`),
  `CountOnlineCTRTotal` + `counterCTRDenominator` (`:713-729`), and an explicit note that
  the `Full` eval widening is **not** implemented and cannot be reached on device (T13
  pins it).
- Regression scope: the 6 + 1 existing `ctr_device_test` oracles and the whole device
  suite (nothing calls the new path yet, so all must be byte-unchanged).

**Validation commands**
```
cargo check -p cb-backend --no-default-features --features rocm
cargo test -p cb-backend --no-default-features --features rocm --lib kernels::ctr_device_test
cargo test -p cb-backend --no-default-features --features rocm
bash ./run_device_tests.sh
```

**Completion evidence**: both Counter oracles green (including bit-exact bins and
permutation independence); every pre-existing oracle green; 23/23 device PASS.

**Rollback**: delete the kernel, the wrapper and the two tests. No call sites yet.

**Risks / guardrails**: the `counts` scratch must be **pre-zeroed by the host**, exactly
as `ordered_ctr_prefix_kernel`'s is (`ctr_device.rs:278-279`); a reused non-zeroed buffer
silently doubles the tally.

---

### T12 — Counter end-to-end oracle with device commitment

- **Specs**: DCTR-10
- **Depends on**: **T00**, T06 (fixture), T09 (dispatch), T10 (gate chain), T11 — **WAVE 4**
- **Parallel**: NO — third link in the gate chain and a second edit to
  `build_ctr_cindex_columns` / `ctr_covered`.

**Goal / observable completion**: a `simple_ctr = Counter:Prior=0.5` fit **commits to the
device** and matches upstream at **≤1e-5**.

**Files / symbols**
- Modify: `crates/cb-backend/src/gpu_runtime/session.rs` `ctr_covered` — widen the
  discriminant list to `matches!(col.ctr_type, 0 | 1 | 4)`.
- Modify: `crates/cb-backend/src/gpu_runtime/session.rs` `build_ctr_cindex_columns` — add
  the `4 =>` arm calling `launch_counter_ctr_resident` (note it takes **neither**
  `permutation` **nor** `target_class`; the surrounding parameters stay for the other
  arms).
- Modify: `crates/cb-train/src/boosting.rs` gate — add `ECtrType::Counter` to the
  `matches!` list.
- Modify: `crates/cb-train/src/device_ctr_combo_config_test.rs` — flip **row 3**
  (`simple / Counter / b=0 / denom=1.0`) of T00's gate-state table to `true`, **in the
  same commit as the gate widening**. T00's contract (`flips_at`) names T12 as the ONE
  task allowed to change that row; leaving it unflipped fails
  `cargo test -p cb-train --lib device_ctr_combo_config_tests` with
  `expected false, got true`, and fixing it in a follow-up commit breaks the
  same-commit invariant that makes the table auditable.
- Create: `crates/cb-train/tests/device_ctr_counter_fit_test.rs`.

**Red**
- Test name: `device_ctr_counter_fit_commits_and_matches_upstream`.
- File: `crates/cb-train/tests/device_ctr_counter_fit_test.rs`.
- Setup: the `ctr_device_counter` fixture; `BoostParams` as T10 but
  `simple_ctr: cb_train::ECtrType::Counter` and **`simple_ctr_priors: vec![0.5]` pinned
  explicitly** (T06's Counter trap — upstream's default Counter prior is `0/1`), and
  `counter_calc_method: cb_train::counter_calc_method_default()` (== `SkipTest`,
  `[V]` `boosting.rs:511-513`).
- Assertions: `grown.get() == params.iterations`; ≤1e-5 per object; the permutation
  divergence guard; `Σ ctr_splits >= 1`.
- Expected initial failure: `assertion `left == right` failed: left: 0, right: 5` on
  `grown.get()` — the gate declines Counter.
- Run: `cargo test -p cb-train --no-default-features --features rocm --test device_ctr_counter_fit_test`

**Green**: the three one-line widenings above (backend list, backend dispatch arm,
cb-train gate list). Nothing else.

**Refactor + regression scope**
- Update the gate doc block for Counter, citing the permutation-independence property and
  the fact that `counter_calc_method` is structurally moot on device (T13 pins it).
- Regression scope: full device suite + `ctr_counter_simple_oracle_test` (CPU).

**Validation commands**
```
cargo test -p cb-train --lib device_ctr_combo_config_tests     # CRITICAL-1: mandatory on every gate task
cargo check -p cb-train --no-default-features --features rocm --test device_ctr_counter_fit_test
cargo test -p cb-train --no-default-features --features rocm --test device_ctr_counter_fit_test
cargo test -p cb-train --test ctr_counter_simple_oracle_test
cargo test -p cb-train --test ctr_counter_full_eval_oracle_test
cargo test -p cb-train --no-default-features --features rocm --test device_ctr_buckets_fit_test
bash ./run_device_tests.sh
```

**Completion evidence**: `grown == 5` and the printed `max |Δpred|` recorded; both CPU
Counter oracles green; 23/23 device PASS.

**Rollback**: remove `4` / `ECtrType::Counter` from the three lists (the kernel stays,
harmlessly unreferenced).

**Risks / guardrails**: a prior mismatch between the fixture (`Counter:Prior=0.5`) and
`BoostParams` (`simple_ctr_priors`) produces a plausible-looking ≈1e-2 divergence with no
type error. If the bar fails, **check the prior on both sides first**.

---

### T13 — `counter_calc_method = Full` + eval set still declines to CPU

- **Specs**: DCTR-11
- **Depends on**: T12 — **WAVE 4**
- **Parallel**: YES with T14/T15 (own test file, no production edit) once T12 is in.

**Goal / observable completion**: a fit with `counter_calc_method = Full` **and a
non-empty eval set** routes to the CPU (`grown.get() == 0`), pinning the P1/P3 boundary
and proving P1 ships **no** eval-widening code.

**Disk evidence**: `[V]` `device_host_eligible` requires `eval_sets.is_empty()`
(`boosting.rs:4480`). `[V]` `Full`'s only effect is `counter_full_eval_columns`, assembled
purely from `eval_sets[*].cat_columns` (`boosting.rs:4093-4107`) and threaded into
`materialize_ctr_feature` as `extra_cat_columns` (`boosting.rs:2242-2244`), consumed by
`online_counter_column`'s `extra_bins` (`online.rs:504-512`). With eval sets empty,
`Full ≡ SkipTest`. `[V]` `CounterCalcMethod::{SkipTest, Full}` (`ctr/mod.rs:257-263`).

**⚠ Correction C-1 applies**: this test **must** use `train_cat_with_eval_sets`
(`boosting.rs:2970-2981`) with a real `&[EvalSet]` (`boosting.rs:2551-2563`). A test
written against `train_cat` would be asserting on a structurally-empty eval-set list and
would pass vacuously.

**Files / symbols**
- Create: `crates/cb-train/tests/device_ctr_type_gate_test.rs` (**new**; **T21** adds
  further functions to this same file — coordinate with T21 only. T22 does **not**: it
  owns `device_ctr_combo_types_diff_test.rs`. See §5.3 for the authoritative ownership
  table.)
- Production files: **none**.

**Red**
- Test name: `counter_full_with_eval_set_declines_to_cpu`.
- Setup: the `ctr_device_counter` fixture split into a learn half and an eval half; params
  as T12 but `counter_calc_method: CounterCalcMethod::Full`; call
  `train_cat_with_eval_sets(&CountingGpu{…}, …, &[EvalSet { feature_values, target, cat_columns }], None)`.
- Assertion: `gpu.grown.get() == 0`.
- **This test is green-on-write.** §2.5 mutation check: temporarily delete
  `&& eval_sets.is_empty()` from `boosting.rs:4480` and confirm the test fails with
  `assertion `left == right` failed: left: 5, right: 0`; revert. Record the message.
- Also add the complement `counter_skiptest_without_eval_set_commits` asserting
  `grown.get() == params.iterations`, so the negative test cannot pass because the fixture
  is broken.
- Run: `cargo test -p cb-train --no-default-features --features rocm --test device_ctr_type_gate_test`

**Green**: nothing — the behaviour is already correct. The Green is the recorded mutation
evidence plus the complement test.

**Refactor + regression scope**
- Add a module comment stating that `counter_calc_method = Full` is **structurally moot**
  on the device in P1 and is deferred to P3, with the `boosting.rs:4093-4107` /
  `:4480` citations. Add the same note to `ctr_types_are_device_covered`'s doc block.
- **P2/P3 inversion annotation (mandatory, checker MINOR-13)**: mark this test
  `// P3 WILL INVERT THIS.` P3 lands device eval-set support and the Counter `Full`
  widening, at which point the correct assertion becomes `grown == params.iterations`.
  A boundary pin that does not say it is a boundary reads to the next phase as a
  requirement, and someone will preserve it instead of flipping it.
- Regression scope: none (test-only).

**Validation commands**
```
cargo check -p cb-train --no-default-features --features rocm --test device_ctr_type_gate_test
cargo test -p cb-train --no-default-features --features rocm --test device_ctr_type_gate_test
cargo test -p cb-train --test ctr_counter_full_eval_oracle_test
```

**Completion evidence**: both functions green; the mutation failure message recorded.

**Rollback**: delete the two test functions.

---

### T14 — Device BinarizedTargetMeanValue accumulator (f32 sum, u32 count)

- **Specs**: DCTR-12
- **Depends on**: T04 (corrected CPU — the parity target), T11 (same file) — **WAVE 5**
- **Parallel**: NO — kernel chain T08 → T11 → **T14**.

**Goal / observable completion**: a device `(f32 Sum, u32 Count)` bucket history
reproduces `online_mean_prefix` exactly, **with the f32 accumulation width proven
load-bearing** by C-2's two detectors.

**Disk evidence**: `[V]` `online_mean_prefix` (`cb-train/src/ctr/online.rs:298-360`):
`TCtrMeanHistory { sum: f32, count: i64 }`; per document read `(s, c)` **before** adding
`class as f32 / divisor` with `divisor = classes.saturating_sub(1).max(1) as f32`;
`value = calc_ctr_online(f64::from(s), c, prior)` — i.e. **the f32 sum is widened to f64
only at the value computation**. `[V]` `ctr_feature.rs:284-293` takes
`nums = prefix.sum.map(f64::from)`, `denoms = prefix.count`, `quantize_in_f32 = true`.

**Design decision (`SPEC.md` §9, locked)**: option **(b) honest accumulator**, not the
Borders alias. It survives multiclass and gives a structural device-vs-CPU oracle. T15
proves (a) and (b) agree at binclf.

**No new quantize kernel** (R-3 refuted by spike Q2: 4,504,501 pairs/prior, 0 mismatches
for every prior in `[0,1]`). Reuse `binarize_ctr_kernel` and the existing
`build_device_ctr_config` border table **unchanged** (C-7).

**CubeCL prerequisites**: `INDEX.md`; `cubecl_error_solution_guide/` on any build error;
`if`/`else` statements; `while` + explicit counter. **The `Array<f32>` accumulator stays
concrete** (§2.4's documented generics-float exception) with the
`TCtrMeanHistory::Sum is float` / `online.rs:294` citation in the comment.

**Files / symbols**
- Create: `#[cube(launch)] fn btmv_ctr_prefix_kernel` in
  `crates/cb-backend/src/kernels/ctr_device.rs`. Parameters mirror
  `ordered_ctr_prefix_kernel` (`perm`, `bins`, `class`, `prior`) plus
  `divisor: &Array<f32>` (length 1) and two scratch arrays `sums: &mut Array<f32>`,
  `cnts: &mut Array<u32>` (length `bucket_count`, host-pre-zeroed). Per position, serial
  on unit 0:
  ```
  read (s, c) BEFORE incrementing;
  out_sum[doc] = s;  out_cnt[doc] = c;
  value[doc]   = (f64::cast_from(s) + pr) / (f64::cast_from(c) + 1.0);   // widen ONLY here
  sums[bucket] = s + (f32::cast_from(class[doc]) / dv);                  // f32 add
  cnts[bucket] = c + 1u32;
  ```
- Create: `pub(crate) fn launch_btmv_ctr_resident(client, perm, bins, class, prior, divisor, bucket_count, n) -> CbResult<ResidentCtrMean>`
  where `ResidentCtrMean { sum: Handle /* f32 */, count: Handle /* u32 */, value: Handle /* f64 */ }`.
  Same host guards, same wgpu rejection, same `n == 0` short-circuit.
- Test: `crates/cb-backend/src/kernels/ctr_device_test.rs`.

**Red**
- Test name A (**output-width pin**, C-2.1 — *not* the width proof):
  `btmv_sum_output_buffer_is_f32_wide`.
  Setup: run the launch at `n = 64`; read the `ResidentCtrMean.sum` handle back and assert
  `bytes.len() == n * std::mem::size_of::<f32>()` and that
  `bytemuck::cast_slice::<u8, f32>(&bytes).len() == n`.
  **Scope, stated in the test's own comment**: this pins the per-document **output**
  buffer, not the per-bucket `sums` accumulator — an `Array<f64>` bucket history feeding an
  `f32` output would still pass. It is a cheap shape-regression guard. v1 claimed "an f64
  accumulator produces `n * 8` bytes"; that assumed the two share a type. **Withdrawn.**
  Detector B below is the actual width proof.
- Test name B (**numeric width detector**, C-2.2):
  `btmv_f32_accumulation_width_is_load_bearing`.
  Setup: `n = 96`, `cardinality = 4`, **`classes = 4` ⇒ `divisor = 3`**, so the added
  values `{0, 1/3, 2/3, 1}` are inexact in f32. Inline CPU reference accumulating in
  **`f32`** (transcribed — cb-backend cannot `use cb_train`). Assert the device `sum`
  vector is **bit-equal** (`f32::to_bits`) to the reference for every object.
  Expected failure under an f64 accumulator: a bit mismatch from the third document of any
  bucket onward, e.g. `left: 0x3F2AAAAB, right: 0x3F2AAAAA`. **The checker measured this
  detector: 22–41 mismatching documents out of 96 at `divisor = 3`** — it is real and
  strongly discriminating, which is why DCTR-12's claim rests on B and not on A.
  **Non-vacuity guard**: assert at least one bucket accumulates ≥3 documents.
- Test name C (**binclf parity**): `btmv_prefix_matches_cpu_reference_at_binclf`
  — `classes = 2, divisor = 1`, assert `sum`/`count`/`value` match the reference, plus a
  bit-exact binarized bin column through the existing border table.
- Expected initial failure (first Red run): **compile error** `cannot find function
  'launch_btmv_ctr_resident' in this scope`.
- Run: `cargo test -p cb-backend --no-default-features --features rocm --lib kernels::ctr_device_test`

**Green**: the kernel, the wrapper, the `ResidentCtrMean` struct. Do **not** wire it into
`build_ctr_cindex_columns` or the gate (T16).

**Refactor + regression scope**
- Doc-comment with `CalcOnlineCTRMean` (`online_ctr.cpp:437-501`, `:762`'s
  `targetClassesCount - 1`), `TCtrMeanHistory::Sum` is `float` (`online_ctr.h:373`), the
  repo pin `online_test::btmv_sum_is_accumulated_in_f32_not_f64`, **and C-2's statement
  that the width is numerically undetectable at binclf** so the multiclass test is the
  real detector.
- Regression scope: every existing `ctr_device_test` oracle + the full device suite.

**Validation commands**
```
cargo check -p cb-backend --no-default-features --features rocm
cargo test -p cb-backend --no-default-features --features rocm --lib kernels::ctr_device_test
cargo test -p cb-backend --no-default-features --features rocm
bash ./run_device_tests.sh
```

**Completion evidence**: all three tests green; the non-vacuity guard satisfied; **the
implementer must additionally record the observed failure of test B under a deliberately
f64-widened accumulator** (a one-line mutation), because that is DCTR-12's central claim.

**Rollback**: delete the kernel, the wrapper, the struct and the three tests.

**Risks / guardrails**: R-4 — widening the accumulator "for precision" is the specified
failure mode. Test B is the guard, and its mutation evidence is mandatory.

---

### T15 — BTMV ≡ Borders@0 at binary classification (structural cross-check)

- **Specs**: DCTR-13
- **Depends on**: T14 — **WAVE 5**
- **Parallel**: YES with T13. Test-only, same file as T14 (serialise with T14, then free).

**Goal / observable completion**: with the same prior and the same input column, the
device **BTMV** path and the device **Borders@0** path emit **bit-identical** `u32` bin
columns at binclf — justifying option (b) while proving it agrees with the alias today.

**Disk evidence**: `[V]` upstream `CalcOnlineCTRMean` adds
`float(permutedTargetClass[docId]) / targetBorderCount` with
`targetBorderCount = targetClassesCount - 1 = 1` at binclf (`online_ctr.cpp:467`, `:762`)
⇒ `Sum` is the count of class-1 documents seen and `Count` is documents seen — exactly the
`good`/`total` the ordered prefix kernel already produces — and **both feed the same
`CalcCTR`**. Mirrored in-repo by `online_mean_prefix`'s `divisor = 1` and
`calc_ctr_online(f64::from(s), c, prior)`.

**Files / symbols**
- Test only: `crates/cb-backend/src/kernels/ctr_device_test.rs`.
- Production files: **none**.

**Red**
- Test name: `btmv_and_borders_emit_identical_bins_at_binclf`.
- Setup: one `synth_fixture(128, 5, 3)` column, `prior = 0.5`, the same border table
  built the way `build_device_ctr_config:2503-2511` builds it
  (`borders[k] = ((k+1)·norm/bc − shift).next_down()`, `bc = 15`). Run the column twice:
  once through `launch_ordered_ctr_resident` with `(ctr_type = 0, target_border_idx = 0)`,
  once through `launch_btmv_ctr_resident` with `divisor = 1`; binarize both with
  `binarize_ctr_column_resident`; assert the two `Vec<u32>` bin columns are **equal**.
- **Non-vacuity guard**: assert the bin column has ≥2 distinct values (otherwise a
  degenerate all-zero column would make the equality trivial).
- **§2.5 status (checker's judgement request, MINOR-14).** This test is green by
  construction once T14 is right, so it is the plan's **eighth** green-on-write candidate.
  A production mutation is **not** required here, and the `≥2 distinct values` guard is
  the accepted substitute: it is what makes the equality non-trivial, and T14's own
  detector-B mutation already proves the accumulator is discriminating. Record that
  reasoning in the completion note rather than silently skipping §2.5.
- Expected initial failure before T14 lands: compile error (no BTMV launch). After T14,
  if the accumulator is wrong, the expected failure is
  `assertion `left == right` failed` on the bin vectors with the first differing index
  printed.
- Run: `cargo test -p cb-backend --no-default-features --features rocm --lib kernels::ctr_device_test`

**Green**: none if T14 is correct. If it fails, the defect is in T14's accumulator — fix
**there**, not here.

**Refactor + regression scope**
- Doc-comment the test with the upstream equivalence proof and the explicit note that this
  identity holds **only at binclf** (`SIMPLE_CLASSES_COUNT == 2`, `online.rs:52`) and will
  break the day multiclass CTR lands — which is precisely why option (b) was chosen over
  aliasing.
- Regression scope: none (test-only).

**Validation commands**
```
cargo test -p cb-backend --no-default-features --features rocm --lib kernels::ctr_device_test
```

**Completion evidence**: the cross-check green with the non-vacuity guard satisfied.

**Rollback**: delete the test.

---

### T16 — BTMV end-to-end oracle with device commitment

- **Specs**: DCTR-14
- **Depends on**: **T00**, T04 (corrected CPU), T07 (fixture), T12 (gate chain), T14 — **WAVE 5**
- **Parallel**: NO — fourth link in the gate chain.

**Goal / observable completion**: a `simple_ctr = BinarizedTargetMeanValue:Prior=0.5` fit
**commits to the device** and matches upstream at **≤1e-5**, against the **corrected**
CPU.

**Files / symbols**
- Modify: `crates/cb-backend/src/gpu_runtime/session.rs` `ctr_covered` — widen to
  `matches!(col.ctr_type, 0 | 1 | 2 | 4)`.
- Modify: `crates/cb-backend/src/gpu_runtime/session.rs` `build_ctr_cindex_columns` — add
  the `2 =>` arm calling `launch_btmv_ctr_resident` with `divisor = 1.0f32` (binclf), then
  the **unchanged** `binarize_ctr_column_resident` on its `value` handle.
- Modify: `crates/cb-train/src/boosting.rs` gate — add
  `ECtrType::BinarizedTargetMeanValue`.
- Modify: `crates/cb-train/src/device_ctr_combo_config_test.rs` — flip **row 5**
  (`simple / BTMV / b=0 / denom=1.0`) of T00's gate-state table to `true`, **in the same
  commit as the gate widening** (same contract and same failure mode as T12's row 3).
- Create: `crates/cb-train/tests/device_ctr_btmv_fit_test.rs`.

**Red**
- Test name: `device_ctr_btmv_fit_commits_and_matches_upstream`.
- Setup: the `ctr_device_btmv` fixture; params as T10 but
  `simple_ctr: cb_train::ECtrType::BinarizedTargetMeanValue`, `simple_ctr_priors: vec![0.5]`.
- Assertions: `grown.get() == params.iterations`; ≤1e-5 per object; permutation
  divergence guard; `Σ ctr_splits >= 1`.
- Expected initial failure: `assertion `left == right` failed: left: 0, right: 5`.
- Run: `cargo test -p cb-train --no-default-features --features rocm --test device_ctr_btmv_fit_test`

**Green**: the three widenings. Nothing else.

**Refactor + regression scope**
- Update the gate doc block for BTMV, citing T14's accumulator, T15's binclf equivalence,
  and **T04 as the parity target** (device == corrected CPU, never the pre-T04 form).
- Regression scope: full device suite + `ctr_btmv_simple_oracle_test` and
  `ctr_btmv_bake_upstream_table_test` (CPU, must be unchanged-green — DCTR-05).

**Validation commands**
```
cargo test -p cb-train --lib device_ctr_combo_config_tests     # CRITICAL-1: mandatory on every gate task
cargo check -p cb-train --no-default-features --features rocm --test device_ctr_btmv_fit_test
cargo test -p cb-train --no-default-features --features rocm --test device_ctr_btmv_fit_test
cargo test -p cb-train --test ctr_btmv_simple_oracle_test
cargo test -p cb-train --test ctr_btmv_bake_upstream_table_test
cargo test -p cb-train --no-default-features --features rocm --test device_ctr_counter_fit_test
cargo test -p cb-train --no-default-features --features rocm --test device_ctr_buckets_fit_test
bash ./run_device_tests.sh
```

**Completion evidence**: `grown == 5`, `max |Δpred|` recorded; both CPU BTMV oracles
green **and unchanged**; 23/23 device PASS.

**Rollback**: remove `2` / `ECtrType::BinarizedTargetMeanValue` from the three lists.

---

### T17 — Per-level combination-eligibility gate on the device (D-1)

- **Specs**: DCTR-15
- **Depends on**: T02 (seam), **T10, T12, T16** (Track D must be last — the gate has to
  cover the Buckets/Counter/BTMV columns those tasks added) — **WAVE 6**
- **Parallel**: NO. Edits `gpu_runtime/mod.rs`'s hottest shared path.

**Goal / observable completion**: pass C skips — **before scoring** — every CTR column
whose `projection_members.len() >= 2` for which no already-chosen projection `q` **in this
tree** satisfies `q.len() + 1 == members.len() && q ⊆ members`; at level 0 of **every**
tree the chosen list is empty ⇒ no ≥2-member column is scored.

**Disk evidence**
- `[V]` the device today scores every column at every level ungated:
  `mod.rs:4907 for c in 0..cs.n_ctr { … }` — no eligibility check anywhere in
  `ResidentCtrSearch` (`mod.rs:4360-4378`, whose doc explicitly claims *"DEVICE SCOPE:
  simple (single-feature) projections only … so no combination-eligibility logic exists
  here"* — that claim becomes false here and must be rewritten).
- `[V]` the CPU mirror to transcribe: `combination_ctr_eligible`
  (`cb-train/src/tree.rs:2896-2906`) and the pre-scoring `continue` in
  `select_level_ctr_aware` (`tree.rs:3134-3146`).
- `[V]` upstream: `AddTreeCtrs` builds `seenProj` from
  `{binAndOneHotFeaturesTree} ∪ currentTree.GetUsedCtrs()` and `continue`s on
  `baseProj.IsEmpty()` (`greedy_tensor_search.cpp:491-551`), called per level from the
  oblivious search — **so no combination candidate exists at level 0 of any tree.**
- **C-5**: `grow_oblivious_tree_resident` runs once per tree ⇒ a function-local `Vec` is
  tree-lifetime for free. `group_used` on `CtrSearchState` (`session.rs:184-186`) is
  **fit-lifetime** and must **not** be reused for this.

**Files / symbols**
- Modify: `crates/cb-backend/src/gpu_runtime/session.rs:178-198` `CtrSearchState` — add
  `projection_members: Vec<Vec<u32>>`.
- Modify: `crates/cb-backend/src/gpu_runtime/session.rs:1768-1776` — populate it from
  `ctr.columns.iter().map(|c| c.projection_members.clone())`.
- Modify: `crates/cb-backend/src/gpu_runtime/mod.rs:4360-4378` `ResidentCtrSearch` — add
  `pub projection_members: &'a [Vec<u32>]` and **rewrite the "simple projections only"
  doc paragraph**.
- Modify: **`crates/cb-backend/src/gpu_runtime/session.rs:2593`** — the
  `ResidentCtrSearch { … }` struct literal (`let mut ctr_search_view = self.ctr_search
  .as_mut().map(|st| ResidentCtrSearch {`). v1 omitted this site; it is the **only**
  construction of the borrowed view and the new field must be wired here or nothing
  reaches pass C. (Checker MINOR-6.)
- Modify: **`crates/cb-backend/src/gpu_runtime/session.rs:1746-1763`** — the
  `bucket_counts` fallback and its **false comment at `:1748`** (checker MAJOR-1 /
  DCTR-20). Today the `bucket_count == 0` branch reads `col.member_bins.first()` only,
  justified by the comment *"(single-member columns; the gate admits only simple
  projections)"* — which T19 makes false. Two required changes:
  1. **Delete the false clause** from the comment and state the new truth: combination
     columns can now reach here.
  2. Make the fallback **correct for a combination**: either fold all members
     (the same `combine_projection_bins` identity `build_ctr_cindex_columns` uses at
     `session.rs:213-219`) or return a typed `CbError::Degenerate` naming the column —
     **never** silently take member 0's cardinality, which under-counts the projection and
     therefore inflates every *other* column's `cat_feature_weight`.
     Note the branch is unreachable in practice today (`build_device_ctr_config` always
     sets `bucket_count = col.bucket_count > 0` from the materialized column), so the
     typed-error option is acceptable and is the cheaper, honest choice; state which was
     taken and why in the completion note.
- Create: `fn resident_combination_eligible(members: &[u32], chosen: &[Vec<u32>]) -> bool`
  in `mod.rs` next to `resident_cat_feature_weight` (`:4383-4388`) — **one** transcription
  read by both this task and T18 (the CPU keeps the same lockstep between
  `select_level_ctr_aware` and `eligible_max_bucket_count`):
  ```rust
  members.len() < 2
      || chosen.iter().any(|q| q.len() + 1 == members.len()
                               && q.iter().all(|m| members.contains(m)))
  ```
- Modify: `crates/cb-backend/src/gpu_runtime/mod.rs` inside `grow_oblivious_tree_resident`,
  above the level loop — `let mut chosen_ctr_projections: Vec<Vec<u32>> = Vec::new();`
  with a comment naming the **tree-lifetime** scope and contrasting it with the
  fit-lifetime `group_used` (R-2).
- Modify: **`mod.rs:4907`** (`for c in 0..cs.n_ctr {` — corrected from v1's `:4919`, C-13)
  — `continue` past ineligible columns **before** `score_partition_over_binsums`, so the
  strict-`>` first-wins order over survivors matches the CPU's.
- Modify: `mod.rs:4980-4990` (the CTR-winner bookkeeping) — in addition to lifting
  `group_used`, push the winner's `projection_members` onto `chosen_ctr_projections`.
- Create: `crates/cb-backend/src/gpu_runtime/ctr_eligibility_test.rs` — new sibling test
  file, mounted in `mod.rs` under `#[cfg(test)] #[path = …] mod ctr_eligibility_test;`
  (match the existing sibling-mount form in that module).

**Red**
- Test name: `resident_combination_eligible_matches_cpu_rule`.
- File: `crates/cb-backend/src/gpu_runtime/ctr_eligibility_test.rs`. **Pure host unit
  test — no GPU required**, so it runs under `cargo test -p cb-backend` on any backend.
- Cases (each an independent assertion):
  1. `members = [3]`, `chosen = []` ⇒ **true** (simple is always eligible,
     `AddSimpleCtrs` is unconditional);
  2. `members = [1,3]`, `chosen = []` ⇒ **false** (the level-0 / `baseProj.IsEmpty()` skip);
  3. `members = [1,3]`, `chosen = [[1]]` ⇒ **true**;
  4. `members = [1,3]`, `chosen = [[2]]` ⇒ **false** (not a subset);
  5. `members = [1,2,3]`, `chosen = [[1]]` ⇒ **false** (`|q|+1 != |p|`);
  6. `members = [1,2,3]`, `chosen = [[1,2]]` ⇒ **true**.
- **Transcribe the case list from the CPU's already-covered tests** (checker MINOR-11)
  rather than inventing it: `crates/cb-train/src/tree_test.rs` covers
  `combination_ctr_eligible` and the `cat_feature_weight` / `phantom_mixed_bucket_count`
  family. Read those cases first and mirror them one-for-one, so the device predicate is
  provably the same function on the same inputs. Any case the CPU covers that the six above
  miss must be added; any divergence in expected value is a **defect in the transcription,
  not a device design choice** — stop and reconcile.
- Expected initial failure: **compile error** `cannot find function
  'resident_combination_eligible' in this scope`.
- Run: `cargo test -p cb-backend --no-default-features --features rocm --lib gpu_runtime::ctr_eligibility_test`

**Green**
- The predicate, the seam→state→view threading, the per-tree local list, the pass-C
  `continue`, and the bookkeeping push. Do **not** touch `eligible_max` (T18) and do
  **not** relax the cb-train arity conjunct (T19).

**Refactor + regression scope**
- Rewrite `ResidentCtrSearch`'s doc paragraph and pass C's `eligible_max` comment
  (`mod.rs:4893-4895`, which currently asserts *"simple columns are always eligible — the
  device gate admits only simple projections"* — half of that becomes false here, the rest
  at T19).
- **Regression scope is the entire device suite**: pass C is shared by every device
  oblivious CTR fit. With the arity conjunct still in place, every column has
  `projection_members.len() == 1` ⇒ the predicate returns `true` unconditionally ⇒ all
  existing behaviour must be **byte-unchanged**. That is this task's safety property and
  must be verified, not assumed.

**Validation commands**
```
cargo check -p cb-backend --no-default-features --features rocm
cargo test -p cb-backend --no-default-features --features rocm --lib gpu_runtime::ctr_eligibility_test
cargo test -p cb-backend --no-default-features --features rocm
cargo test -p cb-train --no-default-features --features rocm --test device_ctr_fit_test
cargo test -p cb-train --no-default-features --features rocm --test device_ctr_buckets_fit_test
cargo test -p cb-train --no-default-features --features rocm --test device_ctr_counter_fit_test
cargo test -p cb-train --no-default-features --features rocm --test device_ctr_btmv_fit_test
bash ./run_device_tests.sh
```

**Completion evidence**: the six predicate cases green; **every existing device CTR e2e's
printed `max |Δpred|` byte-identical to its pre-T17 value**; 23/23 device PASS.

**Rollback**: revert the `continue` first (restores today's scoring), then the state
threading. The predicate itself is inert without the `continue`.

**Risks / guardrails**: **R-2 (fit-lifetime vs tree-lifetime)** is the specified failure
mode. The list must be declared **inside** `grow_oblivious_tree_resident`, never on
`CtrSearchState`. T19's Verify executes the mutation check that detects the violation.

---

### T18 — `eligible_max` uses the same eligibility filter (D-2)

- **Specs**: DCTR-16
- **Depends on**: T17 — **WAVE 6**
- **Parallel**: NO — same function as T17.

**Goal / observable completion**: `eligible_max` maxes `bucket_counts` over **eligible**
columns only, matching `eligible_max_bucket_count`, with a **dedicated unit test** on the
filtered max — because `ctr_device_combo` provably does **not** discriminate this change
(R-20: D-1 alone already passes it at 2.082e-17).

**Disk evidence**: `[V]` `mod.rs:4895`
`let eligible_max = cs.bucket_counts.iter().copied().max().unwrap_or(1).max(1);`
— unfiltered. `[V]` the CPU counterpart `eligible_max_bucket_count`
(`cb-train/src/tree.rs:2919-2933`) filters on
`c.projection.is_simple() || combination_ctr_eligible(&c.projection, used_projections)`
and its own doc comment demands the two gates stay in lockstep. Including an ineligible
combination's larger count inflates `maxCount`, and `(1 + count/maxCount)^(-0.5)` is
**increasing** in `maxCount`, so it raises `cat_feature_weight` for every unused simple
candidate — an independent way to flip the greedy winner.

**⚠ INVARIANT — do NOT filter `phantom_max` (C-16 / checker MINOR-12d).**
`[V]` device `mod.rs:4905`: `let max_bucket_count = eligible_max.max(phantom_max).max(1);`
`[V]` CPU `tree.rs:3116-3125`: `eligible_max_bucket_count(...)` is computed **first**, then
handed to `max_bucket_count_with_phantom(...)`, which ends
`eligible_max.max(phantom_max).max(1)` (`tree.rs:3033`). ⇒ the eligibility filter scopes
**`eligible_max` only**; the phantom mixed float-partition count is folded in **outside**
it (upstream's `binAndOneHotFeaturesTree` is unconditional in `AddTreeCtrs`). T18 replaces
the **inner** max at `:4895` and leaves `:4905`'s `.max(phantom_max).max(1)` untouched.
Filtering the phantom too would diverge from the CPU in the opposite direction.

**Files / symbols**
- Create: `fn resident_eligible_max_bucket_count(bucket_counts: &[usize], projection_members: &[Vec<u32>], chosen: &[Vec<u32>]) -> usize`
  in `mod.rs` next to `resident_combination_eligible`, reading **that same predicate**
  (never a second copy of the rule). Preserve the existing `.unwrap_or(1).max(1)` guard
  verbatim.
- Modify: `crates/cb-backend/src/gpu_runtime/mod.rs:4895` — call it. **Do not touch
  `:4905`.**
- Mirror the CPU's already-covered cases from `crates/cb-train/src/tree_test.rs`
  (`eligible_max_bucket_count` / `max_bucket_count_with_phantom` coverage) rather than
  inventing the table — checker MINOR-11, same rule as T17.
- Test: `crates/cb-backend/src/gpu_runtime/ctr_eligibility_test.rs` (T17's file).

**Red**
- Test name: `eligible_max_excludes_ineligible_combinations`.
- **Pure host unit test — no GPU.**
- Setup: `bucket_counts = [4, 6, 40]`; `projection_members = [[0], [1], [0,1]]` (two
  simple columns and one 2-member combination whose bucket count **dominates**).
  - `chosen = []` ⇒ the combination is ineligible ⇒ expect **6** (the max over the two
    simple columns).
  - `chosen = [[0]]` ⇒ the combination becomes eligible ⇒ expect **40**.
  - `bucket_counts = []` ⇒ expect **1** (the preserved guard).
  - **A fourth case pinning C-16**: with `phantom_max = 100` folded in at the call site,
    the result is `100` for **both** `chosen` states — i.e. the phantom is not filtered.
    Assert this against the composed expression, not the helper alone.
- Expected initial failure: with the unfiltered max, case 1 returns **40** ⇒
  `assertion `left == right` failed: left: 40, right: 6`.
- Run: `cargo test -p cb-backend --no-default-features --features rocm --lib gpu_runtime::ctr_eligibility_test`

**Green**: extract the helper, filter, call it. Nothing else.

**Refactor + regression scope**
- Rewrite the `eligible_max` comment (`mod.rs:4893-4895`) with the
  `CalcMaxFeatureValueCount` (`greedy_tensor_search.cpp:1070-1088`, recomputed **per
  level** over the gated candidate list) and `eligible_max_bucket_count` citations, plus
  the C-16 phantom-scope invariant.
- **Record R-20 in the code**, with its v2 status: no committed *upstream* fixture detects
  this change, and the unit test above proves only that the helper filters — **not** that
  it is wired into pass C. **T22's device-vs-CPU split-sequence differential (DCTR-20) is
  the wiring detector**, and `SPEC.md` R-20 now names it as the primary evidence,
  superseding the unit test. If T22 measures that reverting D-2 does **not** change the
  split sequence, R-20 stays open and that must be recorded — not glossed.
- Regression scope: full device suite. With every column simple, the filter is the
  identity ⇒ byte-unchanged.

**Validation commands**
```
cargo test -p cb-backend --no-default-features --features rocm --lib gpu_runtime::ctr_eligibility_test
cargo test -p cb-backend --no-default-features --features rocm
cargo test -p cb-train --no-default-features --features rocm --test device_ctr_fit_test
bash ./run_device_tests.sh
```

**Completion evidence**: the three max cases green; device suite 23/23; the R-20 gap
comment present in `mod.rs`.

**Rollback**: restore the one-line unfiltered max.

**Risks / guardrails**: **do not skip this task** because T19 passes without it (R-20).
Its correctness claim is CPU parity, not the combo fixture.

---

### T19 — Combination CTR end-to-end: un-ignore, `CountingGpu`, drop the arity conjunct

- **Specs**: DCTR-17
- **Depends on**: **T00**, T16 (gate chain), T18 — **WAVE 6**
- **Parallel**: NO — fifth link in the gate chain.

**Goal / observable completion**: `device_ctr_combo_fit_test` runs **un-ignored**, asserts
`grown.get() == params.iterations`, and matches upstream at ≤1e-5. Measured expectation
**≈2.082e-17** — deliberately **different** from the CPU-fallback `1.388e-17`, which is
independent evidence the device path actually ran.

**Disk evidence**: `[V]` the current `#[ignore]` at
`crates/cb-train/tests/device_ctr_combo_fit_test.rs:179` (its doc block is `:169-178`) with the rationale *"this fit
runs on the CPU grower and the arm-routing assertion below would fail"* — **factually
wrong**: the test's only routing assertions are
`trained.oblivious_trees.len() == params.iterations` and
`non_symmetric_trees.is_empty() && region_trees.is_empty()` (`:119-120` region of the file,
verified), both of which the **CPU** oblivious grower satisfies. The test passes today
under `--ignored` at `1.388e-17`, the CPU number quoted in the gate's own doc comment.
That is the R-8 false-pass class in its purest form.

**Files / symbols**
- Modify: `crates/cb-train/src/boosting.rs:2361` — delete `col.projection.is_simple() &&`.
- Modify: `crates/cb-train/tests/device_ctr_combo_fit_test.rs` — remove `#[ignore = …]`
  and its wrong rationale paragraph; add the `CountingGpu` wrapper (verbatim from
  `device_ctr_gate_test.rs:60-170`); drive the fit through `CountingGpu` instead of
  `GpuBackend::default()`; add `assert_eq!(gpu.grown.get(), params.iterations)`; **keep**
  the existing `assert_ne!(structure, averaging)` divergence guard and the
  `n_ctr_splits >= 1` vacuity guard.
- Modify: `crates/cb-train/src/boosting.rs` — rewrite the FPP-11 escalation block on
  `ctr_types_are_device_covered` (`:2298-2354`): suspect 2 (`bucket_count` semantics) is
  **refuted** (`TOnlineCtrUniqValuesCounts::Count = ComputeReindexHash(topSize = Max<ui64>(), learn range)`,
  `online_ctr.cpp:696-700`, `cat_feature_options.cpp:236`); suspect 1 (`eligible_max`) is
  **confirmed and fixed** by T18; the **primary** cause — a missing per-level candidate
  eligibility gate — was previously unnamed and is fixed by T17.

**Red**
- Test name: `device_ctr_combo_fit_matches_upstream_predictions` (existing function,
  rewritten).
- Setup: unchanged fixture (`ctr_device_combo`), unchanged `combo_params()`
  (`max_ctr_complexity: 2`, `combinations_ctr: ECtrType::Borders`,
  `combinations_ctr_priors: vec![0.5]`).
- Expected initial failure **with the `#[ignore]` removed and `CountingGpu` added but
  before the gate edit**: `assertion `left == right` failed: left: 0, right: 5` on
  `gpu.grown.get()` — the fit is on the CPU grower.
- Expected failure **if T17 were absent** (the measured control arm): the ≤1e-5 assertion
  fires at `obj 0: … |Δ|=2.746e-2`.
- Run: `cargo test -p cb-train --no-default-features --features rocm --test device_ctr_combo_fit_test`

**Green**: delete the arity conjunct, **plus T00's row-2 flip** (below). Nothing else.

**Files / symbols**
- Modify: `crates/cb-train/src/boosting.rs` gate — delete the
  `col.projection.is_simple()` conjunct.
- Modify: `crates/cb-train/src/device_ctr_combo_config_test.rs` — flip **row 2**
  (`combo / Borders / b=0 / denom=1.0`) of T00's gate-state table to `true`, **in the
  same commit as the conjunct deletion** (T00's contract, `flips_at: T19`). Rows 3/4/6
  were already discharged by T12/T10/T01 — do not touch them.

**Refactor + regression scope**
- The doc-block rewrite above, plus a note that the expected value is `≈2.082e-17`, not
  `1.388e-17`.
- Regression scope: full device suite; the `#[ignore]` removal changes
  `run_device_tests.sh`'s `device_ctr_combo_fit_test` line from "1 ignored" to a real run.

**Validation commands**
```
cargo test -p cb-train --lib device_ctr_combo_config_tests     # CRITICAL-1: flips row 2 (arity) ONLY — rows 3/4/6 were already discharged by T12/T10/T01 (V-0c)
cargo check -p cb-train --no-default-features --features rocm --test device_ctr_combo_fit_test
cargo test -p cb-train --no-default-features --features rocm --test device_ctr_combo_fit_test
cargo test -p cb-train --no-default-features --features rocm --test device_ctr_fit_test
cargo test -p cb-train --no-default-features --features rocm --test device_ctr_buckets_fit_test
cargo test -p cb-train --no-default-features --features rocm --test device_ctr_counter_fit_test
cargo test -p cb-train --no-default-features --features rocm --test device_ctr_btmv_fit_test
cargo test -p cb-train --test tensor_ctr_e2e_oracle_test
cargo test -p cb-train --test ctr_mixed_simple_vs_combo_oracle_test
bash ./run_device_tests.sh
```

**MANDATORY mutation checks (both must be executed and recorded)**
1. **R-2 / C-5 scope check**: hoist `chosen_ctr_projections` from the
   `grow_oblivious_tree_resident` local onto `CtrSearchState` (making it fit-lifetime) and
   confirm this test **FAILS** — under a fit-lifetime list, tree ≥ 1 level 0 admits the
   combination, which upstream structurally cannot have. Revert. Record the observed
   `|Δ|`.
2. **R-8 gate-closed check**: temporarily restore `col.projection.is_simple() &&` and
   confirm the test **FAILS on `grown.get()`**, not on the prediction bar. Revert.

**Completion evidence**: `grown == 5`; `max |Δpred|` recorded and compared against the
measured `2.082e-17`; both mutation failures recorded; 23/23 device PASS.

**Rollback**: restore the conjunct and re-add `#[ignore]` with a **corrected** rationale
(the current one is wrong and must not be restored verbatim).

---

### T20 — `CountingGpu` device-commit assertion on `device_ctr_fit_test`

- **Specs**: DCTR-19
- **Depends on**: **T01** — **WAVE 2** (moved out of Wave 0 by checker MINOR-9: this
  task's §2.5 mutation edits the same expression T01 edits, and no mutation may be live
  while another task runs a suite-wide validation)
- **Parallel**: YES with T08 (different files) once T01 has landed. Owns exactly one test file.

**Goal / observable completion**: `device_ctr_fit_test` proves the fit **committed to the
device** rather than merely producing oblivious trees, closing the R-8 false-pass class on
the phase's oldest CTR e2e.

**Disk evidence**: `[V]` `crates/cb-train/tests/device_ctr_fit_test.rs:119-120` asserts
only `trained.oblivious_trees.len() == params.iterations` and
`non_symmetric_trees.is_empty() && region_trees.is_empty()` — both satisfied by the CPU
oblivious grower. There is no `CountingGpu` in the file.

**Files / symbols**
- Modify: `crates/cb-train/tests/device_ctr_fit_test.rs` — add the `CountingGpu` wrapper
  verbatim from `device_ctr_gate_test.rs:60-170`; drive the fit through it; add
  `assert_eq!(gpu.grown.get(), params.iterations, "…")`.
- Production files: **none**.

**Red**
- Test name: the file's existing `device_ctr_fit_matches_upstream_predictions`
  (`:149-150`), extended.
- **This assertion is green-on-write** — the `ctr_device_mixed` fit is already
  device-covered today (simple Borders, `target_border_idx = 0`). §2.5 mutation check:
  temporarily add `&& false` to `ctr_types_are_device_covered`'s closure and confirm the
  test **FAILS with `assertion `left == right` failed: left: 0, right: 5`** (and note the
  prediction bar would still have **passed**, which is exactly the R-8 point). Revert.
- Run: `cargo test -p cb-train --no-default-features --features rocm --test device_ctr_fit_test`

**Green**: the wrapper + one assertion. No production change.

**Refactor + regression scope**
- Add a module comment recording that `oblivious_trees.len()` is **not** a device-commit
  assertion, and pointing at `device_ctr_gate_test.rs:60-170` as the canonical wrapper.
- Regression scope: this one binary.

**Validation commands**
```
cargo check -p cb-train --no-default-features --features rocm --test device_ctr_fit_test
cargo test -p cb-train --no-default-features --features rocm --test device_ctr_fit_test
```

**Completion evidence**: green with `grown == 5`; the mutation failure recorded **with the
observation that the ≤1e-5 assertion still passed under the mutation** — the documented
proof that the old assertion was insufficient.

**Rollback**: remove the wrapper and the assertion.

---

### T21 — One-hot × CTR keeps declining, plus the surviving-clause boundary pins

- **Specs**: DCTR-03 (plus `SPEC.md` §6 acceptance scenario 9's un-ID'd pins)
- **Depends on**: T19 (the gate must be in its final-behaviour state) — **WAVE 7**
- **Parallel**: YES with T22 on **file ownership** (T21 owns
  `crates/cb-train/tests/device_ctr_type_gate_test.rs`; T22 owns
  `crates/cb-train/tests/device_ctr_combo_types_diff_test.rs` — **not**
  `boosting_ctr_gate_test.rs`, whose owners are T01 → T23 per §5.3).
  ⚠ **But the §2.5 isolation rule (see §2.5, "no mutation may be live while another task
  runs a workspace-wide validation") governs this pair and is NOT satisfied by file
  disjointness alone** — the same reasoning that forced T20 to serialise after T01.
  T21 mutates production `boosting.rs:3937`/`:4425`; T22 mutates `mod.rs:4895` and
  `session.rs`'s fallback; and **both** run `bash ./run_device_tests.sh`. Run their
  mutation windows strictly one at a time, or serialise T21 after T22. T13 also wrote to T21's file —
  coordinate or rebase.

**Goal / observable completion**: a pool mixing one-hot columns and CTR features does
**not** commit to the device, with a passing negative test; and each remaining P2/P3
boundary clause has a passing characterization pin.

**Spec decision this task implements (diverging from the research — preserve it)**: the
research proposed **deleting** the `one_hot_bins.is_empty()` device conjunct
(`boosting.rs:4425`) as provably dead, on the grounds that SPEC-OH-26 rejects the mixed
pool earlier in `train_inner` (`[V]` `boosting.rs:3925-3946`, the typed
`CbError::Unsupported`). `SPEC.md` DCTR-03 **retains** it and adds the negative test
instead: deleting it would mean that the day the CPU gains a three-way candidate union,
one-hot × CTR silently reaches the device with **zero** device testing. Retention costs
nothing and removes a latent hazard. **Do not delete the conjunct.**

**Files / symbols**
- Modify: `crates/cb-train/tests/device_ctr_type_gate_test.rs` (T13's file).
- Modify: `crates/cb-train/src/boosting.rs` — doc-comment only, on the
  `one_hot_bins.is_empty()` conjunct at `:4425`: record that it is currently
  **unreachable-true** (case analysis over `partition_cat_columns`: all-one-hot ⇒ no CTR
  columns ⇒ the `materialized_ctr_features.is_empty()` arm is taken; all-CTR ⇒
  `one_hot_bins` is empty; mixed ⇒ `train_inner` already errored at `:3937`), that it is
  **deliberately retained** as defence in depth per DCTR-03, and that
  `one_hot_ctr_mix_declines_to_device` is its pin.

**⚠ A one-hot × CTR decline test ALREADY EXISTS (C-18 / checker MINOR-7).** `[V]`
`crates/cb-train/tests/device_fpp_composition_test.rs:255` `pub fn one_hot_x_ctr_still_declines()`,
driven by `fpp17_one_hot_x_ctr_still_declines` at `:381-385`. v1 would have authored a
third copy. **Do not.** DCTR-03's discharge is: (1) **re-run and record** that test after
T19; (2) execute its §2.5 mutation to prove it is still discriminating in the post-T19
gate state; (3) add the doc rationale below. Only if step 2 shows it has gone vacuous does
this task author a replacement — and then it *replaces*, it does not duplicate.

**Red**
- Primary: re-run `fpp17_one_hot_x_ctr_still_declines` and record the result.
- **§2.5 mutation check** (this is the real work): delete the SPEC-OH-26 rejection at
  `boosting.rs:3937` **and** the `one_hot_bins.is_empty()` conjunct at `:4425` together,
  confirm the existing test fails with a non-zero `grown`, then revert **both**. Record the
  message. If it does **not** fail, the test has gone vacuous post-T19 and this task must
  author a replacement asserting the **observable** — `Err(CbError::Unsupported(_))` **or**
  `gpu.grown.get() == 0`, never the specific layer, because two layers can satisfy it and
  the earlier one currently fires.
- Run: `cargo test -p cb-train --no-default-features --features rocm --test device_fpp_composition_test`

**Additional characterization pins (green-on-write; each needs its own mutation check)**
| pin | asserts | clause pinned | boundary |
|---|---|---|---|
| `cat_only_ctr_pool_declines_to_device` | `grown == 0` for a pool with **zero** float columns | `has_any_scorable_feature` (`boosting.rs:3284-3286`) | P2 (C-5) |
| `non_15_border_count_ctr_pool_declines_to_device` | `grown == 0` when the float borders are not length 15 | `ctr_covered`'s `borders.len() + 1 == n_bins` (`session.rs:155`) | P2 (C-1), R-11 |
| `multi_permutation_ctr_still_declines` | `grown == 0` at `permutation_count = 4` | `learning_folds_for_cycle == 1` (`boosting.rs:4424`) | P3 (C-2) |

The third **already exists** as `multi_permutation_ctr_declines_to_device` in
`device_ctr_gate_test.rs` — re-run it rather than duplicating it, and record that in the
completion note.

**P2/P3 inversion annotations (mandatory, checker MINOR-13).** Every pin above is a
*boundary marker*, not a requirement, and each must carry a comment naming the phase that
inverts it:
- `cat_only_ctr_pool_declines_to_device` — `// P2 WILL INVERT THIS (C-5).` P2's whole
  payoff is making the eight existing cat-only CTR fixtures device-testable.
- `non_15_border_count_ctr_pool_declines_to_device` — `// P2 WILL INVERT THIS (C-1).`
- `multi_permutation_ctr_still_declines` — `// P3 WILL INVERT THIS (C-2), and only for the
  anchored pc=4/seed=0 family (R-17).`
- `one_hot_x_ctr_still_declines` — **NOT** a boundary pin: DCTR-03 retains this clause
  deliberately and indefinitely. Annotate it `// RETAINED BY DESIGN (DCTR-03) — not a
  P2/P3 boundary.` so it is not mistaken for one and flipped.

**Green**: no production change (that is DCTR-03's whole point). The Green is the recorded
mutation evidence for each pin.

**Refactor + regression scope**
- The `one_hot_bins.is_empty()` doc comment above. Regression scope: none (test-only).

**Validation commands**
```
cargo check -p cb-train --no-default-features --features rocm --test device_ctr_type_gate_test
cargo test -p cb-train --no-default-features --features rocm --test device_ctr_type_gate_test
cargo test -p cb-train --no-default-features --features rocm --test device_ctr_gate_test
```

**Completion evidence**: all pins green; **one recorded mutation failure per pin**; the
`one_hot_bins.is_empty()` conjunct still present in the source.

**Rollback**: delete the added test functions and the doc comment.

---

### T22 — Combination × {Buckets, Counter, BTMV}: device-vs-CPU split-sequence differential

- **Specs**: **DCTR-20** (new in the revised SPEC)
- **Depends on**: T19 (combinations must reach the device), and therefore T10/T12/T16 —
  **WAVE 7**. **Blocks T23** (`SPEC.md` DCTR-20: the detector must pass **before** the
  final gate that admits this cross-product ships).
- **Parallel**: YES with T21 on file ownership — **but see T21 for the §2.5 isolation
  caveat**: both tasks mutate production code and both run `bash ./run_device_tests.sh`,
  so their mutation windows must not overlap.

**Why this task exists (checker MAJOR-1).** After T23 the gate admits **any** CPU-supported
type on **any-arity** projection. P1's delivered coverage is simple × {Borders, Buckets,
Counter, BTMV} and combination × **Borders only** (`[V]` `ctr_device_combo/config.json`
pins `combinations_ctr: ["Borders:Prior=0.5"]`). **Nothing exercises combination ×
{Buckets, Counter, BTMV}** — and the first thing that cross-product would hit is exactly
the `bucket_counts` fallback and false comment T17 repairs (`session.rs:1746-1765` — the
false comment at `:1748`, the fallback binding at `:1749-1765`). A
finish line that admits an untested cross-product is not a finish line.

**Goal / observable completion**: for each `combinations_ctr ∈ {Buckets, Counter, BTMV}`,
a device fit and a CPU fit over the **same** `ctr_device_combo` inputs choose the
**identical split sequence**, and their leaf values agree at ε = 1e-4 (D-07).

**Why a differential and not a fourth upstream fixture.** (a) No upstream fixture is
needed — this reuses the frozen `ctr_device_combo` corpus, so no new frozen artifact and
no seed search. (b) It is **strictly stronger** than a ≤1e-5 prediction comparison: it
compares the **chosen splits**, so a wrong `cat_feature_weight` that happens to produce
near-identical predictions still fails. (c) It gives DCTR-16/R-20 the wiring detector the
unit test cannot provide.

**Chain of trust, stated honestly.** This proves **device == CPU**, not device == upstream.
`CPU == upstream` for combination × non-Borders is established separately by the shipped
CPU oracles `ctr_mixed_simple_vs_combo_oracle_test` and `tensor_ctr_e2e_oracle_test`, which
this task must **also run** so the chain is closed end-to-end in one place. Record both
halves in the completion note.

**Files / symbols**
- Create: `crates/cb-train/tests/device_ctr_combo_types_diff_test.rs`.
- Production files: **none expected.** If the differential fails, the fix belongs in the
  owning task (T17's fallback, T18's filter, T09/T12/T16's dispatch) — **not** here.
- `[V]` the CPU arm needs no new machinery: `Runtime`'s `begin_device_training` has a
  default body (`cb-compute/src/runtime.rs:1461-1476`, `let _ = (…)`), so a local
  `struct CpuRefRuntime;` overriding **only** `compute_gradients` inherits the seam
  defaults and the fit runs on the CPU grower — the shipped precedent at
  `crates/cb-train/tests/device_nonsym_fit_test.rs:105-119`.

**Red**
- Test name: `combination_ctr_non_borders_types_match_the_cpu_grower`, one `#[test]` per
  type delegating to a shared `device::run(ctr_type, label)` (the
  `device_nonsym_fit_test.rs` per-policy shape).
- Setup: the `ctr_device_combo` fixture and its `combo_params()`, with
  `combinations_ctr` set to the type under test and `combinations_ctr_priors: vec![0.5]`;
  `simple_ctr` stays `Borders` so the fit mixes a simple and a combination column — which
  is what makes `eligible_max` (D-2) load-bearing. Run twice:
  `train_cat(&CountingGpu { inner: GpuBackend::default(), … }, …)` and
  `train_cat(&CpuRefRuntime, …)`.
- Assertions:
  1. `gpu.grown.get() == params.iterations` — the device arm really committed;
  2. **split-sequence equality**, per tree, over `[V]` `ObliviousTree`'s ordered surfaces
     (`boosting.rs:847-870`): `splits`, `ctr_splits`, `one_hot_splits` and `level_kinds`
     must be equal between the two models. Compare `ctr_splits` on the full
     `CtrSplitSpec` identity (projection, `ctr_type`, `prior_num`, `prior_denom`,
     `target_border_idx`, `border`) — the same identity
     `assign_leaf_over_ctr_columns` keys on (`boosting.rs:2055-2062`);
  3. leaf values equal within **ε = 1e-4** (D-07's device-vs-CPU bar, not the 1e-5
     upstream bar — this is a self-oracle);
  4. vacuity guards: `Σ ctr_splits >= 1` **and** at least one chosen CTR split has a
     ≥2-member projection (otherwise the combination path is untested and the
     differential is trivially satisfied).
- Expected initial failure: **unknown, and that is the point** — this task is a genuine
  probe. The two most likely failures, both already localised: a wrong `bucket_count` for a
  combination column (T17's fallback) surfacing as a **different chosen split** at some
  level; or an `eligible_max` divergence (T18) surfacing the same way. If it passes on
  first run, that is a legitimate green-on-write outcome and §2.5 applies.
- Run: `cargo test -p cb-train --no-default-features --features rocm --test device_ctr_combo_types_diff_test`

**Guard-4 escalation ladder (MANDATORY — the one failure with no owning task)**

Assertion 4 can fail for a reason that is *not* a defect: on this corpus the greedy search
may simply never choose a ≥2-member split for a given type. This is the most likely
non-defect outcome, and T22 gates T23 which gates the phase DoD — so it needs a route.

**The fixture's frozen `X`/`y`/`borders.npy` must NOT change** (R-12), but the *training
params* ARE tunable here, because T22 is a **device-vs-CPU self-oracle** and never
compares against `predictions.npy`. That distinction is what separates T22 from every
other e2e in this plan; do not assume R-12 freezes the params too.

If guard 4 fails, climb in order, recording the rung reached in the test's doc comment:

1. raise `iterations` (5 → 10) — more trees, more chances the combination enters
   `chosen_this_tree` at level ≥1;
2. raise `depth` (2 → 3) — more levels per tree, and combination candidates only exist
   at level ≥1;
3. sweep `combinations_ctr_priors` (e.g. `0.0`, `0.25`, `1.0`) — the prior moves the CTR
   values and therefore the greedy path;
4. if no configuration yields a combination split for a given type, record **that type**
   as structurally unreachable on this corpus, and either generate a targeted fixture for
   it or escalate — do **not** silently drop the type from the loop.

**Never weaken, relax or delete assertion 4.** It is the only thing preventing a
trivially-satisfied differential, and weakening it converts DCTR-20 from a detector into
ceremony. If all four rungs fail for all three types, stop and escalate rather than
shipping T23.

**Green**
- Nothing, if it passes. If it fails, fix the **owning** task's code, re-run that task's
  validation block, and record the defect and its owner here. The two localised device
  defects have owners (T17's fallback, T18's filter); a guard-4 failure has none and takes
  the ladder above instead.

**§2.5 mutation checks — TWO, both mandatory, both recorded**
1. **D-2 wiring (R-20's detector, the checker's stated motivation)**: revert T18's filter
   at `mod.rs:4895` to the unfiltered max and re-run. If the split sequence changes, **R-20
   is closed** — record the differing tree/level and say so in `mod.rs`'s R-20 comment. If
   it does **not** change, **R-20 stays open**: record that explicitly, leave the comment
   in place, and do not claim the differential covers D-2.
2. **`bucket_counts` fallback**: restore the `member_bins.first()`-only fallback T17
   replaced and confirm the differential fails. If the branch is unreachable (T17 may have
   chosen the typed-error option), record that instead — an unreachable branch cannot be
   mutated, and saying so is the honest outcome.

**Refactor + regression scope**
- Note in the test's module doc that it is a **device-vs-CPU self-oracle** at ε = 1e-4,
  that the upstream half of the chain is the two CPU oracles listed below, and that
  `ctr_device_combo`'s frozen artifacts are reused unchanged (no new fixture, R-12).
- Regression scope: full device suite.

**Validation commands**
```
cargo check -p cb-train --no-default-features --features rocm --test device_ctr_combo_types_diff_test
cargo test -p cb-train --no-default-features --features rocm --test device_ctr_combo_types_diff_test
cargo test -p cb-train --no-default-features --features rocm --test device_ctr_combo_fit_test
cargo test -p cb-train --test ctr_mixed_simple_vs_combo_oracle_test
cargo test -p cb-train --test tensor_ctr_e2e_oracle_test
bash ./run_device_tests.sh
```

**Completion evidence**: three types green with all four assertions; both mutation results
recorded (**including a negative D-2 result if that is what happens**); the two CPU
oracles green, closing the upstream half of the chain; the R-20 comment in `mod.rs`
updated to its measured status.

**Rollback**: delete the test file. It adds no production code.

**Risks / guardrails**: the CPU arm must genuinely be the CPU grower — assert it, e.g. by
using a second `CountingGpu` wrapping `CpuRefRuntime` and asserting `grown == 0`, or by
asserting the seam default was taken. A `CpuRefRuntime` that accidentally committed to the
device would make this a device-vs-device tautology — the R-8 class, one level up.

---

### T23 — Final gate form: delegate to `from_i8` / `is_cpu_supported`

- **Specs**: DCTR-18
- **Depends on**: **T00**, T19 (the last conjunct removal), **T22** (DCTR-20's detector
  must pass first) — **WAVE 8**
- **Parallel**: NO — must follow T22.

**Goal / observable completion**: `ctr_types_are_device_covered` contains **no** type,
arity, target-border or prior conjunct; it delegates to the single existing type list; and
T00's gate-state table is replaced by a **final** admitted-set table with positive rows per
admitted type and negative rows for the two GPU-only types and for unknown discriminants.

**⚠ Premise correction (CRITICAL-1, item 5).** v1 said this task "closes the
no-covering-tests gap" on the predicate. **There was no gap** — the predicate had five
covering assertions all along (V-0b), and T00 migrated them into the gate-state table. So
this task's completion criterion is **not** "coverage now exists"; it is:
1. the accumulated `matches!` list is replaced by the `from_i8`/`is_cpu_supported`
   delegation, and
2. **T00's table is retired into its final form** — every row's `flips_at` note is
   discharged, and the table now asserts the *final* admitted set rather than a
   wave-dependent one.
Shipping (1) without (2) would leave a table whose comments promise flips that already
happened — worse than no table.

**Disk evidence**: `[V]` `ECtrType::from_i8` (`ctr/mod.rs:96-108`, `None` for unknown
discriminants) and `ECtrType::is_cpu_supported` (`:155-160`, `true` for exactly
`{Borders, Buckets, BinarizedTargetMeanValue, Counter}`, citing `restrictions.h:18-48`).
Both are **in-crate** for `boosting.rs`, so the Do-Not-Hand-Roll rule applies here (unlike
cb-backend — C-3).

**Files / symbols**
- Modify: `crates/cb-train/src/boosting.rs:2355-2366` — final form per `SPEC.md` §4.7:
  ```rust
  fn ctr_types_are_device_covered(cols: &[crate::ctr::CtrFeatureColumn]) -> bool {
      !cols.is_empty()
          && cols.iter().all(|col| {
              crate::ctr::ECtrType::from_i8(col.ctr_type)
                  .is_some_and(|t| t.is_cpu_supported())
          })
  }
  ```
- Modify: `crates/cb-train/src/boosting_ctr_gate_test.rs` (T01's file).
- Modify: `crates/cb-train/src/device_ctr_combo_config_test.rs` — retire T00's gate-state
  table into its final form: every row's expected value is now the **final** answer, the
  per-row `flips_at` notes are replaced by a single header stating that P1 is complete and
  that **P2/P3 will add rows, not flip these**, and rows 1/7/8 (the never-flipping ones)
  are called out as permanent.

**Red**
- Test name: `gate_admits_exactly_the_cpu_supported_ctr_types`.
- Setup: for each discriminant `0..=5`, build a `CtrFeatureColumn` by hand
  (all fields `pub`, `ctr_feature.rs:71-99`) with a **combination** projection
  (`TProjection::from_features(&[0, 1])`), `target_border_idx: 1` and
  `prior_denom: 1.0`, and assert:
  - `0` Borders, `1` Buckets, `2` BinarizedTargetMeanValue, `4` Counter ⇒ **true**;
  - `3` FloatTargetMeanValue, `5` FeatureFreq ⇒ **false**;
  - an unknown discriminant (e.g. `7`, `-1`) ⇒ **false** (`from_i8` returns `None`);
  - the empty slice ⇒ **false** (the caller's `is_empty()` arm owns that).
  The deliberately non-default `projection` / `target_border_idx` values are what prove
  the arity and target-border conjuncts are **gone**.
- Expected initial failure: **before** the collapse, the `matches!` list left by
  T10/T12/T16 already admits the four types, so the type cases pass — but the **unknown
  discriminant** case (`7`) fails, because a raw `matches!(from_i8(..), Some(Borders|…))`
  written naively over `col.ctr_type` may or may not reject it. The implementer must run
  the test **first**, record the actual failure, and if it is green-on-write treat it under
  §2.5 with a mutation check (add `| ECtrType::FeatureFreq` to the list and confirm the
  `5 ⇒ false` case fails).
- Run: `cargo test -p cb-train --lib boosting_ctr_gate_tests`
  then `cargo test -p cb-train --lib device_ctr_combo_config_tests`

**Green**: replace the accumulated `matches!` list with the `from_i8().is_some_and(is_cpu_supported)`
delegation, and retire T00's table into its final form. **Do not** hand-roll a second type
list — `from_i8`/`is_cpu_supported` are in-crate here (C-3).

**Refactor + regression scope**
- Rewrite the doc block one final time: state that all four conjuncts are gone; that the
  admitted set is exactly the CPU-supported set; that `FloatTargetMeanValue`/`FeatureFreq`
  are rejected earlier by `validate_ctr_types` (`boosting.rs:1372-1387`) and
  `materialize_ctr_feature` (`ctr_feature.rs:311-318`) and have **no** CPU parity surface
  (`restrictions.h:20-32`); and that the surviving `device_host_eligible` clauses
  (`learning_folds_for_cycle == 1`, `one_hot_bins.is_empty()`, `eval_sets.is_empty()`,
  `has_any_scorable_feature`, `ctr_covered`'s width check) are P2/P3 boundaries each pinned
  by a negative test (T13, T21).
- Regression scope: the whole device suite plus every CPU CTR oracle.

**Validation commands**
```
cargo test -p cb-train --lib device_ctr_combo_config_tests     # CRITICAL-1: the table's final retirement
cargo test -p cb-train --lib boosting_ctr_gate_tests
cargo test -p cb-train --lib
cargo test -p cb-train --no-default-features --features rocm --test device_ctr_fit_test
cargo test -p cb-train --no-default-features --features rocm --test device_ctr_combo_types_diff_test
cargo test -p cb-train --no-default-features --features rocm --test device_ctr_buckets_fit_test
cargo test -p cb-train --no-default-features --features rocm --test device_ctr_counter_fit_test
cargo test -p cb-train --no-default-features --features rocm --test device_ctr_btmv_fit_test
cargo test -p cb-train --no-default-features --features rocm --test device_ctr_combo_fit_test
cargo test --workspace
bash ./run_device_tests.sh
```

**Completion evidence** (rewritten per CRITICAL-1 item 5 — the v1 criterion rested on a
gap that never existed):
- all eight gate cases green in `boosting_ctr_gate_tests`;
- **T00's gate-state table retired into its final form**, every `flips_at` note discharged,
  and `cargo test -p cb-train --lib` green **in full** — the CRITICAL-1 regression;
- **T22's DCTR-20 differential green**, so the cross-product this predicate now admits has
  a passing detector (SPEC DCTR-20 ordering requirement);
- every device e2e's `max |Δpred|` unchanged from its own task's recorded value.

**Rollback**: restore the explicit `matches!` list from T16's state.

---

### T24 — Register the new device binaries and prove the phase definition of done

- **Specs**: closes `SPEC.md` §6 acceptance scenarios 6 and 7
- **Depends on**: **every** task above — **WAVE 8 (tail)**
- **Parallel**: NO. **Sole owner of `run_device_tests.sh`** (C-8) — no other task may
  touch it.

**Goal / observable completion** (worded to `SPEC.md` §6 scenario 7 as amended, checker
MINOR-8): `bash ./run_device_tests.sh` is green **at its grown roster** — the 23 binaries
green at `a0a67ec` **plus every binary this phase adds**, all registered in `TESTS=(…)` —
plus the isolated perf lane. The count is **derived, not a target**: 23 + 5 = **28**.
v1's "27/27" both hard-coded a number and omitted T22's binary; scenario 7's amended
wording exists precisely so the phase's own new tests cannot go unregistered.

**Files / symbols**
- Modify: `run_device_tests.sh` `TESTS=(…)` — append, keeping the file's existing grouping
  order, the **five** binaries this phase adds:
  `device_ctr_buckets_fit_test` (T10), `device_ctr_counter_fit_test` (T12),
  `device_ctr_btmv_fit_test` (T16), `device_ctr_type_gate_test` (T13/T21),
  `device_ctr_combo_types_diff_test` (**T22 — new in v2; v1 could not have listed it**).
  `[V]` the array currently holds 23 names and `device_ctr_combo_fit_test` is **already**
  listed (it will now run un-ignored after T19).
- **Registration audit (scenario 7's real requirement)**: `grep -l "fn .*test" crates/cb-train/tests/device_*.rs`
  and confirm every device binary this phase created appears in `TESTS=(…)`. A new binary
  that exists but is unregistered is a scenario-7 failure even if it passes standalone.
- `PERF_TESTS` is unchanged (`kernels::poisson_bootstrap_speed_test`; R-13 — do not chase
  its ratio).

**Red**
- Not a unit test: the Red is the **runner** failing to cover the new binaries. Run
  `bash ./run_device_tests.sh` and confirm the output lists 23 PASS lines with **no**
  mention of the five new binaries — that omission is the failure being fixed. Record the
  23-line output verbatim; it is also the regression baseline.

**Green**: append the five names.

**Refactor + regression scope**
- Update the script's header comment with the new count and the reason the CTR binaries
  must each be a separate `--test` (R-10).
- Regression scope: the whole suite.

**Validation commands**
```
bash ./run_device_tests.sh
cargo test --workspace
cargo test -p cb-backend --no-default-features --features rocm
cargo test -p cb-train --test ctr_btmv_simple_oracle_test
cargo test -p cb-train --test ctr_buckets_simple_oracle_test
cargo test -p cb-train --test ctr_counter_simple_oracle_test
cargo test -p cb-train --test ctr_counter_full_eval_oracle_test
cargo test -p cb-train --test ctr_borders_multiprior_oracle_test
cargo test -p cb-train --test ctr_mixed_simple_vs_combo_oracle_test
cargo test -p cb-train --test tensor_ctr_e2e_oracle_test
cargo test -p cb-train --test plain_ctr_oracle_test
cargo test -p cb-train --test ordered_ctr_oracle_test
cargo test -p cb-train --lib
```

**Completion evidence**: **28/28 PASS** + perf lane PASS (23 baseline + 5 added, every one
registered); `cargo test -p cb-train --lib` green in full; every CPU CTR oracle green; a
completion table recording each device e2e's measured `max |Δpred|` (Buckets, Counter,
BTMV, combination ≈2.082e-17, mixed) and every recorded mutation-check failure message
from §2.5.

**Rollback**: revert the five appended lines.

---

## 5. Coverage

### 5.1 DCTR spec → task (all **20** covered)

| Spec | Task(s) | Track | Kind |
|---|---|---|---|
| DCTR-01 | **T02** | Seam | production + characterization |
| DCTR-02 | **T01** | 1a | deletion + characterization |
| DCTR-03 | **T21** | Hygiene | retention + negative test (re-runs the existing `fpp17_one_hot_x_ctr_still_declines`, C-18) |
| DCTR-04 | **T04** | E | production (CPU) |
| DCTR-05 | **T03** | E | inertness proof (no production change) |
| DCTR-06 | **T08** | A | kernel + self-oracle |
| DCTR-07 | **T09** | A | plumbing + grouping |
| DCTR-08 | **T10** (fixture **T05**) | A | gate + e2e (`CountingGpu`) |
| DCTR-09 | **T11** | C | kernel + self-oracle |
| DCTR-10 | **T12** (fixture **T06**) | C | gate + e2e (`CountingGpu`) |
| DCTR-11 | **T13** | C | negative test (P1/P3 boundary) |
| DCTR-12 | **T14** | B | kernel + self-oracle (C-2 detector B is the proof) |
| DCTR-13 | **T15** | B | structural cross-check |
| DCTR-14 | **T16** (fixture **T07**) | B | gate + e2e (`CountingGpu`) |
| DCTR-15 | **T17** | D | D-1 eligibility gate |
| DCTR-16 | **T18** (+ **T22** as the wiring detector, R-20) | D | D-2 filtered `eligible_max` |
| DCTR-17 | **T19** | D | gate + un-ignore + `CountingGpu` |
| DCTR-18 | **T00** (migration clause) + **T23** (final form) | Gate | **split in v2** — see below |
| DCTR-19 | **T20** | Hygiene | `CountingGpu` on `device_ctr_fit_test` |
| **DCTR-20** | **T22** | D / cross-product | **new in v2** — device-vs-CPU split-sequence differential |

**DCTR-18 is covered by two tasks** because the revised SPEC gave it two obligations: the
*conscious migration* of the five pre-existing covering assertions (**T00**, ordered before
every conjunct deletion) and the *final delegating form* (**T23**). Splitting them is what
keeps the `:124-126` contract executable rather than retrospective.

Fixtures T05/T06/T07 and the runner task T24 carry no DCTR ID of their own; they are the
supporting artifacts and the DoD for DCTR-08/10/14 and acceptance scenarios 6–7.

### 5.2 `SPEC.md` §6 acceptance scenario → task (all **10**)

| # | Scenario | Bar | Closed by |
|---|---|---|---|
| 1 | Buckets fit commits + matches upstream | ≤1e-5 + `grown == iterations` | **T10** (kernel T08, plumbing T09, fixture T05) |
| 2 | Counter fit commits + matches | ≤1e-5 + commitment | **T12** (kernel T11, fixture T06) |
| 3 | BTMV fit commits + matches, vs the **corrected** CPU | ≤1e-5 + commitment | **T16** (T04 → T14 → T15, fixture T07) |
| 4 | Combination fit commits + matches | ≤1e-5 + commitment | **T19** (T17 + T18) |
| 5 | Gate has no type / arity / target-border / prior conjunct | structural | **T23** (built by T01, T10, T12, T16, T19; pinned at every step by T00's table) |
| 6 | Every CTR e2e asserts commitment via `CountingGpu` | structural | **T20** (mixed), **T19** (combo), T10/T12/T16 (new), **T22** (differential) |
| 7 | `run_device_tests.sh` green **at its grown roster** (23 baseline + every binary this phase adds, all registered) + perf lane | regression | **T24** (28 = 23 + 5; the count is derived, not a target — MINOR-8) |
| 8 | `ctr_btmv_simple_oracle_test` passes **unchanged** | regression | **T03** (baseline) + **T04** (Verify) |
| 9 | One-hot×CTR, multi-permutation, eval-set, cat-only, `border_count != 15` all still decline, each with a passing negative test | correctness | **T21** (one-hot via the existing test, cat-only, `border_count`, multi-perm) + **T13** (eval-set / Counter `Full`) |
| **10** | **A combination × non-Borders detector passes before the final gate lands** | device == CPU split sequence, ε=1e-4 | **T22**, ordered before **T23** |

### 5.3 File ownership (serialisation map)

| file | owners, in order |
|---|---|
| `cb-train/src/device_ctr_combo_config_test.rs` | **T00 → T02 → T01 → T10 → T12 → T16 → T19 → T23** — T00 creates the gate-state table; T02 adds a seam test; every gate task flips its own row **in the same commit** as its conjunct edit; T23 retires the table |
| `cb-train/src/boosting.rs` — `ctr_types_are_device_covered` | **T01 → T10 → T12 → T16 → T19 → T23** (strictly serial: one expression) |
| `cb-train/src/boosting.rs` — `build_device_ctr_config` | T02 (only) |
| `cb-train/src/boosting.rs` — `device_host_eligible` doc | T13, T21 (comments only) |
| `cb-compute/src/runtime.rs` | T02 (only) |
| `cb-backend/src/kernels/ctr_device.rs` | **T08 → T11 → T14** (serial) |
| `cb-backend/src/kernels/ctr_device_test.rs` | T08 → T11 → T14 → T15 (serial) |
| `cb-backend/src/gpu_runtime/session.rs` | **T09 → T12 → T16 → T17** (serial; T09 also mounts `session_ctr_type_test.rs` **inside** this file, C-11) |
| `cb-backend/src/gpu_runtime/mod.rs` | **T17 → T18** (serial) |
| `cb-train/src/ctr/ctr_feature.rs` | T04 (only) |
| `cb-train/src/ctr/calc_ctr_test.rs` | T03 (only) |
| `cb-train/tests/device_ctr_fit_test.rs` | T20 (only; after T01 — MINOR-9) |
| `cb-train/tests/device_ctr_type_gate_test.rs` | T13 → T21 |
| `cb-train/tests/device_fpp_composition_test.rs` | T21 (re-run + mutation only; **no new copy**, C-18) |
| `cb-train/tests/device_ctr_combo_types_diff_test.rs` | **T22 only** (new) |
| `cb-train/src/boosting_ctr_gate_test.rs` | T01 → T23 |
| `cb-oracle/tests/ctr_device_{buckets,counter,btmv}_fixture_smoke_test.rs` | T05 / T06 / T07 — **one file each** (C-15), no coordination |
| `run_device_tests.sh` | **T24 only** (C-8) |

Genuinely parallelizable: **{T00, T02, T03, T05, T06, T07}** in Wave 0; **{T01, T04}** in
Wave 1; **{T08, T20}** in Wave 2; **{T21, T22}** in Wave 7. Everything else is serialised
by file or expression ownership above — verified on disk, not assumed. v1 claimed
T05/T06/T07 were parallel while telling them to share one smoke file; C-15 removes that
contradiction, and v1's Wave-0 grouping of T01 with T20 is dissolved by MINOR-9.

---

## 6. Blockers, unverified assumptions, and planner decisions

### BLOCKERS

**None.** `SPEC.md` §9 records that both first-pass blockers (Q2 BTMV quantizer, Q5 FPP-11)
were closed by measurement and Q1/Q4 by user ruling. This plan's verification pass found
no new blocker, and neither did the checker: its CRITICAL-1 is a **plan defect**, not a
requirement gap, and it is resolved by T00. Every symbol, line number and command this
plan cites was confirmed on disk in this worktree — in v2 that includes an independent
re-verification of all 17 checker findings, every one of which held.

### Corrections the implementing agent MUST apply (no decision needed)

| id | source says | verified reality | task |
|---|---|---|---|
| C-1 | a `&[]` in `train_cat` is an empty eval-set list | it is the **weights** argument; eval sets need `train_cat_with_eval_sets` (`boosting.rs:2970-2981`) + `EvalSet` (`:2551-2563`) | T13 |
| C-2 | DCTR-12: "an f64 device sum must FAIL this test" | at binclf the added values are exactly `{0.0, 1.0}` ⇒ f32 and f64 sums are **bit-identical**; the proof is the synthetic `divisor = 3` case (checker measured 22–41/96 mismatches), with the buffer assertion demoted to an output-width pin | T14 |
| C-3 | "reuse `from_i8`/`is_cpu_supported`; do not hand-roll" | applies to **cb-train only**; cb-backend must transcribe inline (it may never depend on cb-train) | T09, T12, T16 |
| C-4 | seam `target_border_idx: u32` | source field is `usize`; convert with `try_from(..).map_err(..)?`, never a silent `unwrap_or(0)` | T02 |
| C-5 | R-1/R-2: tree-lifetime state is new work | `grow_oblivious_tree_resident` runs once per tree ⇒ a function-local `Vec` is tree-scoped for free; the risk is only in hoisting it | T17, T19 |
| C-6 | `cubecl_error_guideline.md` | does not exist; the real path is `cubecl_error_solution_guide/` | T08, T11, T14 |
| C-7 | `ctr_covered` needs per-type shape validation | the border-table shape check is **unchanged** for all four types; only an admission list is added | T09, T12, T16 |
| C-8 | — | `run_device_tests.sh` holds 23 names; **five** new binaries land here (T22's included); single-owner to avoid conflicts | T24 |
| C-9 | DCTR-07 grouping is new behaviour | `build_device_ctr_config:2517` already keys `(ctr_type, projection)` with `target_border_idx` excluded ⇒ green-on-write + mutation check | T09 |
| C-10 | pass C's `eligible_max` comment | asserts "the device gate admits only simple projections" — false after T17/T19; must be rewritten | T17, T18 |
| **C-11** | v1: mount T09's test next to `session_depth_gt1_test`'s mount "in `session.rs`" | that grep is empty — the mount is `gpu_runtime/mod.rs:775`, and `ctr_covered` is private to `mod session` ⇒ **`E0603`**. Mount **inside `session.rs`**; keep the fn private | T09 |
| **C-12** | v1: 9 `DeviceCtrColumn` literals | **8 literals + 1 definition**; 3 use `..Default::default()` ⇒ **5 edits** | T02 |
| **C-13** | v1 line numbers | pass C loop is `mod.rs:4907` (not `:4919`); `max_bucket_count` is `:4905`; `#[ignore]` is `device_ctr_combo_fit_test.rs:179`; `ordered_ctr_prefix_kernel`'s `fn` is `ctr_device.rs:135` | T08, T17, T18, T19 |
| **C-14** | v1: `ctr_covered` has 1 caller | **2** — `session.rs:1471` and `:1701`; the first makes a gate/list mismatch decline the **whole fit** at `:1505`, which is why `grown == iterations` is sufficient evidence | T09 |
| **C-15** | v1: T05/T06/T07 share one smoke file **and** are parallel | contradictory; the shipped convention is **one smoke file per fixture** | T05, T06, T07 |
| **C-16** | — | the eligibility filter scopes `eligible_max` **only**; `phantom_max` is folded in **outside** it (`mod.rs:4905`, CPU `tree.rs:3033`). T18 must not filter the phantom | T18 |
| **C-17** | v1: "guard `norm == 0.0` the same way" | that guard is **unreachable** (`norm >= 1` always); omit it or mark it dead | T04 |
| **C-18** | v1: author a one-hot × CTR negative test | one **already exists** at `device_fpp_composition_test.rs:255`; re-run and mutate it instead | T21 |

### Planner decisions taken (the SPEC delegated or the research left open)

- **Counter kernel is a new sibling `#[cube]` function, not a mode on the prefix kernel**
  (T11). The prefix kernel's loop is permutation-driven and read-before-increment; Counter
  is neither. `SPEC.md` §9's "serial two-pass" shape is honoured; only the packaging
  differs, and it avoids a permanently-dead branch in a serial hot loop.
- **T14's Red uses a synthetic `divisor = 3` multi-class input** (C-2). Production always
  launches `divisor = 1`; the parameter exists so the f32 width is testable at all.
- **The gate is widened incrementally, one type per track, and collapsed once at T23**
  rather than jumping to the final form early. This keeps each track's Red a single
  `grown == 0` failure with one principal cause, and makes each track independently
  revertible (`SPEC.md` §8's rollback contract).
- **T21 asserts the observable (`no device commitment`), not a specific clause**, because
  one-hot × CTR is rejected by two layers (SPEC-OH-26 at `boosting.rs:3937` and the device
  conjunct at `:4425`) and a clause-specific assertion would break when either moves.
- **§2.5's mutation-check protocol is mandatory** for the **nine** green-on-write tests in
  this plan (v1 said seven; T15 and T22 are the additions). Without it, a phase whose
  central risk is the false-pass class (R-8) would ship untested pins.
- **v2: T00 rewrites the existing gate assertions into a state table rather than retiring
  them.** The coordinator allowed either. Retiring hands all coverage to T23 and leaves the
  gate unpinned across T01–T22 — the exact window in which five conjunct edits land.
  Rewriting keeps a discriminating pin live at every wave and makes the `:124-126`
  conscious-act contract *stronger* than it is today.
- **v2: T22 is a device-vs-CPU differential, not a fourth upstream fixture.** It reuses the
  frozen `ctr_device_combo` corpus (no seed search, no new frozen artifact), compares
  **chosen splits** rather than predictions, and closes the upstream half of the chain by
  also running `ctr_mixed_simple_vs_combo_oracle_test` and `tensor_ctr_e2e_oracle_test`.
- **v2: T05's fixture parameters are split into gate-load-bearing vs tunable**, with a
  four-rung escalation ladder ending at `ctr_buckets_simple`'s own shape (2 cat columns,
  10 iterations). v1 froze a shape *below* the only configuration known to satisfy the
  guard and allowed no escape but weakening it.

### Assumptions NOT verified in this planning session

1. **No test was executed.** This is a planning session; every "expected initial failure"
   message is a prediction from reading the code, not an observed run. The implementer
   must record the **actual** message and, where it differs materially, note the deviation
   in the task's completion note (the prior phase's execution-note contract).
2. ~~The `model.json` target-border accessor~~ — **RESOLVED in v2.** `[V]` loaded
   `ctr_buckets_simple/model.json`: each `features_info.ctrs[i]` has a **top-level**
   `target_border_idx` key (full key set:
   `borders, ctr_type, elements, identifier, prior_denomerator, prior_numerator, scale, shift, target_border_idx`).
   v1's guessed `c["ctr"]["identifier"].get("ctr_target_border_idx", 0)` was wrong **and**
   its `.get(…, 0)` default would have made T05's only discriminating guard silently
   vacuous.
3. **T04's Red construction** — that a two-document BTMV column reaches the
   `(good = 1.0, total = 1)` prefix state at a known index — is an arithmetic prediction.
   The task instructs the implementer to widen the column until the state is reached
   rather than weaken the assertion to the clamped value.
4. **Fixture generation** (T05/T06/T07) requires the installed `catboost==1.2.10`; not
   attempted here. Whether **any** rung of T05's ladder yields both `target_border_idx`
   values at `CARDS=(6,)`/64 rows/5 iterations is unverified; `[V]` the only in-repo
   configuration known to achieve it is `ctr_buckets_simple`'s **2 cat columns / 60 rows /
   10 iterations**, which is rung 3–4. Expect to reach rung 3. The escalation rule forbids
   weakening the guard at every rung.
5. **The measured `≈2.082e-17`** for T19 comes from the research's reverted spike on
   `gfx1151`, not from a run in this session. It is an expectation to compare against, not
   an assertion threshold; the assertion is `≤1e-5` plus `grown == iterations`.
6. **R-20 is now addressable but not yet closed.** No *upstream* fixture discriminates
   D-2, and the T18 unit test proves only that the helper filters — not that it is wired
   into pass C. **T22's differential is the candidate wiring detector** (SPEC R-20 as
   amended names it primary), but whether it actually discriminates is **unmeasured**:
   T22's mutation check 1 is what decides. A negative result must be recorded as
   "R-20 still open", not glossed.
7. **T22's outcome is genuinely unknown.** It is the only task in the plan whose Red is a
   probe rather than a prediction. Both likely failure modes are pre-localised (T17's
   `bucket_counts` fallback; T18's filter), and the task forbids fixing them in place.
8. **Which leaf-value path a device Buckets fit takes is unresolved** (T10). `[V]` the host
   gather `assign_leaf_over_ctr_columns` keys on the full identity **including
   `target_border_idx`** (`boosting.rs:2055-2062`) while the device path indexes averaging
   columns by **position** (`session.rs:2723-2784`). Buckets is the first type emitting two
   columns per `(projection, prior)`, so T10 must determine which runs and whether it is
   correct for that layout. **This is the most likely place for an unpredicted T10 failure.**
9. **T00's table rows 3–6 are asserted to be green-on-write** against today's gate. That
   follows from reading the four conjuncts, but was not executed.
10. **Whether `poisson_bootstrap_speed_test` will read above its 5× bar** on the day the
   phase completes is load-dependent (R-13). It is a known do-not-chase flake in the
   isolated lane and is not a phase blocker.

### Carried out of P1 (recorded, not planned here)

`border_count != 15`, cat-only pools, bootstrap/MVS/sampling composition **including**
`leaf_estimation_method = Exact` (user override of R-18) → **P2**. Multi-permutation CTR,
eval sets and Counter `Full` widening → **P3**. One-hot × CTR (SPEC-OH-26) and
CTR × non-symmetric/Region are **CPU-trainer** features with their own specs after P3.
`FloatTargetMeanValue` / `FeatureFreq` (no upstream CPU parity surface), multi-descriptor
`simple_ctr` (SPEC-CTRT-19), and wgpu (WR-02, no f64 channel) are permanently out.

---

## 7. Phase definition of done

All **25** tasks green, every §2.5 mutation-check result recorded (nine candidates), and:

```
cargo test --workspace
cargo test -p cb-train --lib
cargo test -p cb-backend --no-default-features --features rocm
bash ./run_device_tests.sh            # 28/28 PASS + perf lane (23 baseline + 5 added)
cargo test -p cb-oracle --test ctr_device_buckets_fixture_smoke_test
cargo test -p cb-oracle --test ctr_device_counter_fixture_smoke_test
cargo test -p cb-oracle --test ctr_device_btmv_fixture_smoke_test
cargo test -p cb-train --lib boosting_ctr_gate_tests
cargo test -p cb-train --lib device_ctr_combo_config_tests
cargo test -p cb-train --lib ctr::calc_ctr_test
cargo test -p cb-train --lib ctr::ctr_feature_test
cargo test -p cb-train --test ctr_btmv_simple_oracle_test
cargo test -p cb-train --test ctr_btmv_bake_upstream_table_test
cargo test -p cb-train --test ctr_buckets_simple_oracle_test
cargo test -p cb-train --test ctr_counter_simple_oracle_test
cargo test -p cb-train --test ctr_counter_full_eval_oracle_test
cargo test -p cb-train --test ctr_borders_multiprior_oracle_test
cargo test -p cb-train --test ctr_mixed_simple_vs_combo_oracle_test
cargo test -p cb-train --test tensor_ctr_e2e_oracle_test
cargo test -p cb-train --test plain_ctr_oracle_test
cargo test -p cb-train --test ordered_ctr_oracle_test
cargo test -p cb-train --test ctr_feature_materialize_test
```

all green on a real ROCm device (`gfx1151`, ROCm at `/home/user/rocm/opt/rocm`), plus:

- `ctr_types_are_device_covered` contains **no** type, arity, target-border or prior
  conjunct and delegates to `ECtrType::from_i8` / `is_cpu_supported`;
- **every** CTR e2e test asserts `grown.get() == params.iterations` via `CountingGpu`;
- **`cargo test -p cb-train --lib` is green in full** — the CRITICAL-1 regression, and the
  reason the gate-state table exists;
- **T22's DCTR-20 differential is green for all three non-Borders types**, and it landed
  **before** T23 (SPEC scenario 10);
- `run_device_tests.sh`'s `TESTS=(…)` lists **every** device binary this phase created —
  scenario 7's real requirement; the count 28 is derived from that, not aimed at;
- `ctr_btmv_simple_oracle_test` passes **unchanged**, and **zero** files under
  `crates/cb-oracle/fixtures/` were regenerated (only the three new directories added);
- the completion summary records each device e2e's measured `max |Δpred|`, **R-20's
  measured status from T22's mutation 1** (closed or still open — either is acceptable,
  silence is not), T05's escalation rung, T10's leaf-gather-path finding, and every
  mutation-check message.

---

## 8. Disposition of the checker's findings (pass 1 → v2)

| # | Finding | Disposition | Where |
|---|---|---|---|
| **CRITICAL-1** | the gate has five covering assertions; V-0 was false | **Accepted in full.** Independently re-verified. V-0 corrected; **V-0b** added; **T00** created and ordered before every gate task; `device_ctr_combo_config_tests` added to the Validation blocks of T01/T10/T12/T16/T19/T23; §5.3 gained the file with owners `T00 → T02 → T01 → T10 → T12 → T16 → T19 → T23`; T23's completion criterion rewritten off the false premise. **Plus a refinement the checker did not have** — see V-0c: all four negative assertions use *combination* projections, so nothing turns red until **T19**; they go **vacuous**, not red, at T01/T10/T12. The remedy is unchanged (arguably more necessary), and T00's table uses *simple* projections so each row tests exactly one conjunct. | §1 V-0/V-0b/V-0c, T00, T01, T10, T12, T16, T19, T23, §5.3 |
| **MAJOR-1** | combination × non-Borders admitted with zero coverage | **Accepted; option (b) adopted.** New **T22**: device-vs-CPU **split-sequence differential** over `ctr_device_combo` for `combinations_ctr ∈ {Buckets, Counter, BTMV}`, ordered **before T23**. Also assigned the false comment at `session.rs:1748` and the `member_bins.first()`-only `bucket_counts` fallback (`:1746-1763`) to **T17**, which already edits that block, with T22 mutation 2 as their detector. | T22 (new), T17 |
| **MAJOR-2** | T09's test cannot compile | **Accepted.** `[V]` the mount is `gpu_runtime/mod.rs:775`, not `session.rs`; `ctr_covered` is private to `mod session` ⇒ `E0603`. C-11 added; T09 now mounts inside `session.rs`, keeps the fn private, and the test filter is corrected to `gpu_runtime::session::session_ctr_type_test`. | C-11, T09 |
| **MAJOR-4** | T05's fixture recipe can deadlock | **Accepted.** `[V]` `ctr_buckets_simple` = **60 rows, 2 cat columns (6, 5), 10 iterations** — v1's shape was below the only known-good one. Parameters split into **gate-load-bearing** (never change) vs **tunable**, with a four-rung ladder ending at that precedent, and `"escalation_rung"` recorded in `config.json`. Guard 1 may never be weakened. | T05 |
| MINOR-1 | `ctr_covered` has two callers; `:1471` is the safety property | Accepted — C-14; recorded in T09 and required in `ctr_covered`'s doc comment. | C-14, T09 |
| MINOR-2 | T05's accessor is top-level `c["target_border_idx"]` | Accepted — verified by loading `ctr_buckets_simple/model.json`. v1's guess also carried a `.get(…, 0)` default that would have made the guard vacuous. | T05, §6 assumption 2 |
| MINOR-3 | reframe T14 detector A | Accepted — A demoted to an **output-width pin** with its scope stated in-test; the `n*8` claim withdrawn; detector B named as the proof, with the checker's measured 22–41/96 recorded. | C-2, T14 |
| MINOR-4 | 8 literals, 5 need edits | Accepted — C-12 table lists all nine grep hits by form. | C-12, T02 |
| MINOR-5 | line drift | Accepted — C-13; `:4919→:4907`, `:4905` added, `:180-181→:179`, `fn` at `:135`. | C-13, T08, T17, T18, T19 |
| MINOR-6 | T17 must list `session.rs:2593` | Accepted — added; it is the **only** `ResidentCtrSearch` construction. | T17 |
| MINOR-7 | `one_hot_x_ctr_still_declines` already exists | Accepted — C-18; T21 re-runs and mutates it instead of writing a third copy, and only authors a replacement if the mutation shows it went vacuous. | C-18, T21 |
| MINOR-8 | reconcile 27/27 with scenario 7 | Accepted — T24 reworded to the amended "grown roster" phrasing; count is **derived** (23 + 5 = 28) and a **registration audit** step added. | T24, §5.2 row 7, §7 |
| MINOR-9 | serialise T20 after T01; forbid live mutations during suite runs | Accepted — T20 moved to Wave 2 with the reason inline; §2.5 gained the **isolation rule**. | §2.5, T20, §3 |
| MINOR-10 | wave-label and parallelism contradictions | Accepted — all labels recomputed for the 8-wave graph; C-15 gives each fixture its own smoke file, removing the contradiction. | §3, C-15, T05/T06/T07 |
| MINOR-11 | transcribe T17/T18 cases from `tree_test.rs` | Accepted — both tasks now require mirroring the CPU's covered cases one-for-one, with divergence treated as a transcription defect. | T17, T18 |
| MINOR-12 | four bug mitigations | Accepted — (a) fallback + false comment → T17; (b) id-space note → C-4/T02 doc comment; (c) dead `norm == 0.0` guard → C-17/T04; (d) **`phantom_max` outside the filter** → C-16/T18, with a fourth unit case. | C-4, C-16, C-17, T04, T17, T18, T02 |
| MINOR-13 | annotate P2/P3 boundary inversions | Accepted — T13 and each T21 pin carry a `// P2/P3 WILL INVERT THIS` comment; the one-hot pin is explicitly marked **not** a boundary (DCTR-03 retains it by design). | T13, T21 |
| MINOR-14 | T15 is an eighth mutation candidate | Accepted — T15 now states its §2.5 status and why the `≥2 distinct bins` guard is the accepted substitute; the candidate count is **nine** (T15 + T22). | §2.5, T15 |
| MINOR-15 | T10 must settle the leaf-gather path | Accepted — T10 gained the open question with both candidate paths cited, and an instruction to **escalate** rather than patch the gather inside an e2e task. Also listed as unverified assumption 8. | T10, §6 |

**Nothing was declined.** Everything the checker validated is intact: the `CountingGpu`
discipline, the §2.5 protocol, Track E → Track B, Track D last, D-2 shipping with D-1, the
command forms, and C-1…C-10 (only C-2's detector A and C-8's count were touched, per
MINOR-3 and MINOR-8).
