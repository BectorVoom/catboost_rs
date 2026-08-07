---
title: GPU Full Parameter Parity — TDD Implementation Plan
spec: ./SPEC.md
research: ./research.md
status: draft
plan_version: 1
updated_at: 2026-08-07T00:00:00Z
tasks: 38
waves: 8
worktree: /home/user/Documents/workspace/catboost_rs/.claude/worktrees/gpu-string-param-oracle
branch: worktree-gpu-string-param-oracle
base_commit: 1535ac1bd6bd4d7f896f0afbd3f11fb63f77d170
---

# PLAN.md — GPU Full Parameter Parity

This plan converts the LOCKED `SPEC.md` (FPP-01 … FPP-35, SPD-01 … SPD-03, decisions
D1–D7) into executable tasks. It does **not** re-author the specification's WHAT/WHY.

Everything below is `Read`/`grep`/CodeGraph-verified against `base_commit` `1535ac1`
(the current `HEAD` of `worktree-gpu-string-param-oracle`; the prior phase's GDC work has
already merged, so every line number research.md carried from the pre-GDC tree was
re-checked rather than trusted).

**Spike tasks (Wave 7) deliberately carry no Red/Green/Refactor.** Their deliverable is a
design note; fabricating line-precise steps for a kernel whose shape is unknown would be
exactly the failure mode the prior phase's V-7 caught. This follows that phase's own
Track O / T18 "BLOCKED — decision required" precedent, extended per D2/D4/D5 into an
explicit spike → escalation-gate → implement structure.

---

## 0. Reading order for the implementing agent

1. **§1 — verification results.** Read before any task. Ten findings; **four correct
   `research.md`** and two change task shape.
2. §2 — global rules (commands, lints, test conventions, Do Not Hand-Roll).
3. §3 — waves and the dependency graph.
4. §4 — the 38 tasks.
5. §5 — coverage tables.
6. §6 — blockers, escalation gates, and assumptions.
7. §7 — definition of done.

---

## 1. Verification results

`[V]` = verified fact with citation. `[C]` = **correction**: `SPEC.md` or `research.md`
asserts something the code contradicts. All performed this session against `1535ac1`.

### V-1 — `device_host_eligible`'s current line numbers (RESOLVED; research.md was close but stale)

`[V]` `grep -n device_host_eligible crates/cb-train/src/boosting.rs` → the expression is
`boosting.rs:4214-4318` (research.md said "~4214-4318" — correct), consumed at `:4331`,
`:4374`, `:4461`. Individual clause lines, all re-read verbatim:

| clause | line | note |
|---|---|---|
| `group_spans.is_none()` | 4214 | binding at `:3433` |
| `ordered_learning_perm.is_none()` | 4215 | |
| CTR alternation | 4225-4231 | `ctr_types_are_device_covered` at `:2241` |
| `!penalties_active` | 4232 | binding at `:3780-3782` |
| `monotone_constraints.is_empty()` | 4233 | |
| grow-policy match | 4237-4243 | |
| `approx_dimension == 1` / `!is_multiclass` / `!is_multilabel` | 4244-4246 | |
| bootstrap × grow_policy cross-product | 4263-4270 | |
| `random_strength == 0.0` | 4271 | out of scope |
| `eval_sets.is_empty()` | 4272 | |
| `has_any_scorable_feature` | 4288 | fn at `:3099` |
| one-hot cardinality fit | 4294-4299 | |
| `bias == 0.0` | 4312 | comment naming CR-01 at `:4306-4311` |
| `leaf_method ∈ {Gradient, Simple}` | 4318 | comment naming CR-02 at `:4313-4317` |

Supporting bindings: `let bias = starting_approx(params, target);` at `:3641`;
`device_config` construction ends at `:4459`; `begin_device_training` call at `:4462`;
the production `grow_tree_on_device` call at `:4772`; the device branch's terminal
`continue;` at `:5275`.

### V-2 — CR-01's fix point is ONE line, and only the OBLIVIOUS arm needs it (REFINED — this materially shrinks Track A)

`[V]` `crates/cb-backend/src/gpu_runtime/session.rs:1578-1580`:

```rust
// The running approx starts all-zero (the RMSE-from-zero MVP; boost_from_average is
// out of scope, the cross-oracle uses the SAME zero start).
let approx_h = upload_channel_floats(&client, &vec![0.0_f64; n]);
```

`[V]` **but the other three grow arms never read that seed.** Each re-derives `der1`
host-side from the CALLER's `approx` via `host_der1` (`session.rs:670-680`):

| arm | site | reads |
|---|---|---|
| Region | `session.rs:1843-1861` | `host_der1(der_kernel, approx, target)` — caller's approx |
| nonsym (Depthwise/Lossguide) | `session.rs:1881-1899` | idem |
| exact-leaf oblivious | `session.rs:1908-1932` | re-uploads `self.approx_h = upload_channel_floats(&self.client, approx)` EVERY tree |
| oblivious resident | `session.rs:1934-1948` | the resident zero seed — **the only arm affected** |

`[V]` and the caller's `approx` already starts at the bias:
`crates/cb-train/src/boosting.rs:3641` `let bias = starting_approx(params, target);` →
`:3649+` `let mut approx = …` seeded from it.

⇒ **T05's Green is literally `vec![0.0_f64; n]` → `vec![config.bias; n]` plus one new
`DeviceTrainConfig` field.** No per-arm work. This is the highest value-per-line change in
the phase and it also means Track E (exact leaf) is *already* bias-correct by
construction.

### V-3 — `DeviceTrainConfig`, not the argument list, is where new knobs go (CONVENTION, verified)

`[V]` `crates/cb-compute/src/runtime.rs:1353-1355`, the project's own words in the
`begin_device_training` doc comment: *"`config` is the single plain-host
[`DeviceTrainConfig`] carrying the grow-policy / sampling / exact-leaf / CTR knobs …
the config surface widens by mutating this struct, not the argument list."*

`[V]` `begin_device_training` already takes 13 parameters under
`#[allow(clippy::too_many_arguments)]` (`runtime.rs:1359-1374`) and has **36** mentions
across `crates/` including ~9 test wrappers that each re-list every argument.

⇒ SPEC §4's decision to put `bias` on `DeviceTrainConfig` is correct AND is the
project's own stated convention. It also keeps T05's blast radius at **2 files**
(`runtime.rs` + `session.rs`) instead of ~12.

### V-4 — Combination/tensor CTR IS semantically correct under the online statistic (research.md's MEDIUM item, now HIGH — independently re-derived)

research.md flagged this as MEDIUM confidence and explicitly asked for re-derivation. Done:

`[V]` **CPU side**, `crates/cb-train/src/ctr/ctr_feature.rs:190-220`:
1. per document `i ∈ 0..n`, build `feature_hashes[f] = calc_cat_feature_hash(col[i])`;
2. `combined_keys[i] = projection.combined_hash(&feature_hashes)`, which
   (`crates/cb-train/src/projection.rs:191-198`) folds `result = fold_cat_hash(result,
   hash)` over `self.cat_features` **in sorted order**, starting from `result = 0`;
3. remap keys to dense **first-seen bins in OBJECT order** (`:207-220`);
4. run the read-before-increment prefix over `permutation` on those bins.

`[V]` **Device side**, `crates/cb-backend/src/kernels/ctr_device.rs:89-117`
(`combine_projection_bins`): identical structure — per object `0..n`, fold
`key = fold_cat_hash(key, code)` over `member_bins` in order from `key = 0`, then
first-seen dense remap in object order; `bucket_count = remap.len()`. `fold_cat_hash` /
`calc_hash` are transcribed identically on both sides (`ctr_device.rs:66-78` vs
`projection.rs:63-84`, same `MAGIC_MULT = 0x4906_ba49_4954_cb65`, same
`(ui64)(int)hash` sign-extension).

`[V]` The **only** difference: CPU folds `calc_cat_feature_hash(value_string)` (a ui32
string hash) while the device folds `cat_eligible_buckets` codes, which
`cat_eligible_buckets_for` (`boosting.rs:2105-2119`) produces via
`cb_data::perfect_hash_bins` — a dense **first-seen perfect hash per feature**.

⇒ Both are **injective per feature**, so both folds induce the **same partition** of
objects into combined groups; and because both then remap first-seen in the **same object
order**, the resulting `combined_bins` vectors are **integer-identical**. The prefix
statistic (`ordered_ctr_prefix_kernel`, `ctr_device.rs:134-167`) depends only on bucket
identity, never on bucket numbering. **Verdict: combination CTR is `cb-train`-side glue,
full-detail-able — no sub-spike needed.**

`[V]` The plumbing is already double-called: `build_device_ctr_config`'s `build_columns`
closure (`boosting.rs:2292-2343`) is invoked for BOTH permutations at `:2345-2347`, and
`build_ctr_cindex_columns` (`session.rs:197-230`) already branches
`if col.member_bins.len() == 1 { … } else { combine_projection_bins(&col.member_bins, n)? }`
at `:207-213`. ⇒ T07 changes **two expressions** in `boosting.rs`
(`:2244` drop `is_simple()`; `:2299/:2335` map all members) and **zero** lines in
`cb-backend`.

`[C-1]` Residual: the folded KEYS differ between the two sides, so a 64-bit collision
occurring on one side and not the other would diverge the partitions. Negligible at
fixture scale; documented in SPEC §9, detected (not prevented) by T14's ≤1e-5 bar.

### V-5 — `grow_tree_on_device` blast radius is **15 sites across 11 files**, not 10 (CORRECTION)

`[C-2]` research.md carried the prior phase's "10 sites, not 2" figure. Four new test
files have landed since (the GDC merge). `grep -rn "grow_tree_on_device" crates/
--include=*.rs` this session, excluding doc-comment mentions:

| # | file:line | kind |
|---|---|---|
| 1 | `crates/cb-compute/src/runtime.rs:1430` | trait default (signature + `let _ = (...)`) |
| 2 | `crates/cb-backend/src/gpu_backend.rs:310` | the ONE real override |
| 3 | `crates/cb-train/src/boosting.rs:4772` | the ONE production caller |
| 4-6 | `crates/cb-backend/src/gpu_backend_test.rs:189,200,228` | 3 direct calls |
| 7 | `crates/cb-train/tests/bootstrap_dev_oracle_test.rs:261,267` | wrapper → `self.inner` |
| 8 | `crates/cb-train/tests/device_bootstrap_parity_test.rs:241,248` | wrapper |
| 9 | `crates/cb-train/tests/device_ctr_gate_test.rs:119,125` | wrapper *(NEW since GDC)* |
| 10 | `crates/cb-train/tests/device_one_hot_parity_test.rs:193,199` | wrapper |
| 11 | `crates/cb-train/tests/device_poisson_bootstrap_test.rs:195,202` | wrapper |
| 12 | `crates/cb-train/tests/device_seam_test.rs:69` | scripted-return mock |
| 13 | `crates/cb-train/tests/device_fold_count_gate_test.rs:69` | mock *(NEW since GDC)* |
| 14 | `crates/cb-train/tests/device_gate_composition_test.rs:128,134` | wrapper *(NEW since GDC)* |
| 15 | `crates/cb-train/tests/device_weight_gate_test.rs:126,132` | wrapper *(NEW since GDC)* |

`[V]` `CpuRefRuntime` in `device_nonsym_fit_test.rs` / `device_region_fit_test.rs` /
`device_weighted_*_test.rs` overrides only `compute_gradients` and inherits the seam
default — **unaffected**. T04 is sized for 15 and its Red step enumerates them.

### V-6 — Exact-leaf: the reachable loss set is `{Mae, Quantile}` ONLY (CORRECTION to research.md)

`[C-3]` research.md states *"MAE/Quantile/MAPE's documented upstream default IS Exact …
so this gap is reachable by ordinary use"*. Two parts of that are wrong:

1. `[V]` `crates/cb-train/src/boosting.rs:1387-1398` `validate_leaf_method` permits
   `LeafMethod::Exact` for exactly `LogCosh | Mae | Quantile | MultiQuantile` — **`Mape`
   is REJECTED**. So `session.rs:1009-1013`'s `Loss::Mape => Some(DeviceLeafMethod::Exact
   { mape: true })` arm is unreachable from `cb_train::train`. Conversely `LogCosh` IS
   CPU-legal but `map_leaf_method` returns `None` for it (`session.rs:1015`), so the
   device must decline. `MultiQuantile` is multi-dim (`approx_dimension == 1` still
   excludes it). ⇒ the admissible intersection is **`{Mae, Quantile}`**, exactly as
   SPEC FPP-06 specifies.
2. `[V]` `crates/catboost-rs/src/builder.rs:300` — `leaf_method: LeafMethod::Gradient`
   **unconditionally**. This project's builder does NOT auto-select `Exact` for
   MAE/Quantile. ⇒ the gap is reachable only via an **explicit**
   `leaf_estimation_method="Exact"`, not "by ordinary use". Still a genuine parameter
   parity gap (the parameter is IMPLEMENTED and settable), but T06/T13's framing must say
   "explicitly requested", not "default".

`[V]` `device_config.exact_leaf` is genuinely never set:
`grep -n exact_leaf crates/cb-train/src/boosting.rs` → only the COMMENT at `:4442`
("`exact_leaf` / `ctr` likewise stay default"). Default `false` at `runtime.rs:1220`.

### V-7 — Existing quantile fixtures are ALMOST device-shaped but not usable (blocker → new fixture)

`[C-4]` `[V]` `crates/cb-oracle/fixtures/quantile_alpha05_mae/config.json` and
`quantile_alpha07/config.json` pin `leaf_estimation_method: "Exact"`,
`boost_from_average: false`, `bootstrap_type: "No"`, `random_strength: 0`,
`score_function: "L2"`, `depth: 2`, `n_features: 4`, `n_rows: 50`, `n_iterations: 5` —
a near-perfect device recipe. **But**:
- `[V]` `ls` shows only `config.json`, `model.json`, `staged.npy` — **no
  `predictions.npy`** and no `X.npy`/`y.npy` (they reference `input_dataset:
  "numeric_tiny"`).
- `[V]` no `border_count` is pinned ⇒ they inherit upstream's 254-border default, while
  every green device fixture pins `border_count: 15` (16 bins).
- `[V]` their sole Rust consumer `crates/cb-train/tests/wave3_quantile_loss_oracle_test.rs`
  is a `CpuBackend`-importing file that cannot compile under
  `--no-default-features --features rocm`.

⇒ T02 must generate a NEW `exact_leaf_device/{mae,quantile07}/` pair on the
`weighted_device_sym/` artifact template. Reuse is not available.

### V-8 — every device-eligible fixture pins `boost_from_average=False` (research.md's requested per-fixture re-verification: CONFIRMED)

`[V]` `grep -o '"boost_from_average": *true' crates/cb-oracle/fixtures/*/config.json` →
exactly **4** fixtures pin it `true`: `feature_selection/`, `regression_skeleton/`,
`model_sum/`, `ordered_boost_e2e/`. **None** of the four is a device e2e fixture.
`[V]` every fixture the device tests actually consume — `weighted_device_sym/`,
`weighted_device_nonsym/`, `ctr_device_mixed/` — pins `"boost_from_average": false`
(confirmed per-fixture in both `config.json` and the generator source, e.g.
`weighted_device_sym/gen_fixtures.py:76`).

⇒ T09 (removing the `bias == 0.0` clause) **cannot** silently change any currently-green
device test: all of them compute `bias == 0.0` and therefore `config.bias == 0.0`, giving
a byte-identical resident seed. research.md's pitfall #5 is closed, not merely assumed.

`[V]` Side-effect worth carrying to Wave 7: `ordered_boost_e2e/` pins
`boost_from_average: true` — the prior phase's V-8 declared it device-INELIGIBLE for
exactly that reason. **After T09 that reason disappears.** T24 must re-evaluate whether
that fixture becomes reusable rather than pre-generating a new one.

### V-9 — eval_sets-on-device is a `boosting.rs`-only change (research.md's MEDIUM item, now strongly supported)

`[V]` `crates/cb-train/src/boosting.rs:2511-2527`:

```rust
fn last_tree_eval_contribution(
    trees: &[ObliviousTree],
    non_symmetric_trees: &[NonSymmetricTree],
    region_trees: &[RegionTree],
    matrix: &FeatureMatrix,
    obj: usize,
) -> f64 { … }
```

It reads the **folded tree ensembles**, never an approx buffer and never any device state.
`[V]` the device branch folds its grown tree into exactly those three ensembles
(`boosting.rs:4787-4800`, dispatching on `region_path` / `step_nodes` shape).
`[V]` the eval-set update + metric + detector block lives at `boosting.rs:6690-6733`,
inside the CPU body the device branch skips via `continue;` at `:5275`.

⇒ the mechanical hypothesis is *"run the same `if has_test { … }` block before the device
branch's `continue`"*. T31's spike starts from that, not from scratch. Three open
questions remain (guard teardown, snapshot interaction, `LevelKind::Ctr` on an eval matrix
with no CTR columns) — see T31.

### V-10 — Langevin/SGLB has zero CPU surface (D6's rationale: CONFIRMED)

`[V]` `grep -rn "langevin|diffusion_temperature|posterior_sampling" crates/cb-train/src
crates/cb-compute/src crates/catboost-rs/src --include=*.rs` → **zero hits**.
`[V]` `crates/catboost-rs-py/src/params.rs:275-277` lists all three in `VOCABULARY`,
absent from `IMPLEMENTED` (`:66-128`) ⇒ `KnownNotYet` ⇒ rejected at `fit()`.
D6 stands: this is a CPU feature gap, not a GPU-porting gap.

---

## 2. Global rules for every task

### 2.1 Commands (exact; deviations are documented traps)

| purpose | command |
|---|---|
| compute-crate types | `cargo check -p cb-compute` |
| backend build-verify (no GPU needed) | `cargo check -p cb-backend --no-default-features --features rocm` |
| backend self-oracles (real device) | `cargo test -p cb-backend --no-default-features --features rocm` |
| backend self-oracle, filtered | `cargo test -p cb-backend --no-default-features --features rocm -- <filter>` |
| new cb-train device test, build-verify | `cargo check -p cb-train --no-default-features --features rocm --test <name>` |
| new cb-train device test, real run | `cargo test -p cb-train --no-default-features --features rocm --test <name>` |
| CPU regression, single test | `cargo test -p cb-train --test <name>` |
| python-binding tests | `cargo test -p catboost-rs-py` |
| workspace CPU regression | `cargo test --workspace` |

**Traps (unchanged, both previously reproduced with real `cargo` output):**
- NEVER `--features rocm` without `--no-default-features` — `cpu` stays on,
  `SelectedRuntime` resolves to `cubecl-cpu`, and device tests either SKIP silently or
  compare cpu-vs-cpu and report a **false pass**.
- NEVER a package-wide `cargo test -p cb-train --no-default-features --features rocm`
  without `--test <name>` — ~52+ test files `use cb_backend::CpuBackend;` at module scope
  and fail `E0432`, blocking the whole run.

### 2.2 New-test conventions (mandatory)

Every new `crates/cb-train/tests/device_*.rs` file MUST:
1. carry `#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic,
   clippy::indexing_slicing, clippy::float_cmp)]`;
2. put every device-touching item inside
   `#[cfg(any(feature = "rocm", feature = "cuda"))] mod device { … }`;
3. **never** `use cb_backend::CpuBackend` — use a local `struct CpuRefRuntime;` overriding
   only `compute_gradients`;
4. mark shared helpers
   `#[cfg_attr(not(any(feature = "rocm", feature = "cuda")), allow(dead_code))]`;
5. **SKIP by printing and returning** on cpu/wgpu
   (`eprintln!("SKIP …: needs rocm/cuda")`) — never `panic!`, never `#[ignore]`;
6. **assert the RIGHT arm fired** — a `CountingGpu`-style wrapper counting
   `grow_tree_on_device` `Some` returns and asserting the count equals `iterations`
   (precedent: `bootstrap_dev_oracle_test.rs:261-267`), so a silent CPU fallback cannot
   pass.

### 2.3 Source rules

- Source/test separation is mandatory: no `#[cfg(test)] mod tests` in a production file.
  `cb-backend`/`cb-compute` self-oracles live in sibling `*_test.rs` files mounted as
  child modules.
- No `unwrap` / `expect` / `panic` / raw indexing anywhere in `crates/**/src/**`.
- Any host reduction routes through `cb_core::sum_f64` (D-07/D-08).
- **Do Not Hand-Roll** (from research.md, all re-verified present):
  `combine_projection_bins` (`cb-backend/src/kernels/ctr_device.rs:89`),
  `launch_ordered_ctr_resident` + `binarize_ctr_column_resident`,
  `fold_weights_resident` / `vector_mul_kernel`, `host_weighted_der1`
  (`session.rs:693`), `cb_core::sum_f64`, `kernels/exact_quantile.rs`.
- Preserve **D-04** (byte-unchanged CPU path for non-admitted configs), **D-05** (only
  O(1) per-level crossings), **D-10-01** (one per-fit all-or-nothing decision), **D-07**
  (ε=1e-4 device self-oracle / ≤1e-5 vs upstream), **T-10-04** (no `cubecl` and no
  `cb-train` type on any seam-crossing struct).
- **Ordering discipline**: never flip a `device_host_eligible` clause before the
  underlying computation is verified correct. Wave 2 exists solely to enforce this.
- Any CubeCL kernel work: read
  `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md` FIRST, and
  `cubecl_error_guideline.md` on any build error (AGENTS.md mandate).

### 2.4 Documented residuals (SPEC §9, carried forward)

- The fixed-point histogram's `|Σ| < 2^33` precondition has no enforced runtime guard.
  Every new fixture stays under `n · max(weight) · max(|der1|) < 2^33` by construction
  and documents the margin at the generator call site.
- V-4's `fold_cat_hash` collision asymmetry (combination CTR) is documented, not guarded.

---

## 3. Waves and dependency graph

Wave numbering below is finer-grained than SPEC D7's; the D7 mapping is shown per wave.

```text
WAVE 0  — fixtures + the seam type change  [D7 Wave 1; fully parallel]
  T01  bias fixture                  (FPP-03)   cb-oracle only
  T02  exact-leaf fixtures ×2        (FPP-07)   cb-oracle only
  T03  combination-CTR fixture       (FPP-10)   cb-oracle only
  T04  Runtime seam signature ×15    (FPP-15)   cb-compute + 10 more files

WAVE 1  — computations, BEFORE any gate  [D7 Wave 1]
  T05  DeviceTrainConfig.bias + resident seed   (FPP-01)  runtime.rs + session.rs
  T06  device_config.exact_leaf wiring          (FPP-05)  boosting.rs :4400-4459
  T07  combination-CTR gate fn + column builder (FPP-09)  boosting.rs :2241-2343
  T08  nonsym/Region host sample multiplier     (FPP-12)  session.rs :1843,:1881

WAVE 2  — the gate expression  [D7 Wave 1; STRICTLY SERIAL, one expression]
  T09 (FPP-02, :4312) -> T10 (FPP-06, :4318) -> T11 (FPP-13, :4263-4270)

WAVE 3  — e2e oracles  [D7 Wave 1; parallel]
  T12 bias e2e            (FPP-04)   needs T01,T05,T09
  T13 exact-leaf e2e      (FPP-08)   needs T02,T06,T10
  T14 combination-CTR e2e (FPP-11)   needs T03,T07
  T15 sampled-nonsym e2e  (FPP-14)   needs T08,T11

WAVE 4  — task_type  [D7 Wave 2; independently mergeable]
  T16 task_type validation (FPP-16)  catboost-rs-py only

WAVE 5  — benchmark  [D7 Wave 3]
  T17 generator extension (SPD-01)   needs nothing
  T18 grid harness        (SPD-02)   needs T17
  T19 Kaggle P100 run     (SPD-03)   needs T18 + ALL of Wave 3 merged

WAVE 6  — composition guards  [D7 Wave 1 tail]
  T20 cross-gap composition tests (FPP-17)  needs T07,T09,T10,T11

WAVE 7  — spikes and gated implementations  [D7 Wave 4; LAST]
  Ordered            T21 spike -> T22 -> T23 -> T24
  non-Borders CTR    T25 spike -> T26
  monotone           T27 spike -> T28
  penalties          T29 spike -> T30
  eval sets          T31 spike -> T32
  pairwise           T33 spike -> T36
  ranking            T34 spike -> T37
  multi-output       T35 spike -> T38
  (T22/T36/T37/T38 additionally need T04)
```

ASCII dependency edges (acyclic, verified):

```text
T01 ─────────────────► T12
T02 ─────────────────► T13
T03 ─────────────────► T14
T04 ─────────────────► T22, T36, T37, T38
T05 ─► T09 ─► T10 ─► T11
T05 ─────────────────► T12
T06 ─────────────────► T13
T07 ─► T14 ;  T07 ────► T20
T08 ─────────────────► T15
T09 ─► T12 ;  T09 ────► T20 ;  T09 ────► T24 (fixture re-evaluation)
T10 ─► T13 ;  T10 ────► T20
T11 ─► T15 ;  T11 ────► T20
T17 ─► T18 ─► T19 ;  {T12,T13,T14,T15} ─► T19
T21 ─► T22 ─► T23 ─► T24
T25 ─► T26 ;  T27 ─► T28 ;  T29 ─► T30 ;  T31 ─► T32
T33 ─► T36 ;  T34 ─► T37 ;  T35 ─► T38
```

**File-ownership serialisation (the prior phase's V-12 lesson, re-derived):**

| file / region | tasks |
|---|---|
| `boosting.rs` `device_host_eligible` (`:4214-4318`) — ONE expression | **T09 → T10 → T11 strictly serial**; later T23, T26, T28, T30, T32, T36-38 |
| `boosting.rs` `:2241-2343` (CTR coverage fn + column builder) | T07 alone (different function from the gate — parallel-safe with T09-T11) |
| `boosting.rs` `:4400-4459` (device_config) | T05 (adds `bias:`), T06 (adds `exact_leaf:`) — different fields, rebase-safe, coordinate |
| `session.rs` `begin` (`:1034-1700`) | T05 (`:1580`) — later T23 |
| `session.rs` `grow_one` (`:1779-2100`) | T08 (`:1843`, `:1881`) — later T22/T36-38 |
| `cb-compute/src/runtime.rs` | T04 (trait + `FamilyTreeArgs`), T05 (`DeviceTrainConfig.bias`) — different items, rebase-safe |
| `crates/cb-oracle/fixtures/**` | T01, T02, T03 fully disjoint |
| `crates/catboost-rs-py/src/params.rs` | T16 alone |
| `bench/**` | T17, T18, T19 serial |

---

## 4. Tasks

---

### T01 — `bias_device_sym/` upstream fixture

- **Specs**: FPP-03 · **Wave 0** · **Depends on**: none · **Parallel**: YES (cb-oracle only)

**Goal / observable completion**: a frozen fixture directory exists, loadable by a Rust
smoke test, produced by the installed `catboost==1.2.10` with a non-zero starting
approximant.

**Files**
- Create `crates/cb-oracle/fixtures/bias_device_sym/{gen_fixtures.py,config.json,X.npy,
  y.npy,borders.npy,model.json,predictions.npy}`.
- Template: `crates/cb-oracle/fixtures/weighted_device_sym/gen_fixtures.py` (verified: the
  current per-directory authoring convention; the central
  `crates/cb-oracle/generator/gen_fixtures.py` hosts only older scenarios).

**Pinned recipe (every value load-bearing)**
```
catboost==1.2.10, numpy.random.RandomState(0), thread_count=1
n_rows = 64, n_float_features = 2, border_count = 15        # -> 16 bins
loss_function = "RMSE"
iterations = 3, depth = 3, learning_rate = 0.3, l2_leaf_reg = 3.0
boosting_type = "Plain", bootstrap_type = "No", random_strength = 0
boost_from_average = True        # <- THE POINT OF THIS FIXTURE
leaf_estimation_method = "Gradient", leaf_estimation_iterations = 1
score_function = "L2", grow_policy = "SymmetricTree"
sample_weight = uniform (isolate the bias axis from the weight axis)
y drawn so that |mean(y)| > 0.5    # a near-zero mean cannot discriminate the fix
```
- Overflow margin comment (mandatory): `n · max(w) · max(|der1|) = 64 · 1.0 · max|y|
  ≪ 2^33 ≈ 8.6e9`; record the arithmetic verbatim.

**Red**
- Test: `crates/cb-oracle/tests/bias_device_fixture_smoke_test.rs`, fn
  `bias_device_fixture_loads_with_nonzero_bias`.
- Assertions: shapes load; `config.json` has `"boost_from_average": true`;
  `abs(mean(y)) > 0.5`; `model_json.float_feature_borders()` yields 2 columns × 15
  borders.
- Expected failure: directory does not exist ⇒ load error.
- Run: `cargo test -p cb-oracle --test bias_device_fixture_smoke_test`

**Green**
- Run `gen_fixtures.py` OFFLINE against the installed `catboost==1.2.10`; commit the
  artifacts; mark `"note": "FROZEN … NEVER regenerated in CI"` in `config.json`.
- **Anti-false-pass guard in the generator**: also train a `boost_from_average=False`
  sibling in-memory and `assert np.max(np.abs(pred_true - pred_false)) > 1e-6`, failing
  generation otherwise.

**Refactor** — none expected; keep the generator a single flat script like its template.

**Verify**
- `cargo test -p cb-oracle --test bias_device_fixture_smoke_test`
- `git status` shows only new files under `crates/cb-oracle/fixtures/bias_device_sym/`.

**Completion evidence**: directory committed, smoke test green, the >1e-6 discrimination
assertion present and passing during generation.

**Risks**: upstream float-border nondeterminism. **Mitigation**: the Rust side reads
borders from the committed `model.json` (verified precedent
`ordered_boost_e2e_oracle_test.rs:124-127`), never re-derives them.

---

### T02 — `exact_leaf_device/{mae,quantile07}/` upstream fixtures

- **Specs**: FPP-07 · **Wave 0** · **Depends on**: none · **Parallel**: YES

**Goal**: two frozen fixture directories exercising the device Exact order-statistic leaf,
in a device-reachable shape.

**Why new fixtures (V-7)**: `quantile_alpha05_mae/` and `quantile_alpha07/` already pin
`leaf_estimation_method="Exact"` with an otherwise device-shaped recipe, but ship no
`predictions.npy`/`X.npy`/`y.npy`, pin no `border_count` (inheriting upstream's 254), and
their only consumer imports `CpuBackend`. Reuse is not possible.

**Files**
- Create `crates/cb-oracle/fixtures/exact_leaf_device/mae/` and `.../quantile07/`, each
  with the full `weighted_device_sym/` artifact set.

**Pinned recipe** — identical to T01's except:
```
mae/        loss_function = "MAE",                 leaf_estimation_method = "Exact"
quantile07/ loss_function = "Quantile:alpha=0.7",  leaf_estimation_method = "Exact"
both:       boost_from_average = False   # isolate the exact-leaf axis from Track A
            leaf_estimation_iterations = 1
            border_count = 15            # see Risks
```

**Red**
- Test: `crates/cb-oracle/tests/exact_leaf_device_fixture_smoke_test.rs`.
- Assertions: both directories load; both `config.json`s record
  `"leaf_estimation_method": "Exact"` and `"border_count": 15`;
  `max|pred_mae − pred_quantile07| > 1e-6` (proves α is load-bearing — a device path that
  ignored `quantile_alpha` would pass a single-fixture test).
- Run: `cargo test -p cb-oracle --test exact_leaf_device_fixture_smoke_test`

**Green** — generate OFFLINE, commit, freeze.

**Refactor** — factor the shared generator body into one helper inside
`exact_leaf_device/` if duplication exceeds ~30 lines; do NOT move it to the central
generator (current convention is per-directory).

**Verify**
- `cargo test -p cb-oracle --test exact_leaf_device_fixture_smoke_test`

**Risks / open item**: SPEC FPP-07 flags `border_count=15` as the only value proven by
green device fixtures. **Before generating**, confirm the device's packed-cindex line
width admits it for a 2-float pool by reading `session.rs`'s `n_bins` validation and
`pad_hist_line_bins`. If 15 is not admissible for this shape, record the admissible value
in the task's completion note and pin THAT — do not silently generate at a value the gate
will reject.

---

### T03 — `ctr_device_combo/` upstream fixture (mixed float + 2 cat, `max_ctr_complexity=2`)

- **Specs**: FPP-10 · **Wave 0** · **Depends on**: none · **Parallel**: YES

**Goal**: a frozen fixture whose trained model contains at least one CTR split over a
2-member combination projection, in a device-reachable (float + cat) shape.

**Why new**: `tensor_ctr_e2e/` and `ctr_mixed_simple_vs_combo/` are cat-ONLY. A cat-only
pool gives `device_n_float = matrix.n_features() = 0` (`boosting.rs:4350`), so
`has_any_scorable_feature` (`:4288`) and the session's own `n_features == 0` decline both
fire. Only `ctr_device_mixed/` (the prior phase's T13) is device-reachable, and it has ONE
cat column — no combination possible.

**Files**
- Create `crates/cb-oracle/fixtures/ctr_device_combo/{gen_fixtures.py,config.json,X.npy,
  X_cat.npy,y.npy,borders.npy,model.json,predictions.npy}`.
- Template: `crates/cb-oracle/fixtures/ctr_device_mixed/gen_fixtures.py` (the proven
  device-reachable CTR recipe) — copy its pinned values verbatim, then change only the cat
  column count and `max_ctr_complexity`.

**Pinned recipe deltas from `ctr_device_mixed/`**
```
n_cat_features = 2 (small cardinalities, e.g. 3 and 4)
max_ctr_complexity = 2
one_hot_max_size = 1          # force BOTH cat columns to the CTR route
simple_ctr / combinations_ctr = Borders with a single prior, prior_denom = 1
permutation_count = 1
border_count = 15             # the CTR arm requires borders.len()+1 == n_bins (session.rs:151)
```

**Red**
- Test: `crates/cb-oracle/tests/ctr_device_combo_fixture_smoke_test.rs`.
- Assertions: shapes load; the parsed `model.json` contains **at least one CTR split whose
  projection has ≥2 members**. This is the discriminating assertion — a fixture where
  upstream happened to pick only simple projections cannot exercise T07 and must be
  re-seeded, not accepted.
- Also assert `max|pred_combo − pred_complexity1| > 1e-6` against an in-generator
  `max_ctr_complexity=1` sibling.
- Run: `cargo test -p cb-oracle --test ctr_device_combo_fixture_smoke_test`

**Green** — generate OFFLINE, commit, freeze. If the ≥2-member assertion fails, adjust the
seed / cardinalities / `n_rows` until upstream genuinely chooses a combination split;
record the search in the generator's docstring.

**Verify**
- `cargo test -p cb-oracle --test ctr_device_combo_fixture_smoke_test`

**Risks**: CatBoost quantization is run-to-run nondeterministic on categorical routing —
this is why every CTR fixture in this repo is FROZEN and never regenerated in CI. Preserve
that note.

---

### T04 — Extend `Runtime::grow_tree_on_device` with `family: Option<&FamilyTreeArgs>`

- **Specs**: FPP-15 · **Wave 0** · **Depends on**: none
- **Parallel**: YES with T01-T03. Shares `cb-compute/src/runtime.rs` with T05 (different
  items) — rebase-safe, coordinate.

**Goal / observable completion**: the workspace compiles with the new parameter threaded
through all **15** verified sites (V-5); every pre-existing test stays green; no
behavioural change for any fit.

**Files / symbols (all verified, V-5)**
- Modify `crates/cb-compute/src/runtime.rs:1430-1437` — the trait method + its
  `let _ = (approx, target, sample, family);` default.
- Create (same file) `pub enum FamilyTreeArgs<'a>` per SPEC §4 — plain host types only, no
  `cubecl` import (T-10-04).
- Modify `crates/cb-backend/src/gpu_backend.rs:310` — add the parameter, bind it
  (`let _ = family;`) this task; Wave 7 makes it load-bearing.
- Modify `crates/cb-train/src/boosting.rs:4772` — pass `None`.
- Modify (direct calls) `crates/cb-backend/src/gpu_backend_test.rs:189,200,228`.
- Modify (wrappers/mocks, **forwarding verbatim**):
  `crates/cb-train/tests/{bootstrap_dev_oracle_test.rs:261,267,
  device_bootstrap_parity_test.rs:241,248, device_ctr_gate_test.rs:119,125,
  device_one_hot_parity_test.rs:193,199, device_poisson_bootstrap_test.rs:195,202,
  device_seam_test.rs:69, device_fold_count_gate_test.rs:69,
  device_gate_composition_test.rs:128,134, device_weight_gate_test.rs:126,132}`.
- Create `crates/cb-compute/src/runtime_family_args_test.rs` (sibling test file, mounted
  as a child module — source/test separation).

**Red**
- The risk here is a **compile-time** regression, so Red is a build failure by
  construction: change the trait signature FIRST, then run
  `cargo check -p cb-compute && cargo check -p cb-backend --no-default-features --features rocm`.
- Expected failure: `E0050`/`E0061` at every site. **Enumerate the reported sites and
  confirm the count is exactly the 15 in V-5.** A site NOT on that list means the blast
  radius was mis-measured — STOP and re-run
  `grep -rn "grow_tree_on_device" crates/ --include=*.rs`.
- Also add `runtime_family_args_test.rs::family_tree_args_variants_are_plain_host_types`
  constructing all three variants from `&[u32]` / `&[f64]` / `usize` literals and
  asserting field round-trip — fails to compile before the enum exists.

**Green**
- Add the enum and the parameter; thread `None` everywhere.
- The trait default stays `Ok(None)` unconditionally — byte-unchanged.
- **Every wrapper forwards the parameter verbatim**
  (`self.inner.grow_tree_on_device(approx, target, sample, family)`), never a hard-coded
  `None`. A wrapper that swallows it would silently drop family state in Wave 7.

**Refactor**
- Doc-comment the new parameter with the FPP-15 rationale, the per-family variant meaning,
  and the T-10-04 constraint.

**Verify**
- `cargo check -p cb-compute`
- `cargo check -p cb-backend --no-default-features --features rocm`
- `cargo test --workspace` — every pre-existing test green.
- `cargo check -p cb-train --no-default-features --features rocm --test device_nonsym_fit_test`
- `cargo check -p cb-train --no-default-features --features rocm --test bootstrap_dev_oracle_test`
- `git diff --stat` shows exactly the 15 sites + the new enum + the new test file.

**Completion evidence**: clean workspace build; `cargo test --workspace` green; diff
limited to the enumerated sites.

**Risks**: a future family variant needing a `'static` lifetime or an owned buffer would
force a redesign. **Guardrail**: the enum is `#[non_exhaustive]`-free deliberately (adding
a variant is a compile error at every `match`, which is what we want during Wave 7), and
every field is a borrowed slice — document that any owned-data need must be escalated, not
worked around with a `Vec` field.

---

### T05 — `DeviceTrainConfig.bias` + seed the resident approx from it

- **Specs**: FPP-01 · **Wave 1** · **Depends on**: none (rebase-coordinate with T04 on
  `runtime.rs`) · **Parallel**: YES

**Goal / observable completion**: with `config.bias = b`, `GpuTrainSession::begin` seeds
the resident approximant to `vec![b; n]`, so the first tree's resident `der1` matches the
host reference. With `bias == 0.0` the uploaded bytes are identical to today.

**Files / symbols (verified, V-2 / V-3)**
- Modify `crates/cb-compute/src/runtime.rs` — add `pub bias: f64` to `DeviceTrainConfig`
  (near `exact_leaf` at `:1162`) and `bias: 0.0` to its `Default` impl (near `:1220`).
  Plain `f64`, no `cubecl` (T-10-04). **Do NOT add a `begin_device_training` argument** —
  the project's own doc comment at `runtime.rs:1353-1355` states the config surface widens
  by mutating this struct, not the argument list, and the signature already carries 13
  parameters under `#[allow(clippy::too_many_arguments)]` with ~9 test wrappers re-listing
  every one.
- Modify `crates/cb-backend/src/gpu_runtime/session.rs:1578-1580`:
  ```rust
  // FPP-01 (CR-01): the running approx starts at the fit's real starting approximant.
  // `bias == 0.0` is byte-identical to the former hardcoded zero seed (D-04).
  let approx_h = upload_channel_floats(&client, &vec![config.bias; n]);
  ```
  and update the stale comment ("boost_from_average is out of scope") in the same edit.
- **Touch nothing else.** V-2 verified the Region (`:1843`), nonsym (`:1881`) and
  exact-leaf (`:1908`) arms all re-derive `der1` from the CALLER's `approx` via
  `host_der1` (`:670-680`), and the caller's `approx` already starts at the bias
  (`boosting.rs:3641`, `:3649+`). Only the oblivious resident arm reads the seed.

**Red**
- New self-oracle in `crates/cb-backend/src/gpu_runtime/session_residency.rs`'s existing
  sibling test module (extend, do not create a parallel file), fn
  `resident_der1_starts_from_config_bias`.
- Setup: `n = 64`, two float features, 16 bins, RMSE, `config.bias = 1.75`.
- Assertion: after `begin`, the resident `der1` read back equals the host reference
  `target[i] - 1.75` for every `i`, at ε=1e-4. Add a Logloss variant asserting
  `target[i] - sigmoid(1.75)`.
- Expected failure: the device returns `target[i] - 0.0` ⇒ every element off by exactly
  `1.75` (RMSE) — far above 1e-4.
- Add the companion `resident_der1_bias_zero_is_byte_unchanged` asserting bit equality
  against the pre-change behaviour for `bias == 0.0`.
- Run: `cargo test -p cb-backend --no-default-features --features rocm -- resident_der1_starts_from_config_bias`

**Green**
- Implement exactly the two edits above. Nothing else.
- Do NOT relax `boosting.rs:4312` (T09 owns that).
- Do NOT set `device_config.bias` from `cb-train` yet (T09 owns that too — this task
  leaves the field at its `0.0` default from `cb-train`'s perspective, so behaviour is
  observably unchanged end-to-end).

**Refactor**
- Doc-comment the new field with the CR-01 rationale, the `starting_approx` source
  (`boosting.rs:3641`), and the V-2 note that only the oblivious arm consumes it.

**Verify**
- `cargo check -p cb-compute`
- `cargo test -p cb-backend --no-default-features --features rocm -- resident_der1`
- `cargo test -p cb-backend --no-default-features --features rocm` (full backend suite)
- `cargo test --workspace`
- Confirm: no `cb-train` test changes behaviour (the field is `0.0` until T09).

**Completion evidence**: both new assertions green; `git diff` touches exactly
`runtime.rs` (2 lines + doc) and `session.rs` (1 line + comment) plus the test.

**Risks**: `upload_channel_floats` is channel-typed (f32 on wgpu, f64 elsewhere) — a
non-zero bias narrows to f32 on wgpu. **Mitigation**: wgpu is structurally excluded from
device training already (no f64/u64-atomic); note it, do not guard it.

---

### T06 — Wire `device_config.exact_leaf` + `quantile_alpha`/`quantile_delta`

- **Specs**: FPP-05 · **Wave 1** · **Depends on**: none (coordinate with T05 on the same
  `device_config` literal) · **Parallel**: YES

**Goal / observable completion**: a fit with `leaf_method == Exact` and
`loss ∈ {Mae, Quantile}` produces a `DeviceTrainConfig` with `exact_leaf == true` and the
correct α/δ. Every other fit's config is byte-identical to today.

**Files / symbols (verified, V-6)**
- Modify `crates/cb-train/src/boosting.rs`'s `device_config` literal (`:4400-4459`),
  replacing the `exact_leaf`-mentioning comment at `:4442` with real field assignments:
  ```rust
  // FPP-05: activate the device Exact order-statistic leaf ONLY for the intersection of
  // (a) what validate_leaf_method permits (boosting.rs:1387-1398: LogCosh | Mae |
  // Quantile | MultiQuantile) and (b) what map_leaf_method covers (session.rs:996-1017:
  // Mae | Quantile | Mape). LogCosh is CPU-legal but device-uncovered; Mape is
  // device-covered but CPU-rejected; MultiQuantile is multi-dim. => {Mae, Quantile}.
  exact_leaf: matches!(params.leaf_method, LeafMethod::Exact)
      && matches!(params.loss, Loss::Mae | Loss::Quantile { .. }),
  quantile_alpha: match params.loss { Loss::Quantile { alpha, .. } => alpha, _ => <default> },
  quantile_delta: match params.loss { Loss::Quantile { delta, .. } => delta, _ => <default> },
  ```
  (Read the exact `Loss::Quantile` field names and the `DeviceTrainConfig` defaults at
  `runtime.rs:1220` before writing; use the struct's own defaults for the `Mae` case
  rather than re-typing `0.5`/`1e-6`.)

**Red**
- New test file `crates/cb-train/tests/device_exact_leaf_config_test.rs` (CPU-feature
  test — it only inspects config construction, no device needed).
- **Blocker to resolve at execution time**: `device_config` is a local inside
  `train_inner` and is not observable from a test today. Resolve the SAME way T01 of the
  prior phase did: either (a) extract the two-field decision into a
  `#[must_use] pub(crate) fn device_exact_leaf_config(leaf_method, loss) ->
  (bool, f64, f64)` and unit-test THAT, or (b) observe it through a `Runtime` mock that
  records the `&DeviceTrainConfig` it was handed by `begin_device_training`. **(a) is
  preferred** — it is a pure function and needs no device. Record the choice in the
  completion note.
- Cases: `(Gradient, Rmse) → (false, …)`; `(Exact, Mae) → (true, 0.5, 1e-6)`;
  `(Exact, Quantile{0.7, 1e-6}) → (true, 0.7, 1e-6)`; `(Exact, LogCosh) → (false, …)`;
  `(Exact, MultiQuantile) → (false, …)`.
- Expected failure: the function does not exist / `exact_leaf` is always `false`.
- Run: `cargo test -p cb-train --test device_exact_leaf_config_test`

**Green**
- Implement the extraction + the field assignments. Nothing else.
- **Do NOT relax `boosting.rs:4318`** (T10 owns that). This task must leave `device_active`
  unchanged for every fit — an `Exact` fit still falls back to CPU, so the new config is
  built but never reaches `begin`. This ordering is mandatory (research.md pitfall #2:
  relaxing the gate without the config would silently apply the Gradient `calc_average`
  leaf to a Quantile fit — wrong, and worse than today's correct fallback).

**Refactor**
- Doc-comment the extracted function with the V-6 intersection derivation verbatim, so the
  LogCosh/Mape asymmetry is discoverable at the decision site.

**Verify**
- `cargo test -p cb-train --test device_exact_leaf_config_test`
- `cargo test --workspace`
- Confirm every existing device e2e test is unchanged (all use Gradient).

**Completion evidence**: 5 config cases green; `device_active` provably unchanged.

---

### T07 — Admit non-simple CTR projections and populate ALL members

- **Specs**: FPP-09 · **Wave 1** · **Depends on**: none · **Parallel**: YES (different
  function from `device_host_eligible`, so it does not serialise against T09-T11)

**Goal / observable completion**: a `max_ctr_complexity ≥ 2` fit whose CTR columns include
a 2-member combination projection is device-covered, and each such `DeviceCtrColumn`
carries one `member_bins` entry per projection member — never a repeat of the first.

**Files / symbols (verified, V-4)**
- Modify `crates/cb-train/src/boosting.rs:2241-2249` `ctr_types_are_device_covered`:
  drop the `col.projection.is_simple()` conjunct at `:2244`. Keep the other three
  (`ctr_type == Borders`, `target_border_idx == 0`, `prior_denom == 1.0`) — Buckets/BTMV/
  Counter is Track U (T25/T26), not this task.
- Modify `crates/cb-train/src/boosting.rs:2296-2317` inside `build_columns`: replace the
  single-member extraction
  ```rust
  let abs = col.projection.cat_features().first().copied()...;
  let pos = eligible_absolute.iter().position(|&a| a == abs)...;
  let member = cat_eligible_buckets.get(pos).cloned()...;
  ```
  with a `Vec<Vec<u32>>` collected over **all** of `col.projection.cat_features()` in
  projection-sorted order (the order `TProjection` guarantees via
  `projection.rs:168-176`'s sort+dedup, and the SAME order `combined_hash`
  (`projection.rs:191-198`) folds in), preserving the existing per-member
  `eligible_absolute` position lookup and its two typed errors verbatim.
- Modify `:2335` `member_bins: vec![member]` → `member_bins: members`.
- **Touch nothing in `cb-backend`** — `build_ctr_cindex_columns` (`session.rs:197-230`)
  already branches on `col.member_bins.len() == 1` and calls `combine_projection_bins`
  otherwise (`:207-213`), and `ctr_covered` (`:130-159`) requires only
  `!col.member_bins.is_empty()`.
- Both `boosting.rs` edits land in ONE commit: relaxing the gate without the builder would
  score a combination split from ONE member's bins — WRONG, not merely worse
  (research.md pitfall #3).

**Red**
- New test file `crates/cb-train/tests/device_ctr_combo_config_test.rs`, CPU-feature
  (config-level only, no device).
- Same observability question as T06 — prefer extracting/testing `build_device_ctr_config`
  directly (it is already a free function at `boosting.rs:2263`, so it may only need
  `pub(crate)` visibility, not a new extraction). Record the choice.
- Cases:
  - a 2-member projection ⇒ `member_bins.len() == 2` and
    `member_bins[0] != member_bins[1]` (the discriminating assertion — the pre-fix code
    yields `len() == 1`);
  - a simple projection ⇒ `member_bins.len() == 1` and the whole `DeviceCtrColumn` is
    equal to what the pre-change code produced (D-04 regression);
  - `ctr_types_are_device_covered` returns `true` for an all-Borders mixed
    simple+combination column set, and still `false` when any column is non-Borders.
- Expected failure: `member_bins.len() == 1` for the combination case; coverage `false`.
- Run: `cargo test -p cb-train --test device_ctr_combo_config_test`

**Green**
- Implement both edits. Nothing else.
- Keep `unwrap`-free: every `.get`/`.position` keeps its existing typed
  `CbError::OutOfRange` / `CbError::Degenerate`.

**Refactor**
- Replace `ctr_types_are_device_covered`'s doc comment (`:2230-2240`, which currently
  states "SIMPLE (single-feature) projections … Everything else … declines") with the
  corrected statement, and cite V-4's integer-identity derivation at the
  `build_columns` call site so the equivalence is discoverable rather than folklore.

**Verify**
- `cargo test -p cb-train --test device_ctr_combo_config_test`
- `cargo test -p cb-train --test device_ctr_gate_test` (existing gate regression)
- `cargo test --workspace`
- `cargo test -p cb-train --no-default-features --features rocm --test device_ctr_fit_test`
  (the existing simple-CTR device e2e must stay green **unmodified**)

**Completion evidence**: combination columns carry all members; the simple-projection
config is byte-identical; `device_ctr_fit_test` green unmodified.

**Risks**: `[C-1]` — the CPU folds string hashes, the device folds perfect-hash bucket
codes; a 64-bit collision on one side only would diverge. Document at the call site;
T14's ≤1e-5 e2e bar is the detector.

---

### T08 — Host sample multiplier for the nonsym and Region device growers

- **Specs**: FPP-12 · **Wave 1** · **Depends on**: none · **Parallel**: YES vs T05-T07
  (different function); serialise against any later `grow_one` edit.

**Goal / observable completion**: with a length-`n` `sample`, the Region and nonsym device
growers score splits over `der1 ⊙ sample` and `weight ⊙ sample`, while leaf values keep
using the UNSAMPLED channels. With an empty `sample`, behaviour is byte-unchanged.

**Files / symbols (verified this session)**
- `crates/cb-backend/src/gpu_runtime/session.rs:1843-1861` — the Region arm:
  `let der1 = host_weighted_der1(…); return grow_region_tree(…)`.
- `crates/cb-backend/src/gpu_runtime/session.rs:1881-1899` — the nonsym arm, same shape.
- `crates/cb-backend/src/gpu_runtime/session.rs:693` — `host_weighted_der1(der1, weights_uniform, weight)`,
  the existing caller-side substitution helper the prior phase introduced (GDC-03/04).
- **Do NOT edit `crates/cb-backend/src/kernels/nonsym_grow.rs` or `region_device.rs`.**
  The prior phase's V-2 verified both take `der1: &[f64]` / `weight: &[f64]` as plain host
  slices and use `der1` only as a contribution sum; the fix is entirely caller-side, one
  substitution per arm, which covers the leaf value, the split score AND the unsplit-gain
  baseline in one edit.
- The contract to honour verbatim, from `runtime.rs:1418-1423`: *"the backend folds
  [`sample`] into the SPLIT-SCORING stat channels ONLY. Leaf estimation stays on the
  UNSAMPLED derivatives / weights."* ⇒ the two growers need SEPARATE score and leaf
  channels; check whether they already accept distinct ones (the oblivious arm does, via
  `score_der1_h`/`score_weight_h` at `session.rs:1933-1976`). If they do not, add the
  parameter pair to the two grower signatures — that IS an in-kernel-file edit, and this
  task's Green must record it explicitly rather than silently folding the sample into the
  leaf inputs too.

**Red**
- Extend `crates/cb-backend/src/kernels/nonsym_grow_test.rs` and
  `region_device_test.rs` with a sampled variant of each existing device-vs-CPU
  comparison, fn names `*_matches_cpu_with_nontrivial_sample`.
- Setup: `n = 64`, a `sample` with ~30% zeros and the rest in `{0.5, 1.0, 2.0}`; a
  transcribed CPU reference that (i) scores over `der1[i]*sample[i]` /
  `weight[i]*sample[i]` and (ii) computes leaf values over raw `der1`/`weight`.
- Assertion: split choices integer-equal, leaf values at ε=1e-4.
- Expected failure: the device ignores `sample` on these arms, so an object with
  `sample[i] == 0` still contributes to the histogram ⇒ different split choice.
- Run: `cargo test -p cb-backend --no-default-features --features rocm -- with_nontrivial_sample`

**Green**
- Implement the caller-side substitution in both arms, guarded on `sample.is_empty()` so
  the unsampled path launches ZERO extra work (D-04 byte-and-perf-unchanged).
- Route any host reduction through `cb_core::sum_f64`.

**Refactor**
- If the substitution expression is duplicated verbatim across the two arms, factor a
  `host_sampled_channels(der1, weight, sample) -> (Vec<f64>, Vec<f64>)` sibling to
  `host_weighted_der1` at `session.rs:693`. Cosmetic; keep the empty-sample early return.

**Verify**
- `cargo test -p cb-backend --no-default-features --features rocm -- nonsym_grow`
- `cargo test -p cb-backend --no-default-features --features rocm -- region_device`
- `cargo test -p cb-backend --no-default-features --features rocm`
- `cargo test -p cb-train --no-default-features --features rocm --test device_nonsym_fit_test`
- `cargo test -p cb-train --no-default-features --features rocm --test device_region_fit_test`
- `cargo test -p cb-train --no-default-features --features rocm --test device_weighted_nonsym_fit_test`
  (all three must stay green **unmodified** — they pass an empty `sample`)

**Completion evidence**: sampled variants green at ε=1e-4; the three existing e2e tests
green unmodified.

**Risks**: the sample and the weight both multiply the score channel, so a weighted ×
sampled fit multiplies twice — which is CORRECT (it mirrors the oblivious arm's nested
`fold_weights_resident(fold_weights_resident(der1, weight), sample)` at
`session.rs:1933-1976`), but the CPU reference in the Red step must do the same or the
test will chase a phantom. State it explicitly in the test's doc comment.

---

### T09 — Remove `device_host_eligible`'s `bias == 0.0` clause and populate `config.bias`

- **Specs**: FPP-02 · **Wave 2 (SERIAL 1 of 3)** · **Depends on**: T05
- **Parallel**: NO — T09/T10/T11 edit the SAME expression (`boosting.rs:4214-4318`).

**Goal / observable completion**: a default `boost_from_average=true` RMSE fit reaches the
device and trains from the correct starting approximant.

**Files / symbols (verified, V-1 / V-8)**
- Modify `crates/cb-train/src/boosting.rs:4306-4312` — delete the `&& bias == 0.0` clause
  and replace its CR-01 comment with the FPP-01/02 rationale.
- Modify the `device_config` literal (`:4400-4459`) — add `bias,` (the local at `:3641`).
- **Both edits in ONE commit.** Removing the clause without setting the field would seed
  the device from zero on a biased fit — wrong, and worse than today's CPU fallback.

**Red**
- New test file `crates/cb-train/tests/device_bias_gate_test.rs`, following
  `device_weight_gate_test.rs`'s `CountingGpu` structure verbatim.
- fn `boost_from_average_fit_commits_to_device`: an otherwise device-eligible RMSE fit
  with `boost_from_average = true` (so `bias != 0.0`); assert the `CountingGpu`'s
  `grow_tree_on_device` `Some` count equals `iterations` (i.e. `device_active == true`).
- fn `zero_bias_fit_still_commits_to_device`: the same fit with
  `boost_from_average = false`; assert the same count (D-04 regression — the byte-unchanged
  path).
- Expected failure: the first fn records 0 device trees.
- Run: `cargo check -p cb-train --no-default-features --features rocm --test device_bias_gate_test`
  then `cargo test -p cb-train --no-default-features --features rocm --test device_bias_gate_test`

**Green**
- Delete the clause; add the field. Nothing else.
- Do NOT touch `:4318` (T10) or `:4263-4270` (T11).

**Refactor**
- Update the surrounding comment block at `:4300-4312` so it no longer claims CR-01 is
  unrelaxed, and cite V-2's finding (only the oblivious arm consumed the zero seed) so the
  next reader does not re-audit the other three arms.

**Verify**
- `cargo test -p cb-train --no-default-features --features rocm --test device_bias_gate_test`
- **Every existing device e2e test, unmodified** — V-8 verified all of them pin
  `boost_from_average=False`, so `config.bias == 0.0` and the seed is byte-identical:
  `device_nonsym_fit_test`, `device_region_fit_test`, `device_weighted_fit_test`,
  `device_weighted_nonsym_fit_test`, `device_ctr_fit_test`, `bootstrap_dev_oracle_test`,
  `device_bootstrap_parity_test`, `device_poisson_bootstrap_test`,
  `device_one_hot_parity_test`, `device_seam_test`, `device_gate_composition_test`.
- `cargo test --workspace`

**Completion evidence**: the biased fit records `iterations` device trees; all listed
device tests green unmodified.

**Risks**: `Loss::RmseWithUncertainty` has a per-DIMENSION starting approx
(`boosting.rs:3649-3660`, `rmse_uncertainty_starting_approx` at `:1285`), not the scalar
`bias`. It is already excluded by `approx_dimension == 1` (`:4244`), so the scalar
`config.bias` can never be handed a multi-dim fit — **assert this in a comment**, and
re-check it in T38 (multi-output) before relaxing `:4244`.

---

### T10 — Admit `LeafMethod::Exact` for `{Mae, Quantile}`

- **Specs**: FPP-06 · **Wave 2 (SERIAL 2 of 3)** · **Depends on**: T06, T09
- **Parallel**: NO (same expression).

**Goal / observable completion**: an explicit `leaf_estimation_method="Exact"` MAE or
Quantile fit reaches the device; a LogCosh-Exact fit still declines.

**Files / symbols (verified, V-6)**
- Modify `crates/cb-train/src/boosting.rs:4313-4318`:
  ```rust
  && (matches!(params.leaf_method, LeafMethod::Gradient | LeafMethod::Simple)
      // FPP-06: the device Exact ORDER STATISTIC (map_leaf_method, session.rs:996-1017;
      // kernels/exact_quantile.rs) is admitted for exactly the {Mae, Quantile}
      // intersection derived in T06. LogCosh is CPU-legal (validate_leaf_method,
      // boosting.rs:1387-1398) but device-UNCOVERED; Mape is device-covered but
      // CPU-REJECTED; MultiQuantile is multi-dim (excluded at :4244).
      || (matches!(params.leaf_method, LeafMethod::Exact)
          && matches!(params.loss, Loss::Mae | Loss::Quantile { .. })));
  ```

**Red**
- New test file `crates/cb-train/tests/device_exact_leaf_gate_test.rs`, `CountingGpu`
  structure.
- fn `exact_leaf_mae_fit_commits_to_device` — expect `iterations` device trees.
- fn `exact_leaf_quantile_fit_commits_to_device` — same.
- fn `exact_leaf_logcosh_fit_declines_to_cpu` — expect **0** device trees. This is the
  discriminating negative: CPU permits it, the device cannot compute it, and a naive
  `matches!(leaf_method, Exact)` relaxation would wrongly admit it and then silently apply
  `calc_average`.
- fn `gradient_fits_still_commit` — D-04 regression.
- Expected failure: the two positive fns record 0.
- Run: `cargo test -p cb-train --no-default-features --features rocm --test device_exact_leaf_gate_test`

**Green**
- Implement exactly the clause above.
- Do NOT touch `:4263-4270` (T11).

**Refactor**
- None beyond the comment; keep the derivation inline (it is the non-obvious part).

**Verify**
- `cargo test -p cb-train --no-default-features --features rocm --test device_exact_leaf_gate_test`
- `cargo test -p cb-train --test wave3_quantile_loss_oracle_test` (the CPU Exact oracle —
  must stay green; it runs on `CpuBackend` and is untouched)
- `cargo test --workspace`

**Completion evidence**: MAE and Quantile admit, LogCosh declines, Gradient unchanged.

---

### T11 — Relax the bootstrap × grow_policy cross-product (keep Poisson SymmetricTree-only)

- **Specs**: FPP-13 · **Wave 2 (SERIAL 3 of 3)** · **Depends on**: T08, T10
- **Parallel**: NO (same expression).

**Goal / observable completion**: a `{Depthwise, Lossguide, Region} × {Bayesian, Bernoulli,
Mvs}` fit reaches the device; Poisson on any non-SymmetricTree policy still does not.

**Files / symbols (verified, V-1)**
- Modify `crates/cb-train/src/boosting.rs:4263-4270`:
  ```rust
  && (matches!(params.bootstrap_type, EBootstrapType::No)
      // FPP-13: the three HOST-sampled types (Design A — bootstrap() runs host-side and
      // only the per-object multiplier crosses the seam) are now admitted on EVERY
      // covered grow policy; T08 folds the multiplier into the nonsym / Region score
      // channels exactly as the oblivious arm already does.
      || matches!(params.bootstrap_type,
                  EBootstrapType::Bayesian | EBootstrapType::Bernoulli | EBootstrapType::Mvs)
      // POISSON stays SymmetricTree-ONLY: it is the one arm the DEVICE samples resident
      // (sample_from_host = false, :4424-4426), that resident sampler lives in the
      // oblivious path, and :4497+ hard-ERRORS a Poisson fit that fails to commit —
      // widening it without a resident non-symmetric sampler would turn a correct CPU
      // fallback into a user-facing error.
      || (matches!(params.bootstrap_type, EBootstrapType::Poisson)
          && matches!(params.grow_policy, EGrowPolicy::SymmetricTree)))
  ```

**Red**
- New test file `crates/cb-train/tests/device_nonsym_bootstrap_gate_test.rs`.
- fn `depthwise_bernoulli_commits_to_device`, `region_mvs_commits_to_device`,
  `lossguide_bayesian_commits_to_device` — each expects `iterations` device trees.
- fn `region_poisson_still_errors` — expects the existing typed Poisson error
  (`boosting.rs:4497+`), NOT a silent CPU fallback. Discriminating negative.
- fn `symmetric_bernoulli_still_commits` — D-04 regression.
- Run: `cargo test -p cb-train --no-default-features --features rocm --test device_nonsym_bootstrap_gate_test`

**Green** — implement the clause. Nothing else.

**Refactor** — none beyond the comment.

**Verify**
- `cargo test -p cb-train --no-default-features --features rocm --test device_nonsym_bootstrap_gate_test`
- `cargo test -p cb-train --no-default-features --features rocm --test device_bootstrap_parity_test`
- `cargo test -p cb-train --no-default-features --features rocm --test device_poisson_bootstrap_test`
- `cargo test --workspace`

**Completion evidence**: three positives admit, Poisson×Region errors, SymmetricTree
unchanged.

**Risks**: `replay_grow_draws(&mut rng, params.depth, matrix.n_features())`
(`boosting.rs:5261-5266`) restores the RNG phase the device branch skipped — it is written
for the OBLIVIOUS level search's draw count. A non-symmetric level search consumes a
different number of draws. **Before Green, verify** whether `replay_grow_draws` is
policy-aware; if it is not, the sampled nonsym device fit will diverge from the CPU
reference at tree 2, and that divergence — not the multiplier — becomes the principal
failure. **If not policy-aware, STOP and escalate**: this is a distinct defect, and it
should be fixed in its own task before T11's clause is relaxed. Record the finding either
way. (Precedent for this class: the MVS tree-2 upstream gap, resolved in `d7676b5`, whose
root cause was exactly fabricated/missing RNG draws.)

---

### T12 — E2e device oracle: non-zero-bias training

- **Specs**: FPP-04 · **Wave 3** · **Depends on**: T01, T05, T09 · **Parallel**: YES

**Goal**: a `boost_from_average=true` device fit matches upstream at ≤1e-5.

**Files**
- Create `crates/cb-train/tests/device_bias_fit_test.rs`, modelled on
  `device_weighted_fit_test.rs` (module gated `#[cfg(any(feature = "rocm", feature =
  "cuda"))]`, local `CpuRefRuntime`, no `CpuBackend` import, SKIP-by-printing off
  rocm/cuda).

**Red**
- fn `bias_device_sym_matches_upstream`: load `bias_device_sym/`, read float borders from
  the committed `model.json` (never re-derive), train via
  `cb_train::train(&GpuBackend::default(), …)` wrapped in `CountingGpu`, assert
  (i) device tree count == `iterations`, (ii)
  `max|pred − predictions.npy| ≤ 1e-5`.
- Expected failure: before T09 the count is 0; after T09 but before T05 the predictions
  miss by ~`bias × Σ(1-lr)^k`.
- Run: `cargo check -p cb-train --no-default-features --features rocm --test device_bias_fit_test`
  then `cargo test -p cb-train --no-default-features --features rocm --test device_bias_fit_test`

**Green** — none; T05/T09 already implemented the behaviour. If it fails, the fix belongs
in T05 or T09, not here.

**Refactor** — extract the shared fixture-loading helper if it duplicates
`device_weighted_fit_test.rs`'s by more than ~20 lines; keep it inside the new file
(cross-test-file helpers are not the convention here).

**Verify**
- `cargo test -p cb-train --no-default-features --features rocm --test device_bias_fit_test`
- Report the measured `max|diff|` in the completion note (the project's house style).

---

### T13 — E2e device oracle: Exact-leaf training

- **Specs**: FPP-08 · **Wave 3** · **Depends on**: T02, T06, T10 · **Parallel**: YES

**Goal**: MAE and Quantile(α=0.7) device fits match upstream at ≤1e-5, proving the device
Exact order statistic (`kernels/exact_quantile.rs`) reachable and correct end-to-end.

**Files** — create `crates/cb-train/tests/device_exact_leaf_fit_test.rs`.

**Red**
- fn `exact_leaf_mae_matches_upstream` and `exact_leaf_quantile07_matches_upstream`, each:
  `device_active` proven via `CountingGpu`, predictions ≤1e-5 vs the fixture.
- The α=0.7 case is the discriminating one — a device path that silently used
  `calc_average` (or ignored `quantile_alpha`) passes neither, but α=0.7 fails it by the
  widest margin.
- Run: `cargo test -p cb-train --no-default-features --features rocm --test device_exact_leaf_fit_test`

**Green** — none; T06/T10 implemented it.

**Verify** — as T12, plus `cargo test -p cb-backend --no-default-features --features rocm
-- exact_quantile` (the kernel's own self-oracle must stay green).

---

### T14 — E2e device oracle: combination-CTR training

- **Specs**: FPP-11 · **Wave 3** · **Depends on**: T03, T07 · **Parallel**: YES

**Goal**: a combination-projection CTR device fit matches upstream at ≤1e-5, and matches a
`CpuBackend` run of the same fit at ε=1e-4.

**Files** — create `crates/cb-train/tests/device_ctr_combo_fit_test.rs`, modelled on
`device_ctr_fit_test.rs`.

**Red**
- fn `ctr_device_combo_matches_upstream`: `device_active` via `CountingGpu`; predictions
  ≤1e-5 vs `ctr_device_combo/predictions.npy`.
- fn `ctr_device_combo_matches_cpu_reference`: the same fit through a local
  `CpuRefRuntime`; predictions agree at ε=1e-4. **This is the assertion that would catch
  `[C-1]`** (a `fold_cat_hash` collision asymmetry between the two sides).
- Expected failure: before T07 the count is 0.
- Run: `cargo test -p cb-train --no-default-features --features rocm --test device_ctr_combo_fit_test`

**Green** — none; T07 implemented it.

**Verify**
- `cargo test -p cb-train --no-default-features --features rocm --test device_ctr_combo_fit_test`
- `cargo test -p cb-train --no-default-features --features rocm --test device_ctr_fit_test`
  (simple-CTR e2e, unmodified)

---

### T15 — E2e device oracle: sampled non-symmetric training

- **Specs**: FPP-14 · **Wave 3** · **Depends on**: T08, T11 · **Parallel**: YES

**Goal**: `{Depthwise, Lossguide, Region} × {Bayesian, Bernoulli, Mvs}` device fits match a
`CpuBackend` run from the same seed at ε=1e-4, with no mid-run fallback.

**Files** — create `crates/cb-train/tests/device_nonsym_bootstrap_test.rs`, one `#[test]`
per policy delegating to a shared `device::run(policy, bootstrap_type, label)`, mirroring
`device_bootstrap_parity_test.rs`.

**Red**
- 9 scenarios (3 policies × 3 bootstrap types). Each asserts (i) `CountingGpu` device tree
  count == `iterations`, (ii) device-vs-CPU predictions at ε=1e-4 over ALL iterations
  (not just the last — an RNG-phase divergence typically appears at tree 2, so a
  final-only assertion would hide it).
- Run: `cargo test -p cb-train --no-default-features --features rocm --test device_nonsym_bootstrap_test`

**Green** — none; T08/T11 implemented it.

**Verify** — as above, plus the three existing nonsym/region e2e tests unmodified.

**Risks** — see T11's `replay_grow_draws` risk. If T11 escalated it, T15 is BLOCKED behind
that fix.

---

### T16 — `task_type` as an informational/validated parameter

- **Specs**: FPP-16 · **Wave 4 (D7 Wave 2)** · **Depends on**: none
- **Parallel**: YES with everything — `crates/catboost-rs-py` only, independently
  mergeable.

**Goal / observable completion**: `task_type="CPU"` is accepted on any wheel and changes
nothing; `task_type="GPU"` is accepted on a device-feature wheel and errors actionably on a
`cpu`-only wheel; any other value errors listing the two legal values.

**Files / symbols (verified)**
- Modify `crates/catboost-rs-py/src/params.rs`:
  - add `"task_type"` to `IMPLEMENTED` (`:66-128`). It is already in `VOCABULARY`
    (`:233`), so `status_of` flips from `KnownNotYet` to `Implemented` and
    `validate_params` (`:395-423`) stops rejecting it.
  - add a `task_type`-specific value validator invoked from the same ingest path that
    handles the other `Implemented` params, returning `CatBoostParameterError`.
  - the compiled-backend test is expressible in-crate:
    ```rust
    const DEVICE_FEATURE_COMPILED: bool =
        cfg!(any(feature = "wgpu", feature = "cuda", feature = "rocm"));
    ```
    (verified: `crates/catboost-rs-py/Cargo.toml` `[features]` declares
    `cpu`/`rocm`/`wgpu`/`cuda`, each forwarding to `catboost-rs/<backend>`.)
  - `devices` and `device_config` (`:234-235`) stay `KnownNotYet` — do not touch.
- **Do NOT touch `crates/catboost-rs/src/builder.rs`** — D1 is pure input validation; the
  compile-time backend binding at `builder.rs:1183-1190` is unchanged, and no
  `CatBoostBuilder` setter is added.

**Red**
- Extend the existing `crates/catboost-rs-py/src/params_test.rs` sibling with:
  - `task_type_cpu_is_accepted` — `validate_params` returns `Ok(())`.
  - `task_type_gpu_matches_compiled_backend` — `cfg`-split: under a device feature expect
    `Ok(())`; under `cpu` expect `Err` whose message contains both `task_type` and a
    Cargo-feature name (`cuda`/`rocm`/`wgpu`).
  - `task_type_unknown_value_is_rejected_with_legal_values` — `"TPU"` errors and the
    message lists `CPU` and `GPU` (NOT a Levenshtein "unknown parameter" suggestion — the
    parameter NAME is known, only the VALUE is wrong).
  - `task_type_none_is_inert` — Python `None` is accepted (upstream's universal "not set",
    the existing `get()` convention).
  - `task_type_is_no_longer_known_not_yet` — `status_of("task_type") ==
    Some(ParamStatus::Implemented)`.
- Add a Python-level test in the existing pytest suite:
  `test_task_type_cpu_does_not_change_predictions` — fit twice, with and without
  `task_type="CPU"`, on a fixed seed, and assert the predictions are **bit-identical**.
  This is the assertion that proves D1's "pure input validation" claim rather than
  asserting it in prose.
- Expected failure: `validate_params` currently rejects `task_type` outright as a parity
  gap, so every case fails.
- Run: `cargo test -p catboost-rs-py -- task_type` then the pytest.

**Green**
- Implement the registry move + the validator. Nothing else.
- Error message text (pin it in the test, it is the user-facing contract):
  > `task_type="GPU" requires a wheel built with a device backend feature (--features
  > cuda, rocm, or wgpu); this wheel was built with --features cpu. Backend selection in
  > catboost-rs is compile-time only (see CLAUDE.md), so task_type cannot switch backends
  > at runtime.`

**Refactor**
- Extend the module-level "Honesty policy" doc comment (`params.rs:3-18`) with a fourth
  category — **VALIDATED-INFORMATIONAL**: a real upstream parameter this project honours
  by *validating consistency* rather than by changing behaviour, accepted only when
  consistent with the build. Explain why that is honest (it is observable: an inconsistent
  request errors) and cite D1.

**Verify**
- `cargo test -p catboost-rs-py`
- `cargo test --workspace`
- pytest suite green.
- **Both feature arms**: `cargo test -p catboost-rs-py` (default `cpu`) AND
  `cargo check -p catboost-rs-py --no-default-features --features cuda` — the `cfg`-split
  test must compile and pass under both.

**Completion evidence**: five Rust cases + the bit-identical-predictions pytest green
under both feature arms.

**Risks**: `get_all_params()` / `get_params()` may enumerate `IMPLEMENTED` and a pinned
params test may now see a new entry. **Before Green**, grep those surfaces; if they do
enumerate it, decide explicitly whether `task_type` should be echoed (upstream does echo
it) and update the pinned test with a recorded rationale rather than silently.

---

### T17 — Extend the workload generator with categoricals and non-uniform weights

- **Specs**: SPD-01 · **Wave 5 (D7 Wave 3)** · **Depends on**: none · **Parallel**: YES

**Goal / observable completion**: `bench/generator.py` gains `generate_categorical` and
`generate_weights`; the existing `generate` / `binary_target` outputs are byte-unchanged.

**Files**
- Modify `bench/generator.py` — ADD two functions per SPEC §4. Do not alter `generate`
  (`:106`), `binary_target` (`:126`), `CORRECTNESS_CONFIG` (`:56`) or `SPEED_CONFIG`
  (`:61`): the module is the D-06 single source for the committed small-n correctness
  fixtures, and drifting it would invalidate them.

**Red**
- Extend the module's existing `__main__` self-check (it already computes hashes — see the
  `hashlib` import at `:39`) with:
  - `assert sha256(generate(**CORRECTNESS_CONFIG)) == "<pinned>"` — capture the CURRENT
    hash first, pin it, THEN add the new functions. This is the regression that makes the
    D-06 rule enforceable rather than aspirational.
  - `generate_categorical` determinism: same seed twice ⇒ byte-identical.
  - `generate_weights(n, kind="cycled")` cycles `{0.5, 1.0, 2.0, 3.0}`, matching
    `crates/cb-oracle/fixtures/weighted_device_sym/gen_fixtures.py`'s recipe.
- Expected failure: the two functions do not exist.
- Run: `python bench/generator.py --self-check` (add the flag if absent).

**Green**
- Implement both with `numpy.random.RandomState` (the legacy Mersenne-Twister the module
  already mandates for cross-numpy-version stability — see its docstring at `:19-22`).
  numpy only; no new dependency.

**Refactor**
- Document in the module docstring that the D-06 single-source rule now spans three axes
  (float features, categoricals, weights) and that the weight recipe is deliberately
  shared with the committed device fixtures.

**Verify**
- `python bench/generator.py --self-check`
- `cargo test -p cb-oracle` (nothing should change; the committed fixtures are frozen
  artifacts, not regenerated).

---

### T18 — Grid benchmark harness (`bench/full_param_gpu_speed/bench.py`)

- **Specs**: SPD-02 · **Wave 5** · **Depends on**: T17 · **Parallel**: NO

**Goal / observable completion**: a `--dry-run` enumerates the full D3 grid with per-cell
recipes and an eligibility audit, on a machine with no GPU; a real run executes the grid,
3 repeats per cell, inside one Kaggle session, with a running elapsed budget.

**Files**
- Create `bench/full_param_gpu_speed/{bench.py,kernel-metadata.json}`.
- **Do NOT edit `bench/quick_gpu_speed/bench.py`** — it is the frozen record backing the
  r4a/r4b/r4c reports.
- Copy these patterns verbatim from `bench/quick_gpu_speed/bench.py`:
  - `timed_fit` (`:315-336`) — UNTIMED warm fit → TIMED fit → `predict()` on a slice to
    drain the lazy CubeCL queue before stopping the clock;
  - `build_eligibility_audit()` (`:38`) — the static precondition checklist;
  - the honesty caveat that device activation is not observable from Python;
  - the single `maturin build --release --no-default-features --features cuda` build
    (`:220-225`), run ONCE for the whole grid;
  - the `result["timings"]["errors"]` failure-recording discipline (never silently drop a
    failed arm).

**Grid (D3, pinned)**
```
grow_policy   : SymmetricTree, Depthwise
loss          : RMSE, Logloss
weights       : uniform, cycled {0.5,1,2,3}            (T17's generate_weights)
ctr           : none, single-permutation Borders       (float+cat, one_hot_max_size=1)
shape         : (300_000 x 50), (1_000_000 x 50)       (both >= the D-10-09 crossover
                                                        n=100_000 recorded in bench/RESULTS.md)
depth = 6, iterations = 30, border_count = 32, lr = 0.1, l2 = 3.0, seed = 42
repeats       : 3 per cell
=> 2 x 2 x 2 x 2 x 2 = 32 cells; prune the CTR x Logloss x 1M cells if the dry-run's
   budget estimate exceeds the session, down to the D3 16-24 target, and RECORD which
   cells were pruned and why.
```
Additionally include **2 "new-reachability" showcase cells** exercising the Wave-1 wins
that were previously CPU-only: `boost_from_average=True` RMSE, and
`grow_policy=Depthwise, bootstrap_type=Bernoulli`. These are the cells that make this
benchmark measure the phase's own work.

**Red**
- fn/mode `--dry-run`: enumerate every cell, print its full parameter recipe, run
  `build_eligibility_audit()` per cell, print a projected wall-clock budget, and exit 0
  WITHOUT importing torch/catboost/GPU anything.
- Assertion (a local, GPU-free check): every cell the audit marks device-eligible has
  `n >= 100_000`; every "new-reachability" cell is marked ELIGIBLE (if it is not, the
  Wave-1 work is not actually reachable and the benchmark would silently measure CPU).
- Expected failure: the harness does not exist.
- Run: `python bench/full_param_gpu_speed/bench.py --dry-run`

**Green**
- Implement the grid loop, the 3-repeat median/min/max aggregation, and the elapsed-budget
  guard that skips remaining cells with an explicit
  `"BUDGET EXCEEDED, N cells not run"` record rather than being killed mid-write.
- Extend `build_eligibility_audit()` with the new Wave-1 axes (bias, exact leaf,
  combination CTR, sampled non-symmetric).

**Refactor**
- Factor the per-cell model constructors into a table so adding an axis is a data change,
  not a code change.

**Verify**
- `python bench/full_param_gpu_speed/bench.py --dry-run` — grid reviewable before any
  Kaggle session is spent.
- Confirm every cell's official-CatBoost arm uses a recipe official CatBoost GPU actually
  supports; any unsupported cell reports `N/A` and is **never proxied** (the
  `bench/RESULTS.md` Region precedent).

**Risks**: official CatBoost may reject `grow_policy="Depthwise"` under `task_type="GPU"`
for some parameter combinations. The dry-run must surface this; record `N/A`, never
substitute a different recipe and call it a comparison.

---

### T19 — Execute the grid on Kaggle P100 and publish the report

- **Specs**: SPD-03 · **Wave 5** · **Depends on**: T18 **and every Wave-3 task merged**
- **Parallel**: NO

**Goal / observable completion**: one Kaggle P100 session produces
`bench/full_param_gpu_speed/kaggle-output-<date>/{report.md,result.json}` plus a dated run
block appended to `bench/RESULTS.md`.

**Steps**
1. Confirm the dry-run grid is final and reviewed.
2. Push ONE kernel (D3: minimise session churn; account `yensen2` already hit the batch
   session cap once this session).
3. On completion, write `report.md` reporting per cell:
   `median(official)/median(catboost_rs)` with min/max spread, plus the train-set quality
   column (the existing harness's comparability check across tree shapes).
4. **Any cell whose spread crosses `1.0` is labelled "within noise"** — never claimed as a
   win. This is the direct remedy for research.md pitfall #6.
5. Append the run block to `bench/RESULTS.md` under its existing run-log template,
   including the D-10-09 crossover note and the "device activation is not observable from
   Python" caveat.

**Verify / completion evidence**: the report exists, every cell has 3 timings, and the
headline claim is qualified by the exact axes measured — never generalised beyond the
grid.

**Risks**: Kaggle GPU session availability (scheduling, not technical). If the session is
cut short, the budget guard's partial `result.json` is still publishable as a partial run
— publish it as partial, do not re-run blindly.

---

### T20 — Cross-gap composition regression tests

- **Specs**: FPP-17 · **Wave 6** · **Depends on**: T07, T09, T10, T11 · **Parallel**: NO

**Goal / observable completion**: six composition assertions prove no Wave-1 relaxation
widened an unrelated exclusion.

**Files** — create `crates/cb-train/tests/device_fpp_composition_test.rs`. Leave the prior
phase's `device_gate_composition_test.rs` untouched.

**Red** — one fn per item, each asserting the OBSERVABLE (`device_active`, via a
`CountingGpu`-style runtime) rather than a specific clause, because a decline may
legitimately fire at either the host gate or the backend gate (the prior phase's V-11
lesson):
1. `bias_x_weighted_x_ctr_admits_together` — POSITIVE. All three are independently
   correct; nothing about their union is unimplemented.
2. `exact_leaf_x_ctr_still_declines`.
3. `exact_leaf_x_sampling_still_declines`.
4. `combination_ctr_x_one_hot_still_declines` (SPEC-OH-26, untouched).
5. `multi_permutation_x_combination_ctr_still_declines` (the
   `learning_folds_for_cycle == 1` guard, untouched).
6. `random_strength_still_declines` — an exclusion unrelated to this phase
   (`boosting.rs:4271`), proving nothing widened it.

Expected failure: the file does not exist. Run:
`cargo test -p cb-train --no-default-features --features rocm --test device_fpp_composition_test`

**Green** — none expected. **This task must require NO new production code.** Any failing
assertion is a bug in T07/T09/T10/T11 and the fix belongs there.

**Verify** — the new file green, plus `device_gate_composition_test` green unmodified.

---

### T21 — SPIKE: per-segment ordered split-score device kernel design note

- **Specs**: FPP-18 · **Wave 7** · **Depends on**: none (read-only)
- **Status**: **SPIKE — no Red/Green/Refactor. Deliverable is a design note.**

**Goal**: `.planning/plans/gpu-full-parameter-parity/spikes/ordered-split-score.md`
answering, with file:line citations, the five questions in SPEC FPP-18.

**Verified starting points (do not re-derive)**
- `crates/cb-train/src/tree.rs:2383` `score_candidate_ordered`, `:2427`
  `select_level_ordered`, `:2498` `greedy_tensor_search_oblivious_ordered` — all three
  confirmed at these exact lines this session.
- `crates/cb-train/src/fold.rs:105` `body_tail_boundaries(n, multiplier) -> Vec<usize>`,
  `:137` `body_tail_segments(n, multiplier) -> Vec<(usize, usize)>`, `:368`
  `body_sum_weights(n, multiplier, weights) -> Vec<f64>` — all PURE functions of
  `(n, multiplier)`, hence **per-FIT constant** (the prior phase's B-2 finding, still
  true). This matters: they may not need to cross the per-tree seam at all.
- `crates/cb-backend/src/gpu_runtime/session.rs:1073-1076` — the unconditional Ordered
  decline (`let _ordered = map_ordered_coverage(...); return Ok(None);`), and
  `map_ordered_coverage` at `:585` which already requires `config.ctr.is_none()`.
- Host-driven grower precedent (host owns the level loop, device scores candidates):
  `crates/cb-backend/src/kernels/nonsym_grow.rs`, `region_device.rs`, dispatched from
  `session.rs:1843` / `:1881`.
- Resident oblivious substrate: `crates/cb-backend/src/gpu_runtime/mod.rs`
  `grow_oblivious_tree_resident`.
- CubeCL manual (MANDATORY before any kernel sketch):
  `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md`.

**Acceptance bar**: the note names a concrete kernel (or a concrete reuse), a launch shape,
a seam decision (`FamilyTreeArgs` variant vs `DeviceTrainConfig` field vs nothing), and a
size estimate in net-new `#[cube]` lines plus host-driver lines.

**ESCALATION GATE (D2, mandatory)**: if the note concludes the work exceeds
**~3 days-equivalent of net-new kernel work**, **STOP, publish the note, and report to the
user before T22 is written or executed.** The prior phase's own yardstick for this track
was "comparable in size to the ENTIRE CTR-averaging-permutation deliverable".

**Out of scope**: any production code.

---

### T22 — Ordered per-segment split scoring on the device grow path

- **Specs**: FPP-19 · **Wave 7** · **Depends on**: T21 (+ its gate cleared), T04
- **Status**: goal + acceptance bar only — T21's note refines this into executable steps.

**Goal**: for a fixed input, the device Ordered arm chooses the SAME per-level
`(feature, border)` as `greedy_tensor_search_oblivious_ordered`.

**Acceptance bar**: integer equality on the chosen `(feature, border)` at every level
against a transcribed CPU reference on a frozen small fixture (`n = 30`,
`fold_len_multiplier = 2.0` ⇒ segments `[(1,2),(2,4),(4,8),(8,16),(16,30)]` per the worked
example documented at `fold.rs:135`), plus ε=1e-4 on the summed per-segment score.

**Known constraint to honour**: `score_candidate_ordered` hard-codes `l2_split_score` — the
ordered path is **L2-only**. Decline any other score function to CPU rather than
approximating it.

**Do Not Hand-Roll**: `cb_compute::l2_split_score`, `cb_compute::scale_l2_reg`,
`cb_core::sum_f64`.

---

### T23 — Consume `map_ordered_coverage` and relax the Ordered clause

- **Specs**: FPP-20 · **Wave 7** · **Depends on**: T22 · **Status**: goal + acceptance bar

**Goal**: `session.rs:1073-1076` stops discarding the coverage result;
`boosting.rs:4215` admits a covered Ordered fit, guarded so Ordered × CTR never composes.

**Acceptance bar**: a covered Ordered fit reaches the device; an uncovered one declines;
Ordered × CTR declines.

**Mandatory, easy-to-miss**: `crates/cb-backend/src/gpu_runtime/ordered_test.rs`'s
`begin_declines_ordered_to_cpu` is an INTENTIONAL update site. It must be **flipped**
(covered config now succeeds) with its negative case **preserved** (uncovered config still
declines) — never weakened to make it pass.

**Defence in depth (verified)**: `map_ordered_coverage` (`session.rs:585+`) already
requires `config.ctr.is_none()`, so Ordered × CTR declines backend-side even without a new
host guard. The host guard is still required (fail-fast at the single D-10-01 gate), but
the negative TEST must assert the observable (`device_active == false`), not the clause.

---

### T24 — Ordered device fixture + e2e oracle

- **Specs**: FPP-21 · **Wave 7** · **Depends on**: T23, T09 · **Status**: goal + bar

**Goal**: an Ordered device fit matches upstream at ≤1e-5 over all iterations.

**Re-evaluate before generating (V-8)**: `crates/cb-oracle/fixtures/ordered_boost_e2e/`
was ruled device-ineligible by the prior phase's V-8 **solely** because it pins
`"boost_from_average": true` (confirmed this session). **T09 removes that clause.** So
check first whether `ordered_boost_e2e/` becomes reusable as-is (it also needs float
columns and a compatible `border_count` — verify both) before spending a generation cycle
on a new fixture.

**Acceptance bar**: `device_active == true` and predictions ≤1e-5 vs the fixture's
`predictions.npy` over all iterations, in a new
`crates/cb-train/tests/device_ordered_fit_test.rs` (a SIBLING of the existing CPU
`ordered_boost_e2e_oracle_test.rs`, which imports `CpuBackend` and must be left untouched).

---

### T25 — SPIKE: non-Borders CTR accumulation semantics on device

- **Specs**: FPP-22 · **Wave 7** · **Depends on**: none · **Status**: SPIKE

**Goal**: `.../spikes/ctr-non-borders.md` stating, per type (Buckets /
BinarizedTargetMeanValue / Counter), whether it is (a) a parameter change to the existing
kernel, (b) a sibling kernel sharing the resident scratch shape, or (c) a genuinely new
resident design — with a size estimate for each.

**Verified starting points**
- The authoritative gap statement: `ctr_types_are_device_covered`'s doc comment,
  `crates/cb-train/src/boosting.rs:2230-2240`.
- CPU per-type dispatch: `crates/cb-train/src/ctr/ctr_feature.rs:266-330` (producing
  `(nums, denoms, ctr_value, quantize_in_f32)`); `crates/cb-train/src/ctr/online.rs`
  (`TCtrHistory` class counts, `TCtrMeanHistory` float sums, Counter totals — see the
  per-bucket field docs at `:125-139`).
- The device primitive: `crates/cb-backend/src/kernels/ctr_device.rs:134-167`
  (`ordered_ctr_prefix_kernel`, resident `counts: &mut Array<u32>`, read-before-increment).
- **The single highest-risk detail**: `TCtrMeanHistory::Sum` is **`f32`** upstream
  (`online_ctr.h:373`, cited in `crates/cb-train/src/ctr/bake.rs:80-82`), and
  `ctr_feature.rs` carries a `quantize_in_f32` flag. A BTMV/Counter device kernel
  accumulating in f64 will miss ≤1e-5. The note MUST address this explicitly.
- **The second gap inside Buckets**: `boosting.rs:2246` pins `target_border_idx == 0`, but
  Buckets emits 2 columns at binclf (`boosting.rs:2131-2134`). Buckets support therefore
  needs the multi-target-border column dimension too, not just a new accumulator.

**ESCALATION GATE**: if any of the three lands in category (c) at more than
~2 days-equivalent, STOP and report before T26 is written.

**Note (D4)**: carry NO shared-infrastructure assumption between this track and T27/T29 —
research.md's own LOW-confidence finding.

---

### T26 — Implement the non-Borders CTR device accumulators

- **Specs**: FPP-23 · **Wave 7** · **Depends on**: T25 (+ gate cleared)
- **Status**: goal + acceptance bar

**Goal**: `ctr_types_are_device_covered` (`boosting.rs:2245`) admits Buckets /
BinarizedTargetMeanValue / Counter (under `counter_calc_method=SkipTest`), and the device
produces bit-identical binarized cindex columns to the CPU for each.

**Acceptance bar**: a per-type device-vs-CPU **bit-exactness** self-oracle (extending the
existing `ctr_binarized_cindex_column_bit_exact` precedent), plus a per-type e2e ≤1e-5
oracle against a NEW mixed float+cat fixture — `ctr_buckets_simple/`, `ctr_btmv_simple/`
and `ctr_counter_simple/` are all cat-only and device-unreachable (V-9's reasoning,
`boosting.rs:4350`).

---

### T27 — SPIKE: monotone-constrained split scoring on device

- **Specs**: FPP-24 · **Wave 7** · **Depends on**: none · **Status**: SPIKE

**Goal**: `.../spikes/monotone-constraints.md` establishing whether the constraint is a
candidate **FILTER** (cheap: a host-side mask over the candidate set, no kernel change) or
a per-leaf **VALUE clamp** (expensive: touches the leaf reduce), with a size estimate.

**Verified starting points**
- `params.monotone_constraints: Vec<i8>` — `crates/catboost-rs/src/builder.rs:177-182`,
  default `monotone_constraints_default()` at `:329`; IMPLEMENTED in the Python registry
  (`crates/catboost-rs-py/src/params.rs:112`, extractor at `:852`).
- The eligibility clause: `crates/cb-train/src/boosting.rs:4233`.
- The CPU consumption sites: `grep -n monotone crates/cb-train/src/` (pin them in the
  note; this session did not enumerate them).
- **No device precedent whatsoever**: research.md's exhaustive grep for "monotone" across
  `crates/cb-backend/src/` returned only unrelated English usages.

**ESCALATION GATE**: if it is not a candidate filter, STOP and report before T28.

---

### T28 — Implement monotone constraints on the device grow

- **Specs**: FPP-25 · **Wave 7** · **Depends on**: T27 (+ gate) · **Status**: goal + bar

**Goal**: `boosting.rs:4233` admits a monotone-constrained fit; the device chooses the same
splits and produces the same leaf values as the CPU reference.

**Acceptance bar**: device-vs-CPU ε=1e-4, plus an e2e ≤1e-5 oracle against a NEW upstream
fixture (none exists — verified).

---

### T29 — SPIKE: feature-weight penalties on device

- **Specs**: FPP-26 · **Wave 7** · **Depends on**: none · **Status**: SPIKE

**Goal**: `.../spikes/feature-penalties.md` stating, per knob, whether it is a
per-candidate scalar the device scorer can apply from an uploaded per-feature vector
(cheap) or requires per-object device state.

**Verified starting points**
- All four knobs are IMPLEMENTED CPU-side and Python-reachable
  (`crates/catboost-rs-py/src/params.rs:112-116`).
- Builder fields: `feature_weights: Vec<f64>` (MULTIPLICATIVE gain weight,
  `builder.rs:164-166`), `first_feature_use_penalties` / `per_object_feature_penalties`
  (ADDITIVE, `:168-172`), `penalties_coefficient: f64` (`:174-176`). **The multiplicative
  and additive knobs may not share a device shape** — the note must not assume they do.
- `penalties_active` binding: `crates/cb-train/src/boosting.rs:3780-3782`
  (`!feature_weights.is_empty() || !first_feature_use_penalties.is_empty() ||
  !per_object_feature_penalties.is_empty()`).
- CPU consumption sites: `boosting.rs:3211`, `:3254`, `:4595`, `:5853`, `:5918`.
- Eligibility clause: `boosting.rs:4232`.

**ESCALATION GATE**: if `per_object_feature_penalties` requires per-object device state,
STOP and report — and propose splitting the track so the three cheap knobs land without
it, rather than blocking all four.

---

### T30 — Implement feature-weight penalties on the device grow

- **Specs**: FPP-27 · **Wave 7** · **Depends on**: T29 (+ gate) · **Status**: goal + bar

**Goal**: `boosting.rs:4232` admits a penalised fit for whichever subset T29 found
tractable; the device split choice matches the CPU reference.

**Acceptance bar**: integer-equal split choices vs CPU on a frozen fixture, plus an e2e
≤1e-5 oracle against a NEW upstream fixture.

---

### T31 — SPIKE: eval sets + overfitting detector on the device branch

- **Specs**: FPP-28 · **Wave 7** · **Depends on**: none · **Status**: SPIKE

**Goal**: `.../spikes/eval-sets-on-device.md` confirming or refuting tractability.

**Start from V-9's findings, do not re-derive them**:
- `last_tree_eval_contribution` (`boosting.rs:2511-2527`) reads the **folded tree
  ensembles** (`&[ObliviousTree]`, `&[NonSymmetricTree]`, `&[RegionTree]`), never an
  approx buffer and never device state.
- The device branch already folds its grown tree into exactly those three ensembles
  (`boosting.rs:4787-4800`).
- The eval block lives at `boosting.rs:6690-6733`, inside the CPU body the device branch
  skips via `continue;` at `:5275`.
⇒ **Hypothesis**: the change is "run the same `if has_test { … }` block before the device
branch's `continue`", ~50 lines in `boosting.rs`, no kernel work.

**The spike must answer three open questions**:
1. `use_best_model` truncation and the detector's `break` — do they interact correctly
   with `DeviceSessionGuard`'s teardown (`boosting.rs:4489-4492`)?
2. Snapshot/resume interaction on the device branch (the snapshot machinery at
   `boosting.rs:4640-4673` runs before the loop; the device branch writes no per-iteration
   checkpoint today).
3. A CTR device fit's eval matrix has no CTR columns — does `tree_eval_contribution`
   handle a `LevelKind::Ctr` tree at all, or must `eval_sets × CTR` stay excluded?

**ESCALATION GATE**: if (3) forces a device-side eval CTR materialization, STOP and report
— that is a new kernel track, not a wiring change.

---

### T32 — Implement eval sets on the device branch

- **Specs**: FPP-29 · **Wave 7** · **Depends on**: T31 (+ gate) · **Status**: goal + bar

**Goal**: `boosting.rs:4272` admits a fit with eval sets; the device branch updates
`eval_approx`, computes the eval metric, and drives the overfitting detector /
`use_best_model` identically to the CPU branch.

**Acceptance bar**: a device fit with an eval set and `od_type=Iter` stops at the SAME
iteration as the CPU fit from the same seed, and both produce predictions at ε=1e-4.

---

### T33 / T34 / T35 — SPIKES: pairwise / ranking / multi-output grow consumption

- **Specs**: FPP-30 / FPP-31 / FPP-32 · **Wave 7** · **Depends on**: none · **Status**: SPIKE

Each produces `.../spikes/phase13-<family>.md`. Common verified starting points:
- The five decline branches, read verbatim this session:
  `session.rs:1073-1076` (ordered), `:1091-1107` (pairwise), `:1127-1134` (ranking),
  `:1149-1157` (multi-output), `:1168` (langevin, records but does not gate).
- The coverage maps: `map_pairwise_coverage` `:373-409`, `map_ranking_coverage` `:460-504`,
  `map_multiclass_coverage` `:532-556`.
- `FamilyTreeArgs` (T04) is the seam these consume.

**T33 (pairwise)** — the decline comment at `session.rs:1087-1088` names the missing work
verbatim: *"the batched Cholesky SOLVE + the per-tree pair/group descriptor seam … land in
Plan 02 (GPUT-21)"*. Landed: device histogram reuse + `launch_pairwise_assemble_system_into`.
Also relevant: `cb_compute::is_pairwise_scoring` (`crates/cb-compute/src/ranking_der.rs:46-51`)
and the CPU per-tree adjacency re-sample at `boosting.rs:5278-5281`.
**ESCALATION GATE: report before implementation regardless of size** — this is the one
family whose missing half is an explicitly named, wholly unexecuted plan.

**T34 (ranking)** — landed: the der driver over the Plan-03 query-grouping infra,
self-oracled against `cb_compute::calc_ders_for_queries`. The note must state whether
`FamilyTreeArgs::Ranking { group_offsets }` suffices or whether the YetiRank per-tree
adjacency re-sample forces more across the seam. Note the third gap layer:
`QueryCrossEntropy` has no `Loss` variant at all. Also note `boosting.rs:4214`'s
`group_spans.is_none()` is the FIRST clause — relaxing it is a Wave-2-style serial edit.

**T35 (multi-output)** — landed: `grow_multiclass_block` (K-dim Newton block leaf, coupled
softmax vs diagonal separable). Helpfully, `DeviceGrownTree.leaf_values` **already**
carries a `leaf_count × approx_dim` ROW-MAJOR block contract (verified in its doc comment
at `runtime.rs:~950`), so the RETURN type needs no change — only `FamilyTreeArgs::MultiOutput`
inbound. Three eligibility clauses at `boosting.rs:4244-4246` must relax together. Note the
fourth gap layer: `MultiQuantile`'s exact-quantile leaf is not reached by
`map_multiclass_objective`. **Also**: T09's `RmseWithUncertainty` risk note — the scalar
`DeviceTrainConfig.bias` cannot express that loss's per-dimension starting approx
(`boosting.rs:1285`, `:3649-3660`); this spike must decide how, before `:4244` relaxes.

---

### T36 / T37 / T38 — Implement pairwise / ranking / multi-output grow consumption

- **Specs**: FPP-33 / FPP-34 / FPP-35 · **Wave 7** · **Depends on**: T33/T34/T35
  respectively (+ gates), and T04 · **Status**: goal + acceptance bar

**Goal** (each): the family's fits reach the device and match upstream at ≤1e-5 on a NEW
per-family fixture — none of the existing fixtures covers any Phase-13 family end-to-end
on device.

**Acceptance bar** (each): `device_active == true`; predictions ≤1e-5 vs upstream; **every
OTHER family still declines** (D-10-01 all-or-nothing per family — a relaxation for one
family must not leak into another's decline branch).

**Serialisation**: all three edit `session.rs::begin`'s decline chain and
`boosting.rs::device_host_eligible`. Serialise them; do not run two in parallel.

---

## 5. Coverage

### 5.1 Spec → task (all 38 IDs covered)

| Spec | Task(s) | Track | Wave |
|---|---|---|---|
| FPP-01 | T05 | A (bias) | 1 |
| FPP-02 | T09 | A | 2 |
| FPP-03 | T01 | A | 0 |
| FPP-04 | T12 | A | 3 |
| FPP-05 | T06 | E (exact leaf) | 1 |
| FPP-06 | T10 | E | 2 |
| FPP-07 | T02 | E | 0 |
| FPP-08 | T13 | E | 3 |
| FPP-09 | T07 | K (combination CTR) | 1 |
| FPP-10 | T03 | K | 0 |
| FPP-11 | T14 | K | 3 |
| FPP-12 | T08 | P (nonsym sampling) | 1 |
| FPP-13 | T11 | P | 2 |
| FPP-14 | T15 | P | 3 |
| FPP-15 | T04 | S (seam) | 0 |
| FPP-16 | T16 | T (task_type) | 4 |
| FPP-17 | T20 | G (guards) | 6 |
| FPP-18 | T21 (SPIKE) | O (ordered) | 7 |
| FPP-19 | T22 | O | 7 |
| FPP-20 | T23 | O | 7 |
| FPP-21 | T24 | O | 7 |
| FPP-22 | T25 (SPIKE) | U (non-Borders CTR) | 7 |
| FPP-23 | T26 | U | 7 |
| FPP-24 | T27 (SPIKE) | M (monotone) | 7 |
| FPP-25 | T28 | M | 7 |
| FPP-26 | T29 (SPIKE) | N (penalties) | 7 |
| FPP-27 | T30 | N | 7 |
| FPP-28 | T31 (SPIKE) | V (eval sets) | 7 |
| FPP-29 | T32 | V | 7 |
| FPP-30 | T33 (SPIKE) | W (pairwise) | 7 |
| FPP-31 | T34 (SPIKE) | R (ranking) | 7 |
| FPP-32 | T35 (SPIKE) | X (multi-output) | 7 |
| FPP-33 | T36 | W | 7 |
| FPP-34 | T37 | R | 7 |
| FPP-35 | T38 | X | 7 |
| SPD-01 | T17 | SPD | 5 |
| SPD-02 | T18 | SPD | 5 |
| SPD-03 | T19 | SPD | 5 |

### 5.2 Acceptance scenario (SPEC §6) → task

| # | Scenario | Bar | Closed by |
|---|---|---|---|
| 1 | Default `boost_from_average=true` RMSE reaches device, matches upstream | ≤1e-5 | T12 (fixture T01, impl T05+T09) |
| 2 | `bias == 0.0` fits byte-unchanged | regression | T05 Verify + T09 Verify (11 existing device tests, unmodified) |
| 3 | Exact MAE + Quantile reach device, match upstream | ≤1e-5 | T13 (fixtures T02, impl T06+T10) |
| 4 | Exact × LogCosh still declines | correctness | T10 `exact_leaf_logcosh_fit_declines_to_cpu` |
| 5 | Combination CTR reaches device, leaf values match upstream | ≤1e-5 | T14 (fixture T03, impl T07) |
| 6 | Simple-projection CTR byte-unchanged | regression | T07 Verify (`device_ctr_fit_test` unmodified) |
| 7 | Sampled Depthwise/Lossguide/Region match CPU | ε=1e-4 | T15 (impl T08+T11) |
| 8 | Region × Poisson still errors | correctness | T11 `region_poisson_still_errors` |
| 9 | Seam signature change breaks nothing | regression | T04 Verify (`cargo test --workspace`) |
| 10 | `task_type="CPU"` accepted, predictions bit-identical | correctness | T16 pytest |
| 11 | `task_type="GPU"` errors actionably on a cpu wheel | correctness | T16 `task_type_gpu_matches_compiled_backend` |
| 12 | Cross-gap composition (6 items) | correctness | T20 |
| 13 | Ordered device fit matches upstream | ≤1e-5 | T24 (gated on T21) |
| 14 | Buckets/BTMV/Counter bit-exact + e2e | bit-exact / ≤1e-5 | T26 (gated on T25) |
| 15 | Monotone device fit matches upstream | ≤1e-5 | T28 (gated on T27) |
| 16 | Penalised device fit same splits as CPU | integer-equal | T30 (gated on T29) |
| 17 | Device fit with eval sets stops at same iteration | correctness | T32 (gated on T31) |
| 18 | Pairwise / ranking / multi-output reach device | ≤1e-5 | T36 / T37 / T38 (gated on T33/T34/T35) |
| 19 | Generator extension leaves committed bytes unchanged | regression | T17 pinned-hash self-check |
| 20 | 16–24-cell grid in ONE Kaggle session, 3 repeats | methodology | T18 `--dry-run` + T19 |
| 21 | Published claim qualified by axes, with variance | methodology | T19 step 4 |

### 5.3 New test / fixture inventory

| artifact | task | kind |
|---|---|---|
| `crates/cb-oracle/fixtures/bias_device_sym/` | T01 | fixture |
| `crates/cb-oracle/fixtures/exact_leaf_device/{mae,quantile07}/` | T02 | fixture |
| `crates/cb-oracle/fixtures/ctr_device_combo/` | T03 | fixture |
| `crates/cb-oracle/tests/bias_device_fixture_smoke_test.rs` | T01 | smoke |
| `crates/cb-oracle/tests/exact_leaf_device_fixture_smoke_test.rs` | T02 | smoke |
| `crates/cb-oracle/tests/ctr_device_combo_fixture_smoke_test.rs` | T03 | smoke |
| `crates/cb-compute/src/runtime_family_args_test.rs` | T04 | unit (sibling) |
| `crates/cb-train/tests/device_exact_leaf_config_test.rs` | T06 | unit (CPU) |
| `crates/cb-train/tests/device_ctr_combo_config_test.rs` | T07 | unit (CPU) |
| `crates/cb-train/tests/device_bias_gate_test.rs` | T09 | gate (rocm) |
| `crates/cb-train/tests/device_exact_leaf_gate_test.rs` | T10 | gate (rocm) |
| `crates/cb-train/tests/device_nonsym_bootstrap_gate_test.rs` | T11 | gate (rocm) |
| `crates/cb-train/tests/device_bias_fit_test.rs` | T12 | e2e (rocm) |
| `crates/cb-train/tests/device_exact_leaf_fit_test.rs` | T13 | e2e (rocm) |
| `crates/cb-train/tests/device_ctr_combo_fit_test.rs` | T14 | e2e (rocm) |
| `crates/cb-train/tests/device_nonsym_bootstrap_test.rs` | T15 | e2e (rocm) |
| `crates/cb-train/tests/device_fpp_composition_test.rs` | T20 | composition (rocm) |
| extensions to `session_residency.rs` sibling tests | T05 | self-oracle |
| extensions to `nonsym_grow_test.rs`, `region_device_test.rs` | T08 | self-oracle |
| extensions to `params_test.rs` + pytest | T16 | unit + python |
| `bench/full_param_gpu_speed/bench.py` | T18 | harness |

---

## 6. Blockers, escalation gates, and assumptions

### CORRECTIONS to SPEC.md / research.md (no decision needed; apply as written)

| id | source says | verified reality | task that applies it |
|---|---|---|---|
| C-1 | research.md: combination CTR correctness is MEDIUM confidence, "not independently re-derived" | **HIGH** — V-4 derives integer-identical combined bins on both sides; the only residual is a hypothetical 64-bit hash collision asymmetry | T07 (full detail, no sub-spike) |
| C-2 | research.md: "the prior phase found 10 sites" for `grow_tree_on_device` | **15 sites across 11 files** — 4 new test files landed with the GDC merge (V-5) | T04 |
| C-3 | research.md: "MAE/Quantile/**MAPE**'s default IS Exact … reachable by ordinary use" | admissible set is **`{Mae, Quantile}`** only (Mape is CPU-REJECTED at `boosting.rs:1391`; LogCosh is CPU-legal but device-UNCOVERED); and `builder.rs:300` defaults `leaf_method: Gradient` unconditionally, so it is reachable only by an **explicit** request (V-6) | T06, T10, T13 |
| C-4 | research.md implies the quantile fixtures may be reusable | `quantile_alpha05_mae/` + `quantile_alpha07/` ship no `predictions.npy`/`X.npy`/`y.npy`, pin no `border_count`, and their only consumer imports `CpuBackend` (V-7) | T02 |
| — | SPEC §4 shows `bias` threaded via a new `begin_device_training` argument would also work | `runtime.rs:1353-1355` states the project's own convention: *"the config surface widens by mutating this struct, not the argument list"*; the config route keeps T05's blast radius at 2 files instead of ~12 (V-3) | T05 |
| — | research.md pitfall #5: "confirm all currently-passing device tests explicitly pin `boost_from_average=False` … but re-verify per-fixture rather than assuming" | **CONFIRMED per-fixture** (V-8): only 4 fixtures repo-wide pin it `true`, none of them a device e2e fixture | T09 |
| — | prior phase V-8: `ordered_boost_e2e/` is device-ineligible | true TODAY, but **its only disqualifier is the `bias == 0.0` clause T09 removes** — re-evaluate before generating a new Ordered fixture (V-8) | T24 |
| — | research.md: eval-sets-on-device is MEDIUM "probably tractable, not verified" | the update path is `last_tree_eval_contribution` over the **folded tree ensembles** the device branch already populates; the device branch merely `continue`s past the block at `:5275` (V-9) | T31 |

### ESCALATION GATES — each STOPS its track and reports rather than expanding scope

| gate | task | trigger | required action |
|---|---|---|---|
| G1 | T21 (Ordered spike) | net-new kernel work > ~3 days-equivalent | publish the note, report to the user, do NOT start T22 |
| G2 | T25 (non-Borders CTR spike) | any type is a genuinely new resident design > ~2 days-equivalent | report before T26 |
| G3 | T27 (monotone spike) | the constraint is NOT a candidate filter | report before T28 |
| G4 | T29 (penalties spike) | `per_object_feature_penalties` needs per-object device state | report; propose splitting the track so the 3 cheap knobs land alone |
| G5 | T31 (eval-sets spike) | question (3) forces device-side eval CTR materialization | report; that is a new kernel track |
| G6 | T33 (pairwise spike) | **unconditional** — the missing half is a wholly unexecuted named plan (GPUT-21) | report before T36 regardless of the estimate |

### BLOCKERS discovered during verification — resolve before the named task's Green

**B-1 (blocks T11, therefore T15).** `replay_grow_draws(&mut rng, params.depth,
matrix.n_features())` (`boosting.rs:5261-5266`) replays the RNG draws the device branch's
skipped level search would have consumed, so the NEXT tree's `bootstrap()` reads the same
RNG phase as upstream. It is parameterised by `depth` and `n_features` — i.e. written for
the **oblivious** level search's draw count. A Depthwise/Lossguide/Region level search
consumes a different number. **If it is not policy-aware, a sampled non-symmetric device
fit diverges from CPU at tree 2 for a reason no task in this plan owns.** Verify before
T11's Green; if it is not policy-aware, STOP and fix it in a dedicated task first.
(Precedent for this exact failure class: the MVS tree-2 gap, root-caused to fabricated RNG
draws and fixed in `d7676b5`.)

**B-2 (blocks T06 and T07's Red).** `device_config` and `build_device_ctr_config`'s output
are locals inside `train_inner` and are not observable from a test today. Each task states
its preferred resolution (extract a `#[must_use] pub(crate) fn` and unit-test that; or
widen `build_device_ctr_config`'s visibility, since it is already a free function at
`boosting.rs:2263`). **Decide at execution time and record the choice in the task's
completion note** — do not silently fall back to an integration-only assertion, which
would leave the config decision untested.

**B-3 (blocks T02's Green).** The admissible `border_count` for a 2-float device pool is
not independently verified. Every green device fixture uses 15 (16 bins) and
`ctr_covered` (`session.rs:151`) requires `borders.len() + 1 == n_bins` for CTR columns,
but the FLOAT-only histogram width bound was not re-derived this session. Read
`session.rs`'s `n_bins` validation and `pad_hist_line_bins` before generating; pin the
verified value and record it.

**B-4 (blocks T35, therefore T38).** `Loss::RmseWithUncertainty` starts from a
**per-dimension** approx (`rmse_uncertainty_starting_approx`, `boosting.rs:1285`;
`:3649-3660`), which the scalar `DeviceTrainConfig.bias` (T05) cannot express. Today
`approx_dimension == 1` (`:4244`) makes this unreachable. T35's spike must decide how
before `:4244` relaxes; do not relax it and discover this at oracle time.

### Assumptions NOT verifiable in this environment

- No ROCm/CUDA hardware was available for this planning pass, so no
  `cargo test --features rocm` was executed. Only `cargo check`-class verification is
  machine-confirmable here (research.md previously confirmed
  `cargo check -p cb-train --no-default-features --features rocm --test
  device_nonsym_fit_test` succeeds without a GPU, and that a `CpuBackend`-importing test
  fails `E0432` under the same flags).
- Upstream fixture generation (T01/T02/T03, and every Wave-7 fixture) requires a local
  `catboost==1.2.10` install; not attempted here (planning only).
- Whether the existing `session_residency` self-oracle already fails T05's new bias
  assertion is a prediction from reading `session.rs:1580`, not an executed result.
- T18/T19's Kaggle session budget is estimated from the r4a/b/c runs (a single 300k×50
  depth-6/30-iter shape took ~1.2–1.4 s per arm, 8 arms) scaled by the grid size and
  3 repeats. The `--dry-run` budget print is what turns that estimate into a checked
  number before a session is spent.

### Carried residuals (must appear in the completion summary)

1. The fixed-point histogram's `|Σ| < 2^33` precondition has **no enforced runtime
   guard**; this phase does not add one. All three Wave-0 fixtures stay under it by
   construction with the margin documented in their generators. Pre-existing, orthogonal,
   inherited from `.planning/plans/device-bootstrap-parity/SPEC.md` Finding F-D.
2. `[C-1]`'s `fold_cat_hash` collision asymmetry between the CPU's string-hash fold and
   the device's perfect-hash-bucket fold: documented, detected by T14's ≤1e-5 bar, not
   prevented.
3. **Langevin/SGLB remains entirely unimplemented** (D6). This is a CPU feature gap
   (no `BoostParams` field, no `train_inner` path, `KnownNotYet` in the Python registry) —
   state it plainly in the completion summary so "full parameter implementation" is not
   over-claimed.

---

## 7. Definition of done

### Phase-complete (Waves 0–6) — the "full parameter parity, verified" bar

All of T01–T20 green, and:

```
cargo test --workspace
cargo test -p cb-compute
cargo test -p catboost-rs-py
cargo test -p cb-backend --no-default-features --features rocm
cargo check -p catboost-rs-py --no-default-features --features cuda
cargo test -p cb-train --no-default-features --features rocm --test device_bias_gate_test
cargo test -p cb-train --no-default-features --features rocm --test device_exact_leaf_gate_test
cargo test -p cb-train --no-default-features --features rocm --test device_nonsym_bootstrap_gate_test
cargo test -p cb-train --no-default-features --features rocm --test device_bias_fit_test
cargo test -p cb-train --no-default-features --features rocm --test device_exact_leaf_fit_test
cargo test -p cb-train --no-default-features --features rocm --test device_ctr_combo_fit_test
cargo test -p cb-train --no-default-features --features rocm --test device_nonsym_bootstrap_test
cargo test -p cb-train --no-default-features --features rocm --test device_fpp_composition_test
# and, unmodified:
cargo test -p cb-train --no-default-features --features rocm --test device_nonsym_fit_test
cargo test -p cb-train --no-default-features --features rocm --test device_region_fit_test
cargo test -p cb-train --no-default-features --features rocm --test device_weighted_fit_test
cargo test -p cb-train --no-default-features --features rocm --test device_weighted_nonsym_fit_test
cargo test -p cb-train --no-default-features --features rocm --test device_ctr_fit_test
cargo test -p cb-train --no-default-features --features rocm --test device_ctr_gate_test
cargo test -p cb-train --no-default-features --features rocm --test device_weight_gate_test
cargo test -p cb-train --no-default-features --features rocm --test device_fold_count_gate_test
cargo test -p cb-train --no-default-features --features rocm --test device_gate_composition_test
cargo test -p cb-train --no-default-features --features rocm --test bootstrap_dev_oracle_test
cargo test -p cb-train --no-default-features --features rocm --test device_bootstrap_parity_test
cargo test -p cb-train --no-default-features --features rocm --test device_poisson_bootstrap_test
cargo test -p cb-train --no-default-features --features rocm --test device_one_hot_parity_test
cargo test -p cb-train --no-default-features --features rocm --test device_seam_test
```

all green on a real rocm device, plus B-1…B-4 resolved and recorded.

### Speed-complete (Wave 5) — the "beating official CatBoost on Kaggle P100" bar

T19's report exists and, for the grid it measured, reports
`median(official)/median(catboost_rs) > 1.0` with a min/max spread that does **not** cross
`1.0` on the majority of cells — with every straddling cell explicitly labelled "within
noise". A headline claim generalised beyond the measured axes is a failure of this bar, not
a success.

### Parity-complete (Wave 7) — the "full parameter implementation" bar

T21–T38 complete, OR each incomplete track has a published spike note and a recorded
escalation under §6's gate table. **An escalated, documented STOP is a valid completion
state for a Wave-7 track; a silent overrun is not.**
