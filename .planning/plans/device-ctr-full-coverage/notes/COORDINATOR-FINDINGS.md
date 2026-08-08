# Coordinator findings — carried forward between tasks

Findings raised by a completed task that a LATER task must act on. Each executor is
handed this file. Append; never rewrite another task's entry.

## From T03 → **T04**

`calc_normalization(0.0)` returns shift `-0.0`, not `0.0` (`shift = -min(0.0, p)`).
⇒ T04's assertions must compare with `==` (PartialEq), **never** `f64::to_bits` — a bit
comparison fails on *unmutated* code.

T03 also captured the deterministic `--test-threads=1` output of
`cargo test -p cb-train --test ctr_btmv_simple_oracle_test` in `notes/T03.md` as the
**pre-T04 byte-comparison baseline**. T04 must diff against it.

## From T07 → **T05** (already resolved) and **T10/T16**

`model.json`'s `ctrs` array is the *pruned/used* set, so **descriptor count alone does not
discriminate CTR type** — `Buckets:Prior=0.5` on the same data also yields a single
descriptor at `target_border_idx=0`. Assert on `ctr_type` explicitly.

## From T05 → **T10**

`ctr_device_buckets` landed at **escalation rung 0** (`CARDS=(6,)`, 64 rows, 5 iterations,
`data_seed=1`), not the predicted rung 3. `X_cat.npy` is `(64,) i32` (one cat column) —
the shape T10's consumer contract expects. `model.json` carries **2 Buckets descriptors
with `target_border_idx` `{0, 1}`**, which is exactly the two-columns-per-`(projection,
prior)` layout PLAN §6 assumption 8 flags as T10's most likely failure site.

## From T06 → **T12** (raised explicitly by T06; act on it)

On `ctr_device_counter`'s data the Counter prior is **prediction-neutral upstream**:
`Prior=0.5`, the default `0/1`, and `Prior=3` all give **bit-identical** predictions
(upstream compensates by shifting the CTR border — 8.999999 at prior 0.5/0, 3.999999 at
prior 3). ⇒ **the ≤1e-5 e2e cannot police a Rust-side prior mismatch on this fixture.**
The only guard is the explicit pin on both sides: `Counter:Prior=0.5` in the fixture
(asserted by the smoke test) and `simple_ctr_priors = vec![0.5]` in T12's `BoostParams`.
T12 must set that explicitly and say so in its note. See `notes/T06.md` §6.

## From T00 → **T10** (and any task editing the gate-state table)

T00 declined the optional rows 9/10 (`simple/Buckets/b=0`, `simple/Borders/b=1`) to avoid
renumbering rows later tasks are specified against. Cost, documented in-source: **row 4
(`simple/Buckets/b=1`) is the one row with no single-conjunct mutation proof** — it varies
both `ctr_type` and `target_border_idx` vs row 1. T10, which flips row 4, should state in
its note which of the two conjuncts its edit actually removed and how it knows.

T00 also recorded, verbatim, the mutation output that **is T01's predicted Red**:

```
row 6 (simple/Borders/b=0/denom=2.0): expected false, got true
```

## From T02 → **T17 and T19** (MATERIAL — correction to PLAN §6 C-12)

C-12's *form* column is wrong. The **site set** is right (the grep returns the same 9
lines, so C-12's STOP condition correctly did not trigger), but **all 7 test-side
`DeviceCtrColumn` literals end in `..DeviceCtrColumn::default()`** — not just the 3 C-12
lists. Only the production literal (`boosting.rs:2519`, now `:2559`) is a full literal.
⇒ T02 correctly edited **1** literal, not 5, and both `cargo check`s pass.

**Consequence T17/T19 must act on:** the `projection_members: vec![]` re-review set is
**8 sites, not 3**. The dangerous one is
`crates/cb-backend/src/gpu_runtime/session_depth_gt1_test.rs:630`, a genuine **2-member
combination** (`member_bins: vec![cat0, cat1]`) that now defaults to an **empty** member
list — once T17's eligibility gate reads `projection_members`, that fixture will look
*simple*. Production is safe: T02 added a `!projection_members.is_empty()` →
`CbError::Degenerate` guardrail making an empty list unreachable from
`build_device_ctr_config`.

## From T02 → **T24** (pre-existing failure to record, do not chase)

`cargo test --workspace` fails **one** target: `cb-backend --lib` under **default (cpu)**
features, 211 passed / 60 failed. Cause is `plane_inclusive_sum … is not supported on CPU`
in `cubecl-cpu-0.10.0`. T02 proved pre-existence by restoring `runtime.rs` from
`git show HEAD:` read-only and re-running `kernels::sort::` to the identical failures.
Unrelated to this phase. The rocm path (`bash ./run_device_tests.sh`) was **23/23 PASS +
perf lane** at the end of Wave 0, Poisson at 10.5× (no R-13 flake).

DCTR-01 byte identity held: `device_ctr_fit_test` `max |Δpred| = 4.483e-11` before **and**
after T02. That is the number later device tasks should expect to stay put unless they
intentionally move it.

## From T01 → **T10, T12, T16, T19, T22, T23** (the gate covering-test home)

T01 created `crates/cb-train/src/boosting_ctr_gate_test.rs`, mounted at the end of
`boosting.rs`. Its module doc records the **extension contract** every later gate task
follows: **two tests per conjunct** — a structural pin, plus a `gate_body()`-based
"the gate no longer reads X" pin — using the reusable `production_source` /
`code_lines_mentioning` / `gate_body` helpers already in the file. Extend it; do not
start a new file.

**Fragility to know about**: those helpers work by `include_str!("boosting.rs")` and
scanning the source text of `ctr_types_are_device_covered`. They are fail-loud on rename.
**T23 rewrites the gate entirely** (delegating to `ECtrType::from_i8` / `is_cpu_supported`)
— T23 must therefore update or retire these source-scan pins as part of its own change,
not leave them asserting against a function shape that no longer exists.

## From T04 → **T14, T15, T16** (Track B — the device BTMV mirror target)

The CPU BTMV quantizer is now (`ctr_feature.rs`, the `quantize_in_f32` arm):

```rust
let (shift, norm) = crate::ctr::calc_normalization(prior_scalar);   // norm, NOT the denominator
let denom = (total as f32) + 1.0f32;                                // the CTR denominator
let ctr   = (good as f32 + prior_scalar as f32) / denom;
f64::from(((ctr + shift as f32) / norm as f32) * ctr_border_count as f32)
```

Track B's "device == CPU" claims are now provable against the **corrected** CPU. Two things
to carry:

1. **`norm` and `denom` are different quantities and they COINCIDE at `count == 1`.** The
   plan's two-document T04 Red construction reaches `(good = 1.0, total = 1)` as predicted
   — but at that state `denom == 2.0 == norm`, so it cannot distinguish "divide by
   `calc_normalization`'s norm" from "divide by the CTR denominator". T04 widened its
   column to **three** documents (`(sum = 1.0, count = 2)` ⇒ correct bin 7, uncorrected 15,
   conflated 5) and only then did the mutation fail. **Any device BTMV self-oracle that
   drives ≤ 2 documents per bucket inherits the same blind spot.** Drive ≥ 3 documents in at
   least one bucket.
2. **C-17 was applied as "omit/mark dead", not as a live guard.** The device side must not
   introduce a `norm == 0.0` branch either — `norm = max(1, p) - min(0, p) >= 1` always.

DCTR-05 held exactly: `ctr_btmv_simple_oracle_test`'s printed output is byte-identical to
T03's captured baseline across DCTR-04, and **no fixture changed**. `cb-train` as a whole is
**106 targets, all ok** after the change.

## From T04 → **T24** (pre-existing clippy scope is WIDER than recorded)

The "Pre-existing, out of scope" bullet below names only `device_seam_test.rs`. Measured in
T04 with `cargo clippy -p cb-train --lib --tests --keep-going`: **12** `cb-train`
integration-test targets fail the lint gate, not one — `one_hot_draw_accounting_test`,
`learn_set_shuffle_oracle_test`, `yetirank_pairwise_tree_rng_oracle_test`,
`tensor_ctr_oracle_test`, `device_fold_count_gate_test`, `permutation_oracle_test`,
`structure_fold_cycle_oracle_test`, `plain_ctr_oracle_test`, `ordered_ctr_oracle_test`,
`s_order_ctr_bins_oracle_test`, `device_seam_test`, `ordered_boost_oracle_test`. Same class
(`panic`/`expect_used`/`indexing_slicing` in committed `tests/*.rs`), same cause,
`cargo test` unaffected. The **`cb-train` lib / lib-test target itself is clippy-clean**, so
a task whose work lands in `src/` can still verify itself. Without `--keep-going` clippy
stops at whichever target compiles first, which is why different tasks report different
"the" failing file.

## Environment / process facts for every executor

- Settings live at `planning/settings.json` (**no leading dot**); `implementation.use_worktree`
  is `false`. Plans live at `.planning/plans/` (**with** the dot). PLAN.md's frontmatter
  names a `gpu-borders-shared-sample` worktree that **no longer exists** — ignore it. All
  work happens in the primary checkout on branch `feat/device-ctr-full-coverage`.
- ROCm `gfx1151` is present and device tests really run.
- **Pre-existing, out of scope**: `cargo clippy -p cb-train --lib --tests` fails to compile
  the committed `crates/cb-train/tests/device_seam_test.rs` (23 `indexing_slicing` /
  `panic` / `expect_used` errors, from commit `de81a16`). `cargo test` is unaffected. Do
  not fix it inside another task; T24 records it.
- `crates/cb-oracle` has ~5 pre-existing clippy errors unrelated to this phase.
- Wave-parallel artifact: fixture generators carry a corpus-contamination check that
  reports sibling `ctr_device_*` directories created by concurrent tasks. Harmless.
