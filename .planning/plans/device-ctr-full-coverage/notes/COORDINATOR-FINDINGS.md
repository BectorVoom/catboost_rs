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

## From T20 → **T10, T12, T16, T19** (every remaining `CountingGpu` e2e) and **T24**

**1. There is now a measured CPU-fallback fingerprint, and it is cheap to read.** On
`ctr_device_mixed`, forcing the gate closed (`&& false`) flips the printed
`max |Δpred|` from **`4.483e-11` (device)** to **`1.388e-17` (CPU)** — the exact
CPU-fallback value `SPEC.md` DCTR-17 predicts — and collapses the run from **1.9s to
0.01s**. ⇒ *An unexpectedly tiny `max |Δpred|` or an instant device e2e is a
CPU-fallback smell.* Note the direction: **the CPU path scores BETTER against the
upstream oracle than the device path.** No ≤1e-5 bar can ever detect a fallback,
because falling back makes the number improve. Do not read a very small delta as
reassurance.

**2. Put the `CountingGpu` assertion AFTER the ≤1e-5 loop, not right after the fit.**
T20 did, deliberately. Under the §2.5 mutation the run then prints the passing
`max |Δpred|` line *before* panicking on `grown`, so a **single** mutation run yields
both halves of the required completion evidence ("the assertion failed AND the ≤1e-5
bar still passed"). Fail-fast ordering hides exactly the evidence DCTR-08/10/14/17
completion criteria ask for. In-source comments in `device_ctr_fit_test.rs` explain
this so a later reader does not "fix" the ordering.

**3. `&& false` on `ctr_types_are_device_covered` is the universal, cheap mutation for
any device-commitment assertion.** Rebuild + run is ~2.3s + ~2s; it needs no fixture and
no per-task bespoke mutation. Predicted failure shape is always
``assertion `left == right` failed … left: 0, right: <iterations>``. It edits the gate
expression, so the isolation rule applies — focused `--test <name>` only while live.

**4. The `CountingGpu` wrapper now exists in TWO copies** —
`device_ctr_gate_test.rs:82-138` (canonical) and `device_ctr_fit_test.rs` (copied
verbatim per GLOBALS §2.2.6, with a keep-in-sync note). Each new e2e adds a third,
fourth, … copy. Consequence: **any change to the `cb_compute::Runtime` method
signatures this wrapper overrides (`compute_gradients`, `begin_device_training`,
`grow_tree_on_device`, `end_device_training`) now breaks N test files, not one.** No
task in this phase is scheduled to change them, but T24 should record the duplication
count at DoD.

**5. Parallel-wave build hazard (process).** Any `cargo {check,test,clippy} -p cb-train`
transitively compiles `cb-backend`. During Wave 2, T08's in-flight edit left
`crates/cb-backend/src/kernels/ctr_device.rs` non-compiling (`E0061` at `:294`, kernel
signature mid-change) and that surfaced as a **red build in T20's commands** with
nothing wrong in `cb-train`. ⇒ **Before chasing a `cb-backend` compile error, run
`git diff --name-only` and check whether the broken file is yours.** If it is another
task's, record and move on; do not "fix" it.

## From T08 → **T09, T11, T12, T14, T16** (the ordered-prefix kernel seam after DCTR-06)

T08 landed the `(ctr_type, target_border_idx)` numerator selector on
`ordered_ctr_prefix_kernel` (`ctr_device.rs`, `fn` now at `:164`) and on
`launch_ordered_ctr_resident` (`:307`). Four facts later kernel/plumbing tasks must act on:

**1. `launch_ordered_ctr_resident` now REJECTS any `ctr_type` outside `{0 Borders, 1 Buckets}`
with `CbError::OutOfRange`.** This is deliberate and beyond the letter of T08's task text: the
alternative (accept `Counter` / `BTMV` and fall through to the Borders numerator) is a silent
wrong answer, not a worse one. ⇒ **T11 (Counter) and T14 (BTMV) must not route their columns
through this function.** If a later task genuinely needs to, widening the guard is a deliberate,
test-visible act — `ctr_device_test::out_of_range_ctr_mode_is_rejected` pins it and will go red.
`target_border_idx > 1` is rejected the same way (`SIMPLE_CLASSES_COUNT == 2`).

**2. Reuse `CTR_TYPE_BORDERS` / `CTR_TYPE_BUCKETS`** (`pub(crate)` at `ctr_device.rs:277`/`:279`)
rather than re-transcribing the `ECtrType` discriminants. C-3 demands a citation on every
cb-backend transcription; these already carry the full list plus the `restrictions.h` reference.

**3. T09's production edit is two literals, at `session.rs:230-231`** — replacing the
`CTR_TYPE_BORDERS, 0` that T08 pinned there (with an in-source comment naming T09) by
`col.ctr_type, col.target_border_idx`. **No conversion is needed**: `DeviceCtrColumn.ctr_type` is
already `i8` and `.target_border_idx` already `u32` (T02's seam). Combined with finding 1 and
C-14, a cb-train-gate / cb-backend-list mismatch surfaces as a loud `CbError` → `grown == 0`,
never a silent numerator swap.

**4. The `#[cube]` parameter order is now
`perm, bins, class, prior, mode, counts, good, total, value`** — `mode` is the **5th** positional
argument (`Array<u32>`, length 2, `[is_buckets, target_border_idx]`). There is still **exactly
one** `ordered_ctr_prefix_kernel::launch` site (`ctr_device.rs:391`).

**5. Test-side: `cpu_ordered_ctr` (`ctr_device_test.rs:69`) gained
`is_buckets: bool, target_border_idx: usize`**; all 5 pre-existing call sites pass `(false, 0)`,
which reduces to the previous `good = N1` exactly. It is deliberately kept in
`online_class_prefix`'s **generic loop-over-classes** form while the kernel implements the
2-class collapse — that asymmetry is what makes the oracle a cross-check rather than a copy.
`compute_ordered_ctr_host` kept its old signature and delegates to the new
`compute_ordered_ctr_host_mode` (`:496`), so the 4 pre-existing device oracle call sites are
source-unchanged.

**6. Kernel-style note for T11/T14 (cost-free, avoids a warning):** writing the CubeCL
`let mut x = <default>;` + `if`/`else` pattern with an explicit assignment in *every* arm makes
the initializer dead and emits `warning: value assigned to 'x' is never read`. Leaving the
"unreachable/pinned" case to the initializer (and documenting it in a comment) keeps the manual's
pattern and the build warning-free. T08 did this for `Borders@1`.

**D-04 held at T08:** `device_ctr_fit_test` `max |Δpred| = 4.483e-11`, unchanged; 23/23 device
PASS; `cb-backend --lib` under rocm `263 passed / 0 failed`.

**Pre-existing clippy failure to add to T24's list (cb-BACKEND, not cb-train):**
`cargo clippy -p cb-backend --no-default-features --features rocm --lib --tests` fails with
`error: this operation will always return zero` (`clippy::erasing_op`) at
`crates/cb-backend/src/kernels/score_split.rs:374` — `cindex[0 * n + obj] = bin;`. Verified
pre-existing at `HEAD` via `git show HEAD:…/score_split.rs | sed -n '374p'`. `cargo test` and
`cargo check` are unaffected. Do not fix inside another task.

## From T09 → **T10, T12, T16** (the backend admission list is now a live red-maker)

T09 added the per-type admission conjunct
`matches!(col.ctr_type, CTR_TYPE_BORDERS | CTR_TYPE_BUCKETS)` to **both** `.all(..)` closures
of `ctr_covered` (`session.rs`, structure half `:180`, averaging half `:192`) and turned
`build_ctr_cindex_columns`' launch into a `match col.ctr_type` (`:262-282`) whose `_` arm is
`CbError::Unsupported("device CTR type {other} is not implemented")`. Four things to act on:

**1. There is now a covering test that WILL go red for T12 and T16, deliberately.**
`crates/cb-backend/src/gpu_runtime/session_ctr_type_test.rs::ctr_covered_declines_unimplemented_ctr_types`
asserts today that discriminants `{2 BTMV, 3 FloatTMV, 4 Counter, 5 FeatureFreq}` **decline**
and that `{-1, 6, i8::MIN, i8::MAX}` decline too. When T12 admits Counter (or T16 admits BTMV)
it must, in the SAME hunk, (a) add the discriminant to both `ctr_covered` closures, (b) add its
arm to the `build_ctr_cindex_columns` match, and (c) move that discriminant from the test's
decline loop to its admit loop. **That red is the expected Red of those tasks, not a
regression.** The out-of-enum strays and `{3, 5}` must stay in the decline loop forever
(GPU-only upstream, `restrictions.h:20-32`).

**2. The `cb-train` gate and this backend list are still allowed to disagree transiently.**
C-14 is now written into `ctr_covered`'s doc comment in-source: the predicate's first caller
feeds the coverage disjunction whose failure path declines the **whole fit** (`Ok(None)`), so a
gate-admits / backend-declines mismatch surfaces as `grown == 0`, never a silent CTR drop.
T10 may therefore open the cb-train gate for Buckets without touching `session.rs` at all —
the backend already admits it.

**3. `launch_ordered_ctr_resident` is now reached with REAL `(col.ctr_type,
col.target_border_idx)`**, not T08's pinned `(CTR_TYPE_BORDERS, 0)`. Combined with T08's host
guard (which rejects any `ctr_type ∉ {0,1}` and any `target_border_idx > 1` with
`CbError::OutOfRange`), an unimplemented type now has **two** loud stops before it can reach a
wrong numerator. T11/T14 still must not route Counter/BTMV through this function.

**4. Pre-existing doc bug in the region T12/T16/T17 all edit.** The block at
`session.rs:200-215` beginning `/// Compute the ADDITIONAL binarized-CTR cindex columns …`
(including its `# Errors` section) documents `build_ctr_cindex_columns`, but is physically
attached to `struct CtrSearchState`, which was inserted between them at some point. T09 did
not move it (out of scope; three later tasks edit those lines) and instead put the new
`CbError::Unsupported` `# Errors` note on `build_ctr_cindex_columns`' own GDC-09 doc comment.
Whoever next restructures this region should re-attach the orphan.

**D-04 held at T09:** `device_ctr_fit_test` `max |Δpred| = 4.483e-11`, unchanged; `device
grows = 5`; 23/23 device PASS (+ perf lane, Poisson 10.2–10.5×, no R-13 flake); `cb-backend
--lib` under rocm `265 passed / 0 failed` (263 at T08 + T09's 2 new tests); `cb-train --lib`
`390 passed / 0 failed`.

## From T10 → **T12, T16, T19, T22, T23** and **T24** (assumption 8 is SETTLED)

**1. PLAN §6 assumption 8 is resolved: BOTH leaf paths run on a device CTR fit, and they
agree by construction. No escalation, no gather patch.**

* the returned leaf **VALUES** come from the device session's **position-indexed** gather
  (`session.rs:2782-2839`, `avg_bins.get(fu - base)`);
* the per-object leaf **ASSIGNMENT** (main-approx update + leaf weights) comes from the host
  **full-identity** `assign_leaf_over_ctr_columns(&matrix, &averaging_ctr_features, …)`
  (`boosting.rs:5609`; reached because `fused_unit_fold` is `false` whenever a CTR split
  exists, `boosting.rs:5511`).

Position indexing is correct for Buckets' two-columns-per-`(projection, prior)` layout
because `materialize_ctr_columns_for_perm` is the **single** producer of both the structure
and the averaging list (called at `boosting.rs:4254` and `:4284` over the same
`absolute_projections`/`ctr_candidates`), `build_device_ctr_config`'s `build_columns` is an
order-preserving `map` over each, and `ctr_covered` rejects `avg.columns.len() !=
ctr.columns.len()`. ⇒ device tail position `i` ⇔ the same FULL identity on both host lists.
**T17/T19 must preserve that single-producer property**: any task that filters, reorders or
de-duplicates one of the two lists without the other silently breaks the pairing, and the
detector is `device_ctr_buckets_fit_test` (mispairing measured at `|Δ| = 2.506e-1`).

**2. The `ctr_device_buckets` fit selects only `b = 0` splits on the DEVICE** (`[0]×8`),
while the CPU fallback selects `[0,1,0,0,1,1,0,1]` — and **both reproduce upstream at
`max |Δpred| = 2.776e-17`**. Cause, exact: at `Prior = 0.5` and binclf,
`ctr(Buckets@0) + ctr(Buckets@1) = (total + 1)/(total + 1) = 1`, and
`calc_normalization(0.5) = (0, 1)`, so the two columns binarize to mirrored bins and
`bin(b=0) > k` is the complement of `bin(b=1) > 14−k` — the same oblivious partition with one
level bit flipped, hence identical predictions. Two consequences:
  * **a `Prior = 0.5` Buckets fixture can never distinguish "b=0 chosen" from "b=1 chosen" by
    predictions.** A task needing a b-discriminating e2e must use a prior ≠ 0.5.
  * this e2e does **not** exercise the `b = 1` numerator through the *scoring* path. The
    `b = 1` evidence is T08's kernel self-oracle plus T10's MUT-A. Do not cite "the Buckets
    e2e is green" as proof that `b = 1` scoring is right.

**3. T20's CPU-fallback fingerprint has a counterexample — MATERIAL for T12/T16/T19.**
On `ctr_device_buckets` the device and CPU paths print the **same** `max |Δpred| =
2.776e-17`. So "device delta ≠ CPU delta" is NOT a reliable device-commitment signal either.
The two that held: `CountingGpu.grown.get() == iterations`, and runtime (1.7 s device vs
0.01 s CPU). T20's rule survives in its *original* one-way form ("a tiny delta is a smell,
never reassurance"), not as a fingerprint match.

**4. `boosting_ctr_gate_test.rs` now has FOUR more pins, two of them source scans over the
gate body — and one is a trap for the next editor.**
`the_device_gate_no_longer_reads_the_buckets_numerator_selector` asserts the RAW body text
(inline comments included) does not contain `target_border_idx`; the sibling type pin asserts
it does not contain `ECtrType::Borders.as_i8()` and DOES contain `ECtrType::from_i8`.
⇒ **T12/T16 must not spell either phrase in a comment inside the gate body**, and **T23,
which rewrites the predicate to delegate to `from_i8`/`is_cpu_supported`, must update or
retire all six pins in that file** (T01's two + T10's four) as part of its own change — the
`from_i8` conjunct of T10's pin is the one T23 can keep.

**5. The gate-state table gained rows 9 and 10 (APPENDED — rows 1-8 keep their numbers)** and
`gate_admits_exactly_the_current_wave` now reports **all** mismatching rows instead of failing
at the first. Later gate tasks get a complete admitted-set diff in one run; the message format
is `row N (label): expected X, got Y` + the row's `flips_at`, one block per mismatch.

**6. T00's question answered.** T10's edit removed **both** conjuncts (`ctr_type == Borders`
and `target_border_idx == 0`), proved by two single-conjunct restorations: restoring the type
conjunct alone reddens rows 4+9 and leaves row 10 green; restoring the target-border conjunct
alone reddens rows 4+10 and leaves row 9 green. Verbatim output in `notes/T10.md` §3.4.

**7. Pre-existing clippy item for T24's list (NEW — cb-backend, DEFAULT cpu features).**
`cargo clippy -p cb-backend --lib` emits `clippy::duplicated_attributes`: a doubled
`#[allow(clippy::too_many_arguments)]` at `crates/cb-backend/src/gpu_runtime/mod.rs:4401` and
`:4433`. Verified pre-existing at `HEAD` (`git show HEAD:…/mod.rs`); `mod.rs` is in no P1
task's diff so far. Distinct from T08's `erasing_op` (rocm, `score_split.rs:374`) and T04's 12
`cb-train` test targets.

**8. `CountingGpu` is now duplicated THREE times** (`device_ctr_gate_test.rs:82-138` canonical,
`device_ctr_fit_test.rs`, `device_ctr_buckets_fit_test.rs`). T24's DoD count should expect one
more per remaining e2e (T12, T16, T19).

**D-04 held at T10:** `device_ctr_fit_test` `max |Δpred| = 4.483e-11`, unchanged; `device
grows = 5`; 23/23 device PASS (+ perf lane, Poisson 10.8×, no R-13 flake); `cb-backend --lib`
under rocm `265 passed / 0 failed` (identical to T09); `cb-train --lib` `394 passed / 0 failed`;
`cargo test -p cb-train` **107 targets, all ok**.

## From the coordinator, off T10's finding #2 → **T22** (READ BEFORE DESIGNING THE DIFFERENTIAL)

T22 is specified as a device-vs-CPU **chosen-split-sequence** differential, deliberately
comparing splits rather than predictions. T10 measured a case where **the chosen splits
legitimately diverge while the predictions are identical**: on `ctr_device_buckets`
(`Prior = 0.5`, binclf) the device selects `[0]×8` and the CPU selects `[0,1,0,0,1,1,0,1]`,
both at `max |Δpred| = 2.776e-17`, because `Buckets@0` and `Buckets@1` binarize to mirrored
bins and the two split sets describe the **same oblivious partition with one level bit
flipped**.

⇒ A naive "device split sequence == CPU split sequence" assertion **will fail for a benign
reason** on any `Prior = 0.5` Buckets configuration. T22 must either

* choose a prior ≠ 0.5 for its Buckets arm (T10 §2 shows `0.5` is the degenerate point), or
* compare a **partition-invariant** projection of the split set rather than the raw
  `(feature, border)` sequence,

and must say in its note which it did and why. Do **not** discover this as a red test and
"fix" it by loosening the assertion after the fact — that is exactly the class of change
§2.5 exists to prevent. Note this cuts the other way too: because the partitions coincide,
a raw-sequence comparison that happens to pass is not evidence the two paths agree.
