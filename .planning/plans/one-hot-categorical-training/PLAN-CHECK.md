---
title: Plan check (pass 5, final confirmation) — one-hot categorical training
kind: plan-check
pass: 5
spec: .planning/plans/one-hot-categorical-training/SPEC.md (spec_version 4)
plan: .planning/plans/one-hot-categorical-training/PLAN.md (revision 5, 35 tasks)
checked_at: 2026-07-31
verdict: PASS
---

# Plan Check Result — pass 5 (final confirmation)

**VERDICT: PASS**

All three pass-4 findings are closed. I re-verified each against source rather than
against the planner's description, re-ran the regression sweep over everything closed in
passes 1–4, and found nothing new. The plan is ready to implement.

---

## Pass-4 findings — verification

### [MAJOR-1a] `real_folds` empty on the float-only path — **CLOSED**

Fixed in both halves, plus two things I did not ask for that make it durable.

**The single-call-site claim is true.** CodeGraph blast radius: `quantize_feature_major`
(`crates/cb-train/src/boosting.rs:2196`) has **exactly 1 caller**, and I read it:
`boosting.rs:3132` — `let out = quantize_feature_major(feature_values, feature_borders, n);`
— sitting inside the `if device_host_eligible { … }` block that opens at `:3129`
(`let (device_bins, device_n_bins) = if device_host_eligible {`). The plan's citation of
`:3129` names that block; the call is two lines in. So T24's new **Call-site rule** rests
on a verified fact:

> The device-quantize call site is **`boosting.rs:3129`** … and there is exactly ONE. It
> **ALWAYS calls `quantize_feature_major_with_one_hot`** … passing an **empty `cat_bins`
> slice** when the pool has no one-hot columns. It therefore **ALWAYS populates
> `real_folds` to length `n_features`**, including on a float-only pool … **never empty
> on a device-eligible fit**.

**The "UNCHANGED" ambiguity is genuinely removed.** The step now states that
`quantize_feature_major`'s *body and signature are unmodified and it is delegated to* for
the float prefix, then names the wrong reading, spells out its consequence (seven device
oracles plus T29b failing on `CbError::LengthMismatch`), and declares it forbidden. An
implementer cannot now pick the breaking branch by accident.

**The §9b trace is internally consistent.** I read all five rows:

| row | float-only value |
|---|---|
| produce (T24 1b) | `[borders+1, …]` |
| **carry (T27b 1)** | **`[borders+1, …]`, length `n_features` — identical to the produce row; NEVER empty on a device-eligible fit** |
| upload (T27b 3) | uploaded, never read |
| consume (T25) | n/a — `one_hot == false`, eligibility stays `border < max_border` |
| prove | T24 fn 5, T27b Red, T28 fn 2, T29b fn 2 |

produce and carry now agree, and both agree with T24 step 1b and T24 Red fn 5. I grepped
for any surviving `empty (Default)` claim: **the only occurrence is inside the §9b
disposition cell describing the old defect** — a historical record, not a live claim.

**The source-compatibility / runtime-value split is stated correctly.** T27b step 1's
"**Two separate questions, do not conflate them**" heading separates *construction sites*
(all three fields get an empty/`false`/`0` `Default` so every `..Default::default()`
literal keeps compiling) from *runtime value on a device-eligible fit* (never empty),
and closes with "The `Default`-empty value is only ever seen by a construction site that
never reaches `begin_device_training`." That is the correct diagnosis of what produced
the contradiction, and it prevents the same conflation recurring.

**The assertion stayed unconditional, with the right guard rail.** T27b keeps
`real_folds.len() == eff_n_features` unconditional and adds: *"Do NOT weaken it to
`if !real_folds.is_empty()` — that would restore exactly the silently-inert bound [C16]
exists to eliminate."* That is the single most important sentence in the fix: the
tempting local repair is precisely the original defect, and it is now named and banned.

### [MINOR-1] T28's unreachable session read-back — **CLOSED, coverage intact**

The assertion is deleted and replaced by an in-place note recording *why* (`pub(crate)`
on `cb_backend`'s `GpuTrainSession`, itself owned privately by `GpuBackend`,
`gpu_backend.rs:296` `*self.session.borrow_mut() = session;`) and pointing at
`gpu_runtime::one_hot_session_wiring_test` for localization.

**Is the padded-bin production-path guarantee still proven?** Yes, and I checked the
chain rather than the claim. T28 fn 2 is unchanged in the part that matters: it *must*
run `train`/`train_cat` → `device_host_eligible` → `begin_device_training` →
`grow_oblivious_tree_resident`, **never a hand-supplied `real_folds`**, on a 31-border
float column (`n_bins_line = 32`) plus a cardinality-2 one-hot column **with a padded bin
arranged to score highest**, plus the gap-bin sub-case. If `real_folds` were wired to the
padded width, a phantom "all-objects-right" candidate wins device-side only and the
≤1e-5 assertion fails. The deleted read-back only *localized* that failure; it never
carried the guarantee. Producer correctness is proven by T24 Red fn 5, seam transport by
T27b's Red (`real_folds == [32, 2, 2]`, load-bearing assertion `real_folds[1] == 2`, not
`32`). Coverage is complete.

### [MINOR-2] Stale `#[allow(dead_code)]` narrowing — **CLOSED, and correct for exactly the right fields**

T24 step 4 now narrows to **both `first_fold_index` and `folds`** (or keeps the
struct-level allow with a comment naming exactly those two), and states that narrowing to
`first_fold_index` alone *"fails `cargo clippy --workspace --all-targets -- -D warnings`
with `field is never read: folds`"*, with the `#[cfg(test)]`-only caveat for Red fn 1b.

Verified against source: after this task `one_hot_feature` becomes read (via
`device_arrays()`'s 4th element), while `folds` and `first_fold_index` remain unread —
their only occurrences across `crates/cb-backend/src/` are the doc comments at
`cindex.rs:48` / `:54` and the two writes at `:224` / `:225`. The allow-list covers
exactly the two fields that are unread and no others.

### Bonus: the `mod.rs:3122` note

My pass-4 "no finding" is recorded at `PLAN.md:3738-3743` with the reasoning
(`best_c` starts at `u32::MAX` and is only assigned from a `cand` that passed
`>= pass_hi`, and `pass_hi <= n_candidates`) and an explicit **"T25 must not touch it."**
Correct, and it removes a live re-litigation risk.

---

## Regression sweep — nothing previously closed has moved

| item | check performed | result |
|---|---|---|
| **MAJOR-A** (validation filters) | Re-extracted all distinct `cargo test … --lib` filters in revision 5 and diffed against the verified mount set | **22 real filters, all correct**; the only extra match is the header's generic template `gpu_runtime::<name>`, not a command |
| **MAJOR-2** (Green/Design agreement) | `grep "feature_lo as usize + c /"` | **zero hits** — no live instruction prescribes the relative index anywhere |
| **MAJOR-B Design** (absolute `c`, `pass_hi` sentinel) | Untouched by revision 5 | intact |
| **MAJOR-C** (device-test placement) | "Device-test placement rule" section present; the four `gpu_runtime` `src`-sibling Reds unchanged | intact |
| **`real_folds` mechanism** | producer → carrier → uploader → consumer chain re-read | intact and now unambiguous |
| **Comptime elision** (float-only invariance) | Repo precedent re-confirmed: `kernels.rs:4582` / `:4592` `if score_fn == comptime!(SCORE_FN_COSINE)` | argument holds |
| **CTR-padding inertness** | SPEC-OH-26 gates one-hot × CTR ⇒ `eff_n_features == n_features` in the one-hot regime; a CTR column is never `one_hot`, and pass B sweeps only `[n_float·n_bins, n_total·n_bins)` | inert, confirmed |
| **Empty `real_folds` + one-hot present** | T27b's unconditional assertion → `CbError::LengthMismatch` | fail-loud, never a silent inert bound |
| **SPEC** | `spec_version: 4`, md5 `992da4d5…` | untouched this round, as stated |
| **Pass-3 MINORs a–d, pass-2 MINORs a–d** | spot-checked | all still closed |
| Waves / task count / ordering | unchanged (35 tasks); MAJOR-1a's fix lands in T24/T27b, upstream of T25/T28 | no reordering needed |

**Observation, not a finding.** The pass-2 disposition table's own `MINOR-1` cell
(`PLAN.md:3630`) still reads "narrowing the allow to `first_fold_index` alone" — stale
relative to the corrected T24 step 4. It is covered by the standing preamble at `:3698`
("Those tables are the *historical* record … and the current text governs"), and the live
task text is correct and explicit, so no revision is required.

---

## Cumulative disposition across all five passes

| pass | raised | outcome |
|---|---|---|
| **1** | 4 CRITICAL, 11 MAJOR, 7 MINOR | All closed. CRIT-1 `cb_oracle::model_json::SplitJson` blocking A11 → new T02b. CRIT-2 `n_features` fixing both `n_candidates` and `leaf_stride` → `feature_lo`/`feature_hi`. CRIT-3 the host belt as a second trailing-border exclusion → both lifted. CRIT-4 SPEC-OH-27's fallback unschedulable → T01a/T01b split. |
| **2** | 3 MAJOR, 4 MINOR | All closed. MAJOR-A dead `model_json::tests` filter. MAJOR-B relative-vs-absolute candidate index. MAJOR-C four device Reds unreachable from `crates/cb-backend/tests/` → relocated to `gpu_runtime` `src` siblings. |
| **3** | 2 MAJOR, 4 MINOR | All closed. **MAJOR-1: `TCFeature.folds` is the padded line width, so the bound I myself had prescribed in passes 1–2 was inert** → separate `real_folds` array. MAJOR-2: T25's Green still specified the relative form its Design rejected. |
| **4** | 1 MAJOR, 2 MINOR | All closed (this pass). MAJOR-1a: the trace's `carry` row said `real_folds` was empty on float-only while three other statements said otherwise, and T27b's unconditional length assertion turned that reading into a hard failure of seven device oracles. |
| **5** | — | **PASS.** |

Two SPEC defects surfaced and were amended by the coordinator along the way: **v2**
(SPEC-OH-22 named the wrong scorer; SPEC-OH-15 named one of four `float_splits_of`
consumers), **v3** (kernel byte-identity is not a testable property once comptime
parameters are added), **v4** (the real-bin bound must not be `TCFeature.folds`).

**My own errors, for the record:** in passes 1 and 2 I prescribed `TCFeature.folds` as
the real-bin bound having verified only that the field exists, not what the production
caller stores in it — the defect I then had to raise myself in pass 3. And in pass 2 I
told the planner to drive the Red through `score_partition_over_binsums` without checking
that it is a private `fn`, which produced the unbuildable placement I raised in pass 2's
own MAJOR-C.

---

## Why this is a PASS

- **Specification coverage:** all 31 SPEC-OH specifications map to concrete tasks; all 13
  acceptance scenarios (A1–A13) map to a task that can actually execute.
- **Structural verification:** every load-bearing claim about a symbol, signature,
  caller, callee, visibility or impact path has been checked against source across five
  passes — the production scorer, the two trailing-border exclusions, `ModelSplit`'s
  consumer set, `from_trained`'s 24 callers, the four infallible SHAP surfaces,
  `perfect_hash_bins`, `FeatureMatrix::new`, `TCFeature.folds`, `pack_cindex`'s 13
  callers, `DeviceGrownTree`'s 17, the `catboost-rs-py` exhaustive matches, and every
  test-mount path.
- **No unmitigated regression risk:** the blast radius is enumerated file-by-file (22
  serialization rows), every non-oblivious device call site is instructed to pass the
  byte-unchanged default, and the out-of-scope Region/non-symmetric growers are gated by
  `device_region_fit_test` + `device_nonsym_fit_test` on every task that sweeps them.
- **Verification is real, not nominal:** the Reds that matter demonstrate a defect before
  the fix (T04's 20.0↔30.0 leaf transposition, T19's silently-dropped column, T21's
  bin-vs-hash, T25's fn 4/5, T28's production-path padded bin), the scaffolding Reds are
  labelled as such, and every `--lib` filter selects a module that exists.
- **The blind spot that let the worst defect survive three passes is closed** by an
  assertion that cannot be satisfied with hand-supplied inputs.

Remaining unverified items are all explicitly owned and none blocks a start: the
workspace failure baseline (T00 produces it, T29 diffs against it),
`DEVICE_ONE_HOT_MAX_CARDINALITY`'s numeric value (T24, B11), CubeCL's acceptance of a
`#[comptime] u32` loop bound (T25, B5 — with T29b as the standing regression gate),
`TOneHotFeature.Index` semantics (T08, B2), and `is_covered_regime`'s one-hot arm not
perturbing existing device oracles (T27b, B12).

Implementation can begin.
