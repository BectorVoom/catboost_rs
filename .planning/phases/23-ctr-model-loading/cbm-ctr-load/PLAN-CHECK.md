# Plan Check — CTR model loading

**Checker:** `plan-checker` (independent; all symbol/signature/field claims verified via
CodeGraph + source). **Latest verdict:** ✅ **PASS** (pass 2/2). **Date:** 2026-07-17.

| Pass | Verdict | Issues |
|------|---------|--------|
| 1 | ISSUES_FOUND | 2 MAJOR + 4 MINOR (no blocker) |
| 2 | **PASS** | 0 remaining |

## Pass 1 issues (all resolved)
- **MAJOR-1 — fixture provenance.** T3's byte constants depend on exact model bytes but
  T0 under-specified the generator. → T0 now fully pins `gen_fixtures.py` (data
  `RandomState(0)`, `n=200`, layout, `y` formula, all params) and freezes the committed
  `.cbm`/`.npy`/`.json` (CI does not regenerate); T3 dissects the committed file.
- **MAJOR-2 — mean-type CTR decoded but deferred/untested.** → `decode_ctr_model_parts`
  now REJECTS mean ctr_types with a typed error; no `CtrValueTable.mean` from unverified
  bytes. SPEC §2/CTR-03/CTR-05 + T3/T5 updated.
- **MINOR-1 — two generated `ncat_boost_fbs` modules.** → T2 uses
  `model_generated`; T3 uses `ctr_data_generated::{root_as_tctr_value_table,
  TCtrValueTable}`; ctr_type via `ECtrType::from_i8(base.CtrType().0)`.
- **MINOR-2 — dense-hash completeness.** → T3 validates the non-empty idx set is exactly
  `0..bucket_count` (no gap/dup/OOB); added to CTR-05/T5.
- **MINOR-3 — bucket_count authority.** → `bucket_count = #non-empty slots`; width=1 for
  Counter/FeatureFreq; cross-check `blob.len()/4 == bucket_count*width`; added to CTR-05/T5.
- **Tail Option→error.** → T4 passes `buf.get(8+declared..).unwrap_or(&[])`; a
  CtrFeatures-present model with empty tail errors in the parser, never panics.

## Pass 2 — PASS
All six revisions verified internally consistent with the codebase. Apply-compat
confirmed 1:1 (no apply change). Numeric-path regression invariant intact
(`build_combined_bins` preserves float order; empty CtrFeatures → ctr_data None; save
path untouched). Ordering valid. The ≤1e-5 CTR-04 oracle is the objective end-to-end gate
for the hash-fold + f32→f64 border invariants; the frozen cat-driven fixtures make a
wrong decode diverge well beyond 1e-5.

## Non-blocking notes
- Buckets(1)/FeatureFreq(5) decode paths are in-scope but unexercised by the `{0,4}`
  fixtures (they reuse tested Borders/Counter layouts — low risk).
- The `.cbm` CTR byte layout rests on the research's empirical dissection of catboost
  1.2.10 (version-pinned); CTR-03/CTR-04 are the objective gates.
