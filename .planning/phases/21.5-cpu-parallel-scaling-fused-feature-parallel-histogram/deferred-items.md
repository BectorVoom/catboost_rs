# Deferred / Out-of-Scope Items — Phase 21.5

Items discovered during execution that are OUT OF SCOPE for the fused feature-parallel
histogram work (they are not caused by the fusion of `select_level_plain` /
`select_level_perturbed`). Logged per the executor SCOPE BOUNDARY rule; not fixed here.

## 1. Stale Region-rejection assertion in `monotone_oracle_test.rs` (pre-existing since Phase 12)

- **Test:** `monotone_non_symmetric_and_region_are_typed_errors`
  (`crates/cb-train/tests/monotone_oracle_test.rs:286`)
- **Symptom:** panics with `grow_policy=Region must be rejected with a typed error
  (D-6.6-04 "Region OUT")`.
- **Root cause:** the assertion encodes the Phase-6.6 world where `grow_policy=Region`
  had NO CPU path and was rejected by `validate_grow_policy`. Phase 12 (v1.2) BUILT the
  CPU Region grower ("build-CPU-Region-FIRST"), so `train(... grow_policy=Region ...)`
  now succeeds instead of returning a typed error. The test was never updated when the
  CPU Region path landed, so its Region-rejection expectation is stale.
- **Why NOT a 21.5 fusion regression:** the fusion commits (`8c47241`, `400b799`,
  `d20514f`) touch only `select_level_plain`, `select_level_perturbed`, and
  `GrowScratch` — the oblivious symmetric split search. The Region grower is a separate
  non-symmetric dispatch that never routes through the fused per-feature pass, so the
  fusion cannot change whether Region is accepted or rejected. `device_region_fit_test`
  and `region_e2e_test` (CPU Region) pass, confirming Region is a supported CPU path now.
- **Disposition:** DEFERRED. Belongs to a Phase-12 follow-up / monotone-under-Region
  owner: either (a) delete the stale Region-rejection arm (Region is supported), or
  (b) if monotone constraints are genuinely unsupported UNDER the Region grower, change
  the assertion to reject `Region + non-empty monotone_constraints` rather than Region
  itself. Not a parity break; does NOT touch any ≤10⁻⁵ oracle fixture.
