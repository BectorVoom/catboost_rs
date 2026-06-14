# Phase 05 — Deferred Decisions

Auditable record of decisions to retire / defer work within Phase 05, co-located
with the phase artifacts (no orphan `todos/` directory).

---

## 2026-06-14 — Retire `ordered_structure_differs_from_plain` (GAP 1 / ORD-02)

**Plan:** 05-16
**File:** `crates/cb-train/tests/ordered_boost_wiring_test.rs`
**Decision:** RETIRE the `ordered_structure_differs_from_plain` wiring sub-test in
place (renamed `ordered_branch_alive_structural_authority_is_e2e_oracle`,
converted from a failing `assert_ne!` to a passing positive assertion).
**ORD-02 structural authority is delegated to `ordered_boost_e2e_oracle_test`**
(2/2 PASS ≤1e-5 vs catboost 1.2.10).

### Why the assertion was invalidated (not a dead branch)

The retired sub-test asserted `assert_ne!(ordered_splits, plain_splits)` — that
the Ordered model's tree structure differs from Plain's. That premise was
invalidated by **upstream-faithful** behavior introduced in 05-12, not by a dead
Ordered branch:

- The Ordered structure search selects its learning permutation via
  `find(|f| !f.is_averaging)` (`boosting.rs:~1054`), which returns `Folds[0]` =
  the **IDENTITY** (object-order) learning fold for **every** `permutation_count`.
  After 05-12 made the lone learning `Folds[0]` the identity (zero RNG draws,
  upstream `shuffle = foldIdx != 0`, `fold.cpp:54`), the ordered per-segment L2
  scoring walks object order.
- On the test's randomness-free synthetic dataset (`bootstrap=No`,
  `random_strength=0`), Ordered per-segment scoring on object order legitimately
  **coincides** with Plain. Empirically both produce splits
  `[(1, 8.5), (0, 1.5)] × 5`. Asserting divergence here asserts a false premise.

### Why re-keying `permutation_count` cannot fix it

Re-keying the test's `permutation_count` to `>= 2` does **not** change which fold
the ordered search consumes — `find(|f| !f.is_averaging)` still returns the
identity `Folds[0]`. Making the ordered search consume a non-identity learning
fold is a **production change to ordered fold-selection semantics**, which is
**OUT OF SCOPE** for this gap-closure plan. The e2e oracle already validates
ORD-02 ≤1e-5, so altering fold-selection would carry risk with no parity benefit.

### What still guards ORD-02 (no genuine guarantee lost)

- **`ordered_boost_e2e_oracle_test`** — the AUTHORITATIVE ORD-02 structural check:
  FULL multi-tree ordered predictions match catboost 1.2.10 ≤1e-5 (2/2) through
  `cb_model::predict_raw`.
- **`ordered_training_grows_a_full_finite_model`** — the aliveness gate (preserved
  unchanged): proves the Ordered branch grows a real 5-tree model, not a Plain
  fall-through.
- **`plain_path_still_trains`** — preserved unchanged: the shared driver still
  serves the Plain path.

### Scope

Test-only + this DEFERRED note. **No production source file changed.** The
`ordered_boost_e2e` ORD-02 hard gate stays GREEN ≤1e-5.

### Future work (if ever desired)

A genuine ordered-vs-plain structural-divergence unit test would require an
out-of-scope production change letting the ordered structure search consume a
non-identity learning fold (or a dataset engineered to diverge under the identity
fold, which is the least upstream-faithful option). Neither is needed: structural
parity is oracle-locked by `ordered_boost_e2e_oracle_test`.
