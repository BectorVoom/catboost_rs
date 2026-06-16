# Phase 06.3 — Deferred Items

## [06.3-03] PairLogit / PairLogitPairwise leaf-estimation parity gap (DEFERRED)

**Status:** der + Cholesky pairwise-leaf path + `is_pairwise_scoring` routing all LAND
and are unit-tested green; the **end-to-end per-stage oracle** for PairLogit and
PairLogitPairwise is DEFERRED pending a leaf-estimation der2 parity fix.

**Symptom:** PairLogit tree-0 leaf values diverge from catboost 1.2.10.
- depth-1, l2=0, lr=1.0 probe: catboost leaves = `[0.7778 (=7/9), -1.5556 (=-14/9)]`;
  the transcribed der (p=0.5 at exp-approx=1) + standard per-object Newton leaf gives
  `[1.0, -1.3333]`.
- **Root cause (isolated):** the per-leaf SECOND-derivative SUM differs. The total
  `Σ der2` over all objects from the transcribed `competitor.Weight·p·(p-1)` formula
  (`error_functions.h:861-862`) is `-3.5` (7 pairs × 2 × -0.25 at p=0.5), but the value
  catboost's leaves imply is `-3.857 (=-27/7)`. A `Σ der2 = -3.857` is impossible from
  `p(p-1)` with `p ∈ [0,1]` (the quadratic `p(1-p)=0.2755` has no real root) AND the
  pair weights were confirmed to be exactly `1.0` (2-col and 3-col-weight-1 pools give
  identical leaves) AND the leaf assignment matches (leaf_weights `[8,4]`).
- **Conclusion:** catboost computes the PairLogit per-leaf der2 by a path NOT captured by
  the per-object der2 sum — most likely the pairwise-aware `weightedDers` / leaf-der
  reduction (`approx_calcer_querywise.cpp` `AddLeafDersForQueries` over the pairwise
  bucket der2), or an iterative recompute inside the gradient walker. This is a
  leaf-estimation subsystem gap, NOT a der-formula gap (the per-object der1/der2 and the
  Cholesky pairwise-leaf solve are independently unit-test-verified, including bit-exact
  vs `pairwise_leaves_calculation_ut.cpp`).

**What landed (green):**
- `Loss::PairLogit` / `Loss::PairLogitPairwise` der over Competitors (inline exp), unit-tested.
- `pairwise_leaves.rs` Cholesky leaf solve (2×2 + general), bit-exact vs upstream UT.
- `is_pairwise_scoring` / `is_plain_only` predicates + boosting leaf-path routing.
- Frozen catboost 1.2.10 `PairLogit` + `PairLogitPairwise` fixtures (committed for the
  follow-up oracle).

**Follow-up:** add a plan transcribing the pairwise leaf-der reduction
(`approx_calcer_querywise.cpp` `AddLeafDersForQueries` der2 path) so the per-leaf SumDer2
matches upstream, then gate the `pairlogit_oracle` + `pairlogit_pairwise_oracle`
per-stage tests ≤1e-5 against the already-frozen fixtures. The fixtures are committed;
only the leaf-der2 reduction + the two oracle tests remain.

**LambdaMart (Wave B) shipped fully** — per-stage oracle ≤1e-5 GREEN (the listwise
positive-hessian Newton-denominator fix in `leaf.rs::newton_leaf_delta` was the key).
