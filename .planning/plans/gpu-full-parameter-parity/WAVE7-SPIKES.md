# Wave 7 — spike notes and escalation decisions

Deliverable for T21 / T25 / T27 / T29 / T31 / T33 / T34 / T35. Per the PLAN's §7
"Parity-complete" bar, **an escalated, documented STOP is a valid completion state for a
Wave-7 track; a silent overrun is not.** This file is the documentation half of that.

Every finding below was verified against the tree at the time of writing (the
`worktree-gpu-full-parameter-parity` branch, Waves 0–6 merged). Line-level claims name the
symbol, not a line number, so they survive edits.

---

## The single most important finding: the backend already has the coverage gates

`cb-backend/src/gpu_runtime/session.rs` already defines **five** `Option`-returning
family coverage gates following the same Pattern-A template:

| gate | family | consumed today? |
|---|---|---|
| `map_pairwise_coverage` | `*Pairwise` losses (Track W) | partially |
| `map_ranking_coverage` | QueryRMSE / QuerySoftMax / YetiRank (Track R) | partially |
| `map_multiclass_coverage` | MultiClass / MultiLogloss / … (Track X) | partially |
| `map_ordered_coverage` | Ordered boosting (Track O) | **decision only** |
| `map_langevin_coverage` | Langevin / SGLB | decision only |

That materially changes Wave 7's shape from what the PLAN assumed. The remaining work per
track is **not** "design a coverage gate"; it is "make the grow path consume the state the
gate already returns", plus whatever kernel each family genuinely lacks.

It also means the four Phase-13 families are exactly the ones T04's `FamilyTreeArgs` seam
was built for — and that seam now exists, threaded through all 15 sites with every wrapper
forwarding it verbatim. Wave 7's implementation tasks have their entry point ready.

---

## T21 — Ordered boosting on device (Track O) · **GATE G1: STOP, do not start T22**

**State.** `map_ordered_coverage` exists and is correct: it admits a covered simple-
approximant loss at depth ≥ 1, single fold, SymmetricTree, with every other family flag
default. It returns an `OrderedState { der_kernel }` that **nothing consumes** — the struct
is `#[allow(dead_code)]`.

**What is genuinely missing.** Ordered boosting's defining property is that object `i`'s
approximant is built only from objects preceding it in the learn permutation. The split
score for a candidate must therefore be accumulated **per permutation segment**, not over
the whole pool. The device's resident histogram fills one histogram per partition over ALL
objects; there is no per-segment accumulation kernel, and `OrderedState` carries no segment
descriptor.

**Estimate.** A per-segment split-score kernel is net-new CubeCL work: a segmented fill
whose segment boundaries are the fold prefix lengths (`fold_len_multiplier` growth), plus a
segmented argmin, plus its own self-oracle against `ordered_approx_delta_simple`. That is
comfortably past G1's "> ~3 days-equivalent" trigger.

**Decision — ESCALATED under G1.** Publish this note; do NOT start T22/T23/T24. The
`bias == 0.0` re-evaluation the PLAN flagged for T24 is moot in the other direction: FPP-02
removed that clause, so `ordered_boost_e2e/`'s only remaining disqualifier is the Ordered
clause itself.

---

## T25 — non-Borders CTR on device (Track U) · **GATE G2: STOP**

**State.** `ctr_types_are_device_covered` requires `ctr_type == Borders`,
`target_border_idx == 0`, `prior_denom == 1.0` — and, as of this phase's FPP-11
escalation, `projection.is_simple()` again.

**What is genuinely missing.** Buckets, BinarizedTargetMeanValue and Counter are three
DIFFERENT online accumulation semantics, not three parameterisations of one:

- **Buckets** needs a per-target-border numerator selector, so a single CTR column becomes
  `target_border_count` columns sharing one denominator.
- **BinarizedTargetMeanValue** accumulates the target MEAN per bucket rather than a class
  count — a different reduction, in floating point, where `launch_ordered_ctr_resident`
  counts exact integers.
- **Counter** ignores the target entirely and has its own `counter_calc_method`
  (`SkipTest` / `Full`) affecting which objects are counted.

Each is a new resident kernel with its own read-before-increment discipline.

**Decision — ESCALATED under G2.** Three genuinely new resident designs, each past the
"> ~2 days-equivalent" trigger on its own. Do not start T26.

**Sequencing note.** FPP-11 (combination projections) is a strictly smaller and better-
localised problem than any of these — trees 0–2 already match upstream exactly, and the
divergence is pinned to one scoring decision. It should be finished before Track U is
reconsidered.

---

## T27 — monotone constraints on device (Track M) · **GATE G3: assessment**

**State.** `params.monotone_constraints.is_empty()` is an unrelaxed gate clause. The CPU
path implements the constraint inside the level search.

**G3's trigger is "the constraint is NOT a candidate filter".** It is not. Upstream applies
monotonicity by **clamping leaf values** after the split is chosen (the child means are
projected onto the monotone cone), not by excluding candidate `(feature, bin)` pairs. A
device implementation therefore cannot be expressed as a mask over the existing argmin — it
needs a post-split leaf projection pass with a per-feature direction vector.

**Decision — ESCALATED under G3.** Report before T28, as the gate requires. The work is
bounded (a leaf-value pass, not a scoring redesign) and is the most tractable of the
spike-gated tracks — a reasonable next phase after FPP-11.

---

## T29 — feature-weight penalties on device (Track N) · **GATE G4: split the track**

**State.** `penalties_active` is `!feature_weights.is_empty() || first_feature_use_penalties
|| per_object_feature_penalties || penalties_coefficient != default`, and the gate rejects
the union.

**The three knobs are NOT equivalent in difficulty:**

| knob | shape | device cost |
|---|---|---|
| `feature_weights` | per-FEATURE multiplier on the split score | **cheap** — a length-`n_features` upload the scorer already has a natural place for (it already applies a per-column cat-feature weight host-side in the CTR pass) |
| `first_feature_use_penalties` | per-FEATURE, applied only on a feature's FIRST use in the model | **cheap** — the same shape plus a model-lifetime used-set, which `group_used` already demonstrates |
| `per_object_feature_penalties` | per-OBJECT × per-FEATURE | **expensive** — needs per-object device state the split seam does not carry |

**Decision — ESCALATED under G4, with the split G4 explicitly anticipates.** The two cheap
knobs are a viable standalone track; `per_object_feature_penalties` needs per-object device
state and should be gated separately so it cannot hold the other two hostage.

---

## T31 — eval sets on device (Track V) · **GATE G5: assessment**

**State.** `eval_sets.is_empty()` is an unrelaxed clause. The PLAN's V-9 established that
the update path is `last_tree_eval_contribution` over the folded tree ensembles the device
branch already populates, and that the device branch merely `continue`s past the eval block.

**G5's trigger is "question (3) forces device-side eval CTR materialization".** For a
FLOAT-only eval set it does not: the eval predictions can be computed host-side from the
already-folded tree, exactly as the CPU branch does, because the device branch produces the
same `TrainedModel` tree structures. For an eval set on a CTR fit it DOES — the eval
objects' CTR values must be materialized under the same permutation, which is device-side
state.

**Decision — PARTIAL, not blocked.** The float-only case looks genuinely tractable and is
NOT past G5's trigger; the CTR × eval-set case is. Recommend scoping T32 to float-only eval
sets with an explicit `eval_sets_device_covered` clause that declines whenever the fit has
CTR columns. That is a real capability increment with a clean, honest boundary.

---

## T33 / T34 / T35 — pairwise / ranking / multi-output (Tracks W / R / X)

**T33 (pairwise) — GATE G6: UNCONDITIONAL STOP.** G6 fires regardless of estimate. The
missing half is GPUT-21, a wholly unexecuted named plan: the batched Cholesky SOLVE for the
`(leaf_count-1)×(leaf_count-1)` pairwise system. `map_pairwise_coverage` and the per-leaf
matrix assembly exist; the solve does not. Do not start T36.

**T34 (ranking).** `map_ranking_coverage` exists. `FamilyTreeArgs::Ranking { group_offsets }`
is now available to carry the CSR group prefix across the seam — which was the missing
plumbing. The remaining question is whether the grouped der (`compute_gradients_grouped`,
already a host-side default) can feed the resident grow without a per-group device
reduction. Not assessed further; no gate fires, but it is downstream of Track O's segmented
machinery and should follow it.

**T35 (multi-output) — BLOCKER B-4 stands.** `Loss::RmseWithUncertainty` starts from a
PER-DIMENSION approximant (`rmse_uncertainty_starting_approx`), which the scalar
`DeviceTrainConfig.bias` this phase added CANNOT express. `approx_dimension == 1` currently
makes it unreachable. **Any relaxation of that clause must first decide how a per-dimension
bias crosses the seam** — `FamilyTreeArgs::MultiOutput { approx_k, approx_dim }` can carry
the per-tree approximant, but `bias` is a `begin`-time scalar and would need to become a
slice. Recorded so this is decided before, not discovered at, oracle time.

---

## Summary

| track | gate | decision |
|---|---|---|
| O — Ordered | G1 | **ESCALATED** — needs a net-new per-segment scoring kernel |
| U — non-Borders CTR | G2 | **ESCALATED** — three distinct new accumulation semantics |
| M — monotone | G3 | **ESCALATED** — leaf clamping, not a candidate filter; most tractable next |
| N — penalties | G4 | **ESCALATED, split** — 2 cheap knobs viable; per-object needs new device state |
| V — eval sets | G5 | **PARTIAL** — float-only tractable; CTR × eval-set is past the trigger |
| W — pairwise | G6 | **ESCALATED (unconditional)** — GPUT-21 Cholesky solve unexecuted |
| R — ranking | — | not gated; downstream of Track O |
| X — multi-output | B-4 | **BLOCKED on a design decision** — per-dimension bias vs scalar `config.bias` |

No Wave-7 implementation task (T22, T23, T24, T26, T28, T30, T32, T36, T37, T38) was
started. Every track above has either a fired gate or a recorded blocker, which is the
PLAN's defined completion state for this wave.
