---
phase: 05
slug: ordered-boosting-ordered-ctr-categoricals-high-risk-parity-s
status: secured
threats_open: 0
threats_total: 36
threats_closed: 36
asvs_level: 1
block_on: high
created: 2026-06-14
---

# SECURITY.md — Phase 5: Ordered Boosting / Ordered CTR / Categoricals

**Audit date:** 2026-06-14
**ASVS Level:** 1
**block_on:** high
**Disposition:** SECURED — 36/36 threats closed (24 mitigate verified in code, 12 accept rationale confirmed)

## Trust boundaries (this phase)

This is a pure-compute ML library: no network, no auth, no web surface, no
accounts, no secrets. The relevant boundaries are:

1. Untrusted serialized model bytes (`.cbm` binary blobs / `model.json` with
   `ctr_data`) crossing into parsers at load/inference.
2. Committed test fixtures (`.npy`) read by oracle tests (frozen, offline-generated).
3. Internal index/bounds safety in hot numeric loops (permutation indices,
   body/tail boundaries, leaf indices, CTR bucket indices).

Dominant mitigation pattern: checked `.get`/`.get_mut`, no raw indexing,
out-of-range → typed error (`CbError::Degenerate` / `ModelError` / `OracleError`).
CLAUDE.md hard rule (`unwrap()` prohibited in production) holds: no
`unwrap()`/`expect()`/`panic!`/`unreachable!`/`todo!` in any of the 11 verified
production files (only infallible `unwrap_or*` defaults).

## Mitigate threats — verified in code (24/24 CLOSED)

| Threat ID | Category | Status | Evidence (file:line) |
|-----------|----------|--------|----------------------|
| T-05-01-01 | Tampering | closed | model_json.rs:174 `#[serde(default)] ctr_data`; malformed→`OracleError::MalformedModel` 131,143,244; load returns Result 255-259 |
| T-05-02-01 | Tampering | closed | candidates.rs:121-129 `ph.remap()?`→`CbError::OutOfRange`, `CbResult`; no raw index |
| T-05-02-02 | DoS | closed | tree.rs:927/947 one-hot enum over `cat_bins.get`/bounded; `check_depth`/DepthExceeded; no unbounded alloc |
| T-05-03-01 | DoS | closed | fold.rs:117 `min(...,n)` cap; 121-125 monotonicity→`n`; 170 `max(1, saturating_sub)`; checked arith |
| T-05-03-02 | Tampering | closed | permutation.rs:134 `v.swap(i,j)` checked; `j=uniform(i+1)∈[0,i]`<n; no raw index |
| T-05-04-V5 | Tampering | closed | ctr_data.rs:469-482 `BlobReader::take` checked_add+`get(pos..end)`→`ModelError`; 534-542 `bounded_len` cap; unknown type 104-106 |
| T-05-04-01 | Tampering | closed | ctr_data.rs:176 `bucket_for_hash→Option`, 184/190 checked `counts_at`/`mean_at`, 230/242/249/257 not-found→empty; apply.rs:225 `leaf_values.get(leaf).unwrap_or(0.0)` |
| T-05-04-02 | DoS | closed | ctr_data.rs:532 `MAX_DECLARED_LEN` cap pre-alloc; 617 `min(MAX)`; i64 checked decode |
| T-05-05-01 | Tampering | closed | online.rs:200,209,297,312-313,393,396-397 checked bucket `.get`/`.get_mut`; OOR perm→`Degenerate` 287,291,383,387 |
| T-05-05-02 | Info Disclosure | closed | online.rs:298-301 read-before-increment 311-316; ordered 393-395 read before 396-400; per-bucket monotone anchor 425-455 |
| T-05-05-03 | DoS | closed | fold.rs:117 tailFinish capped at n; bounded iter; online.rs walks fixed bins/perm length |
| T-05-06-V5 | Tampering | closed | apply.rs:159-190 `passes_ctr_split` combined-key via `ctr_value_for_combined_projection`; not-found→empty 187; checked member `.get` 167 |
| T-05-06-01 | DoS | closed | projection.rs:213 `max_ctr_complexity.min(cat_feature_count)`; 138-140 `full_projection_length` gate; checked `.get`/`.get_mut` |
| T-05-07-V5 | Tampering | closed | compare.rs:128-135 `compare_permutation` length-check→`PermutationLengthMismatch`; ordered_ctr_oracle_test.rs:94-95 `len==FIXTURE_N` |
| T-05-08-01 | Tampering | closed | tree.rs:622-636 `permutation.get`/`leaf_of.get`/`der1.get`→`Degenerate`; 627 negative idx→`Degenerate`; 641-648 `weight.get` |
| T-05-08-02 | DoS | closed | tree.rs:614 `upper=tail_finish.min(n)`; 698 `scale_l2_reg` guards body_finish==0 (leaf.rs:100-103 returns l2) |
| T-05-08-03 | Tampering | closed | tree.rs:655 `reduce_leaf_stats`; 710 `cb_core::sum_f64`; grep: no `iter().sum()`/`fold(0.0` in tree.rs |
| T-05-09-V5 | Tampering | closed | apply.rs:172-188 `tables.get(&key)` + combined-hash `bucket_for_hash`; absent table→empty 187; empty bucket→empty (ctr_data.rs:230/242/249) |
| T-05-09-01 | DoS | closed | projection.rs:206-220 `enumerate_projections` bounded by `max_ctr_complexity`; `full_projection_length` gate; checked arith |
| T-05-09-02 | Tampering/DoS | closed | ctr_data.rs:359-432 `from_json` panic-free serde, ragged/non-int→`ModelError`; bucket bounds-checked; no unwrap/raw-index |
| T-05-09-03 | Tampering | closed | model.rs:61-67 `ModelSplit{Float,Ctr}`; apply.rs:197-201 exhaustive match; json/shap/fstr/cbm use `as_float`/`float_feature` accessors (Ctr→None); `cargo check --workspace --tests` gate CONFIRMED passing |
| T-05-10-01 | Tampering | closed | boosting.rs:373/395 `permutation.get`; 377-381/399-403 `leaf_of.get`/`der.get`→`Degenerate`; 387/410 `leaf_sum_*.get_mut`; 419 `approx_delta.get_mut`; create_folds Fisher-Yates <n (permutation.rs:134) |
| T-05-10-V5 | Tampering | closed | ordered_boost_oracle_test.rs:78 `len==FIXTURE_N`; production load via `load_model_json`/`load_f64_vec`→typed error |
| T-05-10-02 | DoS | closed | boosting.rs:869 `create_folds` EXACTLY ONCE (grep-confirmed: sole production call site), BEFORE iteration loop (line 975); body/tail precomputed |

## Accept threats — rationale confirmed (12/12 CLOSED)

No new runtime input path and no new crate dependency was introduced this phase.
The Package Legitimacy Audit in each PLAN is N/A (no package installs). Fixture
generators are offline C++/Python harnesses (`crates/cb-oracle/generator/`) pinned
to `catboost==1.2.10`; `.github/workflows/ci.yml` invokes NO fixture generator and
installs NO catboost (grep-confirmed). Frozen `.npy`/`model.json` fixtures are never
a runtime input.

| Threat ID | Rationale |
|-----------|-----------|
| T-05-01-02 | Offline generator only, never in CI |
| T-05-07-01 | Offline generator only, never in CI |
| T-05-01-SC | No package installs / Package Legitimacy Audit N/A |
| T-05-02-SC | No package installs / N/A |
| T-05-03-SC | No package installs / N/A |
| T-05-04-SC | No package installs / N/A |
| T-05-05-SC | No package installs / N/A |
| T-05-06-SC | No package installs / N/A |
| T-05-07-SC | No package installs / N/A |
| T-05-08-SC | No package installs / N/A |
| T-05-09-SC | No package installs / N/A |
| T-05-10-SC | No package installs / N/A |

## Unregistered flags

None. The three SUMMARY files carrying a `## Threat Flags` section (05-06, 05-09,
05-10) each explicitly map their new surface to existing register IDs. No new
attack surface appeared during implementation without a threat mapping.

## Notes on deferred e2e fixtures (not security gaps)

05-09 (`tensor_ctr_e2e`) and 05-10 (`ordered_boost_e2e`) each have one
checkpoint-DEFERRED e2e oracle whose `catboost==1.2.10` fixtures are not
generatable in this environment. The SOURCE for those paths (apply-path CTR eval,
ordered train branch, load paths) IS committed and compiles, and every runtime
mitigation above is verified in that committed source. The deferred fixtures are
frozen offline test data, never a runtime input — they do not affect any runtime
security mitigation.

## Audit Trail

### Security Audit 2026-06-14
| Metric | Count |
|--------|-------|
| Threats found | 36 |
| Closed | 36 |
| Open | 0 |

State B run (no prior SECURITY.md). Register built from the `<threat_model>` blocks
of all 10 PLAN files (`register_authored_at_plan_time: true`) + SUMMARY threat
flags. `gsd-security-auditor` verified all 24 `mitigate` dispositions against the
implementation with file:line evidence and confirmed all 12 `accept` rationales; no
implementation files modified. No BLOCKER, no WARNING.
