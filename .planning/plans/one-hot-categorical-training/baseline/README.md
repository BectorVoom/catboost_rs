# Workspace test baseline (T00 / SPEC-OH-31)

**PLAN_BASE_SHA: `9bf734512d7fccb25a9e8304b34183375ae3e7f5`**
(also in `PLAN_BASE_SHA.txt`; transcript in `workspace-test-baseline.txt`)

Captured with `cargo test --workspace` on the **pre-change** tree, before any
one-hot production edit. The transcript contains **zero compile errors**, which
confirms it compiled the plan-base source rather than a partially-edited tree.

## Result

```
EXIT=101
aggregate: 297 passed, 59 failed
```

The 59 failures are **all in one target** — `cb_backend` lib tests
(`Running unittests src/lib.rs (target/debug/deps/cb_backend-…)`, 184 passed /
59 failed). **Zero failures anywhere else in the workspace.**

## Why T29's bar is "no NEW failing target", not "full green"

`cargo test --workspace` cannot be green on this machine at the plan base. The
`cb-backend` device-kernel tests execute under CubeCL's **cpu** backend (the
workspace default feature), where the kernel lowering is known-broken — the
pre-existing condition recorded in project memory as the cb-backend CubeCL/MLIR
failure. A plan task that demanded a fully green workspace would be unachievable
for reasons entirely unrelated to one-hot.

So the accepted gate is: **no target that passes here may fail later, and no new
failing test may appear in the 59 below.**

## KNOWN DEFECT IN THIS CAPTURE — it is INCOMPLETE

The capture ran `cargo test --workspace` **without `--no-fail-fast`**, so cargo
stopped after the first failing target (`cb_backend`) and **never ran the
`cb-train`, `cb-model`, `catboost-rs`, or `catboost-rs-py` targets at all**. The
transcript therefore enumerates the accepted failures only up to and including
`cb_backend`; it is NOT a complete workspace baseline.

Consequence discovered during T04's regression sweep: `cb-train`'s
`monotone_oracle_test::monotone_non_symmetric_and_region_are_typed_errors` fails
(`grow_policy=Region must be rejected with a typed error (D-6.6-04 "Region OUT")`),
but appears NOWHERE in this transcript, so the transcript alone cannot classify it.

**It was verified pre-existing by direct experiment** — `git stash`ing every Wave-0/1
change and re-running `cargo test -p cb-train --test monotone_oracle_test` on the
clean tree reproduces the identical failure. It matches the known `cb-train monotone`
breakage recorded in project memory. **Accepted, and not caused by this plan.**

Any future task consuming this baseline must therefore:
1. Re-capture with `cargo test --workspace --no-fail-fast` for a genuinely complete
   accepted-failure set, **or**
2. Classify a failure outside the `cb_backend` set by the same stash-and-rerun
   experiment rather than by absence from this transcript.

Absence from this file means "not observed", **not** "newly broken".

## Accepted failures beyond the transcript

| test | verdict | how classified |
|---|---|---|
| `cb-train monotone_oracle_test::monotone_non_symmetric_and_region_are_typed_errors` | pre-existing, accepted | stash + rerun on the clean tree reproduces it |

## The accepted failing set (59, all `cb_backend` lib tests)

| module | count |
|---|---|
| `kernels::scan` | 9 |
| `kernels::pointwise_hist` | 7 |
| `kernels::pairwise_hist` | 7 |
| `kernels::sort` | 5 |
| `kernels::score_split::variants` | 5 |
| `kernels::exact_quantile_test` | 5 |
| `kernels::segmented_sort_test` | 4 |
| `kernels::cindex` | 4 |
| `kernels::partitions` | 3 |
| `kernels::score_split::pairwise` | 2 |
| `kernels::score_split` | 2 |
| `kernels::reduce` | 2 |
| `kernels::grow_loop::partition` | 2 |
| `kernels::score_split::scan` | 1 |
| `kernels::grow_loop::pairwise` | 1 |

Every entry is under `kernels::` — **none is under `gpu_runtime::`**, and none is
outside `cb-backend`. Representative failures:

- `kernels::exact_quantile_test::…` — `device=1.999999 cpu=-0.499999
  abs_div=2.500e0 > 1e-4` (device/CPU divergence under the cpu backend).
- `kernels::cindex::pack_read_bit_exact_large_n` — `large-n pack->read must be
  bit-exact` (`kernels/cindex.rs:196`).

## Note for T00's device capture and T29b

The failing `kernels::cindex::pack_read_bit_exact_*` tests are in
**`kernels/cindex.rs`** — a *different* module from
**`gpu_runtime/cindex.rs::pack_cindex`**, which is the pure host-side bit-packer
T00's device capture uses. The capture is therefore not built on a known-broken
path, but this proximity is worth re-checking if the captured artifact ever looks
wrong: confirm which `cindex` is in play before trusting a diff.

## THIS BASELINE IS FROZEN

Do not regenerate it after a production change — that would silently re-accept
regressions introduced by this plan. Compare against it; never refresh it.
