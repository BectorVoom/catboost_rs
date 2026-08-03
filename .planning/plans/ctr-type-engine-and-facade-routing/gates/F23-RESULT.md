# F23 — workspace baseline gate (acceptance A12)

Run: `cargo test --workspace --no-fail-fast` at the tip of Part 2.
Transcript: `./F23-workspace-transcript.txt`.
Baseline: `.planning/plans/one-hot-categorical-training/baseline/workspace-test-baseline.txt`.

## Result

```
1669 passed, 60 failed across 161 targets
failing targets: 2
```

## The gate

**No target that passes in the baseline fails here.** Diffing the distinct
failing-test sets (`comm -13 baseline now`) leaves exactly ONE name:

```
monotone_non_symmetric_and_region_are_typed_errors
```

which is a **known-accepted pre-existing failure**, named as such in
`PLAN-PART2.md` F23 ("absent from the self-documented incomplete transcript")
and recorded independently in project memory. Verified unrelated:

- its assertion is `grow_policy=Region must be rejected with a typed error
  (D-6.6-04 "Region OUT")` — a grow-policy concern with no categorical content;
- `git log ca4418d..HEAD -- crates/cb-train/tests/monotone_oracle_test.rs`
  returns **0 commits**: Part 2 never touched the file or its subject.

The other 59 failures are the accepted `cb_backend` lib set (191 passed / 59
failed), byte-for-byte the same 59 as the baseline — the CubeCL **cpu**-backend
kernel-lowering condition the baseline README documents. Passing count rose from
184 to 191 because targets were added since the baseline was captured.

## One-hot wave (A12's named targets)

`one_hot_oracle_test`, `one_hot_draw_accounting_test`, `device_one_hot_parity_test`
— all green.

## Clippy

`cargo clippy --workspace --all-targets` reports errors only in
`crates/cb-backend/**` (4, pre-existing) and
`crates/cb-oracle/src/bin/write_skeleton.rs` (2, pre-existing —
`git log ca4418d..HEAD` on that file returns 0 commits). **Zero clippy errors in
any crate Part 2 modified.**
