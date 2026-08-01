# One-hot categorical GPU speed gate (SPEC-OH-30 / plan T30)

`one_hot_bench_colab.py` is the Colab-T4 runner for SPEC-OH-30: on a **matched** config,
catboost-rs must train a one-hot categorical workload faster than official CatBoost with
`task_type='GPU'`, with device activation **observed** rather than assumed.

## Status: BLOCKED — not yet run

The runner exists and is complete, but it **refuses to produce a number** on the current
tree, by design. Its preflight check fails with `BLOCKED-FACADE-ROUTING`:

> The Rust/Python facade does not route `cat_features` into training.
> `crates/catboost-rs/src/builder.rs` calls `cb_train::train`, not `cb_train::train_cat`,
> and pins `one_hot_max_size` to the upstream default with the in-source comment
> *"the facade does not yet surface categorical config"*.

A `.fit(X, y, cat_features=[...])` through the public Python surface would therefore train
a **float-only** model while appearing to measure one-hot training — the worst possible
outcome for a benchmark. Quoting that number would be worse than quoting none.

That routing is a separate, explicitly-not-yet-executed plan (commit `41e7e9c`,
*"cat_features/CTR facade routing — spec+plan (blocked, not yet executed)"*). Once it
lands, the preflight starts passing and this runner needs no change.

**What this does NOT block.** The one-hot device path itself is verified independently of
the facade, through `cb_train::train_cat` directly:

- `crates/cb-train/tests/device_one_hot_parity_test.rs` — device vs CPU grower ≤ 1e-5 on
  three scenarios, with on-device residency and one-hot-split-presence both asserted.
- `crates/cb-backend/src/gpu_runtime/one_hot_{split_score,partition_split,session_wiring}_test.rs`
  — the scorer's equality fold, the split application's equality routing, and the seam
  carrying true per-feature cardinalities.

Only the end-to-end *speed* claim is pending.

## Running it (once unblocked)

```bash
# correctness on local hardware first
cargo test -p cb-train --no-default-features --features rocm --test device_one_hot_parity_test

# then a Colab T4
~/.local/bin/colab new -s onehot-speed --gpu T4
# stage the working tree at /content/cbrs (NOT a git clone — the point is to measure the
# tree under test), then run one_hot_bench_colab.py; it writes
# /content/bench_out/{report.md,result.json,run.log}

# commit the artifacts as bench/one_hot_gpu_speed/colab-t4-<date>/
```

Kaggle (`boomvector`) is the fallback runner; `kernel-metadata.json` follows the
`bench/bootstrap_gpu` pattern.

## What the gate requires

`result.json` must show, for BOTH the RMSE and Logloss arms:

- `speedup_official_catboost_gpu > 1.0`
- `activation_observable == true` (a `CB_GPU_PROF tree` line was actually emitted)

and every knob pinned identically on both sides, with the official side's
`get_all_params()` read back into the report.

`bootstrap_type='No'` and `random_strength=0` are **constrained, not chosen**: catboost-rs
typed-rejects one-hot training with either active (SPEC-OH-27 / T01b Branch B). Both sides
are pinned to the same draw-inert config, which also removes the inflation risk of
comparing against official CatBoost's GPU default (`Bayesian`, strictly more work per
tree).
