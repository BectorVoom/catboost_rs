# Parameter-wireup wave — GPU speed

`bench.py` measures the parameters this wave added that can plausibly cost time, against
official CatBoost `task_type='GPU'` on the same recipe.

Most of the wave's parameters cost nothing measurable — `random_score_type` is inert at the
default `random_strength = 0`, the CTR-mode pair and the logging family do not touch the
grow loop. Three groups can move the clock, and they are the whole grid:

| group | cells | why it is here |
|---|---|---|
| **A. device-decline cliff** | `baseline`, `model_shrink_rate=0.2`, `leaf_estimation_iterations=3` | both parameters route the fit to the **CPU grower**, and neither looks like it would |
| **B. border-build cost** | all 7 `feature_border_type` values | border building is HOST prep, and the exact-DP types (`MinEntropy`, `MaxLogSum`) are asymptotically dearer than the greedy heap |
| **C. nan_mode control** | `Min`, `Max` on a NaN-bearing pool | expected free; measured *because* it is expected free |

## Why group A is the point

`model_shrink_rate != 0` and `leaf_estimation_iterations > 1` are ordinary-looking knobs. A
user reaching for either has no reason to expect a *path* change — but
`device_host_eligible` (`crates/cb-train/src/boosting.rs`) declines both, and the fit
silently falls back to the CPU grower:

- **`model_shrink_rate`** — the shrink rescales the running approx every iteration, but the
  device keeps its approx **resident** and never reads the host copy back per tree, so a
  host-side rescale would be dropped.
- **`leaf_estimation_iterations`** — the accumulate-and-recompute loop lives in the CPU
  leaf-value section, which the device branch `continue`s past. Committing would take ONE
  step and ignore the parameter.

Both declines are correct — the alternative is a wrong model — and both are asserted in
`crates/cb-train/tests/string_param_device_routing_test.rs`. What is NOT documented anywhere
a user would look is what the decline **costs**. That is this grid's headline number.

## Disciplines

Inherited unchanged from `bench/full_param_gpu_speed`:

1. **Device activation is observed, not assumed** — every cell runs a short fit under
   `CB_GPU_PROF=1` first and counts `CB_GPU_PROF tree` lines.
2. **Both sides get the same explicit recipe.**
3. **No proxying** — a recipe official CatBoost GPU cannot express is `N/A` with the reason.
4. **Spread before headline** — median/min/max over 3 repeats; a ratio range spanning 1.0 is
   *within noise*, never a win or a loss.
5. **No invented numbers** — a failed build or cell yields an error row.

Added here:

6. **Activation is asserted in BOTH directions.** A cell expected to commit that shows no
   tree lines is a harness failure — that discipline already existed. A cell expected to
   **decline** that shows tree lines is *also* a harness failure, because it means
   `device_host_eligible` no longer matches what the routing tests assert, and the cell's
   whole interpretation is void. Without this, a regression that made a decline cell start
   committing would show up as a pleasing speedup.

`GPU_UNSUPPORTED_BORDER_TYPES` is deliberately **empty**. An earlier draft declared
`GreedyMinEntropy` GPU-unsupported, but that cannot be checked from the development box — a
ROCm rig with no CUDA driver, where *every* `task_type='GPU'` call fails identically
regardless of the parameter — so the claim would have been an assumption dressed as data,
and a wrongly-declared N/A silently DROPS an official arm that would otherwise have run.
Rejections are discovered on the GPU box and reported with their real message.

## Running it

```bash
# 1. correctness on local hardware first
cargo test -p cb-train --no-default-features --features rocm \
    --test string_param_device_routing_test

# 2. review the grid without a GPU
python bench/param_wave_gpu_speed/bench.py --dry-run

# 3. preflight BOTH arms' kwargs against the real surfaces before spending a session.
#    This is not optional ceremony: it is what caught `leaf_estimation_iterations` being
#    implemented in the engine but absent from the Python surface's IMPLEMENTED list, a
#    seam no Rust-side oracle could see. A kwarg typo discovered on the GPU box costs the
#    whole ~25-minute build.

# 4. on a Colab T4
colab new -s param-wave-speed --gpu T4
# upload the tree under test (git archive HEAD — NOT a clone; the branch is local),
# install rustup, then:
#   CB_BENCH_OUT=/content/bench_out python bench/param_wave_gpu_speed/bench.py
# it writes {report.md, result.json}

# 5. commit the artifacts as bench/param_wave_gpu_speed/colab-t4-<date>/
#    and append the dated block to bench/RESULTS.md
```

### Colab gotcha

A session can be lost mid-run (`404/401`, "appears to be lost") — the orphaned-runtime
failure. Detaching the build with `Popen` and polling it through repeated `colab exec` calls
made this worse; running the build **synchronously inside one long exec** is the shape that
survives. Colab images also do not preinstall official catboost (Kaggle's do), which the
harness handles.
