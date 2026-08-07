# Full-parameter GPU speed grid (SPD-01/02/03)

Measures catboost-rs against official CatBoost `task_type='GPU'` across the parameter axes
the `gpu-full-parameter-parity` phase made device-reachable.

## Status

| task | state |
|---|---|
| SPD-01 (`bench/generator.py` axes) | **done** — `generate_cat` / `generate_weights` / `cat_driven_binary_target`, strictly additive |
| SPD-02 (this harness) | **done** — `--dry-run` verified GPU-free: 34 cells, both showcase cells ELIGIBLE, 28.6 min projected against a 540 min ceiling |
| SPD-03 (execute on Kaggle P100) | **BLOCKED — external** |

### Why SPD-03 is blocked

```
$ kaggle kernels push -p bench/full_param_gpu_speed
Kernel push error: Maximum weekly GPU quota of 30.00 hours reached.
```

The `yensen2` account's **weekly** 30-hour Kaggle GPU quota is exhausted (the last run,
`yensen2/cb-rs-gdc-device-coverage-p100`, completed the day before). This is a scheduling
limit, not a technical failure: nothing about the harness, the branch or the grid needs to
change. Re-push once the weekly quota resets.

Everything the run needs is committed and verified:

- `bench.py` — the grid, the eligibility audit, the `CB_GPU_PROF` residency probe, the
  3-repeat aggregation, the budget guard and the report writer.
- `kaggle_driver.py` — clones the branch (Kaggle kernels take one code file) and hands off
  to `bench.py`, recording the resolved commit SHA as provenance.
- `kernel-metadata.json` — `yensen2/catboost-rs-full-param-gpu-grid`, GPU + internet on.
- The branch `worktree-gpu-full-parameter-parity` is pushed, so the clone resolves.

**No speed number is reported until that run happens.** The phase's parity claims stand on
their own oracles; the supremacy claim does not exist yet, and inventing one from the local
gfx1151 rig would be worse than having none — official CatBoost's GPU trainer is CUDA-only,
so there is no official arm to compare against on ROCm, and a catboost-rs-on-ROCm vs
official-on-CPU number would not be the claim.

## Re-running

```bash
python bench/full_param_gpu_speed/bench.py --dry-run      # review the grid, no GPU needed
kaggle kernels push -p bench/full_param_gpu_speed         # spend a session
kaggle kernels output yensen2/catboost-rs-full-param-gpu-grid -p kaggle-output-<date>/
```

Then write `report.md` + `result.json` into `kaggle-output-<date>/` and append the dated run
block to `bench/RESULTS.md`.

## Disciplines the harness does not relax

1. **Device activation is observed, not assumed.** Every cell runs a short fit under
   `CB_GPU_PROF=1` first; a cell with zero `CB_GPU_PROF tree` lines is reported as a CPU row
   and excluded from any speed claim. (`bench/quick_gpu_speed` could only reason about
   eligibility statically and had to say so; this closes that gap.)
2. **Both sides get the same explicit recipe.** Official CatBoost's GPU default
   `bootstrap_type` is Bayesian — leaving it unset would compare catboost-rs (pushed to `No`
   by its gate) against an official run doing strictly more work per tree.
3. **No proxying.** A cell official CatBoost GPU cannot express is recorded `N/A` with the
   reason, never replaced by a different recipe.
4. **Spread before headline.** Every cell reports median/min/max over 3 repeats; a cell
   whose ratio spread crosses 1.0 is labelled *within noise*, never claimed as a win.
5. **Combination-CTR cells are absent by design.** FPP-11 is escalated and combination
   projections are device-ineligible, so such a cell would silently time a CPU fit.
