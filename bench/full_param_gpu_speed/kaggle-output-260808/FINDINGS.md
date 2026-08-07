# SPD-03 host-prep re-measurement — the fix did NOT deliver the predicted win

**Branch:** `worktree-gpu-host-prep-speed` @ `acefb65de9816648115a0a5f6093dc6883373a65`
**Runners:** Kaggle, account `boomvector` — full grid on **Tesla P100-PCIE-16GB**; a follow-up
diagnostic on both a **Tesla T4** and, matching the grid, the **same P100 class**.
**Verdict: the claim "host-prep fixes should flip the 1M-row comparison" is WRONG.** The
1M-row cells are still ~2.2× slower than official CatBoost GPU, essentially unchanged from
before the fix. The reason is a measurement error in how the fix was validated, not a defect
in the fix's individual pieces.

---

## 1. The claim under test

Three commits (`dcce270`, `3a76a8f`, `d10501f`) attacked the three fixed host-side setup
costs identified from an earlier Colab T4 run (`bench/full_param_gpu_speed/colab-t4-260807/`):

| stage | Colab T4 (pre-fix) | **claimed**, gfx1151 2-vCPU emulation |
|---|---|---|
| `fit-prep copy+borders` | 1374–2290 ms | **~300 ms** |
| `quantize` | 380–732 ms | **0 ms** |
| `begin` (upload) | 568–945 ms | **61–71 ms** |

`d10501f`'s commit message summarized this as a near-order-of-magnitude cut to fixed setup
cost (~2.3–4.0 s → ~0.36 s), and the prior turn in this session extrapolated from it that the
1M-row comparison against official CatBoost GPU (previously 0.44–0.48×) should flip to
catboost-rs being faster. **That extrapolation was never verified on real GPU hardware before
being stated, and it is wrong.**

## 2. What was actually measured

### 2.1 The real grid, Tesla P100

`kaggle-output-260808/result.json`, 5 device-activated cells (`device_prof_tree_lines == 2`,
confirmed — not assumed):

| cell | official (GPU) | catboost_rs (GPU) | ratio official/rs |
|---|---|---|---|
| `SymmetricTree\|RMSE\|unw\|noctr\|300k` | 1.238 s | 1.312 s | 0.944 |
| `SymmetricTree\|Logloss\|unw\|noctr\|300k` | 1.466 s | 1.359 s | **1.078** |
| `SymmetricTree\|RMSE\|unw\|noctr\|1000k` | 1.708 s | 3.840 s | **0.445** |
| `SymmetricTree\|Logloss\|unw\|noctr\|1000k` | 1.840 s | 4.047 s | **0.455** |
| `SHOWCASE-bias\|RMSE\|unw\|noctr\|300k` | 1.238 s | 1.263 s | 0.980 |

300k rows: at parity (0.94–1.08×) — a genuine improvement over the pre-fix Colab T4 reading
of 0.86–0.92× at the same row count.

1000k rows: **0.445× / 0.455×** — statistically indistinguishable from the pre-fix Colab T4
reading of 0.44–0.48×. The fix bought nothing measurable at this scale.

### 2.2 Why: the local validation measured the wrong device

A targeted `CB_GPU_PROF` diagnostic replicated the exact failing cell (1M × 50, RMSE,
SymmetricTree, depth 6, 30 iterations) directly on Kaggle, once on a T4 and once on the same
P100 class the grid used (raw logs: `diag/diag_output_p100.txt`, `diag/diag_output_t4.txt`).

| stage | claimed (gfx1151, 2-vCPU emulation) | **measured, Tesla T4** | **measured, Tesla P100** |
|---|---|---|---|
| `fit-prep copy+borders` | ~300 ms | 561 ms | 792 ms |
| `quantize` | 0 ms | *(folded into begin-raw)* | *(folded into begin-raw)* |
| `begin` / `begin-raw` | 61–71 ms | **1444 ms** | **1222 ms** |

`fit-prep` improved for real, just less than the local number implied (792 ms vs. a claimed
~300 ms — still a genuine ~40–65% cut from the pre-fix 1374–2290 ms).

`begin-raw` — which now folds QPACK-01's on-device quantize+pack into the session-begin
step — is **1222 ms on the real P100**, not 61–71 ms. That is off by a factor of **~17–20×**.

**This is the root cause.** Compare the OLD and NEW costs as combined windows, not as
individual line items (the reorganization moved work between stages, so per-stage comparison
is misleading; combined-window comparison is not):

- OLD: `quantize` (380–732 ms) + `begin` (568–945 ms) ≈ **950–1680 ms** combined
- NEW: `begin-raw` alone ≈ **1222 ms** on the P100 that matters

**That is a wash, not a win.** The on-device quantize+pack kernel and its launch/allocation
overhead cost roughly what the two host-side stages it replaced cost combined. The local
gfx1151 rig could not have caught this: it was throttled to 2 vCPUs to emulate the *host*
side of a Kaggle/Colab session, but gfx1151 is an integrated ROCm APU with device-launch and
memory-transfer characteristics that do not resemble a discrete Kaggle P100's. Throttling host
threads emulates host contention; it does not emulate a different GPU.

### 2.3 A second, larger gap the diagnostic surfaced — still unattributed

`fit-train` (the timer around the whole `train_with_eval_sets` call, starting strictly after
`fit-prep` finishes — confirmed by reading `crates/catboost-rs/src/builder.rs:1176` against
where the `fit-prep` timer is read at line ~1122) reports **5276 ms** on the P100 run. But the
stages it should contain only sum to:

```
begin-raw            1222 ms
30 × tree stage       1926 ms   (sum of every "fill+derive+score+split+stats_read+leaf_apply_der")
----------------------------
accounted             3148 ms
fit-train (measured)  5276 ms
UNACCOUNTED           2128 ms  (~40% of fit-train)
```

This is not explained by anything currently instrumented. It is NOT the fit-prep stage (that
runs before the `fit-train` timer starts, and is logged separately). It is inside
`train_with_eval_sets` — model-assembly overhead, per-iteration bookkeeping outside the device
calls, or queue-drain waits that never surface as their own `CB_GPU_PROF` line are all
plausible; none is confirmed. **This report does not claim a cause for the 2128 ms gap** —
only that it exists, is large, and is currently invisible to the profiling this repo has.

(A smaller, separately-explained gap: `WALLCLOCK_FIT_SECONDS` = 6892 ms vs.
`fit-prep` + `fit-train` = 792 + 5276 = 6068 ms, a ~824 ms residual plausibly attributable to
Python/PyO3 pool ingestion before the Rust `fit-prep` timer starts. Not investigated further
here since it is much smaller than the in-Rust gap above.)

## 3. What this report is NOT claiming

- It is not claiming the `fit-prep` fix (`dcce270`) is wrong — it measurably helped, just less
  than the uncalibrated local number suggested.
- It is not claiming QPACK-01 (`3a76a8f`) is incorrect — its correctness oracle
  (`device_quantize_pack_test`, bit-exact vs. the host pipeline) is a separate concern from its
  *speed*, and this report only speaks to speed.
- It is not claiming a regression — the pre-fix 1M ratio and the post-fix 1M ratio are the
  same within measurement noise; nothing got slower.
- It is not extrapolating a fix for the P100 begin-raw cost or the 2128 ms `fit-train` gap.
  Both need their own profiling pass; this report's job was to correct the record, not to
  re-solve the problem in the same sitting.

## 4. What remains, precisely

1. **The 1M-row comparison against official CatBoost GPU is still ~2.2× UNFAVORABLE** on real
   Kaggle P100 hardware. The `SHOWCASE`/300k-row cells are fine; nothing above 300k rows is.
2. **QPACK-01's on-device quantize+pack cost (~1.2 s at 1M×50) needs its own targeted
   profiling on real GPU hardware** — which kernel(s) inside it dominate, and whether the
   design can be cheaper, is unknown. Local gfx1151 numbers must not be used to validate this
   again without an explicit disclaimer that they do not transfer to discrete cloud GPUs.
3. **A 2128 ms (~40% of `fit-train`) gap inside `train_with_eval_sets` has no attribution.**
   Closing it needs new `CB_GPU_PROF` instrumentation, not another local measurement — it may
   itself turn out to be host/device-transfer-bound in a way gfx1151 cannot represent.
4. **Any future local (gfx1151/ROCm) measurement of this fit-prep/begin/quantize surface must
   be treated as a lower bound on real hardware, not a predictor of it**, until proven
   otherwise by a cloud re-run. This is the operational lesson of this report.

## Artifacts

- `result.json`, `report.md`, `catboost-rs-hostprep-gpu-speed.log` — the full grid run
  (Tesla P100, commit `acefb65`).
- `diag/diag_output_t4.txt` — targeted 1M×50 RMSE `CB_GPU_PROF` profile, Tesla T4.
- `diag/diag_output_p100.txt` — the same, Tesla P100 (the device that matters for §2.2/§2.3).
