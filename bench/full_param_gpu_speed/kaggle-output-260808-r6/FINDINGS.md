# SPD-03 wave 6 — the border-sample fix lands: 1.7–3.3× faster than official CatBoost GPU on every device-active cell

**Branch:** `worktree-gpu-borders-shared-sample` @ `935030e` (grid provenance)
**Runner:** Kaggle, account `boomvector`, **Tesla P100-PCIE-16GB** — the same grid,
protocol and hardware class as every earlier SPD-03 round (untimed in-process warm
fit per arm, then median of 3 timed repeats, device activation probe-confirmed).

## The verdict

`kaggle-output-260808-r6/result.json`, all five device-activated cells:

| cell | official (GPU) | catboost_rs (GPU) | ratio official/rs | r5 ratio |
|---|---|---|---|---|
| `SymmetricTree\|RMSE\|unw\|noctr\|300k` | 1.223 s | 0.435 s | **2.81× — WIN** | 1.77× |
| `SymmetricTree\|Logloss\|unw\|noctr\|300k` | 1.434 s | 0.460 s | **3.12× — WIN** | 2.12× |
| `SymmetricTree\|RMSE\|unw\|noctr\|1000k` | 1.661 s | 0.992 s | **1.68× — WIN** | 1.05× |
| `SymmetricTree\|Logloss\|unw\|noctr\|1000k` | 1.838 s | 0.987 s | **1.86× — WIN** | 1.18× |
| `SHOWCASE-bias\|RMSE\|unw\|noctr\|300k` | 1.243 s | 0.376 s | **3.30× — WIN** | 2.09× |

Progression of the 1M RMSE cell across the rounds, identical protocol:
**0.445× → 0.70× → 0.91× → 1.05× → 1.68×**; 1M Logloss:
**0.455× → 0.74× → 1.03× → 1.18× → 1.86×**. catboost_rs' own 1M fit went
1.706 s → **0.992 s** in one wave; the sub-second 300k fits (0.38–0.46 s) are
now ~3× ahead of official.

## What this wave changed (one commit, `935030e`)

**Shared sorted border sample.** r5's largest host term was fit-prep
(785 ms wall / 3078 ms CPU at 1M — ~all border building). The over-cap 200k
border subsample is drawn from a FIXED-seed `TFastRng64` stream parameterized
only by the object count — so every one of the 50 columns drew the **same**
index set, yet each column rebuilt an O(n) identity permutation (8 MB at 1M
rows), re-ran the 200k-swap partial Fisher–Yates over it, and gathered in
random draw order. The fix draws the index set ONCE per fit, sorts it ascending
(the per-column gather becomes a forward streaming read), and shares it across
columns (`cb_data::sample_indices_for_build_borders` +
`select_borders_greedy_logsum_f32_presampled`).

Border sets are **byte-identical**: the gathered multiset per column is
unchanged (same index set, same subset-before-NaN-drop ordering) and the
sort + greedy tail is a pure function of that multiset (the `{-0.0, +0.0}`
order-insensitivity argument already on file). Pinned by a differential test
that reproduces the pre-wave-6 draw-order algorithm verbatim
(`presampled_borders_match_draw_order_reference`), plus the existing
adversarial-sort and determinism border tests; the full cb-data, catboost-rs
facade and cb-oracle suites pass unchanged.

On-hardware attribution (`diag/diag_repeat.txt`, 1M cell): fit-prep
**785 → 94 ms** wall, borders CPU-time **3078 → 311 ms**; diag wallclock
2.20 → 1.44 s. The timed cells gained more than the prep delta alone —
relieving four-thread memory contention in prep also stopped it bleeding into
the neighboring stages (qpack-fill 223 → 180 ms on the same diag).

## What remains (honest residuals)

- The 1M repeat-process profile is now: py-ingest 307 ms, fit-prep 94 ms,
  begin 213 ms (qpack-fill 180 ms of it), fit-train 1039 ms. The largest
  remaining host term is **py-ingest** (the row-major → SoA f32 transpose,
  already cache-blocked in wave 5). The structural next step, if ever needed,
  is uploading the row-major NumPy buffer directly and letting the device
  transpose/quantize — it would also let border sampling gather whole sampled
  ROWS (contiguous) instead of columns. Not attempted: every device cell
  already wins by ≥1.68×.
- Cold first-fit (fresh machine, empty JIT cache): 2.40 s fit wallclock at 1M
  (`diag/diag_cold.txt`) — down from 4.0 s in r5, now in the same band as
  official's cold fit.
- Weighted / CTR / Depthwise cells still decline to CPU or N/A — device
  coverage of those is tracked elsewhere (`gpu-string-param-oracle-coverage`);
  nothing here changes their status.

## Protocol notes

- Same fairness as every round: both arms JIT-warm via the untimed in-process
  warm fit; explicit `bootstrap_type` recipe on both sides; device activation
  observed via `CB_GPU_PROF tree` lines, never assumed.
- The diag logs carry the per-stage `CB_GPU_PROF` attribution for the next
  investigation.
