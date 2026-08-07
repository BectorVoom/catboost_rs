# SPD-03 attempt — Google Colab Tesla T4 (CUDA 12.8), 2026-08-07

**Verdict: NO SPEED CLAIM IS ESTABLISHED. The run is void as a comparison and is reported as
a defect-finding run instead.**

- **Runner:** Google Colab, Tesla T4 (15360 MiB), driver 580.82.07, CUDA 12.8
- **Why Colab and not Kaggle P100:** the `yensen2` weekly 30 h Kaggle GPU quota is exhausted
  (`Maximum weekly GPU quota of 30.00 hours reached`, retried twice this session). Colab
  offers T4/L4/A100/H100 — no P100. A T4 is still a *CUDA* device, which is the property the
  comparison needs: official CatBoost's GPU trainer is CUDA-only.
- **Tree under test:** `worktree-gpu-full-parameter-parity` @ `4bfc970`, cloned on the VM.
- **Artifacts:** `report-raw.md`, `result.json` (both as the harness emitted them).

## What happened

Every one of the 34 cells was labelled

```
DEVICE NOT ACTIVATED (0 CB_GPU_PROF tree lines) — reported as a CPU row
```

Under the harness's own discipline #1 that is correct behaviour for what it observed, and it
is why no speed number here may be quoted. **But the observation itself is wrong.**

## Root cause: the harness's residency probe crashes on every cell

`bench.py`'s `device_probe` writes a probe script ending in:

```python
m.fit(X, y, sample_weight=w)          # ← unconditional, even when w is None
```

`catboost_rs`'s sklearn surface is
`fit(x, y=None, cat_features=None, eval_set=None)` (`crates/catboost-rs-py/src/regressor.rs:54`,
and the same in `classifier.rs` / `ranker.rs`) — **there is no `sample_weight` parameter.** So
the probe subprocess dies with `TypeError: fit() got an unexpected keyword argument
'sample_weight'` before training a single tree, yields zero `CB_GPU_PROF tree` lines, and the
cell is labelled a CPU row. For all 34 cells.

### The device path is fine — proven directly

Run on the same VM, same wheel, `CB_GPU_PROF=1`, a 20000×10 RMSE fit:

```
CB_GPU_PROF begin covered=true n=20000 nf=10 n_bins=33 elapsed=288.51ms
CB_GPU_PROF tree n=20000 nf=10 bins=64 depth=6 fill=725.20ms derive=61.38ms score=140.71ms …
CB_GPU_PROF tree n=20000 nf=10 bins=64 depth=6 fill=0.62ms  derive=0.20ms score=1.39ms …
CB_GPU_PROF tree n=20000 nf=10 bins=64 depth=6 fill=0.47ms  derive=0.15ms score=1.29ms …
```

`covered=true`, real per-tree device lines. The CUDA wheel works on the T4; only the probe was
broken.

## Defects this run surfaced

| # | defect | kind | blast radius |
|---|---|---|---|
| D1 | `device_probe` passes `sample_weight=` unconditionally to an API that has no such parameter | **harness** | every cell mislabelled "CPU row" → the entire run void |
| D2 | `catboost_rs` sklearn `fit()` accepts no `sample_weight`; weights are reachable only via a `Pool` | **product (API parity)** | every weighted cell `N/A`; sklearn convention is `fit(X, y, sample_weight=…)`, and the project claims sklearn compatibility |
| D3 | `border_count=32` ⇒ 33 bins ⇒ `degenerate training input: pointwise_hist2 one-byte non-binary fill expects n_bins in {32,64,128,256} … got 33` | **product** | every `Depthwise` cell + `SHOWCASE-sampled-nonsym`. The oblivious path pads 33→64 and survives; the pointwise/nonsym fill rejects outright |
| D4 | CTR cells feed a float32 NumPy array plus `cat_features`, which **both** sides reject | **harness** | all 8 CTR cells `N/A` on both arms (official CatBoost errors too, so no proxying is possible) |

D3 is the most interesting product finding: the padding that rescues the oblivious grow is
absent from the nonsymmetric fill, so the two device growers disagree about which
`border_count` values are legal.

## What the numbers would have said (recorded, NOT claimed)

Since the probe lied, the timings that *did* record are `catboost_rs` **CPU** against official
CatBoost **GPU** — not a like-for-like comparison, and quoted here only so the raw data is not
silently discarded:

| cell | official (GPU) | catboost_rs (CPU fallback) |
|---|---|---|
| `SymmetricTree\|RMSE\|unw\|noctr\|1000k` | 3.239 s | 6.418 s |
| `SymmetricTree\|Logloss\|unw\|noctr\|1000k` | 2.998 s | 6.603 s |
| `SHOWCASE-bias\|RMSE\|unw\|noctr\|300k` | 1.591 s | 1.748 s |

**These are not a GPU-vs-GPU result and must not be cited as one.**

Two further confounders that would distort even a fixed run at this grid's settings:

1. **`border_count=32` ⇒ 33 bins ⇒ padded to a 64-wide line**, so the oblivious device fill
   does ~2× the histogram work it would at a true 32. The P100 runs that produced the earlier
   1.14–1.19× advantage did not carry this padding.
2. **JIT warm-up dominates at 30 iterations.** The first tree above cost 725 ms of fill against
   0.5–0.6 ms for every subsequent tree. Over a 30-iteration cell that is a large fixed cost
   attributed to `catboost_rs` and not to official CatBoost.

Any re-run should fix D1–D4 *and* either raise the iteration count or exclude warm-up
explicitly, otherwise the measurement answers a different question than "which trainer is
faster per tree".

## Status

SPD-03 remains **not satisfied**. It is no longer blocked on Kaggle quota — Colab supplies a
usable CUDA device — but it is now blocked on D1–D4. No headline number is published.
