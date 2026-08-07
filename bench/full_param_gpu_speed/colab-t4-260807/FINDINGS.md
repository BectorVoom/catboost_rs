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

One further confounder that would distort even a fixed run at this grid's settings:

- **`border_count=32` ⇒ 33 bins ⇒ padded to a 64-wide line**, so the oblivious device fill
  does ~2× the histogram work it would at a true 32. The P100 runs that produced the earlier
  1.14–1.19× advantage did not carry this padding.

**JIT warm-up is NOT a confounder** — `fit_once` is called once UNTIMED before the timed
repeats specifically to absorb it, and only the subsequent `REPEATS` runs are recorded. (An
earlier draft of this note claimed otherwise; the harness gets this right.)

## Status

SPD-03 remains **not satisfied**. It is no longer blocked on Kaggle quota — Colab supplies a
usable CUDA device — but it is now blocked on D1–D4. No headline number is published.

---

# Round 2 — after fixing D1 (same VM, same wheel)

With the probe fixed, **5 cells activated the device** and produced a genuine GPU-vs-GPU
comparison. `device_prof_tree_lines == 2` on each, i.e. activation was OBSERVED, not assumed.

| cell | official (GPU) | catboost_rs (GPU) | ratio official/rs |
|---|---|---|---|
| `SymmetricTree\|RMSE\|unw\|noctr\|300k` | 1.574 s | 1.780 s | **0.884** |
| `SymmetricTree\|Logloss\|unw\|noctr\|300k` | 1.622 s | 1.884 s | **0.861** |
| `SymmetricTree\|RMSE\|unw\|noctr\|1000k` | 3.028 s | 6.264 s | **0.483** |
| `SymmetricTree\|Logloss\|unw\|noctr\|1000k` | 3.052 s | 6.949 s | **0.439** |
| `SHOWCASE-bias\|RMSE\|unw\|noctr\|300k` | 1.540 s | 1.679 s | **0.917** |

Ratio > 1 would mean catboost_rs is faster. **It is slower in every activated cell** — by
8–14 % at 300 k rows and by ~2.1–2.3× at 1 M rows. The remaining 29 cells stayed `N/A` or
CPU-labelled for D2 / D3 / D4.

**On this hardware and this grid, catboost_rs does not beat official CatBoost GPU.**

## The 33→64 bin-padding hypothesis is REFUTED

A four-point `border_count` A/B on the 1 M × 50 RMSE cell (3 timed repeats each, untimed warm
run first):

| `border_count` | bins | catboost_rs | official | ratio |
|---|---|---|---|---|
| 32 | 33 → padded 64 | 6.99 s | 2.90 s | 0.415 |
| 31 | 32, no padding | 6.59 s | 3.01 s | 0.457 |
| 63 | 64 | 6.42 s | 3.20 s | 0.499 |
| 127 | 128 | 6.38 s | 4.24 s | 0.665 |

Removing the padding bought **6 %**, not the ~2× the gap needs. Note the shape: catboost_rs is
essentially **flat** (6.38–6.99 s) across a 4× bin-width range while official CatBoost *grows*
(2.90 → 4.24 s). A histogram-width-bound workload cannot be flat in histogram width — so the
1 M-row cost is not in the histogram at all.

## Where the time actually goes: host-side setup, not the tree loop

`CB_GPU_PROF` stage attribution at n = 1 M, nf = 50, 30 iterations:

| stage | elapsed |
|---|---|
| `fit-prep copy+borders` | 1374–2290 ms |
| `quantize` | 380–732 ms |
| `begin` (upload/residency) | 568–945 ms |
| **30 × per-tree device work** | **~36 ms/tree ⇒ ~1.1 s** |
| `fit-train` (total) | 4084–4827 ms |

Per-tree steady state is `fill≈25–30 ms, score≈6.6 ms, derive≈1.2 ms, split≈1.2 ms,
stats_read≈1.3 ms, leaf_apply≈0.35 ms`.

**~2.5–3.9 s — 40–60 % of the fit — is spent before a single tree is grown**, while the whole
of official CatBoost's fit is 2.9 s. The grow kernels are competitive; the
prep → quantize → upload pipeline is the bottleneck, and it is a *fixed* cost, which is exactly
why the gap widens from 300 k (0.88) to 1 M (0.44) and why it is flat in bin width.

This corroborates the earlier "round 2" note that the residual gap was "begin/upload + host
prep fixed costs" — now quantified on a T4 at 1 M rows.

## Consequence for the speed goal

Chasing kernel-level tree-growing speed will not close this. The next optimisation target is
`fit-prep copy+borders` (the largest single term, and host-side), then `quantize` and `begin`.
Until then, no "beats official CatBoost" claim is supportable at 1 M rows on a T4.

The earlier 1.14–1.19× advantage recorded on a Kaggle P100 was measured on a different shape
and device; it is not contradicted by this, but it is also not reproduced here, and nothing in
this run supports generalising it.
