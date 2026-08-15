# Perf-critical parameter wave — measured results (CPU)

Machine: 16-core box (AMD Ryzen AI 7 350), Linux. Workload: n = 200 000 rows,
20 float features, 20 iterations, depth 6, `border_count = 254`, RMSE,
`bootstrap_type = No`, `random_strength = 0`. Both engines pinned field-for-field
(`perf_param_bench_test.rs::builder` ≡ `official_thread_count.py::COMMON`) and
fed a byte-identical LCG corpus.

Reproduce:

```
CB_PERF_BENCH=1 BREPS=4 cargo test -p catboost-rs --release \
  --test perf_param_bench_test -- --nocapture --test-threads=1
python bench/perf_param_cpu/official_thread_count.py
```

Timing discipline: untimed warm fit, then best-of-4 with the repetitions
**interleaved** across cells rather than run as a block per cell. That is not
pedantry — the first version of this bench measured each configuration to
completion and reported `rsm = 0.75` at 0.65x; re-running moved the outlier to
`rsm = 0.5` at 0.57x. The anomaly followed the *schedule*, not the parameter.
Interleaved, the `rsm` curve is monotonic.

---

## 1. `thread_count` — we scale as well as upstream; the gap is single-threaded

| threads | cb-rs (s) | speedup | eff. | official (s) | speedup | eff. | cb-rs / official |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1  | 4.857 | 1.00x | 100% | 0.734 | 1.00x | 100% | **6.6x** |
| 2  | 2.736 | 1.78x |  89% | 0.405 | 1.81x |  91% | 6.8x |
| 4  | 1.766 | 2.75x |  69% | 0.268 | 2.74x |  68% | 6.6x |
| 8  | 1.294 | 3.75x |  47% | 0.207 | 3.55x |  44% | 6.3x |
| 16 | 1.178 | 4.12x |  26% | 0.208 | 3.53x |  22% | **5.7x** |

**The two scaling curves are the same, within noise** — 1.78 vs 1.81 at two
threads, 2.75 vs 2.74 at four, 3.75 vs 3.55 at eight, and this engine is
marginally *better* at sixteen (4.12x vs 3.53x; official stops improving past
eight). Parallel efficiency collapses identically on both sides, from ~90% at two
threads to ~25% at sixteen, which is a property of the boosting algorithm — the
per-level barrier and the serial leaf/approx phases — not of either
implementation.

**So the remaining gap is a single-thread constant factor, not threading.** The
ratio is 6.6x at one thread and 5.7x at sixteen: adding threads narrows it
slightly, and no amount of threading closes it.

> ### This corrects an earlier conclusion
>
> The `feature_border_type` wave measured this engine's `GreedyLogSum` at 0.62 s
> against official's 0.17 s, noted official at `thread_count = 1` was 0.58 s, and
> concluded "the whole 3.6x is threading". That measurement was of **border
> building** (quantization), and for that stage the conclusion stands. It does
> **not** generalise to training: on the grow loop both engines are equally
> parallel and this engine is ~6x slower per thread. Optimising toward "we just
> need threads" would have been optimising the wrong thing.

Practical reading of `thread_count` itself: past 8 threads there is almost
nothing left (3.75x → 4.12x for double the cores), so on a busy machine capping
it costs little. It is also the knob that makes a benchmark reproducible, since
`0`/all-cores results depend on whatever else is running.

## 2. `rsm` — a real but modest speedup, paid for in fit quality

| rsm | secs | vs unset | train RMSE |
|---|---:|---:|---:|
| unset | 1.196 | 1.00x | 0.45699 |
| 1.0   | 1.183 | 1.01x | 0.45699 |
| 0.75  | 1.153 | 1.04x | 0.46875 |
| 0.5   | 1.153 | 1.04x | 0.46922 |
| 0.25  | 1.085 | 1.10x | 0.52293 |

`rsm = 1.0` is timed separately from *unset* on purpose: only values **below** 1
enable the per-level candidate draws. The two agree to 1% and produce an
identical RMSE, which is the bench-side echo of the bit-identity the oracle
asserts.

The speedup is far below the naive expectation — scoring a quarter of the
features does **not** make a level 4x cheaper, because the per-level histogram
build and the leaf/approx phases do not shrink with the candidate set, and the
subsampled fit degrades measurably (RMSE 0.457 → 0.523 at `rsm = 0.25`).

**Use `rsm` for regularisation and decorrelation, not for speed.** At 1.10x for a
14% worse training fit, buying wall-clock this way is a poor trade; `depth` or
`border_count` move the time far more.

Caveat: `rsm < 1` **declines the device grower**, so on a GPU-eligible workload
its true cost is losing the device path entirely, which is far larger than
anything in this table.

## 3. `langevin` — 5–6% overhead

| config | secs | overhead |
|---|---:|---:|
| off | 1.190 | — |
| `diffusion_temperature = 10000` (default) | 1.260 | 5.9% |
| `diffusion_temperature = 100` | 1.250 | 5.0% |
| `posterior_sampling` | 1.254 | 5.3% |

Two extra passes per tree — one Gaussian per object over the derivatives, one per
leaf over the leaf sums — against an `O(n · features · bins)` histogram build.
The cost is independent of the temperature (same draw count either way), as it
should be. `posterior_sampling` costs the same as plain Langevin: it only changes
which temperature and shrink rate are used.

5–6% is cheap for what it buys, and it is a floor rather than the whole story:
Langevin also **declines the device grower**, so on GPU-eligible work the real
cost is again the lost device path.

---

## Why there are no GPU numbers here

All three parameters are CPU-only end to end:

* `thread_count` bounds a host thread pool; the device path is unaffected.
* `rsm` declines `device_host_eligible` — the device grower scores every
  quantized feature at every level and has no per-level candidate mask.
* `langevin` declines too — the existing device kernel reseeds **per element**
  while upstream and this engine's CPU path seed **per block of 128**, so it is a
  different stream.

A device benchmark for them would therefore be timing the CPU fallback and
labelling it a GPU result. The device-vs-CPU speedups that already exist
(`bench/RESULTS.md`, `bench/BENCH-03-SIGNOFF.md`) are unaffected by this wave,
since every one of these parameters is inert at its default.
