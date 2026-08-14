# Is the exact-DP border cost OURS, or the algorithm's?

**Date:** 2026-08-15 · **Box:** local dev machine, 16 cores, CPU path only
**Corpus:** `n = 200_000`, `f = 20`, `border_count = 254`, depth 6, 10 iterations,
median of 3. Companion to `FINDINGS-cpu-border-cost.md`, which established that
`MinEntropy` / `MaxLogSum` add ~3.8 s of **one-time border-build** cost.

`~3.8 s` cannot be judged on its own. The exact dynamic program is asymptotically dearer
than the greedy heap, so *some* penalty is expected; the question is whether official
CatBoost pays the same one for the same algorithm.

## Answer: our `E_RLM2` port is at upstream parity

| border type | catboost-rs | official | cb-rs penalty | official penalty |
|---|---|---|---|---|
| `GreedyLogSum` | 0.62 s | 0.17 s | 1.00× | 1.00× |
| `MinEntropy` | 4.45 s | 4.06 s | 7.20× | **23.93×** |
| `MaxLogSum` | 4.12 s | 3.74 s | 6.67× | **22.07×** |

**Read the absolute deltas, not the ratios:**

| border type | cb-rs extra | official extra |
|---|---|---|
| `MinEntropy` | 4.45 − 0.62 = **3.83 s** | 4.06 − 0.17 = **3.89 s** |
| `MaxLogSum` | 4.12 − 0.62 = **3.50 s** | 3.74 − 0.17 = **3.57 s** |

The exact-DP border build costs the **same absolute wall clock on both engines**, within
2%. So the ~3.8 s is the algorithm, not our port — the transcription of upstream's
`E_RLM2` (`binarization.cpp`) reproduces its cost as well as its output.

The wildly different RATIOS (7.2× against 23.9×) are an artifact of the two engines having
different baselines, and are the reason this comparison is stated in absolute terms. A
ratio-only reading would have said "our DP is 3× cheaper than upstream's", which is false
and flattering in the wrong direction.

## The baseline difference is THREADING, not the algorithm

The obvious follow-up: why is our `GreedyLogSum` baseline 0.62 s against official's 0.17 s?

| configuration | median |
|---|---|
| official, `thread_count=16` (its default: all cores) | 0.14 s |
| official, `thread_count=1` | **0.58 s** |
| catboost-rs, default threading | **0.62 s** |

Against **single-threaded** official, catboost-rs is at 0.62 s vs 0.58 s — parity. The
entire 3.6× baseline difference is that official saturates all 16 cores on this path while
catboost-rs effectively does not.

**This is an observation, not a claim about this wave.** CPU training parallelism is
outside the parameter-wireup scope, and `thread_count` is explicitly deferred to its own
wave — this measurement corroborates that deferral rather than contradicting it, and gives
that wave a concrete starting number (a ~4× headroom at this shape). It is recorded here
because it is the honest explanation of a table cell above, and leaving it unexplained
would invite exactly the wrong conclusion about the DP.

## What a user should take away

- Picking `MinEntropy` or `MaxLogSum` costs a fixed ~3.8 s of border building on a
  200k × 20 corpus at `border_count = 254` — the pessimistic end of the axis, since the
  DP's cost grows with the border count.
- That cost is paid **once**, not per iteration, so a long training run barely notices it
  (1.43× at 160 iterations, and falling).
- It is not a catboost-rs tax: official CatBoost pays the same absolute cost.

## Reproducing

```bash
python bench/param_wave_gpu_speed/border_cost_cpu.py vs-upstream
```
