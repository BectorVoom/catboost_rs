#!/usr/bin/env python3
"""Official CatBoost's `thread_count` scaling, on the SAME corpus this engine's
Rust bench uses (`crates/catboost-rs/tests/perf_param_bench_test.rs`).

Why this exists: an earlier wave found this engine's `GreedyLogSum` baseline at
0.62 s against official CatBoost's 0.17 s, but official at `thread_count=1` was
0.58 s -- i.e. essentially the whole 3.6x gap was THREADING, not the algorithm.
`thread_count` is now implemented here, so that claim is finally checkable: run
both engines across the same thread counts and compare the curves.

Two things are compared, and they answer different questions:

* absolute seconds at each count -- "how far behind are we, right now"
* each engine normalised to its OWN `thread_count=1` -- "how well does each one
  scale", which is independent of single-thread constant factors and is the
  number that says whether the remaining gap is parallel efficiency or serial
  work.

The corpus is the Rust bench's LCG, ported verbatim (same constants, same
consumption order), so both engines see byte-identical data. Do not "improve" it
with numpy's RNG -- the point is that it matches.

Usage:
    python bench/perf_param_cpu/official_thread_count.py
Knobs (env): BN rows, BNF features, BITERS, BDEPTH, BREPS.
"""

import os
import sys
import time

import numpy as np

try:
    from catboost import CatBoostRegressor
except ImportError:  # pragma: no cover
    sys.exit("catboost is not installed; this bench compares against it")


def envn(key, default):
    return int(os.environ.get(key, default))


N = envn("BN", 200_000)
NF = envn("BNF", 20)
ITERS = envn("BITERS", 20)
DEPTH = envn("BDEPTH", 6)
REPS = envn("BREPS", 4)

THREAD_COUNTS = [1, 2, 4, 8, 16]

#: Pinned to the Rust `builder()` in perf_param_bench_test.rs, field for field.
COMMON = dict(
    loss_function="RMSE",
    iterations=ITERS,
    depth=DEPTH,
    learning_rate=0.03,
    l2_leaf_reg=3.0,
    random_strength=0,
    boost_from_average=False,
    random_seed=42,
    border_count=254,
    score_function="L2",
    leaf_estimation_method="Gradient",
    leaf_estimation_iterations=1,
    bootstrap_type="No",
    verbose=False,
    allow_writing_files=False,
)


def corpus():
    """The Rust bench's LCG, ported verbatim.

    `s = s * 6364136223846793005 + 1442695040888963407` (wrapping u64), value
    `(s >> 33) / 2^31`, consumed column-major: all of feature 0, then feature 1,
    and so on -- exactly the order `gen()` fills its `Vec<Vec<f64>>`.
    """
    mask = (1 << 64) - 1
    mul = 6_364_136_223_846_793_005
    inc = 1_442_695_040_888_963_407
    s = 0x9E37_79B9_7F4A_7C15

    cols = []
    for _ in range(NF):
        col = np.empty(N, dtype=np.float64)
        for i in range(N):
            s = (s * mul + inc) & mask
            col[i] = ((s >> 33) / float(1 << 31)) * 10.0 - 5.0
        cols.append(col)
    x = np.ascontiguousarray(np.stack(cols, axis=1))
    y = np.sin(x[:, 0] * 0.31) + np.cos(x[:, 1 % NF] * 0.17) * 0.5
    return x, y


def best_of(x, y, thread_count):
    """Best-of-REPS wall time, after an untimed warm fit."""
    CatBoostRegressor(**COMMON, thread_count=thread_count).fit(x, y)
    best = float("inf")
    for _ in range(REPS):
        t = time.perf_counter()
        CatBoostRegressor(**COMMON, thread_count=thread_count).fit(x, y)
        best = min(best, time.perf_counter() - t)
    return best


def main():
    cores = os.cpu_count() or 1
    print("building the shared corpus (%d x %d) ..." % (N, NF))
    x, y = corpus()
    print(
        "\n=== official catboost 1.2.10 thread_count scaling "
        "(n=%d, features=%d, iters=%d, depth=%d, cores=%d) ==="
        % (N, NF, ITERS, DEPTH, cores)
    )
    print("%10s  %10s  %10s  %10s  %10s" % ("threads", "secs", "speedup", "ideal", "efficiency"))

    # Interleave the repetitions across thread counts, for the same reason the
    # Rust bench does: a transient must not land entirely inside one cell.
    results = {tc: float("inf") for tc in THREAD_COUNTS}
    for tc in THREAD_COUNTS:
        CatBoostRegressor(**COMMON, thread_count=tc).fit(x, y)  # warm
    for _ in range(REPS):
        for tc in THREAD_COUNTS:
            t = time.perf_counter()
            CatBoostRegressor(**COMMON, thread_count=tc).fit(x, y)
            results[tc] = min(results[tc], time.perf_counter() - t)

    base = results[THREAD_COUNTS[0]]
    for tc in THREAD_COUNTS:
        ideal = float(min(tc, cores))
        sp = base / results[tc]
        print(
            "%10d  %10.3f  %9.2fx  %9.2fx  %9.0f%%"
            % (tc, results[tc], sp, ideal, sp / ideal * 100.0)
        )

    print(
        "\nCompare against the Rust cell:\n"
        "  CB_PERF_BENCH=1 BREPS=%d cargo test -p catboost-rs --release \\\n"
        "    --test perf_param_bench_test thread_count -- --nocapture --test-threads=1"
        % REPS
    )


if __name__ == "__main__":
    main()
