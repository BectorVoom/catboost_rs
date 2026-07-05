#!/usr/bin/env python
"""SPIKE 002: official CatBoost CPU timing grid, mirroring the Rust harness in
`crates/cb-train/tests/perf_baseline_test.rs`.

Same synthetic generator (splitmix64 hash → uniform [0,1) features, linear
target), same params (RMSE, lr=0.03, l2=3.0, no bootstrap, SymmetricTree). We
time at thread_count=1 (per-core algorithm efficiency, apples-to-apples with the
single-threaded Rust host loop) AND thread_count=max (real-world default gap).

Prints machine-greppable `CBBENCH ...` lines matching the Rust `RSBENCH` rows.
"""
import os
import time
import numpy as np
from catboost import CatBoostRegressor


def splitmix(i, f):
    mask = (1 << 64) - 1
    z = (i * 0x9E3779B97F4A7C15 + f * 0xD1B54A32D192ED03) & mask
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    z ^= z >> 31
    return (z >> 11) / float(1 << 53)


def gen(n, nf):
    # Vectorized splitmix64 over the (i, f) grid, matching the Rust generator.
    mask = (1 << 64) - 1
    ii = np.arange(n, dtype=object)
    X = np.empty((n, nf), dtype=np.float64)
    for f in range(nf):
        z = (ii * 0x9E3779B97F4A7C15 + f * 0xD1B54A32D192ED03) & mask
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        z = z ^ (z >> 31)
        X[:, f] = np.array([(int(v) >> 11) / float(1 << 53) for v in z])
    signs = np.array([1.0 if f % 2 == 0 else -1.0 for f in range(min(nf, 5))])
    y = X[:, : len(signs)] @ signs
    return X, y


def time_row(n, nf, nbins, depth, iters, threads):
    X, y = gen(n, nf)
    m = CatBoostRegressor(
        iterations=iters,
        depth=depth,
        learning_rate=0.03,
        l2_leaf_reg=3.0,
        loss_function="RMSE",
        border_count=nbins - 1,
        bootstrap_type="No",
        boost_from_average=False,
        grow_policy="SymmetricTree",
        thread_count=threads,
        random_seed=42,
        allow_writing_files=False,
        logging_level="Silent",
        feature_border_type="Uniform",
    )
    t = time.perf_counter()
    m.fit(X, y)
    secs = time.perf_counter() - t
    print(
        f"CBBENCH n={n} nf={nf} nbins={nbins} depth={depth} iters={iters} "
        f"threads={threads} train_s={secs:.4f} per_tree_ms={secs*1000.0/iters:.3f}",
        flush=True,
    )


def main():
    base_n, base_nf, base_nbins, base_depth, iters = 20_000, 20, 128, 6, 20
    max_threads = os.cpu_count() or 1
    for threads in (1, max_threads):
        print(f"CBBENCH_META threads={threads} base_n={base_n} base_nf={base_nf} "
              f"base_nbins={base_nbins} base_depth={base_depth} iters={iters}", flush=True)
        time_row(base_n, base_nf, base_nbins, base_depth, iters, threads)
        for n in (5_000, 10_000, 40_000, 80_000):
            time_row(n, base_nf, base_nbins, base_depth, iters, threads)
        for nf in (5, 10, 40, 80):
            time_row(base_n, nf, base_nbins, base_depth, iters, threads)
        for nb in (32, 64, 254):
            time_row(base_n, base_nf, nb, base_depth, iters, threads)
        for d in (2, 4, 8):
            time_row(base_n, base_nf, base_nbins, d, iters, threads)
    print("CBBENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
