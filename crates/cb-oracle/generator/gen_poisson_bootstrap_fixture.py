#!/usr/bin/env python3
"""Freeze the upstream-GPU Poisson bootstrap oracle fixtures.

`bootstrap_type=Poisson` exists ONLY on upstream CatBoost's GPU task type — the CPU
validator rejects it outright ("poisson bootstrap is not supported on CPU",
`bootstrap_options.cpp:29`). There is therefore no Python `catboost` call that can
emit a CPU reference the way `gen_fixtures.py` does for Bayesian / Bernoulli / MVS,
and the per-object GPU bootstrap weights are not observable through any public API.

The reference is instead `poisson_bootstrap_oracle.cpp` — a verbatim host
transcription of upstream's `PoissonBootstrapImpl` + `random_gen.cuh`
(`NextPoisson`/`NextUniform`/`AdvanceSeed`) and of the `PoissonBootstrap` launch
geometry, which decides WHICH seed each object draws from. That geometry is part of
the contract: thread `t` of `numBlocks * 256` owns `seeds[t]` and walks objects
`t, t + stride, ...`.

Each scenario is drawn TWICE over the same, in-place-mutated seed buffer, so the
fixture also gates the cross-tree seed carry-over (`seeds[0] = s` at the end of the
upstream kernel) rather than only a single draw.

Run:
    cd crates/cb-oracle/generator
    python3 gen_poisson_bootstrap_fixture.py
"""

import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = HERE.parent / "fixtures" / "bootstrap_poisson"
SRC = HERE / "poisson_bootstrap_oracle.cpp"

ROUNDS = 2

# (name, n, seeds_size, subsample, seed0)
#
# `seeds_size` is a genuine kernel parameter upstream (`PoissonBootstrap` reads it off
# the seed buffer), and it is what selects the block count together with `n`:
#   numBlocks = min(ceil(seeds_size / 256), ceil(n / 256)),  stride = numBlocks * 256
# The three scenarios pin three DIFFERENT geometries on purpose:
#   * one_pass  — stride > n: every object is the first draw of its own thread.
#   * grid_wrap — stride < n (a deliberately small seed buffer, as upstream allows):
#                 each thread draws for 4 objects in sequence, so a kernel that got
#                 the stride walk wrong would disagree from object `stride` onward.
#   * wide      — the production shape: 79 blocks off the full 65536-seed buffer.
SCENARIOS = [
    ("one_pass", 1000, 65536, 0.66, 20260731),
    ("grid_wrap", 4096, 1024, 0.8, 777),
    ("wide", 20000, 65536, 0.8, 42),
]


def build(tmpdir: pathlib.Path) -> pathlib.Path:
    exe = tmpdir / "poisson_bootstrap_oracle"
    subprocess.run(
        ["g++", "-O2", "-std=c++17", str(SRC), "-o", str(exe)],
        check=True,
    )
    return exe


def main() -> int:
    if not SRC.exists():
        print(f"missing {SRC}", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as td:
        exe = build(pathlib.Path(td))
        for name, n, seeds_size, subsample, seed0 in SCENARIOS:
            proc = subprocess.run(
                [str(exe), str(n), str(seeds_size), str(subsample), str(seed0), str(ROUNDS)],
                check=True,
                capture_output=True,
                text=True,
            )
            counts = np.array(
                [float(line) for line in proc.stdout.split()], dtype=np.float64
            )
            assert counts.shape == (ROUNDS * n,), counts.shape
            assert counts.dtype == np.float64
            assert (counts >= 0).all(), "a Poisson count must be non-negative"
            assert (counts == np.round(counts)).all(), "a Poisson count must be integral"

            block_size = 256
            num_blocks = min(-(-seeds_size // block_size), -(-n // block_size))
            out = FIXTURES / name
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / "weights.npy", counts)
            meta = {
                "source": "poisson_bootstrap_oracle.cpp (verbatim transcription of "
                "catboost/cuda/cuda_util/kernel/bootstrap.cu PoissonBootstrapImpl + "
                "cuda_util/kernel/random_gen.cuh NextPoisson)",
                "n": n,
                "seeds_size": seeds_size,
                "subsample": subsample,
                # `GetPoissonLambda()` = -log(1 - subsample), computed in f32 upstream.
                "lambda": float(np.float32(-np.log(np.float32(1.0) - np.float32(subsample)))),
                "seed0": seed0,
                "rounds": ROUNDS,
                "block_size": block_size,
                "num_blocks": num_blocks,
                "stride": num_blocks * block_size,
                # The seed buffer is SplitMix64(seed0 + GOLDEN * (i + 1)); the Rust side
                # regenerates it with the identical constants (see the oracle .cpp).
                "seed_derivation": "splitmix64",
                "layout": "round-major: round 0 objects 0..n-1, then round 1",
                "mean_count": float(counts.mean()),
                "zero_fraction": float((counts == 0).mean()),
            }
            (out / "config.json").write_text(json.dumps(meta, indent=2) + "\n")
            print(
                f"{name}: n={n} seeds={seeds_size} lambda={meta['lambda']:.6f} "
                f"stride={meta['stride']} mean={meta['mean_count']:.4f} "
                f"zeros={meta['zero_fraction']:.4f} -> {out}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
