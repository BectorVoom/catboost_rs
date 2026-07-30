// poisson_bootstrap_oracle.cpp — authoritative upstream-GPU Poisson bootstrap oracle.
//
// A standalone, dependency-free transcription of Yandex CatBoost's CUDA Poisson
// bootstrap, which is the ONLY definition of `bootstrap_type=Poisson` that upstream
// has: `TBootstrapConfig::Validate` rejects Poisson outright on the CPU task type
// (`catboost/private/libs/options/bootstrap_options.cpp:29`, "poisson bootstrap is
// not supported on CPU"), so there is no CPU sampler to hold a port to. The GPU
// kernel below IS the specification.
//
// Sources transcribed VERBATIM (upstream `master`, fetched 2026-07-31):
//   - catboost/cuda/cuda_util/kernel/random_gen.cuh
//       `AdvanceSeed`, `NextUniform`, `NextNormal`, `NextPoisson`
//   - catboost/cuda/cuda_util/kernel/bootstrap.cu
//       `PoissonBootstrapImpl` (the __global__ body) and `PoissonBootstrap`
//       (the launch geometry: blockSize 256, numBlocks = min(ceilDiv(seedsSize,
//       256), ceilDiv(weightsSize, 256)))
//   - catboost/cuda/gpu_data/bootstrap.h
//       `TBootstrap::BootstrappedWeights` — the weights buffer is filled with 1.0f
//       before the draw, so the emitted weight IS the raw Poisson count.
//   - catboost/private/libs/options/bootstrap_options.h
//       `GetPoissonLambda() = takenFraction < 1 ? -log(1 - takenFraction) : -1`
//
// WHY A HOST TRANSCRIPTION IS A LEGITIMATE ORACLE: every function above is pure
// integer / IEEE-754 arithmetic over an explicit seed word. `AdvanceSeed` is a pair
// of 16-bit multiply-with-carry steps (exact in uint32), `NextUniform` is an exact
// uint32 mix scaled by 2^-32, and `NextPoisson`'s only transcendental is
// `log(double)`. The accumulator `logp` is a FLOAT, so the double `log` result is
// rounded to 24 bits on every iteration; a 1-ulp double-precision disagreement
// between CUDA's `log` and glibc's `log` therefore survives into the float sum with
// probability ~2^-29 per draw, and only matters if it also flips the `logp > L`
// comparison. Bit-for-bit agreement is the expected outcome at fixture scale;
// see the generator script for the recorded draw counts.
//
// The grid geometry is reproduced exactly because the mapping object -> seed depends
// on it: thread `t` of `numBlocks * 256` owns `seeds[t]` and walks objects
// `t, t + stride, t + 2*stride, ...` with `stride = numBlocks * 256`. A different
// block count would hand object `i` a different seed and a different draw position.
//
// I/O: reads a scenario on argv, writes one Poisson count per line to stdout — first
// the `rounds` consecutive draws over the SAME (in-place mutated) seed buffer, one
// round after another, so the caller can also gate the cross-tree seed carry-over.
//
// Build:
//   g++ -O2 -std=c++17 poisson_bootstrap_oracle.cpp -o poisson_bootstrap_oracle
//
// Usage:
//   ./poisson_bootstrap_oracle <n> <seedsSize> <subsample> <seed0> <rounds>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

using ui32 = uint32_t;
using ui64 = uint64_t;

// ---------------------------------------------------------------------------
// random_gen.cuh — VERBATIM
// ---------------------------------------------------------------------------

static inline ui64 AdvanceSeed(ui64* seed) {
    ui32 v = *seed >> 32;
    ui32 u = *seed & 0xFFFFFFFF;
    v = 36969 * (v & 0xFFFF) + (v >> 16);
    u = 18000 * (u & 0xFFFF) + (u >> 16);
    *seed = ((ui64) v << 32) | (ui64) u;
    return *seed;
}

static inline double NextUniform(ui64* seed) {
    ui64 x = AdvanceSeed(seed);
    ui32 v = x >> 32;
    ui32 u = x & 0xFFFFFFFF;

    return ((v << 16) + u) * 2.328306435996595e-10;
}

static inline float NextNormal(ui64* seed) {
    float a = NextUniform(seed);
    float b = NextUniform(seed);
    return sqrtf(-2.0f * logf(a)) * cosf(2.0f * 3.141592654f * b);
}

// `draws` accumulates the number of NextUniform calls consumed, so the generator can
// report the empirical draw count (it is variable per object — the reason a
// host-side "advance the stream by n" model can never track this sampler).
static inline float NextPoisson(ui64* seed, float alpha, ui64* draws) {
    if (alpha > 20) {
        float a = sqrtf(alpha) * NextNormal(seed) + alpha;
        *draws += 2;
        while (a < 0) {
            a = sqrtf(alpha) * NextNormal(seed) + alpha;
            *draws += 2;
        }
        return a;
    }
    float logp = 0.0f, L = -alpha;
    int k = 0;
    do {
        k++;
        logp += log(NextUniform(seed));
        *draws += 1;
    } while (logp > L);
    return k - 1;
}

// ---------------------------------------------------------------------------
// bootstrap.cu — `PoissonBootstrapImpl`, executed over an explicit thread grid
// ---------------------------------------------------------------------------

static ui32 CeilDivide(ui32 x, ui32 y) { return (x + y - 1) / y; }

int main(int argc, char** argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s <n> <seedsSize> <subsample> <seed0> <rounds>\n", argv[0]);
        return 2;
    }
    const ui32 n = (ui32) strtoul(argv[1], nullptr, 10);
    const ui32 seedsSize = (ui32) strtoul(argv[2], nullptr, 10);
    const double subsample = strtod(argv[3], nullptr);
    const ui64 seed0 = strtoull(argv[4], nullptr, 10);
    const ui32 rounds = (ui32) strtoul(argv[5], nullptr, 10);

    // `TBootstrapConfig::GetPoissonLambda()` (bootstrap_options.h:31-34) — VERBATIM.
    const float takenFraction = (float) subsample;
    const float lambda = takenFraction < 1 ? -log(1 - takenFraction) : -1;

    // The seed material. Upstream fills this buffer from its HOST Mersenne stream
    // (`TGpuAwareRandom::FillSeeds` -> `TRandom::NextUniformL`), whose position inside
    // a real fit is not knowable from outside; the oracle therefore PINS the buffer
    // with SplitMix64 so both sides start from identical, reproducible state. Seed
    // provenance is not part of the kernel contract being gated — the object -> weight
    // map given a seed buffer is.
    std::vector<ui64> seeds(seedsSize);
    for (ui32 i = 0; i < seedsSize; ++i) {
        ui64 z = seed0 + 0x9E3779B97F4A7C15ULL * (ui64) (i + 1);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        seeds[i] = z ^ (z >> 31);
    }

    // `PoissonBootstrap` launch geometry (bootstrap.cu:66-70) — VERBATIM.
    const ui32 blockSize = 256;
    const ui32 numBlocks = std::min(CeilDivide(seedsSize, blockSize), CeilDivide(n, blockSize));
    const ui32 stride = numBlocks * blockSize;

    ui64 draws = 0;
    for (ui32 round = 0; round < rounds; ++round) {
        // `FillBuffer(weights, 1.0f)` then the draw (gpu_data/bootstrap.h:88-90).
        std::vector<float> weights(n, 1.0f);
        for (ui32 t = 0; t < stride; ++t) {
            ui64 s = seeds[t];
            ui32 i = t;
            while (i < n) {
                float w = weights[i];
                weights[i] = w * NextPoisson(&s, lambda, &draws);
                i += stride;
            }
            seeds[t] = s;
        }
        for (ui32 i = 0; i < n; ++i) {
            printf("%.0f\n", (double) weights[i]);
        }
    }
    fprintf(stderr, "lambda=%.9g numBlocks=%u stride=%u draws=%llu\n",
            (double) lambda, numBlocks, stride, (unsigned long long) draws);
    return 0;
}
