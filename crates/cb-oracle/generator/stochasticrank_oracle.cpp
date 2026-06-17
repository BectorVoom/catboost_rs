// stochasticrank_oracle.cpp — OFFLINE instrumented ground-truth generator for the
// StochasticRank Monte-Carlo der RNG stream (Plan 06.3-04, Wave C / D-6.3-02).
//
// A standalone, DEPENDENCY-FREE transcription (ZERO catboost includes) of the
// smallest RNG units the StochasticRank parity slice needs — same precedent as
// `yetirank_oracle.cpp` / `ordered_oracle.cpp`. The upstream unit
// (`error_functions.cpp:1008-1102`) cannot be linked in isolation, so we
// TRANSCRIBE the Gaussian-noise draw stream verbatim and SELF-ORACLE against the
// bitstream-validated `cb-core::TFastRng64` + `cb-core::std_normal` Rust
// reproduction (rng_test.rs / normal.rs are the ground truth).
//
// WHAT IS TRANSCRIBED (verbatim, cited):
//   (a) TFastRng64           — util/random/fast.h (shared with yetirank_oracle.cpp).
//   (b) StdNormalDistribution — util/random/normal.h:11-24 (the Marsaglia-polar
//                               rejection loop, the SAME variable-length draw
//                               sequence as cb-core::std_normal).
//   (c) The StochasticRank noise stream — error_functions.cpp:1041-1046:
//                               rng = TFastRng64(randomSeed); per sample, per doc
//                               noise[d] = StdNormalDistribution(rng);
//                               scores[d] = shifted[d] + Sigma·noise[d].
//                               randomSeed = randomSeed + queryIndex
//                               (error_functions.h:1257 GenRandUI64Vector per group).
//
// OUTPUT — JSONL via CB_INSTRUMENT_LOG (env-gated; inert when unset). Schema:
//   {"event":"gauss_draw","group":g,"sample":s,"doc":d,"noise":<f64>}
//   {"event":"score","group":g,"sample":s,"doc":d,"score":<f64>}
//   {"event":"sorted_order","group":g,"sample":s,"order":[...]}
// The committed fixture freezes the noise stream as the RNG ground truth the Rust
// oracle (stochasticrank_oracle_test.rs::compare_stage) gates integer-/f64-exact.
//
// BUILD + RUN (OFFLINE, RUN-ONCE/COMMIT — see instrument_ranking_rng_README.md):
//   clang++ -std=c++20 -O2 stochasticrank_oracle.cpp -o /tmp/stochasticrank_oracle
//   CB_INSTRUMENT_LOG=/tmp/stochasticrank_rng.jsonl /tmp/stochasticrank_oracle
// NO catboost link; builds with a stock C++20 compiler.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>

// ---------------------------------------------------------------------------
// (a) TFastRng64 — verbatim (identical to yetirank_oracle.cpp; see its header for
// the per-line citations). MUST reproduce cb-core::TFastRng64 bit-for-bit.
// ---------------------------------------------------------------------------
// VERBATIM transcription of the oracle-locked Rust port crates/cb-core/src/rng.rs
// (identical to yetirank_oracle.cpp; X = seed + Iterate-then-Mix, 31-bit FixSeq
// mask, low|high<<32 64-bit combine). The Rust reproduction is GROUND TRUTH; the
// self-oracle asserts agreement.
namespace transcribed {

constexpr uint64_t LCG_MULTIPLIER = 6364136223846793005ULL;  // rng.rs:26.

inline uint32_t PcgMix(uint64_t x) {  // rng.rs:31-38.
    uint32_t xorshifted = static_cast<uint32_t>(((x >> 18) ^ x) >> 27);
    uint32_t rot = static_cast<uint32_t>(x >> 59);
    return (xorshifted >> rot) | (xorshifted << ((32 - rot) & 31));
}

struct Lcg32 {
    uint64_t x;
    uint64_t c;
    static Lcg32 New(uint64_t seed, uint32_t seq) {  // rng.rs:82-85.
        Lcg32 r;
        r.c = (static_cast<uint64_t>(seq) << 1) | 1ULL;
        r.x = seed;
        return r;
    }
    static Lcg32 NewReallyFast(uint64_t seed) {  // rng.rs:90-92.
        Lcg32 r;
        r.x = seed;
        r.c = 1;
        return r;
    }
    uint64_t Iterate(uint64_t state) const { return state * LCG_MULTIPLIER + c; }  // rng.rs:96-98.
    uint32_t GenRand32() {  // rng.rs:103-105.
        x = Iterate(x);
        return PcgMix(x);
    }
    uint64_t GenRand64() {  // rng.rs:112-115.
        uint64_t low = GenRand32();
        uint64_t high = GenRand32();
        return low | (high << 32);
    }
};

inline uint32_t FixSeq(uint32_t seq1, uint32_t seq2) {  // rng.rs:128-134.
    const uint32_t mask = (~0u) >> 1;
    if ((seq1 & mask) == (seq2 & mask)) return ~seq2;
    return seq2;
}

struct TFastRng64 {
    Lcg32 r1;
    Lcg32 r2;
    static TFastRng64 New(uint64_t seed1, uint32_t seq1, uint64_t seed2, uint32_t seq2) {
        TFastRng64 r;
        r.r1 = Lcg32::New(seed1, seq1);
        r.r2 = Lcg32::New(seed2, FixSeq(seq1, seq2));
        return r;
    }
    static TFastRng64 FromSeed(uint64_t seed) {  // rng.rs:163-169.
        Lcg32 derive = Lcg32::NewReallyFast(seed);
        uint64_t seed1 = derive.GenRand64();
        uint32_t seq1 = derive.GenRand32();
        uint64_t seed2 = derive.GenRand64();
        uint32_t seq2 = derive.GenRand32();
        return New(seed1, seq1, seed2, seq2);
    }
    uint64_t GenRand() {  // rng.rs:175-178.
        uint64_t x = r1.GenRand32();
        uint64_t y = r2.GenRand32();
        return (x << 32) | y;
    }
    double GenRandReal1() {  // rng.rs:195-197.
        return static_cast<double>(GenRand() >> 11) * (1.0 / 9007199254740991.0);
    }
};

// (b) StdNormalDistribution<double> — util/random/normal.h:11-24. The
// Marsaglia-polar rejection loop, the SAME variable-length gen_rand_real1()
// sequence cb-core::std_normal consumes (a different sampler desyncs every
// subsequent draw — RESEARCH Pitfall 1).
inline double StdNormal(TFastRng64& rng) {
    for (;;) {
        double x = rng.GenRandReal1() * 2.0 - 1.0;
        double y = rng.GenRandReal1() * 2.0 - 1.0;
        double r = x * x + y * y;
        if (!(r > 1.0 || r <= 0.0)) {
            return x * std::sqrt(-2.0 * std::log(r) / r);
        }
    }
}

}  // namespace transcribed

// ---------------------------------------------------------------------------
// JSONL instrumentation log (env-gated by CB_INSTRUMENT_LOG).
// ---------------------------------------------------------------------------
static FILE* g_log = nullptr;
static void OpenLog() {
    const char* path = std::getenv("CB_INSTRUMENT_LOG");
    if (path != nullptr) g_log = std::fopen(path, "w");
}
static void LogLine(const std::string& line) {
    if (g_log != nullptr) std::fprintf(g_log, "%s\n", line.c_str());
}

// ---------------------------------------------------------------------------
// (c) The StochasticRank per-group noise stream, smallest unit (one group, small
// count, num_estimations small). Transcribes the Stage-1 shift/center +
// Stage-2 Monte-Carlo noise draws (error_functions.cpp:1024-1055). We LOG the
// noise/score stream + sorted order — the RNG ground truth — but do NOT
// reproduce the full DCG metric-diff der here (that is the Rust transcription;
// this harness gates the STREAM the der consumes, the parity crux).
// ---------------------------------------------------------------------------
static void StochasticRankNoiseStream(
    const std::vector<double>& approxes,
    const std::vector<double>& targets,
    double sigmaParam,
    double mu,
    int numEstimations,
    uint64_t randomSeed,
    int groupIndex
) {
    const size_t count = approxes.size();
    if (count <= 1) return;

    // Stage 1 — shift to break ties, then center (non-FilteredDCG),
    // error_functions.cpp:1026-1034.
    std::vector<double> shifted(count);
    for (size_t d = 0; d < count; ++d) {
        shifted[d] = approxes[d] - sigmaParam * mu * targets[d];
    }
    // WR-03 / D-08: cb_core::sum_f64 (crates/cb-core/src/reduction.rs) is the parity
    // SOURCE OF TRUTH — a strict left-to-right f64 fold with NO compensated/pairwise
    // summation. The oracle adapts to it: accumulate shifted[] in the SAME
    // doc-ascending sequential order sum_f64 uses (the Rust centering at
    // ranking_der.rs:726-727 is sum_f64(&shifted) / count). std::accumulate could be
    // reordered by an implementation, so transcribe the fold explicitly here.
    double avrg = 0.0;
    for (double s : shifted) avrg += s;
    avrg /= static_cast<double>(count);
    for (auto& s : shifted) s -= avrg;

    // Stage 2 — Monte-Carlo noise stream, error_functions.cpp:1041-1055.
    transcribed::TFastRng64 rng = transcribed::TFastRng64::FromSeed(randomSeed);
    for (int sample = 0; sample < numEstimations; ++sample) {
        std::vector<double> noise(count);
        std::vector<double> scores(count);
        for (size_t d = 0; d < count; ++d) {
            noise[d] = transcribed::StdNormal(rng);
            scores[d] = shifted[d] + sigmaParam * noise[d];
            LogLine("{\"event\":\"gauss_draw\",\"group\":" + std::to_string(groupIndex) +
                    ",\"sample\":" + std::to_string(sample) + ",\"doc\":" + std::to_string(d) +
                    ",\"noise\":" + std::to_string(noise[d]) + "}");
            LogLine("{\"event\":\"score\",\"group\":" + std::to_string(groupIndex) +
                    ",\"sample\":" + std::to_string(sample) + ",\"doc\":" + std::to_string(d) +
                    ",\"score\":" + std::to_string(scores[d]) + "}");
        }
        std::vector<size_t> order(count);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(),
                         [&](size_t a, size_t b) { return scores[a] > scores[b]; });
        std::string ord = "[";
        for (size_t k = 0; k < order.size(); ++k) {
            ord += std::to_string(order[k]);
            if (k + 1 < order.size()) ord += ",";
        }
        ord += "]";
        LogLine("{\"event\":\"sorted_order\",\"group\":" + std::to_string(groupIndex) +
                ",\"sample\":" + std::to_string(sample) + ",\"order\":" + ord + "}");
    }
}

// ---------------------------------------------------------------------------
// SELF-ORACLE: the transcribed StdNormal must reproduce cb-core::std_normal's
// draws. We print the first few normal draws for seed=0; the Rust unit test
// stochasticrank_oracle_test.rs asserts the SAME values (both draw from the SAME
// TFastRng64 + Marsaglia-polar loop, so they MUST agree).
// ---------------------------------------------------------------------------
static int SelfOracle() {
    transcribed::TFastRng64 rng = transcribed::TFastRng64::FromSeed(0);
    for (int i = 0; i < 3; ++i) {
        double n = transcribed::StdNormal(rng);
        std::fprintf(stderr, "[self-oracle] std_normal(seed=0)[%d] = %.17g\n", i, n);
    }
    return 0;
}

int main() {
    OpenLog();
    SelfOracle();

    // Smallest instrumented unit: ONE group, 3 docs, num_estimations small.
    const std::vector<double> approxes = {0.3, -0.4, 0.1};
    const std::vector<double> targets = {2.0, 0.0, 1.0};
    const double sigma = 1.0;          // upstream default.
    const double mu = 0.0;             // upstream default.
    const int numEstimations = 1;      // upstream default.
    const uint64_t randomSeed = 5;     // group 0 seed = randomSeed + 0.

    const int groupIndex = 0;
    StochasticRankNoiseStream(approxes, targets, sigma, mu, numEstimations,
                              randomSeed + groupIndex, groupIndex);

    if (g_log != nullptr) std::fclose(g_log);
    std::fprintf(stderr, "[stochasticrank_oracle] sigma=%.6f mu=%.6f num_estimations=%d (logged)\n",
                 sigma, mu, numEstimations);
    return 0;
}
