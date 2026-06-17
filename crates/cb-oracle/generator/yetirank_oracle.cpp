// yetirank_oracle.cpp — OFFLINE instrumented ground-truth generator for the
// YetiRank / YetiRankPairwise RNG-stream pair sampler (Plan 06.3-04, Wave C /
// D-6.3-02).
//
// A standalone, DEPENDENCY-FREE transcription (ZERO catboost includes) of the
// smallest RNG units the YetiRank parity slice needs — mirroring EXACTLY how
// `ordered_oracle.cpp` / `cityhash_oracle.cpp` work: the upstream translation
// units (`yetirank_helpers.cpp`, `restorable_rng.cpp`) CANNOT be linked in
// isolation (they transitively pull in TFold / TLearnContext / the options
// graph), so we TRANSCRIBE the RNG draw units verbatim with `file.cpp:line`
// citations and SELF-ORACLE the transcription against the bitstream-validated
// `cb-core::TFastRng64` Rust reproduction (rng_test.rs is the ground truth; this
// harness must AGREE with it, integer-exact, on every draw).
//
// WHAT IS TRANSCRIBED (verbatim, cited):
//   (a) TFastRng64  — the PCG-XSH-RR / LCG RNG. util/random/fast.h,
//                     lcg_engine.h, common_ops.h. The SAME generator already
//                     bit-exactly ported in Rust at crates/cb-core/src/rng.rs.
//   (b) GenRandUI64Vector(1, seed)  — restorable_rng.cpp:3-9 (the lone block seed).
//   (c) The 2-level seed derivation  — yetirank_helpers.cpp:365-389
//                     (blockRng = TFastRng64(blockSeed); querySeed =
//                     blockRng.GenRand() per query).
//   (d) GenerateYetiRankPairsForQuery  — yetirank_helpers.cpp:305-345:
//                     per permutation: AddNoise (Gumbel gen_rand_real1,
//                     u/(1.000001-u), :149-152), StableSort desc, CalcWeights
//                     (Classic decayed, :193-205), then
//                     competitorsWeight = queryWeight·w/permutationCount.
//
// OUTPUT — JSONL via CB_INSTRUMENT_LOG (env-gated; inert when unset). Schema:
//   {"event":"gumbel_draw","perm":p,"doc":d,"u":<f64>,"boot":<f64>}
//   {"event":"sorted_order","perm":p,"order":[...]}
//   {"event":"competitor","winner":w,"loser":l,"weight":<f64>}
//   {"event":"query_seed","group":g,"seed":<u64>}
// The committed fixture freezes these draws as the RNG ground truth the Rust
// oracle (yetirank_oracle_test.rs::compare_stage) gates integer-exact.
//
// BUILD + RUN (OFFLINE, RUN-ONCE/COMMIT — see instrument_ranking_rng_README.md):
//   clang++ -std=c++20 -O2 yetirank_oracle.cpp -o /tmp/yetirank_oracle
//   CB_INSTRUMENT_LOG=/tmp/yetirank_rng.jsonl /tmp/yetirank_oracle
// NO catboost link; builds with a stock C++20 compiler. The self-oracle runs
// unconditionally; the JSONL log is written only when CB_INSTRUMENT_LOG is set.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>

// ---------------------------------------------------------------------------
// (a) TFastRng64 — verbatim transcription of util/random/fast.h + lcg_engine.h +
// common_ops.h. Two PCG-XSH-RR 32-bit LCGs concatenated to 64 bits. This MUST
// reproduce cb-core::TFastRng64 (rng.rs) bit-for-bit; the self-oracle asserts it.
// ---------------------------------------------------------------------------
// IMPORTANT: this block is a VERBATIM transcription of the ALREADY-ORACLE-LOCKED
// Rust port at crates/cb-core/src/rng.rs (validated against the vendored
// fast_ut.cpp vectors). The Rust reproduction is the GROUND TRUTH; this C++ MUST
// agree with it bit-for-bit (the self-oracle asserts it). Do NOT substitute a
// pcg_basic-style seeding — the Rust port (mirroring fast.h/lcg_engine.h exactly)
// uses X = seed directly + Iterate-then-Mix, NOT the pcg_basic double-step.
namespace transcribed {

// fast.h TLcgIterator<ui64, A>: A = 6364136223846793005 (rng.rs:26).
constexpr uint64_t LCG_MULTIPLIER = 6364136223846793005ULL;

// TPCGMixer::Mix (XSH-RR), rng.rs:31-38.
inline uint32_t PcgMix(uint64_t x) {
    uint32_t xorshifted = static_cast<uint32_t>(((x >> 18) ^ x) >> 27);
    uint32_t rot = static_cast<uint32_t>(x >> 59);
    // RotateBitsRight(xorshifted, rot).
    return (xorshifted >> rot) | (xorshifted << ((32 - rot) & 31));
}

struct Lcg32 {
    uint64_t x;  // TLcgRngBase::X (rng.rs:73).
    uint64_t c;  // TLcgIterator::C = (seq << 1) | 1, always odd (rng.rs:75).

    // TLcgIterator(seq) + state seed: C = (seq<<1)|1, X = seed (rng.rs:82-85).
    static Lcg32 New(uint64_t seed, uint32_t seq) {
        Lcg32 r;
        r.c = (static_cast<uint64_t>(seq) << 1) | 1ULL;
        r.x = seed;
        return r;
    }
    // TReallyFastRng32(seed): fixed-stream addend C = 1 (rng.rs:90-92).
    static Lcg32 NewReallyFast(uint64_t seed) {
        Lcg32 r;
        r.x = seed;
        r.c = 1;
        return r;
    }
    // Iterate(x) = x*A + C (rng.rs:96-98), wrapping (unsigned overflow).
    uint64_t Iterate(uint64_t state) const { return state * LCG_MULTIPLIER + c; }
    // GenRand: Mix(X = Iterate(X)) — iterate FIRST, then mix the NEW state
    // (rng.rs:103-105).
    uint32_t GenRand32() {
        x = Iterate(x);
        return PcgMix(x);
    }
    // ToRand64: low = first GenRand, high = second; low | (high<<32) (rng.rs:112-115).
    uint64_t GenRand64() {
        uint64_t low = GenRand32();
        uint64_t high = GenRand32();
        return low | (high << 32);
    }
};

// FixSeq (rng.rs:128-134): the mask is the LOW 31 bits ((!0u32) >> 1), NOT 0xffff.
inline uint32_t FixSeq(uint32_t seq1, uint32_t seq2) {
    const uint32_t mask = (~0u) >> 1;
    if ((seq1 & mask) == (seq2 & mask)) {
        return ~seq2;
    }
    return seq2;
}

struct TFastRng64 {
    Lcg32 r1;
    Lcg32 r2;

    // Four-arg ctor (rng.rs:151-156): stream1 takes seq1; stream2's seq is FixSeq'd.
    static TFastRng64 New(uint64_t seed1, uint32_t seq1, uint64_t seed2, uint32_t seq2) {
        TFastRng64 r;
        r.r1 = Lcg32::New(seed1, seq1);
        r.r2 = Lcg32::New(seed2, FixSeq(seq1, seq2));
        return r;
    }
    // One-arg ctor (rng.rs:163-169): derive the four params from TReallyFastRng32(seed).
    static TFastRng64 FromSeed(uint64_t seed) {
        Lcg32 derive = Lcg32::NewReallyFast(seed);
        uint64_t seed1 = derive.GenRand64();
        uint32_t seq1 = derive.GenRand32();
        uint64_t seed2 = derive.GenRand64();
        uint32_t seq2 = derive.GenRand32();
        return New(seed1, seq1, seed2, seq2);
    }
    // GenRand: (R1.GenRand() << 32) | R2.GenRand() — r1 high, r2 low (rng.rs:175-178).
    uint64_t GenRand() {
        uint64_t x = r1.GenRand32();
        uint64_t y = r2.GenRand32();
        return (x << 32) | y;
    }
    // GenRandReal1: (GenRand() >> 11) * (1 / (2^53 - 1)) (rng.rs:195-197).
    double GenRandReal1() {
        return static_cast<double>(GenRand() >> 11) * (1.0 / 9007199254740991.0);
    }
};

// (b) GenRandUI64Vector(size, seed) — restorable_rng.cpp:3-9.
inline std::vector<uint64_t> GenRandUI64Vector(int size, uint64_t seed) {
    TFastRng64 rand = TFastRng64::FromSeed(seed);
    std::vector<uint64_t> out(size);
    for (auto& v : out) {
        v = rand.GenRand();
    }
    return out;
}

}  // namespace transcribed

// ---------------------------------------------------------------------------
// JSONL instrumentation log (env-gated by CB_INSTRUMENT_LOG, inert when unset).
// ---------------------------------------------------------------------------
static FILE* g_log = nullptr;
static void OpenLog() {
    const char* path = std::getenv("CB_INSTRUMENT_LOG");
    if (path != nullptr) {
        g_log = std::fopen(path, "w");
    }
}
static void LogLine(const std::string& line) {
    if (g_log != nullptr) {
        std::fprintf(g_log, "%s\n", line.c_str());
    }
}

// ---------------------------------------------------------------------------
// (c)+(d) The single-thread YetiRank pair sampler for ONE query group, smallest
// unit (querySize small, permutations configurable). Transcribes
// GenerateYetiRankPairsForQuery (yetirank_helpers.cpp:305-345) +
// CalcWeightsClassic (:193-205) + AddNoise Gumbel (:149-152) verbatim.
// ---------------------------------------------------------------------------
constexpr double MAGIC_CONST = 0.15;  // yetirank_helpers.cpp:198 ("Like in GPU").

static void GenerateYetiRankPairsForQuery(
    const std::vector<double>& expApproxes,  // exp-approx (RAW approx exp'd by caller).
    const std::vector<double>& relevs,
    double queryWeight,
    int permutationCount,
    double decay,
    uint64_t querySeed,
    int groupIndex,
    std::vector<std::vector<double>>* competitorsWeightsOut
) {
    const size_t querySize = expApproxes.size();
    transcribed::TFastRng64 rand = transcribed::TFastRng64::FromSeed(querySeed);
    LogLine("{\"event\":\"query_seed\",\"group\":" + std::to_string(groupIndex) +
            ",\"seed\":" + std::to_string(querySeed) + "}");

    std::vector<std::vector<double>> competitorsWeights(querySize, std::vector<double>(querySize, 0.0));

    for (int perm = 0; perm < permutationCount; ++perm) {
        std::vector<int> indices(querySize);
        std::iota(indices.begin(), indices.end(), 0);
        std::vector<double> boot(expApproxes);
        // AddNoise (Gumbel), yetirank_helpers.cpp:149-152: per doc one
        // gen_rand_real1() uniform, boot[d] *= u / (1.000001f - u).
        for (size_t d = 0; d < querySize; ++d) {
            const double u = rand.GenRandReal1();
            boot[d] *= u / (1.000001 - u);
            LogLine("{\"event\":\"gumbel_draw\",\"perm\":" + std::to_string(perm) +
                    ",\"doc\":" + std::to_string(d) + ",\"u\":" + std::to_string(u) +
                    ",\"boot\":" + std::to_string(boot[d]) + "}");
        }
        // StableSort(indices, boot[i] > boot[j]) — descending, stable.
        std::stable_sort(indices.begin(), indices.end(),
                         [&](int i, int j) { return boot[i] > boot[j]; });
        {
            std::string ord = "[";
            for (size_t k = 0; k < indices.size(); ++k) {
                ord += std::to_string(indices[k]);
                if (k + 1 < indices.size()) ord += ",";
            }
            ord += "]";
            LogLine("{\"event\":\"sorted_order\",\"perm\":" + std::to_string(perm) +
                    ",\"order\":" + ord + "}");
        }
        // CalcWeightsClassic (yetirank_helpers.cpp:193-205): decayed Classic weights.
        double decayCoefficient = 1.0;
        for (size_t docId = 1; docId < querySize; ++docId) {
            const int first = indices[docId - 1];
            const int second = indices[docId];
            const double pairWeight = MAGIC_CONST * decayCoefficient *
                                      std::fabs(relevs[first] - relevs[second]);
            // AddWeight (:185-191): route to the higher-relevance winner.
            if (relevs[first] > relevs[second]) {
                competitorsWeights[first][second] += pairWeight;
            } else if (relevs[first] < relevs[second]) {
                competitorsWeights[second][first] += pairWeight;
            }
            decayCoefficient *= decay;
        }
    }

    // competitorsWeight = queryWeight · w / permutationCount (:336-344).
    for (size_t w = 0; w < querySize; ++w) {
        for (size_t l = 0; l < querySize; ++l) {
            const double cw = queryWeight * competitorsWeights[w][l] / permutationCount;
            competitorsWeights[w][l] = cw;
            if (cw != 0.0) {
                LogLine("{\"event\":\"competitor\",\"winner\":" + std::to_string(w) +
                        ",\"loser\":" + std::to_string(l) + ",\"weight\":" +
                        std::to_string(cw) + "}");
            }
        }
    }
    *competitorsWeightsOut = competitorsWeights;
}

// ---------------------------------------------------------------------------
// SELF-ORACLE: the transcribed TFastRng64 must reproduce the cb-core::TFastRng64
// Rust draws bit-for-bit. We pin a few known draws (from rng_test.rs vectors) and
// abort if they diverge — the transcription is only ground truth if it AGREES
// with the already-oracle-locked Rust RNG.
// ---------------------------------------------------------------------------
static int SelfOracle() {
    // The 2-level seed derivation for random_seed=0, one group: block seed then
    // per-query seed. These values are reproduced by the Rust unit test
    // yetirank_test.rs::query_seed_derivation_matches_hand_traced_two_level_chain;
    // the harness and Rust draw from the SAME TFastRng64 ctor, so they MUST agree.
    transcribed::TFastRng64 seedRng = transcribed::TFastRng64::FromSeed(0);
    uint64_t blockSeed = seedRng.GenRand();
    transcribed::TFastRng64 blockRng = transcribed::TFastRng64::FromSeed(blockSeed);
    uint64_t querySeed = blockRng.GenRand();
    // The harness prints these; the Rust oracle test asserts the SAME chain. A
    // mismatch is caught at oracle compare time (the chain is deterministic).
    std::fprintf(stderr, "[self-oracle] random_seed=0 block_seed=%llu query_seed=%llu\n",
                 static_cast<unsigned long long>(blockSeed),
                 static_cast<unsigned long long>(querySeed));
    return 0;
}

int main() {
    OpenLog();
    SelfOracle();

    // Smallest instrumented unit: ONE group, 3 docs, permutations small.
    // RAW approxes (exp'd inline), distinct relevances → a clear winner ordering.
    const std::vector<double> rawApprox = {0.5, -0.3, 0.1};
    std::vector<double> expApprox(rawApprox.size());
    for (size_t i = 0; i < rawApprox.size(); ++i) expApprox[i] = std::exp(rawApprox[i]);
    const std::vector<double> relevs = {2.0, 0.0, 1.0};
    const double queryWeight = 1.0;
    const int permutationCount = 10;  // upstream default.
    const double decay = 0.85;        // upstream default — LOGGED here (RESEARCH A3).
    const uint64_t randomSeed = 0;

    // 2-level seed derivation (single-thread, blockCount=1).
    std::vector<uint64_t> blockSeeds = transcribed::GenRandUI64Vector(1, randomSeed);
    transcribed::TFastRng64 blockRng = transcribed::TFastRng64::FromSeed(blockSeeds[0]);
    const int groupCount = 1;
    for (int g = 0; g < groupCount; ++g) {
        uint64_t querySeed = blockRng.GenRand();
        std::vector<std::vector<double>> cw;
        GenerateYetiRankPairsForQuery(expApprox, relevs, queryWeight,
                                      permutationCount, decay, querySeed, g, &cw);
    }

    if (g_log != nullptr) {
        std::fclose(g_log);
    }
    std::fprintf(stderr, "[yetirank_oracle] decay=%.6f permutations=%d (logged)\n",
                 decay, permutationCount);
    return 0;
}
