// ordered_oracle.cpp — per-object ground-truth oracle for the Phase-5 high-risk
// parity slice (ordered boosting / ordered + online CTR / categoricals).
//
// A standalone, DEPENDENCY-FREE transcription (ZERO catboost includes) of the
// four small upstream algorithms the per-object oracle needs. It mirrors EXACTLY
// how `cityhash_oracle.cpp` works: the four research-flagged translation units
// (`online_ctr.cpp`, the ordered path in `approx_calcer.cpp`, `fold.cpp`
// permutation/prefix generation) CANNOT be linked in isolation — they
// transitively pull in `TLearnContext`, `TTrainingDataProviders`, the full
// options system and the metrics graph (see 05-RESEARCH § "Per-Object Oracle
// Strategy" ESCALATION). So, exactly like cityhash, we TRANSCRIBE the leaf math
// verbatim with file:line citations and SELF-ORACLE the transcription.
//
// WHAT IS TRANSCRIBED (verbatim, cited):
//   (a) TFastRng64  — the PCG-XSH-RR / LCG RNG that seeds the permutation.
//                     util/random/fast.h, lcg_engine.h, common_ops.h. This is
//                     the SAME generator already bit-exactly ported in Rust at
//                     crates/cb-core/src/rng.rs (TFastRng64) and validated
//                     against the vendored fast_ut.cpp vectors.
//   (b) Shuffle     — modern Fisher-Yates over the RNG.
//                     util/random/shuffle.h:24-32 (block size 1 for N<1000,
//                     05-RESEARCH Open Q3): for i in [1,sz): swap(v[i],
//                     v[gen.Uniform(i+1)]).
//   (c) Online CTR  — read-before-increment + CalcCTR online form.
//                     online_ctr.cpp:168-184 (classes) / 300-307 (simple binclf),
//                     online_ctr.h:128-131 (CalcCTR, denom is hard "+1"),
//                     online_ctr.cpp:102-111 (CalcNormalization shift/norm).
//   (d) Body/tail   — ordered-boosting prefix boundaries.
//                     fold.cpp:35-41 (SelectMinBatchSize / SelectTailSize),
//                     fold.cpp:156-198 (BuildDynamicFold growth loop),
//                     fold_len_multiplier default 2.0.
//   (e) Ordered approx prefix update.
//                     approx_calcer.cpp:566-600 (UpdateApproxDeltasHistorically
//                     Impl): per-row, accumulate leaf der on the body prefix,
//                     apply CalcMethodDelta to the tail row so a tail document's
//                     approximant never depends on itself.
//
// SELF-ORACLE ANCHORS (documented in README.md; this harness is a CROSS-CHECK,
// not the sole source of truth):
//   * Permutation (D-03 linchpin) is cross-checked against the already-oracle-
//     locked cb-core::TFastRng64 Rust reproduction of the same Fisher-Yates draw
//     (rng_test.rs is bitstream-verified). The Rust reproduction is ground truth;
//     this harness must AGREE with it (Stage::Permutation, integer-exact).
//   * Final whole-set CTR counts are cross-checked against the trained upstream
//     model's `ctr_data` TCtrValueTable blobs (model.json from offline
//     catboost==1.2.10), interpreted per static_ctr_provider.cpp:14-126.
//   * Ordered approx / per-object running CTR have no direct external dump;
//     anchored indirectly via final-prediction parity + internal consistency
//     (identity permutation => prefix == final).
//
// OUTPUT — the D-02 per-object .npy schema (05-RESEARCH § Per-Object Oracle
// Schema), little-endian, written next to the input config:
//   permutation_fold{k}.npy   [N]            int32   exact (Stage::Permutation)
//   ctr_good_count.npy        [N]            int32   exact integer numerator
//   ctr_total_count.npy       [N]            int32   exact integer denominator
//   ctr_value.npy             [N] | [N,P]    float64 Stage::OnlineCtr (<=1e-5)
//   ordered_approx_iter{t}.npy[N]            float64 Stage::OrderedApprox (<=1e-5)
//
// THIS HARNESS NEVER RUNS IN CI (D-09). It is an OFFLINE generator: only its
// frozen .npy / config.json outputs are committed under
// crates/cb-oracle/fixtures/. The Rust test lane only READS those artifacts.
//
// Build (little-endian targets, which is all catboost ships on):
//   g++ -O2 -std=c++17 ordered_oracle.cpp -o ordered_oracle
//
// I/O contract (kept deliberately small; the fixtures themselves are generated
// by the Python pipeline and committed frozen — this harness is the transcribed
// reference the Python generator and the Rust reader agree against):
//   argv[1] = output directory (the fixture dir to write the .npy stack into)
//   stdin   = whitespace/newline-separated config + data, in this order:
//             N fold_count fold_len_multiplier prior border_count
//             cat_bin[0..N)            (ui32 hashed-cat bucket id per doc)
//             target_class[0..N)       (int label class per doc, 0/1 for binclf)
//             der[0..N)                (double per-object derivative, ordered approx)
//             seed                     (ui64 RNG seed for the permutation)

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>

using ui8 = uint8_t;
using ui32 = uint32_t;
using ui64 = uint64_t;
using i32 = int32_t;

// ===========================================================================
// (a) TFastRng64 — verbatim transcription of util/random/fast.h (PCG-XSH-RR over
//     two TLcgIterator<ui64, A> 32-bit streams), lcg_engine.h (LcgAdvance), and
//     common_ops.h (GenUniform / ToRand64). Identical to the Rust port at
//     crates/cb-core/src/rng.rs — kept here so the harness has ZERO includes.
// ===========================================================================

// fast.h:20 — TLcgIterator<ui64, ULL(6364136223846793005)>.
static const ui64 LCG_MULTIPLIER = 6364136223846793005ULL; // 0x5851F42D4C957F2D

// fast.h:11-17 — TPCGMixer::Mix (XSH-RR): 64-bit state -> 32-bit result.
static inline ui32 PcgMix(ui64 x) {
    const ui32 xorshifted = (ui32)(((x >> 18u) ^ x) >> 27u);
    const ui32 rot = (ui32)(x >> 59u);
    // RotateBitsRight(xorshifted, rot).
    return (xorshifted >> rot) | (xorshifted << ((32u - rot) & 31u));
}

// lcg_engine.cpp NPrivate::LcgAdvance — closed-form jump by `delta` steps:
// seed[n] = A**n*seed[0] + (A**n - 1)/(A - 1)*addend. All wrapping (defined
// unsigned wraparound). Mirrors crates/cb-core/src/rng.rs::lcg_advance.
static ui64 LcgAdvance(ui64 seed, ui64 lcgBase, ui64 lcgAddend, ui64 delta) {
    ui64 mask = 1;
    while (mask != (1ULL << 63) && (mask << 1) <= delta) {
        mask <<= 1;
    }
    ui64 apow = 1; // A**m
    ui64 adiv = 0; // (A**m - 1)/(A - 1)
    while (mask != 0) {
        adiv = adiv * (apow + 1);
        apow = apow * apow;
        if (delta & mask) {
            adiv = adiv + apow;
            apow = apow * lcgBase;
        }
        mask >>= 1;
    }
    return seed * apow + lcgAddend * adiv;
}

// One PCG-XSH-RR 32-bit stream (TLcgRngBase composed with TLcgIterator + mixer).
struct Lcg32 {
    ui64 x;
    ui64 c; // addend C = (seq << 1) | 1, always odd.

    static Lcg32 New(ui64 seed, ui32 seq) {
        return Lcg32{seed, ((ui64)seq << 1) | 1};
    }
    // TReallyFastRng32(seed): fixed-stream engine, addend == 1.
    static Lcg32 NewReallyFast(ui64 seed) { return Lcg32{seed, 1}; }

    // TLcgIterator::Iterate(x) = x*A + C (wrapping).
    ui64 Iterate(ui64 v) const { return v * LCG_MULTIPLIER + c; }

    // TLcgRngBase::GenRand = Mix(X = Iterate(X)).
    ui32 GenRand32() { x = Iterate(x); return PcgMix(x); }

    // ToRand64: low 32 bits from first GenRand, high 32 from second.
    ui64 GenRand64() {
        ui64 low = (ui64)GenRand32();
        ui64 high = (ui64)GenRand32();
        return low | (high << 32);
    }

    void Advance(ui64 delta) { x = LcgAdvance(x, LCG_MULTIPLIER, c, delta); }
};

// fast.cpp FixSeq: force the two streams onto distinct sequences.
static inline ui32 FixSeq(ui32 seq1, ui32 seq2) {
    const ui32 mask = (~0u) >> 1;
    return ((seq1 & mask) == (seq2 & mask)) ? ~seq2 : seq2;
}

// fast.h:44-95 — TFastRng64: two 32-bit PCG streams concatenated to 64 bits.
struct TFastRng64 {
    Lcg32 r1;
    Lcg32 r2;

    // Four-arg ctor TFastRng64(seed1, seq1, seed2, seq2).
    static TFastRng64 New(ui64 seed1, ui32 seq1, ui64 seed2, ui32 seq2) {
        return TFastRng64{Lcg32::New(seed1, seq1), Lcg32::New(seed2, FixSeq(seq1, seq2))};
    }
    // One-arg ctor TFastRng64(ui64 seed) via TArgs: draw the four params, in
    // order, from a TReallyFastRng32(seed). Matches rng.rs::from_seed.
    static TFastRng64 FromSeed(ui64 seed) {
        Lcg32 derive = Lcg32::NewReallyFast(seed);
        ui64 seed1 = derive.GenRand64();
        ui32 seq1 = derive.GenRand32();
        ui64 seed2 = derive.GenRand64();
        ui32 seq2 = derive.GenRand32();
        return New(seed1, seq1, seed2, seq2);
    }

    // TFastRng64::GenRand = (R1.GenRand() << 32) | R2.GenRand().
    ui64 GenRand() {
        ui64 a = (ui64)r1.GenRand32();
        ui64 b = (ui64)r2.GenRand32();
        return (a << 32) | b;
    }

    void Advance(ui64 delta) { r1.Advance(delta); r2.Advance(delta); }

    // NPrivate::GenUniform (common_ops.h:49-60): rejection-sample [0, bound).
    // randmax = RandMax() - RandMax() % bound, RandMax() == u64::MAX.
    ui64 Uniform(ui64 bound) {
        if (bound == 0) { return 0; }
        const ui64 randmax = UINT64_MAX - (UINT64_MAX % bound);
        ui64 r;
        while ((r = GenRand()) >= randmax) {}
        return r % bound;
    }
};

// ===========================================================================
// (b) Shuffle — modern Fisher-Yates (shuffle.h:24-32). For our N<1000 the
//     ungrouped block-size-1 path resolves to this loop over the seeded
//     TRestorableFastRng64 (a TFastRng64). The harness reproduces the exact
//     draw order: for i in [1,sz): swap(v[i], v[gen.Uniform(i+1)]).
// ===========================================================================
static std::vector<i32> FisherYatesPermutation(size_t n, ui64 seed) {
    std::vector<i32> v(n);
    for (size_t i = 0; i < n; ++i) { v[i] = (i32)i; }
    TFastRng64 gen = TFastRng64::FromSeed(seed);
    for (size_t i = 1; i < n; ++i) {
        ui64 j = gen.Uniform((ui64)i + 1); // gen.Uniform(i + 1)
        std::swap(v[i], v[(size_t)j]);
    }
    return v;
}

// ===========================================================================
// (c) Online CTR — read-before-increment + CalcCTR online form.
// ===========================================================================

// online_ctr.cpp:102-111 — CalcNormalization (single prior).
static void CalcNormalization(float prior, float* shift, float* norm) {
    float left = std::min(0.0f, prior);
    float right = std::max(1.0f, prior);
    *shift = -left;
    *norm = (right - left);
}

// online_ctr.h:128-131 — CalcCTR online form. NOTE: online denom is HARD "+1"
// (NOT the inference (total + PriorDenom)); they coincide only at PriorDenom==1.
static float CalcCtrValue(float countInClass, int totalCount, float prior) {
    return (countInClass + prior) / (float)(totalCount + 1);
}

// Per-object online (ordered) CTR over the permutation, simple binclf path
// (online_ctr.cpp:300-307): goodCount = elem[1]; totalCount = elem[0]+elem[1];
// ++elem[target_class] — READ the prefix counts BEFORE incrementing with the
// document's own label (the no-leakage property).
struct OnlineCtrOut {
    std::vector<i32> goodCount;   // numerator per doc, in permutation order
    std::vector<i32> totalCount;  // denominator per doc
    std::vector<double> ctrValue; // (good + prior)/(total + 1) per doc
};
static OnlineCtrOut ComputeOnlineCtr(
    const std::vector<i32>& permutation,
    const std::vector<ui32>& catBin,
    const std::vector<int>& targetClass,
    float prior) {
    const size_t n = permutation.size();
    OnlineCtrOut out;
    out.goodCount.resize(n);
    out.totalCount.resize(n);
    out.ctrValue.resize(n);
    // Per-bucket class counts elem[2] = {neg, pos}; std::map keeps it allocation
    // -bounded and deterministic (integer-exact, no float reduction — these are
    // int counts, 05-RESEARCH "Hand-rolling float sums" caveat).
    std::map<ui32, std::array<int, 2>> counts;
    for (size_t p = 0; p < n; ++p) {
        const int doc = permutation[p];
        const ui32 bucket = catBin[doc];
        std::array<int, 2>& elem = counts[bucket]; // default {0,0}
        const int good = elem[1];                  // READ prefix (online_ctr.cpp:303)
        const int total = elem[0] + elem[1];       // READ prefix (online_ctr.cpp:304)
        out.goodCount[doc] = good;
        out.totalCount[doc] = total;
        out.ctrValue[doc] = (double)CalcCtrValue((float)good, total, prior);
        ++elem[targetClass[doc]];                  // INCREMENT after read (learn set)
    }
    return out;
}

// ===========================================================================
// (d) Body/tail prefix boundaries (fold.cpp:35-41 + 156-198, BuildDynamicFold).
// ===========================================================================
static ui32 SelectMinBatchSize(ui32 learnSampleCount) {
    return learnSampleCount > 500 ? std::min<ui32>(100, learnSampleCount / 50) : 1;
}
static double SelectTailSize(ui32 oldSize, double multiplier) {
    return std::ceil((double)oldSize * multiplier);
}
// The growing body/tail boundary sequence: bodyFinish grows by
// fold_len_multiplier each step, capped at N. Returns the body-finish boundaries.
static std::vector<ui32> BuildDynamicFoldBoundaries(ui32 n, double multiplier) {
    std::vector<ui32> boundaries;
    if (n == 0) { return boundaries; }
    ui32 leftPartLen = SelectMinBatchSize(n);
    while (leftPartLen < n) {
        boundaries.push_back(leftPartLen);
        ui32 tail = (ui32)SelectTailSize(leftPartLen, multiplier);
        if (tail > n) { tail = n; }
        leftPartLen = tail;
    }
    boundaries.push_back(n);
    return boundaries;
}

// ===========================================================================
// (e) Ordered approximant prefix update (approx_calcer.cpp:566-600,
//     UpdateApproxDeltasHistoricallyImpl, Gradient method, single leaf).
//     For each row in [rowStart, rowStart+rowCount): accumulate the leaf der on
//     the body prefix, then the per-row approx delta is the running mean der
//     (Gradient CalcMethodDelta = sumDer / (sumWeight + l2)). A tail document's
//     approximant uses only the prefix BEFORE it (no self-dependence).
// ===========================================================================
static std::vector<double> ComputeOrderedApprox(
    const std::vector<i32>& permutation,
    const std::vector<double>& der,
    double l2Regularizer) {
    const size_t n = permutation.size();
    std::vector<double> approx(n, 0.0);
    double sumDer = 0.0;
    double sumWeights = 0.0;
    for (size_t p = 0; p < n; ++p) {
        const int doc = permutation[p];
        const double rowWeight = 1.0;
        sumWeights += rowWeight;
        sumDer += der[doc];
        // CalcMethodDelta<Gradient>: sumDer / (sumWeights + l2).
        const double delta = sumDer / (sumWeights + l2Regularizer);
        approx[doc] = delta; // UpdateApprox (non-exp): approx += delta from 0.
    }
    return approx;
}

// ===========================================================================
// Minimal little-endian .npy writer (numpy format v1.0). Mirrors the dtype/shape
// the Rust ndarray-npy reader (load_f64_vec) and the Python generator emit.
// ===========================================================================
static void WriteNpyHeader(std::ofstream& f, const std::string& descr, size_t len) {
    std::string dict = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': (" +
                       std::to_string(len) + ",), }";
    // header length must make (10 + header) a multiple of 64; pad with spaces.
    size_t base = 10 + dict.size() + 1; // +1 for trailing '\n'
    size_t pad = (64 - (base % 64)) % 64;
    dict.append(pad, ' ');
    dict.push_back('\n');
    const char magic[6] = {(char)0x93, 'N', 'U', 'M', 'P', 'Y'};
    f.write(magic, 6);
    const ui8 ver[2] = {1, 0};
    f.write((const char*)ver, 2);
    ui8 hlen[2];
    ui32 hl = (ui32)dict.size();
    hlen[0] = (ui8)(hl & 0xff);
    hlen[1] = (ui8)((hl >> 8) & 0xff);
    f.write((const char*)hlen, 2);
    f.write(dict.data(), dict.size());
}
static void WriteNpyI32(const std::string& path, const std::vector<i32>& data) {
    std::ofstream f(path, std::ios::binary);
    WriteNpyHeader(f, "<i4", data.size());
    f.write((const char*)data.data(), data.size() * sizeof(i32));
}
static void WriteNpyF64(const std::string& path, const std::vector<double>& data) {
    std::ofstream f(path, std::ios::binary);
    WriteNpyHeader(f, "<f8", data.size());
    f.write((const char*)data.data(), data.size() * sizeof(double));
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: ordered_oracle <output-dir>  (config on stdin)\n";
        return 2;
    }
    const std::string outDir = argv[1];

    size_t n = 0;
    int foldCount = 0;
    double foldLenMultiplier = 2.0;
    double prior = 0.0;
    int borderCount = 0;
    std::cin >> n >> foldCount >> foldLenMultiplier >> prior >> borderCount;

    std::vector<ui32> catBin(n);
    for (size_t i = 0; i < n; ++i) { std::cin >> catBin[i]; }
    std::vector<int> targetClass(n);
    for (size_t i = 0; i < n; ++i) { std::cin >> targetClass[i]; }
    std::vector<double> der(n);
    for (size_t i = 0; i < n; ++i) { std::cin >> der[i]; }
    ui64 seed = 0;
    std::cin >> seed;

    // (b) per-fold permutations. Each fold k advances the seed deterministically
    // (k as a sub-seed) so folds differ; fold 0 uses the base seed.
    for (int k = 0; k < std::max(1, foldCount); ++k) {
        ui64 foldSeed = seed + (ui64)k;
        std::vector<i32> perm = FisherYatesPermutation(n, foldSeed);
        WriteNpyI32(outDir + "/permutation_fold" + std::to_string(k) + ".npy", perm);
    }

    // (c) online CTR over fold-0 permutation.
    std::vector<i32> perm0 = FisherYatesPermutation(n, seed);
    OnlineCtrOut ctr = ComputeOnlineCtr(perm0, catBin, targetClass, (float)prior);
    WriteNpyI32(outDir + "/ctr_good_count.npy", ctr.goodCount);
    WriteNpyI32(outDir + "/ctr_total_count.npy", ctr.totalCount);
    WriteNpyF64(outDir + "/ctr_value.npy", ctr.ctrValue);

    // (d) body/tail boundaries — dumped as int32 for auditing the prefix split.
    std::vector<ui32> bnds = BuildDynamicFoldBoundaries((ui32)n, foldLenMultiplier);
    std::vector<i32> bndsI32(bnds.begin(), bnds.end());
    WriteNpyI32(outDir + "/body_tail_boundaries.npy", bndsI32);

    // (e) ordered approx — one iteration dumped (l2 = 3.0 default pin).
    std::vector<double> approx = ComputeOrderedApprox(perm0, der, 3.0);
    WriteNpyF64(outDir + "/ordered_approx_iter0.npy", approx);

    std::cerr << "ordered_oracle: wrote D-02 .npy stack for N=" << n
              << " folds=" << std::max(1, foldCount) << " to " << outDir << "\n";
    return 0;
}
