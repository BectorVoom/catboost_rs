// cityhash_oracle.cpp — authoritative CalcCatFeatureHash oracle.
//
// A standalone, dependency-free transcription of Yandex CatBoost's CityHash 1.0
// variant (`catboost-master/util/digest/city.cpp` / `city.h`) and its
// `CalcCatFeatureHash = CityHash64(s) & 0xffffffff` reduction
// (`catboost-master/catboost/libs/cat_feature/cat_feature.cpp:6-8`).
//
// WHY THIS EXISTS (oracle correctness): the previous cat_hash fixtures were
// extracted from a trained model's `ctr_data` `hash_map`, which stores
// CTR-PROJECTION hashes (CalcHashes over projections combined via MultiHash and
// priors — see catboost/private/libs/algo/index_hash_calcer.h), NOT the raw
// `CalcCatFeatureHash(string)`. Those values are therefore the wrong oracle
// target for a CityHash64 port. This tool computes the true `CalcCatFeatureHash`
// directly from the same algorithm the live catboost library compiles, making it
// the authoritative source of truth for the (string -> ui32/ui64) vectors.
//
// I/O: reads one UTF-8 string per line from stdin (newline-terminated, the
// newline stripped; the string bytes themselves are hashed verbatim) and writes
// `<ui64_decimal>\t<ui32_decimal>` per line to stdout, preserving input order.
//
// Build (little-endian targets, which is what catboost ships on):
//   g++ -O2 -std=c++17 cityhash_oracle.cpp -o cityhash_oracle

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <utility>
#include <string>
#include <iostream>

using uint8 = uint8_t;
using uint32 = uint32_t;
using uint64 = uint64_t;
using uint128 = std::pair<uint64, uint64>;

// UNALIGNED_LOAD64/32 = ReadUnaligned<uiN> (city.cpp:45-46): little-endian load
// (catboost ships only on little-endian targets, so the raw load is a LE decode).
static inline uint64 U64(const char* p) { uint64 v; memcpy(&v, p, 8); return v; }
static inline uint32 U32(const char* p) { uint32 v; memcpy(&v, p, 4); return v; }

// Primes (city.cpp:50-54).
static const uint64 k0 = 0xc3a5c85c97cb3127ULL;
static const uint64 k1 = 0xb492b66fbe98f273ULL;
static const uint64 k2 = 0x9ae16a3b2f90404fULL;
static const uint64 k3 = 0xc949d7c7509e6557ULL;

static uint64 Rotate(uint64 val, int shift) {
    return shift == 0 ? val : ((val >> shift) | (val << (64 - shift)));
}
static uint64 RotateByAtLeast1(uint64 val, int shift) {
    return (val >> shift) | (val << (64 - shift));
}
static uint64 ShiftMix(uint64 val) { return val ^ (val >> 47); }

static inline uint64 Hash128to64(const uint128& x) {
    const uint64 kMul = 0x9ddfea08eb382d69ULL;
    uint64 a = (x.first ^ x.second) * kMul;
    a ^= (a >> 47);
    uint64 b = (x.second ^ a) * kMul;
    b ^= (b >> 47);
    b *= kMul;
    return b;
}
static uint64 HashLen16(uint64 u, uint64 v) { return Hash128to64(uint128(u, v)); }

static uint64 HashLen0to16(const char* s, size_t len) {
    if (len > 8) {
        uint64 a = U64(s);
        uint64 b = U64(s + len - 8);
        return HashLen16(a, RotateByAtLeast1(b + len, (int)len)) ^ b;
    }
    if (len >= 4) {
        uint64 a = U32(s);
        return HashLen16(len + (a << 3), U32(s + len - 4));
    }
    if (len > 0) {
        uint8 a = s[0];
        uint8 b = s[len >> 1];
        uint8 c = s[len - 1];
        uint32 y = (uint32)a + ((uint32)b << 8);
        uint32 z = (uint32)len + ((uint32)c << 2);
        return ShiftMix(y * k2 ^ z * k3) * k2;
    }
    return k2;
}

static uint64 HashLen17to32(const char* s, size_t len) {
    uint64 a = U64(s) * k1;
    uint64 b = U64(s + 8);
    uint64 c = U64(s + len - 8) * k2;
    uint64 d = U64(s + len - 16) * k0;
    return HashLen16(Rotate(a - b, 43) + Rotate(c, 30) + d,
                     a + Rotate(b ^ k3, 20) - c + len);
}

static std::pair<uint64, uint64> WeakHashLen32WithSeeds(
    uint64 w, uint64 x, uint64 y, uint64 z, uint64 a, uint64 b) {
    a += w;
    b = Rotate(b + a + z, 21);
    uint64 c = a;
    a += x;
    a += y;
    b += Rotate(a, 44);
    return std::make_pair(a + z, b + c);
}
static std::pair<uint64, uint64> WeakHashLen32WithSeeds(const char* s, uint64 a, uint64 b) {
    return WeakHashLen32WithSeeds(U64(s), U64(s + 8), U64(s + 16), U64(s + 24), a, b);
}

static uint64 HashLen33to64(const char* s, size_t len) {
    uint64 z = U64(s + 24);
    uint64 a = U64(s) + (len + U64(s + len - 16)) * k0;
    uint64 b = Rotate(a + z, 52);
    uint64 c = Rotate(a, 37);
    a += U64(s + 8);
    c += Rotate(a, 7);
    a += U64(s + 16);
    uint64 vf = a + z;
    uint64 vs = b + Rotate(a, 31) + c;
    a = U64(s + 16) + U64(s + len - 32);
    z = U64(s + len - 8);
    b = Rotate(a + z, 52);
    c = Rotate(a, 37);
    a += U64(s + len - 24);
    c += Rotate(a, 7);
    a += U64(s + len - 16);
    uint64 wf = a + z;
    uint64 ws = b + Rotate(a, 31) + c;
    uint64 r = ShiftMix((vf + ws) * k2 + (wf + vs) * k0);
    return ShiftMix(r * k0 + vs) * k2;
}

uint64 CityHash64(const char* s, size_t len) {
    if (len <= 32) {
        if (len <= 16) {
            return HashLen0to16(s, len);
        } else {
            return HashLen17to32(s, len);
        }
    } else if (len <= 64) {
        return HashLen33to64(s, len);
    }
    uint64 x = U64(s);
    uint64 y = U64(s + len - 16) ^ k1;
    uint64 z = U64(s + len - 56) ^ k0;
    auto v = WeakHashLen32WithSeeds(s + len - 64, len, y);
    auto w = WeakHashLen32WithSeeds(s + len - 32, len * k1, k0);
    z += ShiftMix(v.second) * k1;
    x = Rotate(z + x, 39) * k1;
    y = Rotate(y, 33) * k1;
    len = (len - 1) & ~static_cast<size_t>(63);
    do {
        x = Rotate(x + y + v.first + U64(s + 16), 37) * k1;
        y = Rotate(y + v.second + U64(s + 48), 42) * k1;
        x ^= w.second;
        y ^= v.first;
        z = Rotate(z ^ w.first, 33);
        v = WeakHashLen32WithSeeds(s, v.second * k1, x + w.first);
        w = WeakHashLen32WithSeeds(s + 32, z + w.second, y);
        std::swap(z, x);
        s += 64;
        len -= 64;
    } while (len != 0);
    return HashLen16(HashLen16(v.first, w.first) + ShiftMix(y) * k1 + z,
                     HashLen16(v.second, w.second) + x);
}

int main() {
    std::string line;
    while (std::getline(std::cin, line)) {
        uint64 h = CityHash64(line.data(), line.size());
        uint32 h32 = (uint32)(h & 0xffffffffULL);
        std::cout << (unsigned long long)h << '\t' << (unsigned)h32 << '\n';
    }
    return 0;
}
