//! Categorical hashing: a bit-exact Rust port of Yandex CatBoost's CityHash64
//! (`catboost-master/util/digest/city.cpp` / `city.h`), the
//! `CalcCatFeatureHash = CityHash64(bytes) & 0xffffffff` reduction
//! (`catboost-master/catboost/libs/cat_feature/cat_feature.cpp:6-8`), and the
//! first-seen perfect-hash remap that turns category bytes into dense bin
//! indices (`catboost-master/catboost/libs/data/cat_feature_perfect_hash_helper.cpp`).
//!
//! # Non-cryptographic
//!
//! **`city_hash_64` is NOT cryptographically secure.** It is a deterministic
//! port of CityHash 1.0 whose sole purpose is to reproduce upstream CatBoost's
//! categorical-feature hashes bit-for-bit for parity (the categorical bins, and
//! later the CTR statistics of Phase 5, are keyed on it). Its output is fully
//! predictable from the input bytes. **Never** use it for secrets, tokens,
//! nonces, MACs, or anything where collision-resistance or unpredictability
//! matters (RESEARCH Security Domain V6, threat T-02-10). Note also that this is
//! the *Yandex* CityHash 1.0 variant (`city.h:8-9`: "results are *different* from
//! the mainline version of CityHash") — a generic `cityhash` crate would produce
//! different tail mixing and silently break categorical parity, so the algorithm
//! is transcribed here, not crate-sourced (RESEARCH Pitfall 4 / Open Q4).
//!
//! # Parity contract
//!
//! Every multiply / add uses `wrapping_*` so the arithmetic matches C++'s defined
//! unsigned wraparound and cannot trigger a debug-overflow panic. Multi-byte
//! little-endian loads mirror `ReadUnaligned<uiN>` on the little-endian targets
//! CatBoost ships (`city.cpp:45-46`). Validated bit-exactly against the
//! `(string -> ui32)` / `(string -> ui64)` vectors extracted from upstream
//! catboost 1.2.10 (`cb-oracle/fixtures/cat_hash/config.json`; see `cat_hash_test.rs`).

use cb_core::CbResult;
use cb_core::CbError;
use std::collections::HashMap;

// Some primes between 2^63 and 2^64 for various uses (`city.cpp:50-54`).
const K0: u64 = 0xc3a5_c85c_97cb_3127;
const K1: u64 = 0xb492_b66f_be98_f273;
const K2: u64 = 0x9ae1_6a3b_2f90_404f;
const K3: u64 = 0xc949_d7c7_509e_6557;

/// `UNALIGNED_LOAD64(p) = ReadUnaligned<uint64>(p)` (`city.cpp:45`): little-endian
/// 64-bit load of `bytes[off..off+8]`. CatBoost ships on little-endian targets, so
/// the raw memory load is a little-endian decode.
#[inline]
fn load64(bytes: &[u8], off: usize) -> u64 {
    let mut buf = [0u8; 8];
    if let Some(slice) = bytes.get(off..off + 8) {
        buf.copy_from_slice(slice);
    }
    u64::from_le_bytes(buf)
}

/// `UNALIGNED_LOAD32(p) = ReadUnaligned<uint32>(p)` (`city.cpp:46`): little-endian
/// 32-bit load of `bytes[off..off+4]`.
#[inline]
fn load32(bytes: &[u8], off: usize) -> u32 {
    let mut buf = [0u8; 4];
    if let Some(slice) = bytes.get(off..off + 4) {
        buf.copy_from_slice(slice);
    }
    u32::from_le_bytes(buf)
}

/// `Rotate(val, shift)` (`city.cpp:58-61`): bitwise right rotate. The C++ guards
/// `shift == 0` to avoid the undefined `>> 64`; Rust's `rotate_right` is defined
/// for every shift (and returns `val` unchanged at `shift == 0`), so it is the
/// exact, panic-free equivalent.
#[inline]
fn rotate(val: u64, shift: u32) -> u64 {
    // return shift == 0 ? val : ((val >> shift) | (val << (64 - shift)));
    val.rotate_right(shift)
}

/// `RotateByAtLeast1(val, shift)` (`city.cpp:66-68`): rotate requiring `shift != 0`.
/// `rotate_right` is the exact equivalent for every non-zero shift (and also for
/// the C++-undefined `shift == 0`, which this caller never produces).
#[inline]
fn rotate_by_at_least1(val: u64, shift: u32) -> u64 {
    // return (val >> shift) | (val << (64 - shift));
    val.rotate_right(shift)
}

/// `ShiftMix(val)` (`city.cpp:70-72`).
#[inline]
fn shift_mix(val: u64) -> u64 {
    // return val ^ (val >> 47);
    val ^ (val >> 47)
}

/// `Hash128to64(uint128 x)` (`city.h:36-45`): Murmur-inspired 128->64 mix.
#[inline]
fn hash_128_to_64(low: u64, high: u64) -> u64 {
    // const ui64 kMul = 0x9ddfea08eb382d69ULL;
    const K_MUL: u64 = 0x9ddf_ea08_eb38_2d69;
    // ui64 a = (Uint128Low64(x) ^ Uint128High64(x)) * kMul;
    let mut a = (low ^ high).wrapping_mul(K_MUL);
    // a ^= (a >> 47);
    a ^= a >> 47;
    // ui64 b = (Uint128High64(x) ^ a) * kMul;
    let mut b = (high ^ a).wrapping_mul(K_MUL);
    // b ^= (b >> 47);
    b ^= b >> 47;
    // b *= kMul;
    b = b.wrapping_mul(K_MUL);
    b
}

/// `HashLen16(u, v)` (`city.cpp:74-76`).
#[inline]
fn hash_len16(u: u64, v: u64) -> u64 {
    hash_128_to_64(u, v)
}

/// `HashLen0to16(s, len)` (`city.cpp:78-97`).
#[inline]
fn hash_len0to16(s: &[u8], len: usize) -> u64 {
    if len > 8 {
        // uint64 a = UNALIGNED_LOAD64(s);
        let a = load64(s, 0);
        // uint64 b = UNALIGNED_LOAD64(s + len - 8);
        let b = load64(s, len - 8);
        // return HashLen16(a, RotateByAtLeast1(b + len, len)) ^ b;
        return hash_len16(a, rotate_by_at_least1(b.wrapping_add(len as u64), len as u32)) ^ b;
    }
    if len >= 4 {
        // uint64 a = UNALIGNED_LOAD32(s);
        let a = load32(s, 0) as u64;
        // return HashLen16(len + (a << 3), UNALIGNED_LOAD32(s + len - 4));
        return hash_len16(
            (len as u64).wrapping_add(a << 3),
            load32(s, len - 4) as u64,
        );
    }
    if len > 0 {
        // uint8 a = s[0]; uint8 b = s[len >> 1]; uint8 c = s[len - 1];
        let a = *s.first().unwrap_or(&0);
        let b = *s.get(len >> 1).unwrap_or(&0);
        let c = *s.get(len - 1).unwrap_or(&0);
        // uint32 y = a + (b << 8);
        let y = (a as u32).wrapping_add((b as u32) << 8);
        // uint32 z = len + (c << 2);
        let z = (len as u32).wrapping_add((c as u32) << 2);
        // return ShiftMix(y * k2 ^ z * k3) * k2;
        return shift_mix((y as u64).wrapping_mul(K2) ^ (z as u64).wrapping_mul(K3))
            .wrapping_mul(K2);
    }
    // return k2;
    K2
}

/// `HashLen17to32(s, len)` (`city.cpp:101-108`).
#[inline]
fn hash_len17to32(s: &[u8], len: usize) -> u64 {
    // uint64 a = UNALIGNED_LOAD64(s) * k1;
    let a = load64(s, 0).wrapping_mul(K1);
    // uint64 b = UNALIGNED_LOAD64(s + 8);
    let b = load64(s, 8);
    // uint64 c = UNALIGNED_LOAD64(s + len - 8) * k2;
    let c = load64(s, len - 8).wrapping_mul(K2);
    // uint64 d = UNALIGNED_LOAD64(s + len - 16) * k0;
    let d = load64(s, len - 16).wrapping_mul(K0);
    // return HashLen16(Rotate(a - b, 43) + Rotate(c, 30) + d,
    //                  a + Rotate(b ^ k3, 20) - c + len);
    hash_len16(
        rotate(a.wrapping_sub(b), 43)
            .wrapping_add(rotate(c, 30))
            .wrapping_add(d),
        a.wrapping_add(rotate(b ^ K3, 20))
            .wrapping_sub(c)
            .wrapping_add(len as u64),
    )
}

/// `WeakHashLen32WithSeeds(w, x, y, z, a, b)` (`city.cpp:112-121`): returns the
/// `(first, second)` pair.
#[inline]
fn weak_hash_len32_with_seeds_raw(
    w: u64,
    x: u64,
    y: u64,
    z: u64,
    mut a: u64,
    mut b: u64,
) -> (u64, u64) {
    // a += w;
    a = a.wrapping_add(w);
    // b = Rotate(b + a + z, 21);
    b = rotate(b.wrapping_add(a).wrapping_add(z), 21);
    // uint64 c = a;
    let c = a;
    // a += x; a += y;
    a = a.wrapping_add(x);
    a = a.wrapping_add(y);
    // b += Rotate(a, 44);
    b = b.wrapping_add(rotate(a, 44));
    // return make_pair(a + z, b + c);
    (a.wrapping_add(z), b.wrapping_add(c))
}

/// `WeakHashLen32WithSeeds(s, a, b)` (`city.cpp:124-132`): the byte-string form.
#[inline]
fn weak_hash_len32_with_seeds(s: &[u8], off: usize, a: u64, b: u64) -> (u64, u64) {
    weak_hash_len32_with_seeds_raw(
        load64(s, off),
        load64(s, off + 8),
        load64(s, off + 16),
        load64(s, off + 24),
        a,
        b,
    )
}

/// `HashLen33to64(s, len)` (`city.cpp:135-156`).
#[inline]
fn hash_len33to64(s: &[u8], len: usize) -> u64 {
    // uint64 z = UNALIGNED_LOAD64(s + 24);
    let mut z = load64(s, 24);
    // uint64 a = UNALIGNED_LOAD64(s) + (len + UNALIGNED_LOAD64(s + len - 16)) * k0;
    let mut a = load64(s, 0).wrapping_add(
        (len as u64)
            .wrapping_add(load64(s, len - 16))
            .wrapping_mul(K0),
    );
    // uint64 b = Rotate(a + z, 52);
    let mut b = rotate(a.wrapping_add(z), 52);
    // uint64 c = Rotate(a, 37);
    let mut c = rotate(a, 37);
    // a += UNALIGNED_LOAD64(s + 8);
    a = a.wrapping_add(load64(s, 8));
    // c += Rotate(a, 7);
    c = c.wrapping_add(rotate(a, 7));
    // a += UNALIGNED_LOAD64(s + 16);
    a = a.wrapping_add(load64(s, 16));
    // uint64 vf = a + z;
    let vf = a.wrapping_add(z);
    // uint64 vs = b + Rotate(a, 31) + c;
    let vs = b.wrapping_add(rotate(a, 31)).wrapping_add(c);
    // a = UNALIGNED_LOAD64(s + 16) + UNALIGNED_LOAD64(s + len - 32);
    a = load64(s, 16).wrapping_add(load64(s, len - 32));
    // z = UNALIGNED_LOAD64(s + len - 8);
    z = load64(s, len - 8);
    // b = Rotate(a + z, 52);
    b = rotate(a.wrapping_add(z), 52);
    // c = Rotate(a, 37);
    c = rotate(a, 37);
    // a += UNALIGNED_LOAD64(s + len - 24);
    a = a.wrapping_add(load64(s, len - 24));
    // c += Rotate(a, 7);
    c = c.wrapping_add(rotate(a, 7));
    // a += UNALIGNED_LOAD64(s + len - 16);
    a = a.wrapping_add(load64(s, len - 16));
    // uint64 wf = a + z;
    let wf = a.wrapping_add(z);
    // uint64 ws = b + Rotate(a, 31) + c;
    let ws = b.wrapping_add(rotate(a, 31)).wrapping_add(c);
    // uint64 r = ShiftMix((vf + ws) * k2 + (wf + vs) * k0);
    let r = shift_mix(
        vf.wrapping_add(ws)
            .wrapping_mul(K2)
            .wrapping_add(wf.wrapping_add(vs).wrapping_mul(K0)),
    );
    // return ShiftMix(r * k0 + vs) * k2;
    shift_mix(r.wrapping_mul(K0).wrapping_add(vs)).wrapping_mul(K2)
}

/// Bit-exact port of Yandex CatBoost's `CityHash64(const char* s, size_t len)`
/// (`util/digest/city.cpp:158-196`). This is the CityHash **1.0** variant whose
/// results differ from mainline CityHash (`city.h:8-9`).
///
/// See the module-level docs for the non-cryptographic caveat and parity contract.
#[must_use]
pub fn city_hash_64(bytes: &[u8]) -> u64 {
    let mut len = bytes.len();
    if len <= 32 {
        if len <= 16 {
            // return HashLen0to16(s, len);
            return hash_len0to16(bytes, len);
        }
        // return HashLen17to32(s, len);
        return hash_len17to32(bytes, len);
    } else if len <= 64 {
        // return HashLen33to64(s, len);
        return hash_len33to64(bytes, len);
    }

    // For strings over 64 bytes we hash the end first, then loop keeping 56
    // bytes of state: v, w, x, y, z (`city.cpp:169-196`).
    // uint64 x = UNALIGNED_LOAD64(s);
    let mut x = load64(bytes, 0);
    // uint64 y = UNALIGNED_LOAD64(s + len - 16) ^ k1;
    let mut y = load64(bytes, len - 16) ^ K1;
    // uint64 z = UNALIGNED_LOAD64(s + len - 56) ^ k0;
    let mut z = load64(bytes, len - 56) ^ K0;
    // pair v = WeakHashLen32WithSeeds(s + len - 64, len, y);
    let mut v = weak_hash_len32_with_seeds(bytes, len - 64, len as u64, y);
    // pair w = WeakHashLen32WithSeeds(s + len - 32, len * k1, k0);
    let mut w = weak_hash_len32_with_seeds(bytes, len - 32, (len as u64).wrapping_mul(K1), K0);
    // z += ShiftMix(v.second) * k1;
    z = z.wrapping_add(shift_mix(v.1).wrapping_mul(K1));
    // x = Rotate(z + x, 39) * k1;
    x = rotate(z.wrapping_add(x), 39).wrapping_mul(K1);
    // y = Rotate(y, 33) * k1;
    y = rotate(y, 33).wrapping_mul(K1);

    // len = (len - 1) & ~63;  // nearest multiple of 64
    len = (len - 1) & !63usize;
    // `s` advances by 64 each iteration — tracked as `pos`.
    let mut pos = 0usize;
    loop {
        // x = Rotate(x + y + v.first + UNALIGNED_LOAD64(s + 16), 37) * k1;
        x = rotate(
            x.wrapping_add(y)
                .wrapping_add(v.0)
                .wrapping_add(load64(bytes, pos + 16)),
            37,
        )
        .wrapping_mul(K1);
        // y = Rotate(y + v.second + UNALIGNED_LOAD64(s + 48), 42) * k1;
        y = rotate(
            y.wrapping_add(v.1).wrapping_add(load64(bytes, pos + 48)),
            42,
        )
        .wrapping_mul(K1);
        // x ^= w.second;
        x ^= w.1;
        // y ^= v.first;
        y ^= v.0;
        // z = Rotate(z ^ w.first, 33);
        z = rotate(z ^ w.0, 33);
        // v = WeakHashLen32WithSeeds(s, v.second * k1, x + w.first);
        v = weak_hash_len32_with_seeds(bytes, pos, v.1.wrapping_mul(K1), x.wrapping_add(w.0));
        // w = WeakHashLen32WithSeeds(s + 32, z + w.second, y);
        w = weak_hash_len32_with_seeds(bytes, pos + 32, z.wrapping_add(w.1), y);
        // DoSwap(z, x);
        std::mem::swap(&mut z, &mut x);
        // s += 64;
        pos += 64;
        // len -= 64;
        len -= 64;
        // } while (len != 0);
        if len == 0 {
            break;
        }
    }
    // return HashLen16(HashLen16(v.first, w.first) + ShiftMix(y) * k1 + z,
    //                  HashLen16(v.second, w.second) + x);
    hash_len16(
        hash_len16(v.0, w.0)
            .wrapping_add(shift_mix(y).wrapping_mul(K1))
            .wrapping_add(z),
        hash_len16(v.1, w.1).wrapping_add(x),
    )
}

/// `CalcCatFeatureHash(feature) = CityHash64(feature) & 0xffffffff`
/// (`catboost-master/catboost/libs/cat_feature/cat_feature.cpp:6-8`).
///
/// The input `s` is hashed by its UTF-8 bytes. Integer-coded categories must be
/// pre-stringified with [`stringify_int_category`] (the A4-resolved PLAIN-integer
/// form) before being passed here.
#[must_use]
pub fn calc_cat_feature_hash(s: &str) -> u32 {
    // return CityHash64(feature) & 0xffffffff;
    (city_hash_64(s.as_bytes()) & 0xffff_ffff) as u32
}

/// Stringify an integer-coded categorical value to the form CatBoost hashes
/// (Assumption A4, `cb-oracle/fixtures/cat_hash/config.json`): a PLAIN base-10
/// integer with no decimal point — `3 -> "3"` (ui32 `2658984922`), distinct from
/// the float form `"3.0"` (ui32 `1187060909`).
#[must_use]
pub fn stringify_int_category(value: i64) -> String {
    // Rust's i64 Display is the plain base-10 form with no fractional part,
    // matching CatBoost's integer stringification (A4).
    value.to_string()
}

/// First-seen perfect-hash remap (RESEARCH Pattern 4,
/// `cat_feature_perfect_hash_helper.cpp:111-131`): assigns each distinct ui32
/// hash a dense bin index in first-seen iteration order — `bin = map.size()` for
/// each new hash (`cat_feature_perfect_hash_helper.cpp:120`), repeat hashes reuse
/// their assigned bin.
///
/// The `TMap`'s sorted (RB-tree) order in upstream matters only for the deferred
/// most-frequent-value-to-0 tiebreak (`mapMostFrequentValueTo0`), which is out of
/// Phase 2 scope; plain training is first-seen, so an insertion-counter +
/// `HashMap` lookup reproduces the bin assignment exactly.
#[derive(Debug, Default, Clone)]
pub struct PerfectHash {
    /// ui32 hash -> assigned dense bin.
    map: HashMap<u32, u32>,
}

/// The upstream uniq-cat bound: `MAX_UNIQ_CAT_VALUES = Max<ui32>() + 1` on 64-bit
/// hosts (`cat_feature_perfect_hash_helper.cpp:53-54`). The map must never reach
/// this size; the `(u32)map.size()` bin index would otherwise overflow. We bound
/// at `u32::MAX` distinct values and refuse the next insert (Security V5,
/// threat T-02-11) with a typed [`CbError`] rather than panicking.
const MAX_UNIQ_CAT_VALUES: usize = u32::MAX as usize;

impl PerfectHash {
    /// Construct an empty perfect-hash map.
    #[must_use]
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    /// Number of distinct hashes seen so far (`perfectHashMap.GetSize()`).
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether no hash has been inserted yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Map one ui32 `hash` to its dense bin, assigning a new first-seen bin if
    /// unseen (`processNonDefaultValue`, `cat_feature_perfect_hash_helper.cpp:111-131`).
    ///
    /// # Errors
    ///
    /// Returns [`CbError::OutOfRange`] if the map already holds
    /// [`MAX_UNIQ_CAT_VALUES`] distinct hashes (mirrors the `CB_ENSURE` at
    /// `cat_feature_perfect_hash_helper.cpp:114-119`) — no panic.
    pub fn remap(&mut self, hash: u32) -> CbResult<u32> {
        if let Some(&bin) = self.map.get(&hash) {
            // it != end: reuse the assigned bin (`it->second.Value`).
            return Ok(bin);
        }
        // it == end: CB_ENSURE(map.size() != MAX_UNIQ_CAT_VALUES, ...) before insert.
        if self.map.len() >= MAX_UNIQ_CAT_VALUES {
            return Err(CbError::OutOfRange(format!(
                "categorical feature has more than {MAX_UNIQ_CAT_VALUES} unique values, which is currently unsupported"
            )));
        }
        // const ui32 bin = (ui32)perfectHashMap.GetSize();
        let bin = self.map.len() as u32;
        self.map.insert(hash, bin);
        Ok(bin)
    }
}

/// Hash a whole categorical column (already in the A4 string form) and remap it
/// to dense first-seen bins in one pass over `column`, returning the per-object
/// bins. Each string is hashed with [`calc_cat_feature_hash`] then routed through
/// [`PerfectHash::remap`] in iteration order.
///
/// # Errors
///
/// Propagates [`CbError::OutOfRange`] from [`PerfectHash::remap`] if the column
/// has more than [`MAX_UNIQ_CAT_VALUES`] distinct values.
pub fn perfect_hash_bins(column: &[&str]) -> CbResult<Vec<u32>> {
    let mut ph = PerfectHash::new();
    let mut bins = Vec::with_capacity(column.len());
    for value in column {
        let hash = calc_cat_feature_hash(value);
        bins.push(ph.remap(hash)?);
    }
    Ok(bins)
}
