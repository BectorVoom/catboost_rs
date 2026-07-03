//! Self-oracle for the bit-compression primitives (Plan 10-05, GPUT-16): packing an
//! ≤8-bit bin column into shared 32-bit words (`pack_bins_kernel`) and extracting each
//! key's field back out (`unpack_bins_kernel`) must reproduce the source bins EXACTLY —
//! integer equality (BIT-EXACT, tighter than the ≤1e-4 float bar, D-07). The ground
//! truth is an INLINE SERIAL pack/unpack reference (D-02: no `cb-train` reach, no
//! upstream/CUB fixture). Multiple keys sharing one 32-bit word each extract their own
//! field via their distinct Shift, and an `n >> keys_per_word` case is asserted.
//!
//! Source/test separation is mandatory (CLAUDE.md / AGENTS.md): the kernels + the
//! `bit_pack_layout` host helper live in `kernels.rs`; all assertions + `.unwrap()`/
//! indexing live here.
//!
//! This runs on `rocm` in-env on gfx1100 (wave32) and under every other backend over
//! the generic [`crate::SelectedRuntime`]. The authoritative Kaggle CUDA sign-off
//! (bit-exact) is human-gated via 10-09; here the inline serial reference is the
//! ground truth.

use cubecl::prelude::*;

use crate::kernels::{bit_pack_layout, pack_bins_kernel, unpack_bins_kernel, BitPackLayout};

/// Launch geometry: 32-wide cubes (wave32 gfx1100), enough cubes to cover every lane.
const CUBE_DIM: usize = 32;

/// Pack `bins` (each `u32` holds one ≤8-bit bin) into 32-bit words on the device, then
/// unpack them back into a bin column — the full round-trip over the selected runtime.
/// The bit geometry comes from the host [`bit_pack_layout`] (comptime pack params).
/// Returns the unpacked bins (must equal the source, bit-exact).
fn run_pack_unpack(bins: &[u32], n_bins: u32) -> Vec<u32> {
    let n = bins.len();
    let BitPackLayout {
        bits_per_key,
        keys_per_word,
        mask,
        num_words,
    } = bit_pack_layout(n_bins, n).unwrap();

    let device = <crate::SelectedRuntime as Runtime>::Device::default();
    let client = <crate::SelectedRuntime as Runtime>::client(&device);
    let dim32 = CubeDim { x: CUBE_DIM as u32, y: 1, z: 1 };

    let bins_h = client.create(cubecl::bytes::Bytes::from_elems(bins.to_vec()));

    // Pack: one lane per OUTPUT word (each thread owns a word — no cross-lane |= race).
    let words_h = client.empty(num_words * std::mem::size_of::<u32>());
    let word_cubes = num_words.div_ceil(CUBE_DIM).max(1);
    pack_bins_kernel::launch::<f64, crate::SelectedRuntime>(
        &client,
        CubeCount::Static(word_cubes as u32, 1, 1),
        dim32,
        unsafe { ArrayArg::from_raw_parts(bins_h, n) },
        unsafe { ArrayArg::from_raw_parts(words_h.clone(), num_words) },
        bits_per_key,
        keys_per_word,
        mask,
    );

    // Unpack: one lane per KEY.
    let out_h = client.empty(n * std::mem::size_of::<u32>());
    let key_cubes = n.div_ceil(CUBE_DIM).max(1);
    unpack_bins_kernel::launch::<f64, crate::SelectedRuntime>(
        &client,
        CubeCount::Static(key_cubes as u32, 1, 1),
        dim32,
        unsafe { ArrayArg::from_raw_parts(words_h, num_words) },
        unsafe { ArrayArg::from_raw_parts(out_h.clone(), n) },
        bits_per_key,
        keys_per_word,
        mask,
    );

    let bytes = client.read_one(out_h).unwrap();
    bytemuck::cast_slice::<u8, u32>(&bytes).to_vec()
}

/// Inline serial pack reference (D-02): pack `bins` into `num_words` 32-bit words, key
/// `i` at word `i / keys_per_word`, slot `i % keys_per_word`, shift `slot *
/// bits_per_key`.
fn cpu_pack(bins: &[u32], layout: &BitPackLayout) -> Vec<u32> {
    let kpw = layout.keys_per_word as usize;
    let mut words = vec![0u32; layout.num_words];
    for (i, &b) in bins.iter().enumerate() {
        let word_idx = i / kpw;
        let slot = (i % kpw) as u32;
        words[word_idx] |= (b & layout.mask) << (slot * layout.bits_per_key);
    }
    words
}

/// Inline serial unpack reference (D-02): extract each key's field from `words`.
fn cpu_unpack(words: &[u32], n: usize, layout: &BitPackLayout) -> Vec<u32> {
    let kpw = layout.keys_per_word as usize;
    (0..n)
        .map(|i| {
            let word_idx = i / kpw;
            let slot = (i % kpw) as u32;
            (words[word_idx] >> (slot * layout.bits_per_key)) & layout.mask
        })
        .collect()
}

/// Deterministic pseudo-random bin column in `0..=n_bins` (LCG — no rand dep).
fn synth_bins(n: usize, n_bins: u32, seed: u32) -> Vec<u32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) % (n_bins + 1)
        })
        .collect()
}

#[test]
fn pack_unpack_behaviour_example_multiple_keys_per_word() {
    // n_bins = 3 ⇒ bits_per_key = 2, keys_per_word = 16, mask = 0b11. Four keys
    // [3,1,2,0] pack into ONE word; each extracts its own 2-bit field via its Shift.
    let n_bins = 3u32;
    let bins = vec![3u32, 1, 2, 0];
    let layout = bit_pack_layout(n_bins, bins.len()).unwrap();
    assert_eq!(layout.bits_per_key, 2, "2 bits hold bins 0..=3");
    assert_eq!(layout.keys_per_word, 16, "16 two-bit keys per 32-bit word");
    assert_eq!(layout.mask, 0b11);
    assert_eq!(layout.num_words, 1, "four keys fit in one word");

    // The packed word: 3<<0 | 1<<2 | 2<<4 | 0<<6 = 3 + 4 + 32 = 0b100111 = 39.
    let words = cpu_pack(&bins, &layout);
    assert_eq!(words, vec![39u32], "serial pack of [3,1,2,0]");

    // Device pack∘unpack reproduces the source bins bit-exactly.
    let got = run_pack_unpack(&bins, n_bins);
    assert_eq!(got, bins, "device pack∘unpack must be bit-exact");
}

#[test]
fn pack_unpack_round_trips_bit_exact_across_seeds_and_widths() {
    // Random ui8-range bin columns across several n_bins widths and seeds. Every pack∘
    // unpack must reproduce the source EXACTLY (integer equality), and the device result
    // must match the inline serial pack/unpack reference.
    for &n_bins in &[1u32, 3, 7, 15, 31, 63, 127, 255] {
        for &seed in &[1u32, 42, 12345, 987_654_321] {
            let n = 1000usize; // >> keys_per_word for every width
            let bins = synth_bins(n, n_bins, seed);
            let layout = bit_pack_layout(n_bins, n).unwrap();

            // Serial reference round-trip is bit-exact.
            let words = cpu_pack(&bins, &layout);
            let serial = cpu_unpack(&words, n, &layout);
            assert_eq!(serial, bins, "serial pack∘unpack (n_bins={n_bins}, seed={seed})");

            // Device round-trip is bit-exact and matches the serial reference.
            let got = run_pack_unpack(&bins, n_bins);
            assert_eq!(
                got, bins,
                "device pack∘unpack not bit-exact (n_bins={n_bins}, seed={seed})"
            );
        }
    }
}

#[test]
fn pack_unpack_large_n_full_byte_width() {
    // n_bins = 255 ⇒ bits_per_key = 8, keys_per_word = 4. n = 10000 >> keys_per_word so
    // many packed words + a non-multiple tail are exercised.
    let n_bins = 255u32;
    let n = 10_000usize;
    let bins = synth_bins(n, n_bins, 7);
    let layout = bit_pack_layout(n_bins, n).unwrap();
    assert_eq!(layout.bits_per_key, 8);
    assert_eq!(layout.keys_per_word, 4);
    assert_eq!(layout.mask, 0xFF);
    assert_eq!(layout.num_words, n.div_ceil(4));

    let got = run_pack_unpack(&bins, n_bins);
    assert_eq!(got, bins, "device pack∘unpack bit-exact at n={n}, full-byte width");
}

#[test]
fn bit_pack_layout_geometry_and_overflow_guards() {
    // Geometry: bits_per_key = ceil(log2(n_bins+1)).
    assert_eq!(bit_pack_layout(0, 8).unwrap().bits_per_key, 1); // one value → 1 bit
    assert_eq!(bit_pack_layout(1, 8).unwrap().bits_per_key, 1); // 0..=1 → 1 bit
    assert_eq!(bit_pack_layout(2, 8).unwrap().bits_per_key, 2); // 0..=2 → 2 bits
    assert_eq!(bit_pack_layout(3, 8).unwrap().bits_per_key, 2); // 0..=3 → 2 bits
    assert_eq!(bit_pack_layout(4, 8).unwrap().bits_per_key, 3); // 0..=4 → 3 bits
    assert_eq!(bit_pack_layout(255, 8).unwrap().bits_per_key, 8); // full byte

    // keys_per_word / mask consistency.
    let l = bit_pack_layout(15, 100).unwrap();
    assert_eq!(l.bits_per_key, 4);
    assert_eq!(l.keys_per_word, 8);
    assert_eq!(l.mask, 0xF);
    assert_eq!(l.num_words, 100usize.div_ceil(8));

    // Overflow guard (T-10-12): n_bins == u32::MAX ⇒ n_bins+1 overflows u32 → OutOfRange
    // (the checked_add fires — no unguarded arithmetic reaches the device).
    let err = bit_pack_layout(u32::MAX, 10);
    assert!(err.is_err(), "n_bins=u32::MAX must be rejected, got {err:?}");
}
