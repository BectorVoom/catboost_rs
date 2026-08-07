//! SPD-03 wave 3: the ingestion-attached f32 narrowing cache
//! (`Pool::float_features_f32`) lets `fit` skip the full f64→f32 narrowing pass.
//! The cache is only legitimate because the narrowing is bit-exactly invertible
//! for f32-origin data — so a fit THROUGH the cache must produce a bit-identical
//! model to the same fit WITHOUT it. This differential is the whole safety
//! argument; if it ever drifts, the cache is silently changing training data.

use catboost_rs::{CatBoostBuilder, IngestSource, Loss, OwnedColumns};

/// Deterministic f32-origin columns (the cache contract's precondition: the f64
/// pool values are exact f32 round-trips, as the NumPy f32 ingestion guarantees).
fn f32_origin_data(n: usize, nf: usize) -> (Vec<Vec<f32>>, Vec<f64>) {
    let mut s: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f32) / ((1u64 << 31) as f32) * 10.0 - 5.0
    };
    let cols: Vec<Vec<f32>> = (0..nf).map(|_| (0..n).map(|_| next()).collect()).collect();
    let label: Vec<f64> = (0..n)
        .map(|i| {
            let a = cols.first().and_then(|c| c.get(i)).copied().unwrap_or(0.0);
            let b = cols.get(1 % nf).and_then(|c| c.get(i)).copied().unwrap_or(0.0);
            f64::from((a * 0.31).sin() + (b * 0.17).cos() * 0.5)
        })
        .collect();
    (cols, label)
}

#[test]
fn fit_through_the_f32_cache_is_bit_identical() {
    let (f32_cols, label) = f32_origin_data(4096, 5);
    let f64_cols: Vec<Vec<f64>> = f32_cols
        .iter()
        .map(|c| c.iter().map(|&v| f64::from(v)).collect())
        .collect();

    let pool_plain = OwnedColumns::new(f64_cols.clone(), label.clone())
        .into_pool()
        .expect("plain pool");
    let pool_cached = OwnedColumns::new(f64_cols, label)
        .with_float_f32_cache(f32_cols.clone())
        .into_pool()
        .expect("cached pool");

    let builder = || {
        CatBoostBuilder::new()
            .loss(Loss::Rmse)
            .iterations(8)
            .depth(4)
            .learning_rate(0.1)
            .border_count(32)
            .random_seed(7)
    };
    let m_plain = builder().fit(&pool_plain).expect("plain fit");
    let m_cached = builder().fit(&pool_cached).expect("cached fit");

    // Predict over the training pool; every prediction must agree on BITS.
    let p_plain = m_plain.predict(&pool_plain).expect("plain predict");
    let p_cached = m_cached.predict(&pool_plain).expect("cached predict");
    assert_eq!(p_plain.len(), p_cached.len());
    for (i, (a, b)) in p_plain.iter().zip(p_cached.iter()).enumerate() {
        assert!(
            a.to_bits() == b.to_bits(),
            "prediction {i} differs through the f32 cache: {a:?} vs {b:?}"
        );
    }
}

/// A WRONG-SHAPE cache must be rejected at ingestion (never silently attached).
#[test]
fn wrong_shape_f32_cache_is_a_typed_ingest_error() {
    let (f32_cols, label) = f32_origin_data(64, 3);
    let f64_cols: Vec<Vec<f64>> = f32_cols
        .iter()
        .map(|c| c.iter().map(|&v| f64::from(v)).collect())
        .collect();
    let short_cache: Vec<Vec<f32>> = f32_cols.iter().take(2).cloned().collect();
    let result = OwnedColumns::new(f64_cols, label)
        .with_float_f32_cache(short_cache)
        .into_pool();
    assert!(result.is_err(), "a 2-column cache on a 3-column pool must not validate");
}
