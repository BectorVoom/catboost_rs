//! FPP-05 (T06) unit self-oracle for `device_exact_leaf_config` — the pure decision that
//! populates `DeviceTrainConfig::{exact_leaf, quantile_alpha, quantile_delta}`.
//!
//! Mounted as a sibling `#[path]` submodule of `boosting` (source/test separation,
//! CLAUDE.md; the `boosting_device_fold_test.rs` precedent), so it reaches the private
//! `super::device_exact_leaf_config` directly. PLAN blocker **B-2 is resolved here by
//! option (a)**: the two-field decision is extracted as a pure function and unit-tested,
//! rather than observed through a `Runtime` mock — it needs no device and no fit.
//!
//! # The asymmetry this file pins (PLAN V-6)
//!
//! The admissible set is the INTERSECTION of what the CPU permits and what the device
//! covers, and the two disagree in BOTH directions:
//!
//! | loss | `validate_leaf_method` (CPU) | `map_leaf_method` (device) | admitted |
//! |---|---|---|---|
//! | `Mae` | Exact legal | Exact covered | **yes** |
//! | `Quantile` | Exact legal | Exact covered | **yes** |
//! | `LogCosh` | Exact legal | NOT covered | no |
//! | `Mape` | Exact **REJECTED** | Exact covered | no |
//! | `MultiQuantile` | Exact legal | multi-dim | no |
//!
//! A test over a single loss would miss both asymmetries, which is why all five appear.

use cb_compute::{LeafMethod, Loss, QUANTILE_ALPHA, QUANTILE_DELTA};

use super::device_exact_leaf_config;

#[test]
fn gradient_leaf_never_activates_the_device_exact_arm() {
    // The overwhelmingly common case: `builder.rs` defaults `leaf_method: Gradient`
    // unconditionally, so Exact is reachable only by an EXPLICIT request.
    let (exact, alpha, delta) = device_exact_leaf_config(LeafMethod::Gradient, &Loss::Rmse);
    assert!(!exact, "Gradient/RMSE must not activate the device exact leaf");
    assert!((alpha - QUANTILE_ALPHA).abs() < f64::EPSILON, "inert alpha stays at the default");
    assert!((delta - QUANTILE_DELTA).abs() < f64::EPSILON, "inert delta stays at the default");

    // Gradient over a loss that WOULD be admissible under Exact must still decline.
    let (exact, _, _) = device_exact_leaf_config(LeafMethod::Gradient, &Loss::Mae);
    assert!(!exact, "Gradient/MAE must not activate the device exact leaf");
}

#[test]
fn exact_mae_activates_with_the_default_median_quantile() {
    let (exact, alpha, delta) = device_exact_leaf_config(LeafMethod::Exact, &Loss::Mae);
    assert!(exact, "Exact/MAE is in the admitted intersection");
    // MAE's exact leaf IS the weighted median — the struct's own defaults, not a
    // re-typed 0.5/1e-6 literal.
    assert!(
        (alpha - QUANTILE_ALPHA).abs() < f64::EPSILON,
        "MAE must carry the default median alpha {QUANTILE_ALPHA}, got {alpha}"
    );
    assert!(
        (delta - QUANTILE_DELTA).abs() < f64::EPSILON,
        "MAE must carry the default delta {QUANTILE_DELTA}, got {delta}"
    );
}

#[test]
fn exact_quantile_carries_its_own_alpha_and_delta() {
    let loss = Loss::Quantile {
        alpha: 0.7,
        delta: 1e-6,
    };
    let (exact, alpha, delta) = device_exact_leaf_config(LeafMethod::Exact, &loss);
    assert!(exact, "Exact/Quantile is in the admitted intersection");
    assert!(
        (alpha - 0.7).abs() < f64::EPSILON,
        "the LOSS's alpha must reach the device config, got {alpha}"
    );
    assert!((delta - 1e-6).abs() < f64::EPSILON, "the LOSS's delta must reach it, got {delta}");

    // A different alpha must produce a different config — otherwise a device path that
    // hardcoded the median would pass every single-alpha test.
    let (_, alpha_03, _) = device_exact_leaf_config(
        LeafMethod::Exact,
        &Loss::Quantile {
            alpha: 0.3,
            delta: 2e-6,
        },
    );
    assert!(
        (alpha_03 - alpha).abs() > 1e-9,
        "alpha must be load-bearing: 0.3 and 0.7 produced the same config value"
    );
}

#[test]
fn exact_logcosh_declines_because_the_device_does_not_cover_it() {
    // CPU-LEGAL (validate_leaf_method admits LogCosh under Exact) but device-UNCOVERED
    // (map_leaf_method has no LogCosh arm). Admitting it would silently apply the WRONG
    // leaf; declining keeps today's correct CPU fallback.
    let (exact, _, _) = device_exact_leaf_config(LeafMethod::Exact, &Loss::LogCosh);
    assert!(
        !exact,
        "Exact/LogCosh is CPU-legal but device-uncovered — it must fall back to the CPU"
    );
}

#[test]
fn exact_mape_declines_because_the_cpu_rejects_it() {
    // The mirror asymmetry: device-COVERED (map_leaf_method has a Mape arm with
    // `mape: true`) but CPU-REJECTED by validate_leaf_method, so no fit can reach here
    // with this pair at all. Pinned so a future validate_leaf_method relaxation does not
    // silently change the device decision without this test being revisited.
    let (exact, _, _) = device_exact_leaf_config(LeafMethod::Exact, &Loss::Mape);
    assert!(!exact, "Exact/MAPE is CPU-rejected — the device config must not activate");
}

#[test]
fn exact_multiquantile_declines_because_it_is_multi_dimensional() {
    let loss = Loss::MultiQuantile {
        alpha: vec![0.3, 0.7],
        delta: 1e-6,
    };
    let (exact, _, _) = device_exact_leaf_config(LeafMethod::Exact, &loss);
    assert!(
        !exact,
        "MultiQuantile is multi-dimensional; the scalar exact_leaf arm cannot express it"
    );
}
