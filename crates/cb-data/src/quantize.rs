//! `pool.quantize(&params) -> QuantizedPool` — the float-feature quantization
//! driver (DATA-02 / D-01) that composes border selection (Plan 02) with NanMode
//! sentinel handling + strict `value > border` bin assignment (Task 1) into the
//! immutable binned [`QuantizedPool`].
//!
//! # Composition (D-01)
//!
//! For each float feature column the driver:
//! 1. determines the per-feature [`NanMode`] (a column containing `NaN` is
//!    quantized under [`NanMode::Min`] — the catboost 1.2.10 default; a NaN-free
//!    column is [`NanMode::Forbidden`], no sentinel);
//! 2. selects borders via [`crate::select_borders_greedy_logsum`]
//!    (`borders::`), which already prepends the [`f32::MIN`] sentinel for `Min`
//!    so the result matches the standalone-quantizer oracle verbatim;
//! 3. assigns each object value to a bin via [`crate::bin_of`]
//!    (`nan_mode::`, strict `>`), routing `NaN` values to [`crate::nan_bin`];
//! 4. stores the bins in the width-selected [`ColumnBins`]
//!    ([`crate::quantized_pool::pack_bins`]).
//!
//! # f32 discipline
//!
//! Borders and values are compared in `f32` (CatBoost's storage type): each `f64`
//! column value is narrowed to `f32` before [`crate::bin_of`], matching the f32
//! borders. Mixing widths would shift boundary values into the wrong bin.
//!
//! # Fallibility (D-14)
//!
//! Returns [`CbResult`]: the width selector rejects a float feature that would
//! need `> u16` bins (threat T-02-09), surfacing a [`cb_core::CbError`] rather
//! than panicking.

use cb_core::{CbError, CbResult};

use crate::nan_mode::{bin_of, nan_bin, NanMode};
use crate::quantized_pool::{pack_bins, ColumnBins, FeatureKind, QuantizedPool};
use crate::Pool;

/// Quantization parameters consumed by [`Pool::quantize`] (D-01). Mirrors the
/// catboost 1.2.10 defaults resolved in 02-01 (A2): `border_count = 254`,
/// `GreedyLogSum`, `nan_mode = Min`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantizeParams {
    /// Total border budget per feature (catboost default 254). For a `Min`/`Max`
    /// feature the binarizer reserves one border for the sentinel internally.
    pub border_count: usize,
    /// The [`NanMode`] to apply to float features that contain `NaN`. NaN-free
    /// features are always [`NanMode::Forbidden`] (no sentinel) regardless.
    pub nan_mode: NanMode,
}

impl Default for QuantizeParams {
    fn default() -> Self {
        // catboost 1.2.10 defaults (02-01 A2).
        Self {
            border_count: 254,
            nan_mode: NanMode::Min,
        }
    }
}

impl Pool {
    /// Quantize every float feature of this [`Pool`] into an immutable
    /// [`QuantizedPool`] (D-01). See the module docs for the per-feature pipeline.
    ///
    /// # Errors
    /// Returns [`CbError::OutOfRange`] if a float feature would require more than
    /// `u16` bins (a float feature is hard-capped at `< 65536` borders,
    /// `utils.h:175-181`; threat T-02-09).
    pub fn quantize(&self, params: &QuantizeParams) -> CbResult<QuantizedPool> {
        let n_features = self.n_float_features();
        let mut float_bins: Vec<ColumnBins> = Vec::with_capacity(n_features);
        let mut float_borders: Vec<Vec<f32>> = Vec::with_capacity(n_features);
        let mut float_nan_modes: Vec<NanMode> = Vec::with_capacity(n_features);

        for fi in 0..n_features {
            let column = self.float_feature(fi).unwrap_or(&[]);

            // (1) Per-feature NanMode: a column with any NaN uses the requested
            // nan_mode (Min/Max); a NaN-free column is Forbidden (no sentinel).
            let has_nan = column.iter().any(|v| v.is_nan());
            let feature_mode = if has_nan {
                params.nan_mode
            } else {
                NanMode::Forbidden
            };

            // (2) Borders via the GreedyLogSum binarizer; the Min sentinel is
            // prepended here so the result matches the standalone-quantizer
            // oracle verbatim (borders_quant fixtures, A1/A3).
            let prepend_min_sentinel = feature_mode.prepends_min_sentinel();
            let borders_f64 = crate::select_borders_greedy_logsum(
                column,
                params.border_count,
                prepend_min_sentinel,
            );
            // Narrow the (already f32-valued, f64-stored) borders back to f32 for
            // the f32 bin comparison.
            let borders_f32: Vec<f32> = borders_f64.iter().map(|&b| b as f32).collect();

            // (3) Assign each value to a bin (strict `>`); NaN -> nan_bin.
            let mut bins: Vec<u32> = Vec::with_capacity(column.len());
            for &value in column {
                let v32 = value as f32;
                let bin = if v32.is_nan() {
                    nan_bin(feature_mode, &borders_f32)
                } else {
                    bin_of(&borders_f32, v32)
                };
                bins.push(bin);
            }

            // (4) Pack into the width-selected ColumnBins (float -> u8/u16 only).
            let column_bins = pack_bins(borders_f32.len(), FeatureKind::Float, &bins)
                .ok_or_else(|| {
                    CbError::OutOfRange(format!(
                        "float feature {fi} needs {} borders (> u16 limit); float bins are capped at u16",
                        borders_f32.len()
                    ))
                })?;

            float_bins.push(column_bins);
            float_borders.push(borders_f32);
            float_nan_modes.push(feature_mode);
        }

        Ok(QuantizedPool::new(
            self.n_rows(),
            float_bins,
            float_borders,
            float_nan_modes,
        ))
    }
}
