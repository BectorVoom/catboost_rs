//! Exact standard-normal draw — RED stub (Task 1 TDD). Replaced by the
//! Marsaglia-polar implementation in the GREEN step.

use crate::rng::TFastRng64;

/// RED stub: deliberately wrong so the draw-sequence tests fail before the real
/// Marsaglia-polar loop is ported.
#[must_use]
pub fn std_normal(_rng: &mut TFastRng64) -> f64 {
    0.0
}
