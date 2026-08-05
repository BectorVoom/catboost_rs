//! Training snapshot / resume (ORCH-03) — the on-disk checkpoint of the boosting
//! loop's carried state, and the compat key that decides whether a snapshot may
//! be resumed against the run being started.
//!
//! # Why a DTO and not `serde` on the live types
//!
//! `Model` / `ObliviousTree` stay serde-free (D-04): the trained-model wire
//! formats (`.cbm`, `model.json`) are owned by `cb-model` and pinned by
//! byte-identity oracles, and a `#[derive(Serialize)]` on the training types
//! would make an unrelated field addition a silent snapshot-format change. The
//! snapshot instead mirrors the loop state into DTOs defined here and converts at
//! the boundary. `cb-model` is NEVER referenced from this module — `cb-train` sits
//! BELOW `cb-model` in the build graph (`cb-model` depends on `cb-train`; the
//! reverse edge exists only as a `[dev-dependencies]` entry for integration
//! tests), so reaching for `cb_model` here would be a dependency cycle.
//!
//! # Scope
//!
//! Slice 1 supports exactly the regime whose loop-carried mutable state is
//! `{approx, trees, rng}` — the audit in
//! `.planning/plans/snapshot-resume/TASK-01-findings.md` establishes that set
//! against the current `boosting.rs`. Every other regime (ranking, eval sets,
//! categorical/CTR, ordered boosting, sampling, multi-dimension, penalties,
//! non-symmetric grow policies, device training, and a requested staged-prediction
//! buffer) is REFUSED with a typed error rather than snapshotted incorrectly.

use std::path::PathBuf;
use std::time::Duration;

use cb_core::{CbError, CbResult};
use serde::{Deserialize, Serialize};

use crate::boosting::ObliviousTree;
use crate::tree::Split;

/// The on-disk snapshot format version. Bumped whenever the meaning or the layout
/// of any [`TrainSnapshot`] field changes; [`decode`] REFUSES any other value, so
/// an older binary can never read a newer file with today's field meanings.
///
/// `2`: [`fingerprint`] now folds the per-object `weights` column (field 17). A
/// v1 snapshot's stored fingerprint was computed WITHOUT the weights, so it would
/// compare equal across two runs that differ only in their weighting — exactly
/// the silent corruption the fingerprint exists to prevent. Refusing v1 files
/// outright is the only way to guarantee that never happens on an upgrade.
pub const SNAPSHOT_FORMAT_VERSION: u32 = 2;

/// Caller-facing snapshot configuration. Mirrors upstream's `snapshot_file` /
/// `snapshot_interval` parameters; upstream's `save_snapshot=true` corresponds to
/// passing `Some(_)` here.
///
/// Resume is automatic, matching upstream: it triggers exactly when
/// [`Self::snapshot_file`] already exists AND the snapshot it holds carries a
/// fingerprint equal to the starting run's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotConfig {
    /// Destination the checkpoint is written to, and the source a resume reads.
    pub snapshot_file: PathBuf,
    /// Minimum wall-clock time between checkpoint writes. Upstream's default is
    /// 600s; a zero interval writes on every completed iteration (the setting the
    /// deterministic tests use).
    pub snapshot_interval: Duration,
}

/// Serialize every `f64` as its IEEE-754 BIT PATTERN rather than as a JSON number.
///
/// This is not a micro-optimization — it is a correctness requirement. `serde_json`
/// writes an `f64` as a decimal string and parses it back through its own decimal
/// converter, and that round-trip is NOT bit-exact for every value: a real trained
/// leaf value from this crate's own test corpus comes back one ULP off. A resumed
/// run must be BIT-identical to a straight-through run, so a codec that perturbs
/// the restored `approx` in the last bit would make the whole slice's guarantee
/// false in a way no approximate comparison would ever catch.
///
/// Storing `u64` bits removes the decimal conversion entirely: what is written is
/// what is read, for every finite value and every NaN payload alike.
mod f64_bits {
    use serde::{Deserialize, Deserializer, Serializer};

    pub(super) fn serialize<S: Serializer>(v: &f64, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u64(v.to_bits())
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
        u64::deserialize(d).map(f64::from_bits)
    }
}

/// The `Vec<f64>` counterpart of [`f64_bits`], with the same rationale.
mod f64_bits_vec {
    use serde::{Deserialize, Deserializer, Serializer};

    pub(super) fn serialize<S: Serializer>(v: &[f64], s: S) -> Result<S::Ok, S::Error> {
        s.collect_seq(v.iter().map(|x| x.to_bits()))
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<f64>, D::Error> {
        Ok(Vec::<u64>::deserialize(d)?.into_iter().map(f64::from_bits).collect())
    }
}

/// One float split, mirrored for serialization ([`crate::tree::Split`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SplitDto {
    /// The float feature this split tests.
    pub feature: usize,
    /// The split border; an object passes when `value > border`. Stored as bits —
    /// see [`f64_bits`].
    #[serde(with = "f64_bits")]
    pub border: f64,
}

/// One trained oblivious tree, mirrored for serialization.
///
/// Carries ONLY the float-split shape: the live [`ObliviousTree`] additionally has
/// `ctr_splits`, `one_hot_splits` and `level_kinds`, all of which are EMPTY on the
/// snapshottable regime. [`dto_from_tree`] refuses a tree where any of them is
/// non-empty rather than dropping it — a dropped categorical split would resume
/// into a silently different model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObliviousTreeDto {
    /// The ordered float splits defining the symmetric structure.
    pub splits: Vec<SplitDto>,
    /// Leaf values in canonical forward-bit-order, length `2^depth`. Stored as
    /// bits — see [`f64_bits`].
    #[serde(with = "f64_bits_vec")]
    pub leaf_values: Vec<f64>,
    /// Per-leaf summed training-document weights, same order as `leaf_values`.
    /// Stored as bits — see [`f64_bits`].
    #[serde(with = "f64_bits_vec")]
    pub leaf_weights: Vec<f64>,
}

/// The persisted boosting-loop state: everything a resumed run needs to continue
/// at iteration `completed_iters` and finish bit-identically to a straight-through
/// run of the same configuration.
///
/// `approx` is stored VERBATIM rather than reconstructed by re-applying the
/// persisted trees — re-application would re-associate the per-iteration sums and
/// could shift the resumed run's derivatives in the last bits.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainSnapshot {
    /// [`SNAPSHOT_FORMAT_VERSION`] at write time.
    pub format_version: u32,
    /// The compat key of the run that wrote this file ([`fingerprint`]).
    pub fingerprint: u64,
    /// `K` — the number of iterations already completed. A resumed run starts at
    /// iteration `K` (0-based), i.e. `K` trees are already in `trees`.
    pub completed_iters: usize,
    /// The starting approximant (`boost_from_average` / explicit bias). Stored as
    /// bits — see [`f64_bits`].
    #[serde(with = "f64_bits")]
    pub bias: f64,
    /// The training `approx_dimension` (always 1 on the snapshottable regime;
    /// persisted so a resume can reject a dimension change).
    pub approx_dimension: usize,
    /// The live approximant after `completed_iters` iterations, length `n`. Stored
    /// as bits — see [`f64_bits`]; restoring this value ONE ULP off would make the
    /// resumed run's derivatives differ from the straight-through run's.
    #[serde(with = "f64_bits_vec")]
    pub approx: Vec<f64>,
    /// The `completed_iters` trees grown so far, in order.
    pub trees: Vec<ObliviousTreeDto>,
    /// The persistent sampling RNG's raw stream state
    /// ([`cb_core::TFastRng64::raw_state`]).
    pub rng_raw_state: [u64; 4],
    /// The persistent sampling RNG's consumed-draw count.
    pub rng_call_count: u64,
}

/// Serialize a snapshot to JSON bytes.
///
/// Floats are written as bit patterns ([`f64_bits`]), so the round-trip is exact.
///
/// # Errors
/// [`CbError::Snapshot`] if any `f64` in the snapshot is non-finite, or if
/// serialization fails.
///
/// The finiteness check does NOT exist for encodability — bit storage handles NaN
/// and infinity perfectly well. It exists because a non-finite `approx` or leaf
/// value means the fit has already diverged, and a checkpoint of a diverged fit can
/// only ever resume into more divergence. Failing at the write, where the run and
/// its parameters are still in hand, beats resuming hours later from a file full of
/// NaNs.
pub fn encode(snapshot: &TrainSnapshot) -> CbResult<Vec<u8>> {
    check_finite(snapshot)?;
    serde_json::to_vec(snapshot)
        .map_err(|e| CbError::Snapshot(format!("failed to serialize snapshot: {e}")))
}

/// Deserialize a snapshot from JSON bytes, rejecting an unknown format version.
///
/// # Errors
/// [`CbError::Snapshot`] on malformed input (never a panic) or on a
/// `format_version` other than [`SNAPSHOT_FORMAT_VERSION`].
pub fn decode(bytes: &[u8]) -> CbResult<TrainSnapshot> {
    let snapshot: TrainSnapshot = serde_json::from_slice(bytes)
        .map_err(|e| CbError::Snapshot(format!("failed to parse snapshot: {e}")))?;
    if snapshot.format_version != SNAPSHOT_FORMAT_VERSION {
        return Err(CbError::Snapshot(format!(
            "unsupported snapshot format_version {} (this build reads {SNAPSHOT_FORMAT_VERSION})",
            snapshot.format_version
        )));
    }
    Ok(snapshot)
}

/// Every `f64` the snapshot carries must be finite — see [`encode`].
fn check_finite(snapshot: &TrainSnapshot) -> CbResult<()> {
    let non_finite = |what: &str| CbError::Snapshot(format!("snapshot carries a non-finite {what}"));

    if !snapshot.bias.is_finite() {
        return Err(non_finite("bias"));
    }
    if snapshot.approx.iter().any(|v| !v.is_finite()) {
        return Err(non_finite("approx value"));
    }
    for tree in &snapshot.trees {
        if tree.splits.iter().any(|s| !s.border.is_finite()) {
            return Err(non_finite("split border"));
        }
        if tree.leaf_values.iter().any(|v| !v.is_finite()) {
            return Err(non_finite("leaf value"));
        }
        if tree.leaf_weights.iter().any(|v| !v.is_finite()) {
            return Err(non_finite("leaf weight"));
        }
    }
    Ok(())
}

/// Mirror a live tree into its DTO.
///
/// # Errors
/// [`CbError::Snapshot`] if the tree carries categorical structure (`ctr_splits`,
/// `one_hot_splits` or `level_kinds`), which slice 1 cannot represent. TASK-06's
/// scope guard refuses such a run before any tree is grown, so reaching this is a
/// defect; failing loudly beats writing a checkpoint that resumes into a different
/// model.
pub fn dto_from_tree(tree: &ObliviousTree) -> CbResult<ObliviousTreeDto> {
    if !tree.ctr_splits.is_empty() {
        return Err(CbError::Snapshot(
            "cannot snapshot a tree carrying CTR splits (categorical training is out of the \
             snapshot regime)"
                .to_owned(),
        ));
    }
    if !tree.one_hot_splits.is_empty() {
        return Err(CbError::Snapshot(
            "cannot snapshot a tree carrying one-hot splits (categorical training is out of the \
             snapshot regime)"
                .to_owned(),
        ));
    }
    if !tree.level_kinds.is_empty() {
        return Err(CbError::Snapshot(
            "cannot snapshot a tree carrying an explicit level-kind order (interleaved split \
             kinds are out of the snapshot regime)"
                .to_owned(),
        ));
    }
    Ok(ObliviousTreeDto {
        splits: tree
            .splits
            .iter()
            .map(|s| SplitDto { feature: s.feature, border: s.border })
            .collect(),
        leaf_values: tree.leaf_values.clone(),
        leaf_weights: tree.leaf_weights.clone(),
    })
}

/// Rebuild a live tree from its DTO — the exact inverse of [`dto_from_tree`] over
/// the float-only shape (the three categorical vectors come back EMPTY, which is
/// what `dto_from_tree` required them to be).
#[must_use]
pub fn tree_from_dto(dto: &ObliviousTreeDto) -> ObliviousTree {
    ObliviousTree {
        splits: dto
            .splits
            .iter()
            .map(|s| Split { feature: s.feature, border: s.border })
            .collect(),
        ctr_splits: Vec::new(),
        one_hot_splits: Vec::new(),
        level_kinds: Vec::new(),
        leaf_values: dto.leaf_values.clone(),
        leaf_weights: dto.leaf_weights.clone(),
    }
}

// ---------------------------------------------------------------------------
// Compat fingerprint (ORCH-03-S4)
// ---------------------------------------------------------------------------

/// FNV-1a 64-bit offset basis.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Fold bytes into a running FNV-1a hash.
///
/// FNV-1a is used rather than `std::collections::hash_map::DefaultHasher` because
/// the standard hasher's output is explicitly NOT guaranteed stable across
/// toolchain versions — and a fingerprint that changes when the compiler changes
/// would reject every pre-existing snapshot after an upgrade.
fn feed(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h ^= u64::from(b);
        *h = h.wrapping_mul(FNV_PRIME);
    }
}

/// Fold a stable per-variant tag for the loss, plus every numeric payload field of
/// the parametric variants.
///
/// The match is EXHAUSTIVE by design — no `_` arm. A `Loss` variant added later is
/// then a COMPILE ERROR here rather than a silently discriminant-only hash, which
/// is exactly the corruption shape three separate plan-review passes each caught a
/// fresh instance of. The tags are explicit literals, never `mem::discriminant`,
/// whose numeric value is not a stable contract.
#[allow(clippy::match_same_arms)]
fn feed_loss(h: &mut u64, loss: &cb_compute::Loss) {
    use cb_compute::Loss as L;

    let mut tag = |t: u16| feed(h, &t.to_le_bytes());
    match loss {
        // --- non-parametric, single-dimension, in-scope -----------------------
        L::Rmse => tag(1),
        L::Logloss => tag(2),
        L::CrossEntropy => tag(3),
        L::Mae => tag(4),
        L::LogCosh => tag(5),
        L::Poisson => tag(6),
        L::Mape => tag(7),

        // --- parametric, single-dimension, in-scope: tag AND payload ----------
        // Each payload is load-bearing for the per-object gradient / leaf math yet
        // SHARES a tag with a differently-parameterized sibling, so hashing the tag
        // alone would accept a changed parameter silently.
        L::Focal { alpha, gamma } => {
            tag(8);
            feed(h, &alpha.to_bits().to_le_bytes());
            feed(h, &gamma.to_bits().to_le_bytes());
        }
        L::Quantile { alpha, delta } => {
            tag(9);
            feed(h, &alpha.to_bits().to_le_bytes());
            feed(h, &delta.to_bits().to_le_bytes());
        }
        L::Lq { q } => {
            tag(10);
            feed(h, &q.to_bits().to_le_bytes());
        }
        L::Huber { delta } => {
            tag(11);
            feed(h, &delta.to_bits().to_le_bytes());
        }
        L::Expectile { alpha } => {
            tag(12);
            feed(h, &alpha.to_bits().to_le_bytes());
        }
        L::Tweedie { variance_power } => {
            tag(13);
            feed(h, &variance_power.to_bits().to_le_bytes());
        }

        // --- multi-dimensional: scope-guard-rejected before a fingerprint is ever
        // computed. `MultiQuantile` is the one variant that can be SINGLE-dimension
        // (`approx_dimension == alpha.len()`), so its payload is folded in full;
        // `alpha.len()` is fed too, so a length change cannot collide.
        L::MultiQuantile { alpha, delta } => {
            tag(14);
            feed(h, &(alpha.len() as u64).to_le_bytes());
            for a in alpha {
                feed(h, &a.to_bits().to_le_bytes());
            }
            feed(h, &delta.to_bits().to_le_bytes());
        }
        L::MultiClass => tag(15),
        L::MultiClassOneVsAll => tag(16),
        L::MultiLogloss => tag(17),
        L::MultiCrossEntropy => tag(18),
        L::RmseWithUncertainty => tag(19),

        // --- grouped / ranking: scope-guard-rejected (`is_grouped_loss`). Payloads
        // are folded anyway so that widening the regime later cannot reintroduce the
        // tag-only gap.
        L::QueryRmse => tag(20),
        L::QuerySoftMax { lambda, beta } => {
            tag(21);
            feed(h, &lambda.to_bits().to_le_bytes());
            feed(h, &beta.to_bits().to_le_bytes());
        }
        L::PairLogit => tag(22),
        L::PairLogitPairwise => tag(23),
        L::LambdaMart { metric, sigma, top, norm } => {
            tag(24);
            feed(h, &format!("{metric:?}").into_bytes());
            feed(h, &sigma.to_bits().to_le_bytes());
            feed(h, &top.to_le_bytes());
            feed(h, &[u8::from(*norm)]);
        }
        L::YetiRank { permutations, decay } => {
            tag(25);
            feed(h, &permutations.to_le_bytes());
            feed(h, &decay.to_bits().to_le_bytes());
        }
        L::YetiRankPairwise { permutations, decay } => {
            tag(26);
            feed(h, &permutations.to_le_bytes());
            feed(h, &decay.to_bits().to_le_bytes());
        }
        L::StochasticRank { metric, sigma, mu, num_estimations } => {
            tag(27);
            feed(h, &format!("{metric:?}").into_bytes());
            feed(h, &sigma.to_bits().to_le_bytes());
            feed(h, &mu.to_bits().to_le_bytes());
            feed(h, &num_estimations.to_le_bytes());
        }

        // A user objective is an `Arc<dyn CustomObjective>` whose only identity is
        // process-local — it CANNOT be fingerprinted across runs, which is why the
        // scope guard rejects it outright (a resumed run could silently pair a
        // snapshot with a different objective). The tag is fed for completeness;
        // the guard, not this hash, is the protection.
        L::Custom(_) => tag(28),
    }
}

/// The compat key of a training run: a stable 64-bit hash over every input the
/// scoped boosting path reads. A resume whose stored fingerprint differs from the
/// starting run's is REJECTED ([`check_resume`]) rather than silently continued
/// against different data or hyperparameters.
///
/// # Hashed field order (part of the contract — changing it invalidates every
/// existing snapshot, which is what [`SNAPSHOT_FORMAT_VERSION`] exists to signal)
///
/// 1. the loss tag AND every parametric payload ([`feed_loss`])
/// 2. `iterations`  3. `depth`  4. `learning_rate`  5. `l2_leaf_reg`
/// 6. `random_seed`  7. `boosting_type`  8. `leaf_method`  9. `score_function`
/// 10. `min_data_in_leaf`  11. `monotone_constraints` (length + elements)
/// 12. `boost_from_average`  13. `auto_learning_rate`  14. `n`
/// 15. `feature_borders` (per-feature length + every border's bits)
/// 16. `target` (length + every value's bits)
/// 17. `weights` (length + every value's bits)
///
/// `weights` (field 17) is a DATA input on exactly the same footing as `target`:
/// the scoped boosting path reduces leaf statistics over it every iteration, so
/// two runs that differ only in their per-object weights produce different trees.
/// It is the EFFECTIVE weight column the trainer consumes — i.e. the vector after
/// `class_weights` / `auto_class_weights` / `scale_pos_weight` resolution — so a
/// resume that changes any of those inputs is rejected even though none of them
/// is a `BoostParams` field. An unweighted run passes the all-`1.0` column the
/// trainer actually uses, which is a fixed function of `n` and therefore does not
/// depend on how the caller spelled "no weights".
///
/// Floats are folded via `to_bits`, never via their decimal rendering, so the hash
/// is exact rather than round-trip-dependent. Every variable-length collection
/// feeds its LENGTH before its elements, so a regrouping that leaves the flattened
/// byte sequence unchanged still moves the hash.
///
/// `min_data_in_leaf` (field 10) is folded DEFENSIVELY: the scoped SymmetricTree
/// dispatch does not read it. Over-fingerprinting can only cause a spurious — and
/// safe — rejection; under-fingerprinting causes silent corruption.
#[must_use]
pub fn fingerprint(
    params: &crate::BoostParams,
    n: usize,
    feature_borders: &[Vec<f64>],
    target: &[f64],
    weights: &[f64],
) -> u64 {
    let mut h = FNV_OFFSET;

    // 1. loss (tag + parametric payloads)
    feed_loss(&mut h, &params.loss);
    // 2..6
    feed(&mut h, &(params.iterations as u64).to_le_bytes());
    feed(&mut h, &(params.depth as u64).to_le_bytes());
    feed(&mut h, &params.learning_rate.to_bits().to_le_bytes());
    feed(&mut h, &params.l2_leaf_reg.to_bits().to_le_bytes());
    feed(&mut h, &params.random_seed.to_le_bytes());
    // 7..9 — explicit tags, not `mem::discriminant` (see `feed_loss`).
    feed(&mut h, &[boosting_type_tag(params.boosting_type)]);
    feed(&mut h, &[leaf_method_tag(params.leaf_method)]);
    feed(&mut h, &[score_function_tag(params.score_function)]);
    // 10..13
    feed(&mut h, &(params.min_data_in_leaf as u64).to_le_bytes());
    feed(&mut h, &(params.monotone_constraints.len() as u64).to_le_bytes());
    for &c in &params.monotone_constraints {
        feed(&mut h, &c.to_le_bytes());
    }
    feed(&mut h, &[u8::from(params.boost_from_average)]);
    feed(&mut h, &[u8::from(params.auto_learning_rate)]);
    // 14..16 — the data inputs
    feed(&mut h, &(n as u64).to_le_bytes());
    feed(&mut h, &(feature_borders.len() as u64).to_le_bytes());
    for borders in feature_borders {
        feed(&mut h, &(borders.len() as u64).to_le_bytes());
        for b in borders {
            feed(&mut h, &b.to_bits().to_le_bytes());
        }
    }
    feed(&mut h, &(target.len() as u64).to_le_bytes());
    for t in target {
        feed(&mut h, &t.to_bits().to_le_bytes());
    }
    // 17 — the EFFECTIVE per-object weight column (see the doc above).
    feed(&mut h, &(weights.len() as u64).to_le_bytes());
    for w in weights {
        feed(&mut h, &w.to_bits().to_le_bytes());
    }

    h
}

/// Stable tag for [`crate::EBoostingType`] — exhaustive, so a new variant is a
/// compile error rather than a silent collision.
fn boosting_type_tag(t: crate::EBoostingType) -> u8 {
    match t {
        crate::EBoostingType::Plain => 1,
        crate::EBoostingType::Ordered => 2,
    }
}

/// Stable tag for [`cb_compute::LeafMethod`] — exhaustive by design.
fn leaf_method_tag(m: cb_compute::LeafMethod) -> u8 {
    match m {
        cb_compute::LeafMethod::Gradient => 1,
        cb_compute::LeafMethod::Newton => 2,
        cb_compute::LeafMethod::Simple => 3,
        cb_compute::LeafMethod::Exact => 4,
    }
}

/// Stable tag for [`cb_compute::EScoreFunction`] — exhaustive by design.
fn score_function_tag(s: cb_compute::EScoreFunction) -> u8 {
    match s {
        cb_compute::EScoreFunction::Cosine => 1,
        cb_compute::EScoreFunction::L2 => 2,
        cb_compute::EScoreFunction::SolarL2 => 3,
        cb_compute::EScoreFunction::NewtonL2 => 4,
        cb_compute::EScoreFunction::NewtonCosine => 5,
        cb_compute::EScoreFunction::LOOL2 => 6,
        cb_compute::EScoreFunction::SatL2 => 7,
    }
}

/// Gate a resume on fingerprint equality.
///
/// # Errors
/// [`CbError::Snapshot`] when the snapshot's stored fingerprint differs from the
/// starting run's — the run's data or hyperparameters changed, so continuing would
/// produce a model that is neither the snapshot's nor the new configuration's.
pub fn check_resume(stored: u64, current: u64) -> CbResult<()> {
    if stored == current {
        return Ok(());
    }
    Err(CbError::Snapshot(format!(
        "snapshot fingerprint {stored} does not match this run's fingerprint {current}: the \
         training data or parameters changed, so the snapshot cannot be resumed"
    )))
}

// ---------------------------------------------------------------------------
// Checkpoint write (ORCH-03-S5)
// ---------------------------------------------------------------------------

/// Write a snapshot to `path` ATOMICALLY: serialize into a sibling temporary file,
/// then `rename` it into place.
///
/// A checkpoint is overwritten repeatedly while training runs, so a plain
/// truncate-and-write leaves a window in which the file on disk is neither the old
/// snapshot nor the new one. A process killed in that window — which is exactly the
/// scenario snapshots exist for — would find a torn file on resume. `rename` within
/// one directory is atomic on every platform this crate targets, so a reader sees
/// either the previous complete snapshot or the new one.
///
/// The temporary file is a SIBLING (same directory) because `rename` across
/// filesystems is not atomic and may fail outright.
///
/// # Errors
/// [`CbError::Snapshot`] if the snapshot cannot be serialized (see [`encode`]), if
/// `path` has no parent directory, or on any I/O failure.
pub fn write_atomic(path: &std::path::Path, snapshot: &TrainSnapshot) -> CbResult<()> {
    let bytes = encode(snapshot)?;

    let parent = path.parent().ok_or_else(|| {
        CbError::Snapshot(format!("snapshot path {} has no parent directory", path.display()))
    })?;
    // A fixed `.tmp` sibling: one writer per snapshot path is the training-loop
    // contract, and a stable name leaves no debris behind on a crash.
    let mut tmp_name = path.file_name().unwrap_or(std::ffi::OsStr::new("snapshot")).to_os_string();
    tmp_name.push(".tmp");
    let tmp = parent.join(tmp_name);

    std::fs::write(&tmp, &bytes).map_err(|e| {
        CbError::Snapshot(format!("failed to write snapshot to {}: {e}", tmp.display()))
    })?;
    std::fs::rename(&tmp, path).map_err(|e| {
        CbError::Snapshot(format!(
            "failed to move snapshot into place at {}: {e}",
            path.display()
        ))
    })
}

/// Read a snapshot back from `path`.
///
/// # Errors
/// [`CbError::Snapshot`] on an I/O failure or on any [`decode`] failure.
pub fn read_from(path: &std::path::Path) -> CbResult<TrainSnapshot> {
    let bytes = std::fs::read(path).map_err(|e| {
        CbError::Snapshot(format!("failed to read snapshot {}: {e}", path.display()))
    })?;
    decode(&bytes)
}

/// Build the checkpoint for a completed iteration boundary.
///
/// `completed_iters` is the number of trees already grown — i.e. `iter + 1` at the
/// end of iteration `iter` — and is exactly the iteration a resumed run starts at.
///
/// # Errors
/// [`CbError::Snapshot`] if any tree carries categorical structure the slice-1 DTO
/// cannot represent (see [`dto_from_tree`]).
pub fn capture(
    completed_iters: usize,
    fingerprint: u64,
    bias: f64,
    approx_dimension: usize,
    approx: &[f64],
    trees: &[ObliviousTree],
    rng: &cb_core::TFastRng64,
) -> CbResult<TrainSnapshot> {
    let dtos = trees.iter().map(dto_from_tree).collect::<CbResult<Vec<_>>>()?;
    Ok(TrainSnapshot {
        format_version: SNAPSHOT_FORMAT_VERSION,
        fingerprint,
        completed_iters,
        bias,
        approx_dimension,
        approx: approx.to_vec(),
        trees: dtos,
        rng_raw_state: rng.raw_state(),
        rng_call_count: rng.call_count(),
    })
}

#[cfg(test)]
#[path = "snapshot_test.rs"]
mod tests;
