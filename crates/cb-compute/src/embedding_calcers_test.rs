//! Tests for the LDA embedding calcer ([`crate::embedding_calcers`]).
//!
//! These lock the per-class [`IncrementalCloud`] scatter accumulation +
//! [`total_scatter`] + [`between_matrix`] against the instrumented upstream
//! ground truth (`fixtures/embedding_calcers/LDA/scatter_projection_gt.json`,
//! captured by the 06.5-05 `lda_scatter` hook), and the full
//! [`LdaCalcer::fit`] → [`LdaCalcer::compute`] pipeline against the
//! reference-faithful eigensolve.
//!
//! Parity note (06.5-05): the SCATTER matches upstream to ≤5e-10; the eigensolve
//! is reference-faithful (matches f64 scipy and f32 LAPACK `ssygv`). The upstream
//! vendored-CLAPACK `ssyev_` iterate differs by 4.9e-2 on the dominant eigenvector
//! — but that divergence does not cross any split border, so the model oracle is
//! byte-identical (verified in `cb-oracle::lda_oracle_test`).

use crate::embedding_calcers::{
    between_matrix, total_scatter, IncrementalCloud, KnnCalcer, KnnCloud, LdaCalcer,
};

const DIM: usize = 4;
const REG: f32 = 0.05;

/// The frozen corpus embeddings (16 rows × 4 dims) — mirrors the
/// `text_embedding_inputs` corpus the instrumented GT was captured over.
#[rustfmt::skip]
const EMB: [[f32; 4]; 16] = [
    [0.8480936288833618, 1.3459652662277222, -0.8181768655776978, -0.9245402812957764],
    [-1.6112185716629028, -1.1183204650878906, 0.9326959252357483, 1.0004271268844604],
    [0.9156883955001831, 0.9867314696311951, -1.1083682775497437, -1.257594108581543],
    [-0.8664456009864807, -0.8088169097900391, 0.7051903605461121, 0.5060048699378967],
    [1.053161859512329, 1.1903846263885498, -1.2830679416656494, -0.8254095315933228],
    [-0.9955786466598511, -1.659306287765503, 1.1641911268234253, 1.08004629611969],
    [1.0400522947311401, 1.4191780090332031, -1.4298564195632935, -1.3421956300735474],
    [-1.0222734212875366, -1.1926385164260864, 0.951580286026001, 1.170548439025879],
    [1.1097509860992432, 1.0400322675704956, -0.656127393245697, -1.319750189781189],
    [-1.2417240142822266, -0.7331587672233582, 1.3103097677230835, 0.8206236362457275],
    [1.2133253812789917, 1.0596122741699219, -1.0053892135620117, -0.5424982309341431],
    [-0.8437013030052185, -1.1891505718231201, 0.7384138107299805, 0.947220504283905],
    [1.2349718809127808, 1.4978270530700684, -0.9429530501365662, -1.0751142501831055],
    [-1.1465598344802856, -0.5449184775352478, 0.941191554069519, 0.8506897687911987],
    [1.2464923858642578, 1.2335710525512695, -0.8773979544639587, -0.9625028371810913],
    [-1.3708170652389526, -0.7448427081108093, 1.0651146173477173, 0.7684383392333984],
];
/// Per-row class label (1 = positive cloud, 0 = negative cloud).
const LAB: [usize; 16] = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0];

/// Final-Flush upstream betweenMatrix B (regularized, row-major) from the frozen
/// GT (`scatter_projection_gt.json`, `final_flush_scatter_inner`).
#[rustfmt::skip]
const GT_SCATTER_INNER: [f32; 16] = [
    0.08977694809436798, -0.008402512408792973, -0.007163579575717449, 0.002264200011268258,
    -0.008402512408792973, 0.12173843383789062, -0.006083576008677483, -0.02397523634135723,
    -0.007163578644394875, -0.006083576008677483, 0.09581366181373596, 0.006580251269042492,
    0.0022642009425908327, -0.02397523634135723, 0.006580251269042492, 0.10182788968086243,
];
/// Final-Flush upstream totalScatter A (row-major) from the frozen GT
/// (`final_flush_scatter_total`).
#[rustfmt::skip]
const GT_SCATTER_TOTAL: [f32; 16] = [
    1.232080101966858, 1.2323991060256958, -1.105136513710022, -1.067922592163086,
    1.2323991060256958, 1.2327182292938232, -1.1054227352142334, -1.0681991577148438,
    -1.105136513710022, -1.1054227352142334, 0.9912723302841187, 0.957892656326294,
    -1.067922592163086, -1.0681991577148438, 0.957892656326294, 0.9256369471549988,
];

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Build the two per-class clouds over the corpus in object order (the identity
/// permutation; the ordered-prefix variant is exercised in cb-train).
fn build_clouds() -> (Vec<IncrementalCloud>, f32) {
    let mut clouds = vec![IncrementalCloud::new(DIM), IncrementalCloud::new(DIM)];
    let mut size = 0.0f32;
    for (row, &lab) in EMB.iter().zip(LAB.iter()) {
        clouds
            .get_mut(lab)
            .expect("label in range")
            .add_vector(row)
            .expect("dim ok");
        size += 1.0;
    }
    (clouds, size)
}

#[test]
fn scatter_inner_matches_upstream_gt() {
    // The regularized betweenMatrix (B) must match the instrumented upstream dump.
    let (clouds, size) = build_clouds();
    let between = between_matrix(&clouds, size, DIM, REG).expect("between ok");
    let err = max_abs_err(&between, &GT_SCATTER_INNER);
    assert!(
        err <= 1e-5,
        "betweenMatrix vs upstream GT err {err:e} > 1e-5; got {between:?}"
    );
}

#[test]
fn scatter_total_matches_upstream_gt() {
    // The totalScatter (A) must match the instrumented upstream dump.
    let (clouds, size) = build_clouds();
    let total = total_scatter(&clouds, size, DIM).expect("total ok");
    let err = max_abs_err(&total, &GT_SCATTER_TOTAL);
    assert!(
        err <= 1e-5,
        "totalScatter vs upstream GT err {err:e} > 1e-5; got {total:?}"
    );
}

#[test]
fn fit_produces_unit_norm_projection() {
    // The eigensolve projection is the (Euclidean) unit-norm dominant eigenvector
    // of the reduced problem.
    let (clouds, size) = build_clouds();
    let calcer = LdaCalcer::fit(&clouds, size, DIM, 1, REG).expect("fit ok");
    let proj = calcer.projection_matrix();
    assert_eq!(proj.len(), DIM);
    let norm: f32 = proj.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-4, "projection norm {norm} != 1");
}

#[test]
fn compute_projects_via_sgemv() {
    // compute(embed) == projection · embed (the cblas_sgemv project step).
    let (clouds, size) = build_clouds();
    let calcer = LdaCalcer::fit(&clouds, size, DIM, 1, REG).expect("fit ok");
    let proj = calcer.projection_matrix().to_vec();
    for row in EMB.iter() {
        let out = calcer.compute(row).expect("compute ok");
        assert_eq!(out.len(), 1);
        let manual: f32 = proj.iter().zip(row.iter()).map(|(p, e)| p * e).sum();
        let got = *out.first().expect("one output");
        assert!((got - manual).abs() < 1e-5, "{got} != {manual}");
    }
}

#[test]
fn compute_rejects_dim_mismatch() {
    let (clouds, size) = build_clouds();
    let calcer = LdaCalcer::fit(&clouds, size, DIM, 1, REG).expect("fit ok");
    assert!(calcer.compute(&[1.0, 2.0, 3.0]).is_err());
    assert!(calcer.compute(&[1.0, 2.0, 3.0, 4.0, 5.0]).is_err());
}

#[test]
fn add_vector_rejects_wrong_dim() {
    let mut cloud = IncrementalCloud::new(DIM);
    assert!(cloud.add_vector(&[1.0, 2.0]).is_err());
}

#[test]
fn feature_count_is_projection_dim() {
    let (clouds, size) = build_clouds();
    let calcer = LdaCalcer::fit(&clouds, size, DIM, 1, REG).expect("fit ok");
    assert_eq!(calcer.feature_count(), 1);
}

#[test]
fn between_matrix_regression_path_single_class() {
    // One cloud (regression) -> betweenMatrix = that cloud's scatter + reg diagonal.
    let mut cloud = IncrementalCloud::new(DIM);
    for row in EMB.iter() {
        cloud.add_vector(row).expect("dim ok");
    }
    let size = EMB.len() as f32;
    let between = between_matrix(std::slice::from_ref(&cloud), size, DIM, REG).expect("between ok");
    // Diagonal must be scatter diag + reg; off-diagonal must equal scatter.
    for i in 0..DIM {
        for j in 0..DIM {
            let expect = cloud.scatter().get(i * DIM + j).copied().unwrap_or(0.0)
                + if i == j { REG } else { 0.0 };
            let got = between.get(i * DIM + j).copied().unwrap_or(0.0);
            assert!((got - expect).abs() < 1e-6, "cell ({i},{j}): {got} != {expect}");
        }
    }
}

// ---------------------------------------------------------------------------
// KNN calcer (brute-force-exact L2-squared, spike-validated 06.5-06).
// ---------------------------------------------------------------------------

/// The frozen corpus binary labels (alternating 1,0 — matches `labels.npy`).
const LABELS: [usize; 16] = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0];

#[test]
fn knn_cloud_empty_prefix_yields_no_neighbors() {
    let cloud = KnnCloud::new(DIM);
    assert!(cloud.is_empty());
    let nb = cloud.nearest_neighbors(&EMB[0], 5).expect("ok");
    assert!(nb.is_empty(), "empty cloud must return no neighbors");
}

#[test]
fn knn_cloud_prefix_smaller_than_k_returns_all_sorted_by_distance() {
    // Insert docs 0,1,2 (insertion order), query doc 3 -> all 3 sorted by L2.
    let mut cloud = KnnCloud::new(DIM);
    for i in 0..3 {
        cloud.add_vector(&EMB[i]).expect("dim ok");
    }
    let nb = cloud.nearest_neighbors(&EMB[3], 5).expect("ok");
    // Per the instrumented dump (doc index 3, prefix {0,1,2}): [1, 0, 2].
    assert_eq!(nb, vec![1, 0, 2], "neighbor order must match the HNSW dump");
}

#[test]
fn knn_cloud_self_is_nearest_at_distance_zero() {
    // Whole-set insert; query a member -> it is its own nearest (distance 0).
    let mut cloud = KnnCloud::new(DIM);
    for row in EMB.iter() {
        cloud.add_vector(row).expect("dim ok");
    }
    let nb = cloud.nearest_neighbors(&EMB[7], 3).expect("ok");
    assert_eq!(nb.first().copied(), Some(7), "self must be the nearest neighbor");
    assert_eq!(nb.len(), 3, "k=3 neighbors returned");
}

#[test]
fn knn_cloud_rejects_dim_mismatch_on_add_and_query() {
    let mut cloud = KnnCloud::new(DIM);
    assert!(cloud.add_vector(&[1.0, 2.0, 3.0]).is_err(), "short add must error");
    cloud.add_vector(&EMB[0]).expect("dim ok");
    assert!(
        cloud.nearest_neighbors(&[1.0, 2.0, 3.0], 1).is_err(),
        "short query must error"
    );
}

#[test]
fn knn_calcer_classification_vote_counts_width_is_num_classes() {
    // Online insert docs 0..2 (read-before-update prefix), compute doc 3.
    let mut calcer = KnnCalcer::new(DIM, 3, true, 2).expect("knn calcer");
    assert_eq!(calcer.feature_count(), 2, "clf width = num_classes");
    for i in 0..3 {
        calcer
            .update(LABELS[i] as f32, &EMB[i])
            .expect("update ok");
    }
    // doc3 neighbors over {0,1,2} = [1,0,2]; classes [0,1,1] -> votes class0=1, class1=2.
    let feat = calcer.compute(&EMB[3]).expect("compute ok");
    assert_eq!(feat, vec![1.0, 2.0], "per-class vote counts");
}

#[test]
fn knn_calcer_classification_empty_prefix_is_all_zero() {
    let calcer = KnnCalcer::new(DIM, 3, true, 2).expect("knn calcer");
    let feat = calcer.compute(&EMB[0]).expect("compute ok");
    assert_eq!(feat, vec![0.0, 0.0], "no neighbors -> zero votes");
}

#[test]
fn knn_calcer_regression_mean_target_width_one() {
    // Regression: width=1, result[0] = mean(neighbor targets).
    let mut calcer = KnnCalcer::new(DIM, 3, false, 2).expect("knn calcer");
    assert_eq!(calcer.feature_count(), 1, "reg width = 1 (Pitfall 5)");
    // Insert docs 0,1,2 with targets 10,20,30.
    calcer.update(10.0, &EMB[0]).expect("ok");
    calcer.update(20.0, &EMB[1]).expect("ok");
    calcer.update(30.0, &EMB[2]).expect("ok");
    // doc3 neighbors [1,0,2] -> mean(20,10,30) = 20.
    let feat = calcer.compute(&EMB[3]).expect("compute ok");
    assert_eq!(feat.len(), 1);
    assert!((feat[0] - 20.0).abs() < 1e-6, "neighbor target mean");
}

#[test]
fn knn_calcer_regression_empty_prefix_is_zero() {
    let calcer = KnnCalcer::new(DIM, 3, false, 2).expect("knn calcer");
    let feat = calcer.compute(&EMB[0]).expect("compute ok");
    assert_eq!(feat, vec![0.0], "no neighbors -> zero mean");
}

#[test]
fn knn_calcer_whole_set_separates_classes_at_border_half() {
    // Whole-set (offline Plain) insert; the k=3 class0 vote perfectly separates
    // classes (class0 docs -> [3,0], class1 docs -> [0,3]) at the 0.5 border.
    let mut calcer = KnnCalcer::new(DIM, 3, true, 2).expect("knn calcer");
    for (i, row) in EMB.iter().enumerate() {
        calcer.update(LABELS[i] as f32, row).expect("update ok");
    }
    for (i, row) in EMB.iter().enumerate() {
        let feat = calcer.compute(row).expect("compute ok");
        let class0_vote = feat.first().copied().unwrap_or(0.0);
        if LABELS[i] == 0 {
            assert!(class0_vote > 0.5, "doc{i} (class0) vote {class0_vote} > 0.5");
        } else {
            assert!(class0_vote < 0.5, "doc{i} (class1) vote {class0_vote} < 0.5");
        }
    }
}

#[test]
fn knn_calcer_update_rejects_dim_mismatch() {
    let mut calcer = KnnCalcer::new(DIM, 3, true, 2).expect("knn calcer");
    assert!(calcer.update(0.0, &[1.0, 2.0]).is_err());
}
