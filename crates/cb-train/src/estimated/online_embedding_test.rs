//! Tests for the online-LDA embedding seam ([`crate::estimated::online_embedding`]).
//!
//! These lock the read-before-update prefix discipline (D-03): the FIRST document
//! in the permutation projects to zero (no prefix), and the OFFLINE whole-set fit
//! projects every document against one shared projection. The scatter/projection
//! NUMERIC parity vs upstream is gated in `cb-oracle::lda_oracle_test`.

use crate::estimated::online_embedding::{
    lda_projection_dim, offline_lda_features, online_lda_prefix,
};

const DIM: usize = 4;
const REG: f32 = 0.05;

#[rustfmt::skip]
fn corpus() -> (Vec<Vec<f32>>, Vec<usize>) {
    let emb: Vec<Vec<f32>> = vec![
        vec![0.8480936, 1.3459653, -0.8181769, -0.9245403],
        vec![-1.6112186, -1.1183205, 0.9326959, 1.0004271],
        vec![0.9156884, 0.9867315, -1.1083683, -1.2575941],
        vec![-0.8664456, -0.8088169, 0.7051904, 0.5060049],
        vec![1.0531619, 1.1903846, -1.2830679, -0.8254095],
        vec![-0.9955786, -1.6593063, 1.1641911, 1.0800463],
        vec![1.0400523, 1.4191780, -1.4298564, -1.3421956],
        vec![-1.0222734, -1.1926385, 0.9515803, 1.1705484],
        vec![1.1097510, 1.0400323, -0.6561274, -1.3197502],
        vec![-1.2417240, -0.7331588, 1.3103098, 0.8206236],
        vec![1.2133254, 1.0596123, -1.0053892, -0.5424982],
        vec![-0.8437013, -1.1891506, 0.7384138, 0.9472205],
        vec![1.2349719, 1.4978271, -0.9429531, -1.0751143],
        vec![-1.1465598, -0.5449185, 0.9411916, 0.8506898],
        vec![1.2464924, 1.2335711, -0.8773980, -0.9625028],
        vec![-1.3708171, -0.7448427, 1.0651146, 0.7684383],
    ];
    let lab: Vec<usize> = vec![1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
    (emb, lab)
}

#[test]
fn projection_dim_is_min_classes_dim() {
    assert_eq!(lda_projection_dim(2, 4), 1);
    assert_eq!(lda_projection_dim(3, 4), 2);
    assert_eq!(lda_projection_dim(5, 2), 1);
    assert_eq!(lda_projection_dim(2, 1), 0);
}

#[test]
fn offline_projects_every_document() {
    let (emb, lab) = corpus();
    let cols = offline_lda_features(&emb, &lab, 2, REG).expect("offline ok");
    assert_eq!(cols.len(), 1); // proj_dim = 1
    let col = cols.first().expect("one column");
    assert_eq!(col.len(), emb.len());
    // The two class clouds are well-separated -> projections differ in sign by class.
    let pos: Vec<f32> = (0..emb.len()).filter(|&i| lab[i] == 1).map(|i| col[i]).collect();
    let neg: Vec<f32> = (0..emb.len()).filter(|&i| lab[i] == 0).map(|i| col[i]).collect();
    let pos_mean: f32 = pos.iter().sum::<f32>() / pos.len() as f32;
    let neg_mean: f32 = neg.iter().sum::<f32>() / neg.len() as f32;
    assert!(
        (pos_mean - neg_mean).abs() > 1.0,
        "class projections not separated: pos {pos_mean} neg {neg_mean}"
    );
}

#[test]
fn online_first_document_has_no_prefix() {
    // Read-before-update: the FIRST document in the permutation projects against an
    // empty prefix (no projection fitted yet) -> zero.
    let (emb, lab) = corpus();
    let perm: Vec<i32> = (0..emb.len() as i32).collect();
    let out = online_lda_prefix(&perm, &emb, &lab, 2, REG).expect("online ok");
    let first = out.projection_in_order.first().expect("at least one doc");
    assert_eq!(first.len(), 1);
    assert_eq!(*first.first().expect("one value"), 0.0);
}

#[test]
fn online_columns_object_indexed() {
    // A non-identity permutation must still scatter projections OBJECT-indexed.
    let (emb, lab) = corpus();
    let mut perm: Vec<i32> = (0..emb.len() as i32).collect();
    perm.reverse();
    let out = online_lda_prefix(&perm, &emb, &lab, 2, REG).expect("online ok");
    assert_eq!(out.columns.len(), 1);
    assert_eq!(out.columns.first().expect("col").len(), emb.len());
    assert_eq!(out.projection_in_order.len(), emb.len());
    // The LAST permutation position (object 0) sees the full prefix -> nonzero.
    let last = out.projection_in_order.last().expect("last");
    assert_ne!(*last.first().expect("one value"), 0.0);
}

#[test]
fn online_rejects_out_of_range_permutation() {
    let (emb, lab) = corpus();
    let mut perm: Vec<i32> = (0..emb.len() as i32).collect();
    if let Some(slot) = perm.first_mut() {
        *slot = 999;
    }
    assert!(online_lda_prefix(&perm, &emb, &lab, 2, REG).is_err());
}

#[test]
fn online_rejects_length_mismatch() {
    let (emb, lab) = corpus();
    let perm: Vec<i32> = (0..(emb.len() as i32 - 1)).collect();
    assert!(online_lda_prefix(&perm, &emb, &lab, 2, REG).is_err());
}

#[test]
fn offline_rejects_length_mismatch() {
    let (emb, _lab) = corpus();
    let short = vec![1usize; emb.len() - 1];
    assert!(offline_lda_features(&emb, &short, 2, REG).is_err());
}
