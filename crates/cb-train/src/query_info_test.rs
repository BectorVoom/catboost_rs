//! Unit tests for the [`super::build_query_info`] grouped-view builder
//! (LOSS-04, D-6.3-03). Mirrors upstream `GroupSamples` (query.h:48-67) run
//! detection + the `data_providers.cpp:315-340` pairs→competitors mapping.
//!
//! Source/test separation (INFRA-06): these tests live in this dedicated file,
//! linked from `query_info.rs` via the `#[path]` footer — never inline.

use super::{build_query_info, Competitor};
use cb_data::Pair;

#[test]
fn multi_group_contiguous_runs_produce_correct_spans() {
    // Three contiguous groups: [0,2), [2,5), [5,6).
    let group_id = [10, 10, 20, 20, 20, 30];
    let groups = build_query_info(6, &group_id, &[], &[], &[]).unwrap();
    assert_eq!(groups.len(), 3);
    assert_eq!((groups[0].begin, groups[0].end), (0, 2));
    assert_eq!((groups[1].begin, groups[1].end), (2, 5));
    assert_eq!((groups[2].begin, groups[2].end), (5, 6));
    assert_eq!(groups[0].size(), 2);
    assert_eq!(groups[1].size(), 3);
    assert_eq!(groups[2].size(), 1);
}

#[test]
fn empty_group_id_yields_single_full_span_group() {
    let groups = build_query_info(5, &[], &[], &[], &[]).unwrap();
    assert_eq!(groups.len(), 1);
    assert_eq!((groups[0].begin, groups[0].end), (0, 5));
    // No weights supplied → every group weight defaults to 1.0.
    assert!((groups[0].weight - 1.0).abs() < 1e-12);
}

#[test]
fn empty_dataset_yields_no_groups() {
    let groups = build_query_info(0, &[], &[], &[], &[]).unwrap();
    assert!(groups.is_empty());
}

#[test]
fn subgroup_id_is_carried_group_local() {
    let group_id = [1, 1, 2];
    let subgroup_id = [100, 101, 200];
    let groups = build_query_info(3, &group_id, &subgroup_id, &[], &[]).unwrap();
    assert_eq!(groups[0].subgroup_id, vec![100, 101]);
    assert_eq!(groups[1].subgroup_id, vec![200]);
}

#[test]
fn group_weight_is_mean_of_member_weights() {
    let group_id = [1, 1, 2];
    let weights = [2.0, 4.0, 9.0];
    let groups = build_query_info(3, &group_id, &[], &[], &weights).unwrap();
    // Group 0: mean(2,4) = 3.0; group 1: single member → 9.0.
    assert!((groups[0].weight - 3.0).abs() < 1e-12);
    assert!((groups[1].weight - 9.0).abs() < 1e-12);
}

#[test]
fn explicit_pairs_map_to_group_local_competitors() {
    // Group [0,3): winner 0 over loser 1, winner 1 over loser 2.
    let group_id = [7, 7, 7, 8, 8];
    let pairs = [
        Pair {
            winner_id: 0,
            loser_id: 1,
        },
        Pair {
            winner_id: 1,
            loser_id: 2,
        },
    ];
    let groups = build_query_info(5, &group_id, &[], &pairs, &[]).unwrap();
    // Group 0 competitors: winner-local 0 → loser-local 1; winner-local 1 →
    // loser-local 2; winner-local 2 → none.
    assert_eq!(
        groups[0].competitors[0],
        vec![Competitor {
            id: 1,
            weight: 1.0
        }]
    );
    assert_eq!(
        groups[0].competitors[1],
        vec![Competitor {
            id: 2,
            weight: 1.0
        }]
    );
    assert!(groups[0].competitors[2].is_empty());
    // Group 1 has no pairs → all-empty rows.
    assert_eq!(groups[1].competitors.len(), 2);
    assert!(groups[1].competitors.iter().all(Vec::is_empty));
}

#[test]
fn non_contiguous_group_id_is_degenerate_error() {
    // Group id 1 reappears after 2 intervened → upstream "queryIds should be
    // grouped" → typed Degenerate error, NOT a panic.
    let group_id = [1, 1, 2, 1];
    let err = build_query_info(4, &group_id, &[], &[], &[]).unwrap_err();
    assert!(matches!(err, cb_core::CbError::Degenerate(_)));
}

#[test]
fn cross_group_pair_is_out_of_range_error() {
    // Winner in group 0, loser in group 1.
    let group_id = [1, 1, 2, 2];
    let pairs = [Pair {
        winner_id: 0,
        loser_id: 2,
    }];
    let err = build_query_info(4, &group_id, &[], &pairs, &[]).unwrap_err();
    assert!(matches!(err, cb_core::CbError::OutOfRange(_)));
}

#[test]
fn out_of_range_pair_is_out_of_range_error() {
    let group_id = [1, 1];
    let pairs = [Pair {
        winner_id: 0,
        loser_id: 9,
    }];
    let err = build_query_info(2, &group_id, &[], &pairs, &[]).unwrap_err();
    assert!(matches!(err, cb_core::CbError::OutOfRange(_)));
}

#[test]
fn mismatched_column_length_is_degenerate_error() {
    let group_id = [1, 1, 2];
    let err = build_query_info(2, &group_id, &[], &[], &[]).unwrap_err();
    assert!(matches!(err, cb_core::CbError::Degenerate(_)));
}
