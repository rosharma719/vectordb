use crate::utils::types::{DistanceMetric, Vector};
use crate::vector::hnsw::HNSWIndex;
use crate::utils::errors::DBError;

/// Helper to build a test vector
fn vecf(v: &[f32]) -> Vector {
    v.to_vec()
}

fn test_insert_and_basic_search() {
    let mut hnsw = HNSWIndex::new(DistanceMetric::Euclidean, 16, 50, 16, 2);

    hnsw.insert(1, vecf(&[0.0, 0.0])).unwrap();
    hnsw.insert(2, vecf(&[1.0, 1.0])).unwrap();
    hnsw.insert(3, vecf(&[5.0, 5.0])).unwrap();
    hnsw.insert(4, vecf(&[-1.0, -1.0])).unwrap();

    let results = hnsw.search(&vecf(&[0.0, 0.0]), 2).unwrap();
    let ids: Vec<_> = results.iter().map(|r| r.id).collect();

    assert!(ids.contains(&1));
    assert!(ids.iter().any(|&id| id == 2 || id == 4));
}

fn test_empty_search_returns_nothing() {
    let hnsw = HNSWIndex::new(DistanceMetric::Cosine, 16, 50, 16, 2);
    let results = hnsw.search(&vecf(&[0.0, 0.0]), 3).unwrap();
    assert!(results.is_empty());
}

fn test_insertion_is_idempotent() {
    let mut hnsw = HNSWIndex::new(DistanceMetric::Dot, 16, 50, 16, 2);

    let vec = vecf(&[2.0, 2.0]);
    hnsw.insert(42, vec.clone()).unwrap();
    hnsw.insert(42, vec.clone()).unwrap(); // duplicate insert

    let results = hnsw.search(&vec, 1).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, 42);
}

fn test_search_respects_top_k() {
    let mut hnsw = HNSWIndex::new(DistanceMetric::Euclidean, 16, 50, 16, 2);

    for i in 0..20 {
        hnsw.insert(i, vecf(&[i as f32, i as f32])).unwrap();
    }

    let results = hnsw.search(&vecf(&[0.0, 0.0]), 5).unwrap();
    assert_eq!(results.len(), 5);

    let scores: Vec<_> = results.iter().map(|r| r.score).collect();
    for w in scores.windows(2) {
        assert!(w[0] <= w[1]);
    }
}

fn test_high_dimensional_vectors() {
    let mut hnsw = HNSWIndex::new(DistanceMetric::Cosine, 16, 50, 16, 128);

    hnsw.insert(1, vec![1.0; 128]).unwrap();
    hnsw.insert(2, vec![-1.0; 128]).unwrap();
    hnsw.insert(3, vec![0.0; 128]).unwrap();

    let results = hnsw.search(&vec![1.0; 128], 1).unwrap();
    assert_eq!(results[0].id, 1);
}

fn test_dimensionality_mismatch_errors() {
    let mut hnsw = HNSWIndex::new(DistanceMetric::Cosine, 16, 50, 16, 3);

    // Insert a valid point to ensure `entry_point` is not None
    hnsw.insert(42, vecf(&[0.0, 0.0, 0.0])).unwrap();

    // Invalid insert: length 2 instead of 3
    let insert_result = hnsw.insert(1, vecf(&[1.0, 2.0]));
    assert!(matches!(insert_result, Err(DBError::VectorLengthMismatch { .. })));

    // Invalid search: also length 2 instead of 3
    let search_result = hnsw.search(&vecf(&[1.0, 2.0]), 1);
    match search_result {
        Err(DBError::VectorLengthMismatch { expected, actual }) => {
            assert_eq!(expected, 3);
            assert_eq!(actual, 2);
        }
        other => panic!("Expected VectorLengthMismatch, got: {:?}", other),
    }
}


pub fn run_hnsw_tests() {
    println!("Running HNSW tests...");

    test_insert_and_basic_search();
    test_empty_search_returns_nothing();
    test_insertion_is_idempotent();
    test_search_respects_top_k();
    test_high_dimensional_vectors();
    test_dimensionality_mismatch_errors();

    println!("✅ All HNSW tests passed");
}
