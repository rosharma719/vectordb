use vectordb::utils::errors::DBError;
use vectordb::utils::types::*;
use vectordb::vector::metric::*;

#[test]
fn test_distance_metrics() {
    let v1 = vec![1.0, 2.0, 3.0];
    let v2 = vec![4.0, 5.0, 6.0];

    let cosine = distance(&v1, &v2, DistanceMetric::Cosine);
    assert!(
        cosine >= 0.0 && cosine <= 2.0,
        "Cosine distance out of range: {cosine}"
    );

    let dot = distance(&v1, &v2, DistanceMetric::Dot);
    assert!(
        dot <= 0.0,
        "Dot product distance should be non-positive: {dot}"
    );

    let euclidean = distance(&v1, &v2, DistanceMetric::Euclidean);
    assert!(euclidean > 0.0, "Euclidean distance should be positive: {euclidean}");
}

#[test]
fn test_error_display() {
    let not_found = DBError::NotFound(42);
    assert_eq!(format!("{}", not_found), "Point with ID 42 not found");

    let mismatch = DBError::VectorLengthMismatch { expected: 3, actual: 5 };
    assert!(
        format!("{}", mismatch).contains("expected 3"),
        "Mismatch message incorrect"
    );

    let wal = DBError::WALCorrupt("bad header".to_string());
    assert!(format!("{}", wal).contains("bad header"));
}

#[test]
fn test_cosine_distance_exact() {
    let v1 = vec![1.0, 0.0];
    let v2 = vec![0.0, 1.0];

    let dist = distance(&v1, &v2, DistanceMetric::Cosine);
    assert!((dist - 1.0).abs() < 1e-6);
}
