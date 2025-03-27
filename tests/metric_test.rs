use vectordb::vector::metric::distance;
use vectordb::utils::types::DistanceMetric;

#[test]
#[should_panic(expected = "Vectors must be the same length")]
fn test_distance_vector_length_mismatch_panics() {
    let v1 = vec![1.0, 2.0];
    let v2 = vec![1.0]; // mismatched length
    let _ = distance(&v1, &v2, DistanceMetric::Dot);
}
