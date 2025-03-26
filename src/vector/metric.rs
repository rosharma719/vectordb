use crate::utils::types::{Vector, DistanceMetric};

/// Main distance dispatcher
pub fn distance(a: &Vector, b: &Vector, metric: DistanceMetric) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must be the same length");

    match metric {
        DistanceMetric::Cosine => cosine_distance(a, b),
        DistanceMetric::Dot => dot_product_distance(a, b),
        DistanceMetric::Euclidean => euclidean_distance(a, b),
    }
}

/// Cosine distance: 1 - cosine similarity
fn cosine_distance(a: &Vector, b: &Vector) -> f32 {
    let dot = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    1.0 - (dot / (norm_a * norm_b + 1e-10))  // + epsilon to avoid NaNs
}

/// Dot product similarity (inverted to behave like a distance)
fn dot_product_distance(a: &Vector, b: &Vector) -> f32 {
    -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
}

/// Euclidean distance
fn euclidean_distance(a: &Vector, b: &Vector) -> f32 {
    a.iter().zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}
