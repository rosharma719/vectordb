use std::collections::HashSet;
use std::env;

use rand::distr::{Distribution, Uniform};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use vectordb::segment::segment::Segment;
use vectordb::utils::types::{DistanceMetric, Vector};
use vectordb::vector::hnsw::HNSWIndex;
use vectordb::vector::metric::score;

mod common;
use common::generate_vector_dim;

fn parse_usize_list(env_key: &str, default: &[usize]) -> Vec<usize> {
    env::var(env_key)
        .ok()
        .map(|v| {
            v.split(',')
                .filter_map(|s| s.trim().replace('_', "").parse::<usize>().ok())
                .collect::<Vec<_>>()
        })
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| default.to_vec())
}

fn random_vector(rng: &mut SmallRng, dim: usize, noise: f32) -> Vector {
    // Fast uniform filler avoids trig and keeps RNG overhead low.
    let uniform = Uniform::new(-1.0f32, 1.0f32).unwrap();
    let mut base: Vec<f32> = uniform.sample_iter(&mut *rng).take(dim).collect();
    if noise == 0.0 {
        return base;
    }
    let noise_dist = Uniform::new(-noise, noise).unwrap();
    for v in base.iter_mut() {
        *v += rng.sample(noise_dist);
    }
    base
}

#[test]
#[ignore]
fn recall_unfiltered_euclidean() {
    let size = env::var("VECTORDB_RECALL_SIZE")
        .ok()
        .and_then(|v| v.replace('_', "").parse().ok())
        .unwrap_or(20_000);
    let dim = env::var("VECTORDB_RECALL_DIM")
        .ok()
        .and_then(|v| v.replace('_', "").parse().ok())
        .unwrap_or(1536);
    let ef_values = parse_usize_list("VECTORDB_RECALL_EF_SEARCH", &[32, 64, 128]);
    let top_k = env::var("VECTORDB_RECALL_TOPK")
        .ok()
        .and_then(|v| v.replace('_', "").parse().ok())
        .unwrap_or(20);
    let num_queries = env::var("VECTORDB_RECALL_QUERIES")
        .ok()
        .and_then(|v| v.replace('_', "").parse().ok())
        .unwrap_or(20);
    let seed = env::var("VECTORDB_RECALL_SEED")
        .ok()
        .and_then(|v| v.replace('_', "").parse().ok())
        .unwrap_or(42);
    let noise = env::var("VECTORDB_RECALL_NOISE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.002);
    let use_random = env::var("VECTORDB_RECALL_RANDOM")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(true);
    let metric = DistanceMetric::Euclidean;
    let m = 16;
    let ef_construct = 200;

    println!(
        "\n🧪 Building recall harness: size={}, dim={}, top_k={}, queries={}, ef_search={:?}, ef_construct={}",
        size, dim, top_k, num_queries, ef_values, ef_construct
    );

    // Build dataset and index once; keep stored vectors for anchored queries.
    let mut data_rng = SmallRng::seed_from_u64(seed);
    let mut dataset: Vec<(u64, Vector)> = Vec::with_capacity(size);
    let hnsw = HNSWIndex::new(
        metric,
        m,
        ef_values.iter().copied().max().unwrap_or(32).max(top_k),
        16,
        dim,
    );
    let mut segment = Segment::new(hnsw);
    segment.hnsw_mut().set_ef_construct(ef_construct);
    for i in 0..size {
        let v = if use_random {
            random_vector(&mut data_rng, dim, 0.0)
        } else {
            generate_vector_dim(i, dim)
        };
        let id = segment.insert(v.clone(), None).unwrap();
        dataset.push((id, v));
    }

    // Anchored queries: sample stored vectors and add optional noise.
    let mut query_rng = SmallRng::seed_from_u64(seed ^ 0x9E3779B97F4A7C15);
    let mut queries: Vec<Vector> = Vec::with_capacity(num_queries);
    while queries.len() < num_queries {
        let idx = query_rng.random_range(0..size);
        let mut q = dataset[idx].1.clone();
        if noise > 0.0 {
            for v in q.iter_mut() {
                *v += query_rng.random_range(-noise..noise);
            }
        }
        queries.push(q);
    }

    let mut summary: Vec<(usize, f64)> = Vec::new();

    for &ef_requested in &ef_values {
        let ef_search = ef_requested.max(top_k);
        segment.hnsw_mut().set_ef_search(ef_search);
        println!("ℹ️  Running sweep entry with ef_search={} (ef_construct={})", ef_search, ef_construct);

        let mut avg_recall = 0.0;
        for (qi, q) in queries.iter().enumerate() {
            // ground truth via brute force using the exact metric implementation
            let mut brute: Vec<(u64, f32)> = dataset
                .iter()
                .map(|(id, v)| (*id, score(q, v, metric)))
                .collect();
            brute.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            brute.truncate(top_k);
            let truth: HashSet<u64> = brute.iter().map(|(id, _)| *id).collect();

            let approx = segment.search(q, top_k).unwrap();
            let hits = approx.iter().filter(|r| truth.contains(&r.id)).count() as f64;
            let recall = hits / top_k as f64;
            avg_recall += recall;
            println!(
                "[ef_search={}] Query {} recall: {:.3} (hits {}/{})",
                ef_search,
                qi,
                recall,
                hits as usize,
                top_k
            );
        }
        avg_recall /= num_queries as f64;
        println!(
            "✅ [ef_search={}] Average recall over {} queries: {:.3}",
            ef_search, num_queries, avg_recall
        );
        summary.push((ef_search, avg_recall));
    }

    println!("\nSummary (ef_search -> avg recall):");
    for (ef, r) in summary {
        println!("  {} -> {:.3}", ef, r);
    }
}
