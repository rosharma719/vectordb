use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::Path;
use std::time::Instant;

use ndarray::Array2;
use ndarray_npy::read_npy;
use serde_json::from_slice;

use vectordb::segment::segment::Segment;
use vectordb::utils::types::{DistanceMetric, Vector};
use vectordb::vector::hnsw::HNSWIndex;

fn load_vectors(path: &Path) -> Vec<Vector> {
    let arr: Array2<f32> = read_npy(path).expect("failed to read .npy");
    arr.rows().into_iter().map(|r| r.to_vec()).collect()
}

fn load_ground_truth(path: &Path) -> Vec<Vec<usize>> {
    let data = fs::read(path).expect("failed to read ground truth json");
    from_slice(&data).expect("failed to parse ground truth json")
}

fn parse_usize_list(env_key: &str) -> Option<Vec<usize>> {
    env::var(env_key)
        .ok()
        .map(|v| {
            v.split(',')
                .filter_map(|s| s.trim().replace('_', "").parse::<usize>().ok())
                .collect::<Vec<_>>()
        })
        .filter(|v| !v.is_empty())
}

fn ensure_exists(path: &Path) {
    assert!(
        path.exists(),
        "missing required dataset file: {} (run the download script in README)",
        path.display()
    );
}

/// Uses the ANN-Benchmarks NYTimes 256-d Angular dataset from Hugging Face.
/// Requires local files:
///   base.npy, queries.npy, ground_truth.json in data/nytimes-256-angular (by default).
#[test]
#[ignore]
fn nytimes_256_angular_perf_and_recall() {
    let t0 = Instant::now();
    let data_dir = env::var("VECTORDB_NYT_DATA_DIR")
        .unwrap_or_else(|_| "data/nytimes-256-angular".to_string());
    let top_k = env::var("VECTORDB_NYT_TOPK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20usize);
    let ef_values = parse_usize_list("VECTORDB_NYT_EF_SEARCH_LIST").unwrap_or_else(|| {
        env::var("VECTORDB_NYT_EF_SEARCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .map(|v| vec![v])
            .unwrap_or_else(|| vec![128usize])
    });
    let queries_cap = env::var("VECTORDB_NYT_QUERIES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1000usize);
    let base_cap = env::var("VECTORDB_NYT_BASE_LIMIT")
        .ok()
        .and_then(|v| v.parse().ok());
    let ef_construct = env::var("VECTORDB_NYT_EF_CONSTRUCT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100usize);

    let base_path = Path::new(&data_dir).join("base.npy");
    let queries_path = Path::new(&data_dir).join("queries.npy");
    let truth_path = Path::new(&data_dir).join("ground_truth.json");
    ensure_exists(&base_path);
    ensure_exists(&queries_path);
    ensure_exists(&truth_path);

    println!(
        "\n📚 Loading NYTimes dataset from {} (top_k={}, ef_search_list={:?}, max_queries={}, base_cap={:?}, ef_construct={})",
        data_dir, top_k, ef_values, queries_cap, base_cap, ef_construct
    );
    let mut base = load_vectors(&base_path);
    if let Some(cap) = base_cap {
        if cap < base.len() {
            base.truncate(cap);
        }
    }
    let queries = load_vectors(&queries_path);
    let ground_truth = load_ground_truth(&truth_path);
    assert_eq!(ground_truth.len(), queries.len(), "ground truth length must match queries");
    println!("⏱️  Data loaded in {:?}", t0.elapsed());

    let dim = base.first().map(|v| v.len()).unwrap_or(0);
    assert_eq!(dim, 256, "expected 256-d vectors");

    let metric = DistanceMetric::Cosine; // angular dataset
    let m = 16;
    let mut segment = Segment::new(HNSWIndex::new(
        metric,
        m,
        ef_values.iter().copied().max().unwrap_or(1).max(top_k),
        16,
        dim,
    ));
    segment.hnsw_mut().set_ef_construct(ef_construct);

    println!("🚀 Inserting {} vectors with dataset-aligned IDs...", base.len());
    let start_insert = Instant::now();
    for (i, v) in base.iter().enumerate() {
        let dataset_id = i as u64; // align with ground-truth neighbor IDs (0-based)
        segment.insert_with_id(dataset_id, v.clone(), None).unwrap();
        if i != 0 && i % 10_000 == 0 {
            println!("Inserted {} vectors", i);
        }
    }
    let insert_dur = start_insert.elapsed();
    let insert_ms = insert_dur.as_secs_f64() * 1000.0 / base.len() as f64;
    println!(
        "✅ Inserted {} vectors in {:?} (~{:.3} ms/insert)",
        base.len(),
        insert_dur,
        insert_ms
    );

    let num_queries = queries.len().min(queries_cap);
    println!(
        "🔍 Sweeping ef_search over {:?} for {} queries (top_k={})...",
        ef_values, num_queries, top_k
    );

    let mut summary: Vec<(usize, f64, f64)> = Vec::new();
    for &ef in &ef_values {
        let ef_search = ef.max(top_k);
        segment.hnsw_mut().set_ef_search(ef_search);
        let mut hits = 0usize;
        let mut total_targets = 0usize;
        let start_search = Instant::now();
        for (qi, q) in queries.iter().take(num_queries).enumerate() {
            let approx = segment.search(q, top_k).unwrap();
            let truth = &ground_truth[qi];
            let truth_k = truth.len().min(top_k);
            let truth_set: HashSet<_> = truth
                .iter()
                .take(truth_k)
                // Ground truth IDs are 0-based row indices; we insert with the same IDs.
                .filter(|&&id| (id as usize) < base.len())
                .map(|&id| id as u64)
                .collect();
            total_targets += truth_set.len();
            hits += approx.iter().filter(|r| truth_set.contains(&r.id)).count();
            if (qi + 1) % 100 == 0 || qi + 1 == num_queries {
                let partial_recall = hits as f64 / total_targets.max(1) as f64;
                println!(
                    "  progress: query {}/{} (ef_search={}) cumulative recall={:.3}",
                    qi + 1,
                    num_queries,
                    ef_search,
                    partial_recall
                );
            }
        }
        let search_dur = start_search.elapsed();
        let avg_ms = search_dur.as_secs_f64() * 1000.0 / num_queries as f64;
        let recall = hits as f64 / total_targets.max(1) as f64;
        println!(
            "🎯 [ef_search={}] recall@{}: {:.3} over {} queries (hits {}/{}) | avg {:.3} ms/query",
            ef_search,
            top_k,
            recall,
            num_queries,
            hits,
            total_targets,
            avg_ms
        );
        summary.push((ef_search, recall, avg_ms));
    }

    println!("\nSummary (ef_search -> recall, ms/query):");
    for (ef, recall, ms) in summary {
        println!("  {} -> {:.3}, {:.3} ms/query", ef, recall, ms);
    }
}
