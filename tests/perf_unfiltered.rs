use std::env;
use std::time::Instant;

mod common;
use common::generate_vector_dim;
use vectordb::segment::segment::Segment;
use vectordb::utils::types::DistanceMetric;
use vectordb::vector::hnsw::HNSWIndex;

/// Heavier unfiltered search latency check to approximate production scale.
#[test]
#[ignore]
fn bench_unfiltered_large_scale() {
    let size = 20_000;
    let metric = DistanceMetric::Euclidean;
    let dim = 1536;
    let hnsw = HNSWIndex::new(metric, 16, 64, 16, dim);
    let mut segment = Segment::new(hnsw);
    segment.hnsw_mut().set_ef_construct(100);

    println!("\n🚀 Inserting {} vectors for large-scale unfiltered benchmark", size);
    let start_insert = Instant::now();
    for i in 0..size {
        segment.insert(generate_vector_dim(i, dim), None).unwrap();
        if i != 0 && i % 1000 == 0 {
            println!("Inserted {} vectors... (+{:?})", i, start_insert.elapsed());
        }
    }
    let insert_dur = start_insert.elapsed();
    println!("✅ Inserted {} vectors in {:?}", size, insert_dur);

    let queries: Vec<_> = (0..5).map(|i| generate_vector_dim(i + size, dim)).collect();
    let start_search = Instant::now();
    for q in &queries {
        let res = segment.search(q, 20).unwrap();
        assert!(!res.is_empty(), "Search returned empty result");
    }
    let search_dur = start_search.elapsed();
    let avg_ms = search_dur.as_secs_f64() * 1000.0 / queries.len() as f64;
    println!(
        "🔍 Ran {} unfiltered searches in {:?} (avg {:.3} ms/search)",
        queries.len(),
        search_dur,
        avg_ms
    );
}

/// Parametrizable unfiltered benchmark; size/dim can be set via env vars.
#[test]
#[ignore]
fn bench_unfiltered_param_scale() {
    let size = env::var("VECTORDB_BENCH_SIZE")
        .ok()
        .and_then(|v| v.replace('_', "").parse().ok())
        .unwrap_or(100_000);
    let dim = env::var("VECTORDB_BENCH_DIM")
        .ok()
        .and_then(|v| v.replace('_', "").parse().ok())
        .unwrap_or(1536);
    let metric = DistanceMetric::Euclidean;
    let hnsw = HNSWIndex::new(metric, 16, 64, 16, dim);
    let mut segment = Segment::new(hnsw);
    segment.hnsw_mut().set_ef_construct(100);

    println!(
        "\n🚀 Inserting {} vectors (dim = {}) for parametrized unfiltered benchmark",
        size, dim
    );
    let start_insert = Instant::now();
    for i in 0..size {
        segment.insert(generate_vector_dim(i, dim), None).unwrap();
        if i != 0 && i % 1000 == 0 {
            println!("Inserted {} vectors... (+{:?})", i, start_insert.elapsed());
        }
    }
    let insert_dur = start_insert.elapsed();
    println!("✅ Inserted {} vectors in {:?}", size, insert_dur);

    let queries: Vec<_> = (0..5).map(|i| generate_vector_dim(i + size, dim)).collect();
    let start_search = Instant::now();
    for q in &queries {
        let res = segment.search(q, 20).unwrap();
        assert!(!res.is_empty(), "Search returned empty result");
    }
    let search_dur = start_search.elapsed();
    let avg_ms = search_dur.as_secs_f64() * 1000.0 / queries.len() as f64;
    println!(
        "🔍 Ran {} unfiltered searches in {:?} (avg {:.3} ms/search)",
        queries.len(),
        search_dur,
        avg_ms
    );
}

/// Very large-scale unfiltered benchmark (opt-in; runs longer).
#[test]
#[ignore]
fn bench_unfiltered_million_scale() {
    let size = 1_000_000;
    let metric = DistanceMetric::Euclidean;
    let dim = 1536;
    let hnsw = HNSWIndex::new(metric, 16, 64, 16, dim);
    let mut segment = Segment::new(hnsw);
    segment.hnsw_mut().set_ef_construct(100);

    println!("\n🚀 Inserting {} vectors for million-scale unfiltered benchmark", size);
    let start_insert = Instant::now();
    for i in 0..size {
        segment.insert(generate_vector_dim(i, dim), None).unwrap();
        if i != 0 && i % 1000 == 0 {
            println!("Inserted {} vectors... (+{:?})", i, start_insert.elapsed());
        }
    }
    let insert_dur = start_insert.elapsed();
    println!("✅ Inserted {} vectors in {:?}", size, insert_dur);

    let queries: Vec<_> = (0..5).map(|i| generate_vector_dim(i + size, dim)).collect();
    let start_search = Instant::now();
    for q in &queries {
        let res = segment.search(q, 20).unwrap();
        assert!(!res.is_empty(), "Search returned empty result");
    }
    let search_dur = start_search.elapsed();
    let avg_ms = search_dur.as_secs_f64() * 1000.0 / queries.len() as f64;
    println!(
        "🔍 Ran {} unfiltered searches in {:?} (avg {:.3} ms/search)",
        queries.len(),
        search_dur,
        avg_ms
    );
}
