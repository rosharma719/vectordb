/// Synthetic benchmark mirroring NYTimes 256-angular characteristics:
/// 290,000 256-dim cosine-metric vectors, m=16, m0=32, ef_construct=100.
/// Measures insert throughput and search QPS/recall at ef_search=[32,64,128,256,512].
/// Ground truth is computed by brute-force exact scan on 1,000 query vectors.
use std::time::Instant;

use vectordb::segment::segment::Segment;
use vectordb::utils::types::{DistanceMetric, Vector};
use vectordb::vector::hnsw::HNSWIndex;

const N_BASE: usize = 290_000;
const N_QUERIES: usize = 1_000;
const DIM: usize = 256;
const TOP_K: usize = 20;
const M: usize = 16;
const M0: usize = 32;
const EF_CONSTRUCT: usize = 100;

fn lcg(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
}

fn gen_unit_vector(seed: u64) -> Vector {
    let mut state = seed ^ 0xdeadbeefcafe1234;
    let mut v: Vec<f32> = (0..DIM).map(|_| lcg(&mut state)).collect();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        let inv = 1.0 / norm;
        for x in &mut v {
            *x *= inv;
        }
    }
    v
}

fn brute_force_top_k(queries: &[Vector], base: &[Vector], k: usize) -> Vec<Vec<usize>> {
    queries
        .iter()
        .map(|q| {
            let mut scored: Vec<(usize, f32)> = base
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let dot: f32 = q.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                    (i, 1.0 - dot)
                })
                .collect();
            scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            scored.into_iter().take(k).map(|(i, _)| i).collect()
        })
        .collect()
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx]
}

fn main() {
    println!("=== Synthetic NYT-style Benchmark ===");
    println!(
        "N={} dim={} M={} M0={} ef_construct={} top_k={}",
        N_BASE, DIM, M, M0, EF_CONSTRUCT, TOP_K
    );

    // Generate base vectors
    print!("Generating {} base vectors... ", N_BASE);
    let _ = std::io::Write::flush(&mut std::io::stdout());
    let base: Vec<Vector> = (0..N_BASE as u64).map(gen_unit_vector).collect();
    println!("done");

    // Generate query vectors
    let queries: Vec<Vector> = (N_BASE as u64..N_BASE as u64 + N_QUERIES as u64)
        .map(gen_unit_vector)
        .collect();

    // Build index
    println!("Building index...");
    let mut segment = Segment::new(HNSWIndex::new(
        DistanceMetric::Cosine,
        M,
        EF_CONSTRUCT,
        16,
        DIM,
    ));
    segment.hnsw_mut().set_m0(M0);
    segment.hnsw_mut().set_ef_construct(EF_CONSTRUCT);

    let insert_start = Instant::now();
    let log_every = 50_000;
    for (i, v) in base.iter().enumerate() {
        segment.insert_with_id(i as u64, v.clone(), None).unwrap();
        if (i + 1) % log_every == 0 {
            let elapsed = insert_start.elapsed().as_secs_f64();
            let rate = (i + 1) as f64 / elapsed;
            println!(
                "  inserted {} / {} ({:.0} vec/s, {:.3} ms/vec)",
                i + 1,
                N_BASE,
                rate,
                1000.0 / rate
            );
        }
    }
    let insert_dur = insert_start.elapsed();
    let ms_per_insert = insert_dur.as_secs_f64() * 1000.0 / N_BASE as f64;
    println!(
        "Insert: {:?} total, {:.3} ms/insert ({:.0} vec/s)",
        insert_dur,
        ms_per_insert,
        N_BASE as f64 / insert_dur.as_secs_f64()
    );

    // Compute ground truth
    print!("Computing ground truth (brute force)... ");
    let _ = std::io::Write::flush(&mut std::io::stdout());
    let gt_start = Instant::now();
    let ground_truth = brute_force_top_k(&queries, &base, TOP_K);
    println!("done ({:?})", gt_start.elapsed());

    // Search sweep
    println!("\nSearch sweep (ef_search -> QPS, latency, recall@{}):", TOP_K);
    let ef_list = [32usize, 64, 128, 256, 512];

    let mut summary: Vec<(usize, f64, f64, f64)> = Vec::new();
    for &ef in &ef_list {
        let ef_search = ef.max(TOP_K);
        segment.hnsw_mut().set_ef_search(ef_search);

        // Warmup
        for q in queries.iter().take(50) {
            let _ = segment.search(q, TOP_K).unwrap();
        }

        let mut latencies: Vec<f64> = Vec::with_capacity(N_QUERIES);
        let sweep_start = Instant::now();
        for q in queries.iter() {
            let t = Instant::now();
            let _ = segment.search(q, TOP_K).unwrap();
            latencies.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        let total = sweep_start.elapsed().as_secs_f64();
        let qps = N_QUERIES as f64 / total;
        let avg_ms = latencies.iter().sum::<f64>() / latencies.len() as f64;

        let mut sorted_lat = latencies.clone();
        sorted_lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = percentile(&sorted_lat, 50.0);
        let p99 = percentile(&sorted_lat, 99.0);

        // Compute recall
        let mut hits = 0usize;
        let mut total_targets = 0usize;
        for (q, truth) in queries.iter().zip(ground_truth.iter()) {
            let results = segment.search(q, TOP_K).unwrap();
            let truth_set: std::collections::HashSet<u64> =
                truth.iter().map(|&id| id as u64).collect();
            total_targets += truth.len();
            hits += results.iter().filter(|r| truth_set.contains(&r.id)).count();
        }
        let recall = hits as f64 / total_targets.max(1) as f64;

        println!(
            "  ef={:3}  qps={:7.1}  avg={:.3}ms  p50={:.3}ms  p99={:.3}ms  recall={:.4}",
            ef_search, qps, avg_ms, p50, p99, recall
        );
        summary.push((ef_search, qps, avg_ms, recall));
    }

    println!("\nSummary:");
    for (ef, qps, ms, recall) in &summary {
        println!(
            "  ef={:3}  {:.1} qps  {:.3} ms/q  recall={:.4}",
            ef, qps, ms, recall
        );
    }
}
