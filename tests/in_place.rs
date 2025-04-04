use std::collections::HashSet;
use std::time::Instant;

use vectordb::segment::segment::Segment;
use vectordb::utils::types::{DistanceMetric, Vector};
use vectordb::utils::payload::{Payload, PayloadValue, ScalarComparisonOp};
use vectordb::vector::hnsw::HNSWIndex;
use vectordb::payload_storage::filters::Filter;

fn make_payload(group: &str, score: i64) -> Payload {
    let mut payload = Payload::default();
    payload.set("group", PayloadValue::Str(group.to_string()));
    payload.set("score", PayloadValue::Int(score));
    payload
}

fn make_random_vec(seed: u64, dim: usize) -> Vector {
    (0..dim).map(|i| ((seed + i as u64 * 7919) % 97) as f32 / 10.0).collect()
}

fn generate_segment(metric: DistanceMetric, num: usize, dim: usize) -> (Segment, Vec<Vector>) {
    let hnsw = HNSWIndex::new(metric, 16, 128, 16, dim);
    let mut segment = Segment::new(hnsw);
    let mut inserted_vecs = Vec::new();

    let start = Instant::now();
    for i in 0..num {
        let vec = make_random_vec(i as u64, dim);
        let group = if i % 2 == 0 { "even" } else { "odd" };
        let payload = make_payload(group, i as i64);
        segment.insert(vec.clone(), Some(payload)).unwrap();
        inserted_vecs.push(vec);
    }
    let elapsed = start.elapsed().as_millis();
    println!(
        "[BENCH] Inserted {num} vectors in {elapsed}ms ({:.3}ms/insert)",
        elapsed as f64 / num as f64
    );
    (segment, inserted_vecs)
}

#[test]
fn benchmark_segment_ops_large() {
    const NUM_POINTS: usize = 5000;
    const DIM: usize = 32;
    const TOP_K: usize = 10;

    for &metric in &[DistanceMetric::Euclidean, DistanceMetric::Cosine, DistanceMetric::Dot] {
        println!("\n--- Benchmarking segment with {:?} ---", metric);
        let (segment, inserted_vecs) = generate_segment(metric, NUM_POINTS, DIM);
        let query = inserted_vecs[123].clone(); // Arbitrary known query

        // Complex AND filter: group = "even" AND score >= 3000
        let filter = Filter::And(vec![
            Filter::Match {
                key: "group".into(),
                value: PayloadValue::Str("even".into()),
            },
            Filter::Compare {
                key: "score".into(),
                op: ScalarComparisonOp::Gte,
                value: PayloadValue::Int(3000),
            },
        ]);

        // === Unfiltered Search ===
        let t0 = Instant::now();
        let res_unfiltered = segment.search(&query, TOP_K).unwrap();
        let t1 = Instant::now();
        println!(
            "[{:?}] search (no filter): {} results in {:?}ms",
            metric,
            res_unfiltered.len(),
            (t1 - t0).as_millis()
        );

        // === In-place Filtered Search ===
        let t2 = Instant::now();
        let res_filtered = segment.search_with_filter(&query, TOP_K, Some(&filter)).unwrap();
        let t3 = Instant::now();
        println!(
            "[{:?}] search_with_filter: {} results in {:?}ms",
            metric,
            res_filtered.len(),
            (t3 - t2).as_millis()
        );

        // === Brute-force Post Filtered ===
        let t4 = Instant::now();
        let res_post = segment.post_filter(&query, TOP_K, Some(&filter)).unwrap();
        let t5 = Instant::now();
        println!(
            "[{:?}] post_filter: {} results in {:?}ms",
            metric,
            res_post.len(),
            (t5 - t4).as_millis()
        );

        // === Validate Filtered Results Match Filter ===
        for r in &res_filtered {
            let p = segment.get_payload(r.id).unwrap();
            assert!(matches!(p.get("group"), Some(PayloadValue::Str(s)) if s == "even"));
            assert!(matches!(p.get("score"), Some(PayloadValue::Int(s)) if *s >= 3000));
        }
        for r in &res_post {
            let p = segment.get_payload(r.id).unwrap();
            assert!(matches!(p.get("group"), Some(PayloadValue::Str(s)) if s == "even"));
            assert!(matches!(p.get("score"), Some(PayloadValue::Int(s)) if *s >= 3000));
        }

        // === Recall Check ===
        let ids_filtered: HashSet<_> = res_filtered.iter().map(|r| r.id).collect();
        let ids_post: HashSet<_> = res_post.iter().map(|r| r.id).collect();

        let missed: Vec<_> = ids_post.difference(&ids_filtered).collect();
        let recall = if res_post.is_empty() {
            1.0
        } else {
            (TOP_K - missed.len()) as f64 / TOP_K as f64
        };

        println!(
            "[{:?}] Recall: {:.2}% ({} correct / {} expected)",
            metric,
            100.0 * recall,
            TOP_K - missed.len(),
            TOP_K
        );

        assert!(
            missed.is_empty(),
            "[{:?}] Recall miss: filtered search missed {:?}",
            metric,
            missed
        );

        println!("[{:?}] ✅ All assertions passed for TOP_K = {}", metric, TOP_K);
    }
}
