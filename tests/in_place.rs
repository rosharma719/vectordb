use std::collections::HashSet;
use std::time::Instant;

use vectordb::segment::segment::Segment;
use vectordb::utils::types::{DistanceMetric, Vector};
use vectordb::utils::payload::{Payload, PayloadValue, ScalarComparisonOp};
use vectordb::payload_storage::filters::evaluate_filter;
use vectordb::vector::hnsw::{HNSWIndex, ScoredPoint};
use vectordb::payload_storage::filters::Filter;
use vectordb::vector::metric::score;

fn make_payload(group: &str, score: i64) -> Payload {
    let mut payload = Payload::default();
    payload.set("group", PayloadValue::Str(group.to_string()));
    payload.set("score", PayloadValue::Int(score));
    payload
}

fn make_random_vec(seed: u64, dim: usize) -> Vector {
    (0..dim)
        .map(|i| ((seed + i as u64 * 7919) % 97) as f32 / 10.0)
        .collect()
}

fn generate_segment(
    metric: DistanceMetric,
    num: usize,
    dim: usize,
) -> (Segment, Vec<Vector>) {
    let hnsw = HNSWIndex::new(metric, 16, 32, 16, dim);
    let mut segment = Segment::new(hnsw);
    let mut inserted_vecs = Vec::with_capacity(num);

    let start = Instant::now();
    for i in 0..num {
        let vec = make_random_vec(i as u64, dim);
        let group = if i % 2 == 0 { "even" } else { "odd" };
        let payload = make_payload(group, i as i64);
        segment.insert(vec.clone(), Some(payload)).unwrap();
        inserted_vecs.push(vec);
    }
    let took = start.elapsed();
    println!(
        "[BENCH][{:?}] Inserted {} vectors in {:?} ({:.3}ms/insert)",
        metric,
        num,
        took,
        took.as_secs_f64() * 1e3 / num as f64,
    );
    (segment, inserted_vecs)
}

#[test]
fn benchmark_segment_ops_large() {
const NUM_POINTS: usize = 1_500;
const DIM: usize        = 1536;
const TOP_K: usize      = 10;

    // collect *all* failures here
    let mut failures: Vec<String> = Vec::new();

    for &metric in &[
        DistanceMetric::Euclidean,
        DistanceMetric::Cosine,
        DistanceMetric::Dot,
    ] {
        println!("\n=== Testing metric {:?} ===", metric);
        let (segment, inserted) = generate_segment(metric, NUM_POINTS, DIM);
        let query = &inserted[123];

        // complex filter (threshold scales down with dataset size to keep matches)
        let score_threshold = (NUM_POINTS as i64 * 2 / 3) as i64;
        let filter = Filter::And(vec![
            Filter::Match {
                key: "group".into(),
                value: PayloadValue::Str("even".into()),
            },
            Filter::Compare {
                key: "score".into(),
                op: ScalarComparisonOp::Gte,
                value: PayloadValue::Int(score_threshold),
            },
        ]);

        // 0) brute-force ground truth
        let mut ground: Vec<ScoredPoint> = segment
            .hnsw()
            .iter_vectors()
            .filter_map(|(&id, vec)| {
                if segment.is_deleted(id) { return None; }
                let p = segment.get_payload(id).unwrap();
                if !evaluate_filter(&filter, p).unwrap() { return None; }
                let raw = score(query, vec, metric);
                let sort_key = match metric {
                    DistanceMetric::Dot => -raw,
                    _ => raw,
                };
                Some(ScoredPoint { id, raw_score: raw, sort_key })
            })
            .collect();
        ground.sort_by(|a, b| a.sort_key.partial_cmp(&b.sort_key).unwrap());
        ground.truncate(TOP_K);
        let truth_ids: HashSet<_> = ground.iter().map(|r| r.id).collect();

        // 1) unfiltered
        let t1 = Instant::now();
        let res_unf = segment.search(query, TOP_K).unwrap();
        let dt_unf = t1.elapsed();
        println!("[{:?}] unfiltered={}  took {:?}", metric, res_unf.len(), dt_unf);

        // 2) in-place
        let t2 = Instant::now();
        let res_inp = segment.search_with_filter(query, TOP_K, Some(&filter)).unwrap();
        let dt_inp = t2.elapsed();
        println!("[{:?}] in-place-filter={}  took {:?}", metric, res_inp.len(), dt_inp);

        // 3) filtered (formerly post-filter)
        let t3 = Instant::now();
        let res_post = segment.search_with_filter(query, TOP_K, Some(&filter)).unwrap();
        let dt_post = t3.elapsed();
        println!("[{:?}] filtered={}  took {:?}", metric, res_post.len(), dt_post);
        println!(
            "[{:?}] search timings -> unfiltered: {:.3} ms, in-place: {:.3} ms, filtered: {:.3} ms",
            metric,
            dt_unf.as_secs_f64() * 1e3,
            dt_inp.as_secs_f64() * 1e3,
            dt_post.as_secs_f64() * 1e3
        );

        // 4) sanity: every returned item matches
        for (name, set) in &[
            ("ground",   &ground),
            ("in-place", &res_inp),
            ("filtered", &res_post),
        ] {
            for r in *set {
                let p = segment.get_payload(r.id).unwrap();
                if !matches!(p.get("group"), Some(PayloadValue::Str(s)) if s == "even")
                    || !matches!(p.get("score"), Some(PayloadValue::Int(s)) if *s >= score_threshold)
                {
                    failures.push(format!(
                        "[{:?}][{}] returned id={} that does not match filter",
                        metric, name, r.id
                    ));
                }
            }
        }

        // 5) compare to truth
        let in_ids:  HashSet<_> = res_inp.iter().map(|r| r.id).collect();
        let missed_in: Vec<_> = truth_ids.difference(&in_ids).cloned().collect();
        if !missed_in.is_empty() {
            failures.push(format!(
                "[{:?}][in-place] missed ground-truth ids {:?}",
                metric, missed_in
            ));
        }

        let post_ids: HashSet<_> = res_post.iter().map(|r| r.id).collect();
        let missed_post: Vec<_> = truth_ids.difference(&post_ids).cloned().collect();
        if !missed_post.is_empty() {
            failures.push(format!(
                "[{:?}][filtered] missed ground-truth ids {:?}",
                metric, missed_post
            ));
        }

        println!("[{:?}] ✅ done.", metric);
    }

    // at the end, fail if any
    if !failures.is_empty() {
        panic!(
            "Some filter tests failed:\n{}",
            failures.join("\n")
        );
    }
}
