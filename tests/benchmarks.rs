use std::time::Instant;
use std::convert::TryInto;

use vectordb::segment::segment::Segment;
use vectordb::utils::types::{DistanceMetric, Vector};
use vectordb::utils::payload::{Payload, PayloadValue, ScalarComparisonOp};
use vectordb::vector::hnsw::HNSWIndex;
use vectordb::payload_storage::filters::Filter;

fn vecf(v: &[f32]) -> Vector {
    v.to_vec()
}

fn generate_vector(i: usize) -> Vector {
    vecf(&[
        (i as f32).sin() * 5.0,
        ((i * 3) as f32).cos() * 3.0,
        ((i % 7) as f32).sqrt(),
    ])
}

fn generate_payload(i: usize) -> Payload {
    let mut payload = Payload::default();
    payload.set("index", PayloadValue::Int(i.try_into().unwrap()));
    payload.set("animal", PayloadValue::Str(match i % 4 {
        0 => "dog",
        1 => "cat",
        2 => "bird",
        _ => "fish",
    }.to_string()));
    payload.set("age", PayloadValue::Int((i % 8 + 1).try_into().unwrap()));
    payload.set("score", PayloadValue::Float((60.0 + (i % 40) as f64).into()));
    payload.set("tags", PayloadValue::ListStr(if i % 2 == 0 {
        vec!["cheap".to_string(), "small".to_string()]
    } else {
        vec!["expensive".to_string(), "large".to_string()]
    }));
    payload.set("active", PayloadValue::Bool(i % 3 == 0));
    payload
}

// === Insertion Benchmark ===
pub fn bench_insertion(metric: DistanceMetric, size: usize) -> Segment {
    println!("\n🛠️ Inserting {} points with {:?} metric", size, metric);
    let hnsw = HNSWIndex::new(metric, 16, 50, 16, 3);
    let mut segment = Segment::new(hnsw);

    let start = Instant::now();
    for i in 0..size {
        let vec = generate_vector(i);
        let payload = generate_payload(i);
        segment.insert(vec, Some(payload)).unwrap();
    }
    println!("✅ Insertion took {:?}", start.elapsed());
    segment
}

// === Search Benchmark ===
pub fn bench_search(segment: &Segment, query: &Vector) {
    println!("\n🔍 Basic search...");
    let start = Instant::now();
    let _ = segment.search(query, 10).unwrap();
    println!("Basic search took {:?}", start.elapsed());

    let filter = Filter::And(vec![
        Filter::Match {
            key: "animal".into(),
            value: PayloadValue::Str("dog".into()),
        },
        Filter::Compare {
            key: "age".into(),
            op: ScalarComparisonOp::Gte,
            value: PayloadValue::Int(6),
        },
        Filter::Compare {
            key: "score".into(),
            op: ScalarComparisonOp::Lt,
            value: PayloadValue::Float(90.0.into()),
        },
    ]);

    println!("🧃 Filtered search...");
    let start = Instant::now();
    let _ = segment.post_filter(query, 10, Some(&filter)).unwrap();
    println!("Filtered search took {:?}", start.elapsed());

    let tag_filter = Filter::Compare {
        key: "tags".into(),
        op: ScalarComparisonOp::Eq,
        value: PayloadValue::Str("cheap".into()),
    };

    println!("📦 List match search...");
    let start = Instant::now();
    let _ = segment.post_filter(query, 10, Some(&tag_filter)).unwrap();
    println!("List filter took {:?}", start.elapsed());
}

// === Deletion Benchmark ===
pub fn bench_deletion(segment: &mut Segment, size: usize) {
    println!("\n❌ Deleting every 7th point...");
    let start = Instant::now();
    for i in (0..size).step_by(7) {
        let _ = segment.delete((i + 1) as u64); // IDs start at 1
    }
    println!("Sparse deletion took {:?}", start.elapsed());

    println!("🧹 Full deletion...");
    let start = Instant::now();
    for i in 0..size {
        let _ = segment.delete((i + 1) as u64);
    }
    println!("Full deletion (purge) took {:?}", start.elapsed());
}


#[test]
fn bench_all_euclidean_100() {
    let size = 100;
    let metric = DistanceMetric::Euclidean;
    let mut segment = bench_insertion(metric, size);
    let query = vecf(&[1.0, 0.0, 0.0]);
    bench_search(&segment, &query);
    bench_deletion(&mut segment, size);
}

#[test]
fn bench_all_cosine_1000() {
    let size = 1000;
    let metric = DistanceMetric::Cosine;
    let mut segment = bench_insertion(metric, size);
    let query = vecf(&[1.0, 0.0, 0.0]);
    bench_search(&segment, &query);
    bench_deletion(&mut segment, size);
}

#[test]
fn bench_all_dot_500() {
    let size = 500;
    let metric = DistanceMetric::Dot;
    let mut segment = bench_insertion(metric, size);
    let query = vecf(&[1.0, 0.0, 0.0]);
    bench_search(&segment, &query);
    bench_deletion(&mut segment, size);
}
