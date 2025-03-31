use vectordb::segment::segment::Segment;
use vectordb::vector::hnsw::HNSWIndex;
use vectordb::utils::types::{DistanceMetric, Vector};
use vectordb::utils::payload::{Payload, PayloadValue};
use vectordb::payload_storage::filters::Filter;

fn vecf(v: &[f32]) -> Vector {
    v.to_vec()
}

#[test]
fn test_segment_insert_and_search() {
    let hnsw = HNSWIndex::new(DistanceMetric::Euclidean, 16, 50, 16, 2);
    let mut segment = Segment::new(hnsw);

    let payload = {
        let mut p = Payload::default();
        p.set("category", PayloadValue::Str("dog".into()));
        p
    };

    let id = segment.insert(vecf(&[1.0, 2.0]), Some(payload)).unwrap();
    let results = segment.search(&vecf(&[1.0, 2.0]), 1).unwrap();
    assert_eq!(results[0].id, id);
}

#[test]
fn test_segment_logical_delete() {
    let hnsw = HNSWIndex::new(DistanceMetric::Dot, 16, 50, 16, 2);
    let mut segment = Segment::new(hnsw);

    let id = segment.insert(vecf(&[1.0, 0.0]), None).unwrap();
    segment.delete(id).unwrap();

    let result = segment.search(&vecf(&[1.0, 0.0]), 1);
    assert!(matches!(result, Err(vectordb::utils::errors::DBError::SearchError(_))));
}

#[test]
fn test_segment_search_after_partial_deletion() {
    let hnsw = HNSWIndex::new(DistanceMetric::Dot, 16, 50, 16, 2);
    let mut segment = Segment::new(hnsw);

    let id1 = segment.insert(vecf(&[1.0, 0.0]), None).unwrap();
    let id2 = segment.insert(vecf(&[0.0, 1.0]), None).unwrap();
    segment.delete(id1).unwrap();

    let results = segment.search(&vecf(&[0.0, 1.0]), 1).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, id2);
}

#[test]
fn test_segment_payload_metadata() {
    let hnsw = HNSWIndex::new(DistanceMetric::Cosine, 16, 50, 16, 2);
    let mut segment = Segment::new(hnsw);

    let mut payload = Payload::default();
    payload.set("label", PayloadValue::Str("dog".to_string()));
    let id = segment.insert(vecf(&[0.5, 0.5]), Some(payload.clone())).unwrap();

    let retrieved = segment.get_payload(id).unwrap();
    assert_eq!(retrieved, &payload);
}

#[test]
fn test_segment_id_auto_increment() {
    let hnsw = HNSWIndex::new(DistanceMetric::Euclidean, 16, 50, 16, 2);
    let mut segment = Segment::new(hnsw);

    let id1 = segment.insert(vecf(&[1.0, 0.0]), None).unwrap();
    let id2 = segment.insert(vecf(&[0.0, 1.0]), None).unwrap();
    assert_eq!(id2, id1 + 1);
}

#[test]
fn test_segment_unfiltered_search() {
    let hnsw = HNSWIndex::new(DistanceMetric::Euclidean, 16, 50, 16, 2);
    let mut segment = Segment::new(hnsw);

    let id = segment.insert(vecf(&[3.0, 3.0]), None).unwrap();
    segment.delete(id).unwrap();

    let results = segment.search_unfiltered(&vecf(&[3.0, 3.0]), 1).unwrap();
    assert!(
        results.is_empty() || results[0].id == id,
        "Expected either empty or the deleted point in results"
    );
}

#[test]
fn test_segment_purge_removes_deleted() {
    let hnsw = HNSWIndex::new(DistanceMetric::Euclidean, 16, 50, 16, 2);
    let mut segment = Segment::new(hnsw);

    let id1 = segment.insert(vecf(&[1.0, 1.0]), None).unwrap();
    let id2 = segment.insert(vecf(&[2.0, 2.0]), None).unwrap();
    segment.delete(id1).unwrap();
    segment.purge().unwrap();

    let results = segment.search_unfiltered(&vecf(&[1.0, 1.0]), 5).unwrap();
    assert!(!results.iter().any(|r| r.id == id1));
    assert!(results.iter().any(|r| r.id == id2));
}

#[test]
fn test_segment_post_filter_large_vector_set() {
    let hnsw = HNSWIndex::new(DistanceMetric::Dot, 16, 50, 16, 2);
    let mut segment = Segment::new(hnsw);

    for i in 0..100 {
        let mut payload = Payload::default();
        let label = if i % 2 == 0 { "even" } else { "odd" };
        payload.set("parity", PayloadValue::Str(label.to_string()));
        segment.insert(vecf(&[i as f32, 0.0]), Some(payload)).unwrap();
    }

    let filter = Filter::Match {
        key: "parity".into(),
        value: PayloadValue::Str("even".into()),
    };

    let results = segment.post_filter(&vecf(&[0.0, 0.0]), 10, Some(&filter)).unwrap();
    assert!(results.iter().all(|sp| {
        let payload = segment.get_payload(sp.id).unwrap();
        matches!(payload.get("parity"), Some(PayloadValue::Str(s)) if s == "even")
    }));
    assert!(results.len() <= 10);
}
