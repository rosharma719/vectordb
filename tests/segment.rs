use vectordb::segment::segment::Segment;
use vectordb::vector::hnsw::HNSWIndex;
use vectordb::utils::types::{DistanceMetric, Vector};
use vectordb::utils::payload::{Payload, PayloadValue};
use vectordb::payload_storage::filters::Filter;
use vectordb::utils::errors::DBError;

fn vecf(v: &[f32]) -> Vector {
    v.to_vec()
}

#[test]
fn test_segment_basic_insert_and_search() {
    let hnsw = HNSWIndex::new(DistanceMetric::Euclidean, 16, 50, 16, 2);
    let mut segment = Segment::new(hnsw);

    let mut payload = Payload::default();
    payload.set("animal", PayloadValue::Str("dog".into()));

    let id = segment.insert(vecf(&[1.0, 2.0]), Some(payload.clone())).unwrap();
    let results = segment.search(&vecf(&[1.0, 2.0]), 1).unwrap();
    
    assert_eq!(results[0].id, id);
    let retrieved = segment.get_payload(id).unwrap();
    assert_eq!(retrieved, &payload);
}

#[test]
fn test_segment_logical_deletion_prevents_search() {
    let hnsw = HNSWIndex::new(DistanceMetric::Dot, 16, 50, 16, 2);
    let mut segment = Segment::new(hnsw);

    let id = segment.insert(vecf(&[1.0, 0.0]), None).unwrap();
    segment.delete(id).unwrap();

    let result = segment.search(&vecf(&[1.0, 0.0]), 1);
    assert!(matches!(result, Err(DBError::SearchError(_))));
}

#[test]
fn test_segment_partial_deletion_preserves_valid_entries() {
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
fn test_segment_payload_multiple_fields() {
    let hnsw = HNSWIndex::new(DistanceMetric::Cosine, 16, 50, 16, 2);
    let mut segment = Segment::new(hnsw);

    let mut payload = Payload::default();
    payload.set("species", PayloadValue::Str("canine".to_string()));
    payload.set("age", PayloadValue::Int(5));
    payload.set("tags", PayloadValue::Str("pet".into()));

    let id = segment.insert(vecf(&[0.5, 0.5]), Some(payload.clone())).unwrap();
    let retrieved = segment.get_payload(id).unwrap();

    assert_eq!(retrieved.get("species"), Some(&PayloadValue::Str("canine".to_string())));
    assert_eq!(retrieved.get("age"), Some(&PayloadValue::Int(5)));
    assert_eq!(retrieved.get("tags"), Some(&PayloadValue::Str("pet".to_string())));
}

#[test]
fn test_segment_auto_increment_id_behavior() {
    let hnsw = HNSWIndex::new(DistanceMetric::Euclidean, 16, 50, 16, 2);
    let mut segment = Segment::new(hnsw);

    let id1 = segment.insert(vecf(&[1.0, 0.0]), None).unwrap();
    let id2 = segment.insert(vecf(&[0.0, 1.0]), None).unwrap();
    let id3 = segment.insert(vecf(&[1.0, 1.0]), None).unwrap();

    assert_eq!(id2, id1 + 1);
    assert_eq!(id3, id2 + 1);
}

#[test]
fn test_segment_unfiltered_allows_deleted_entries() {
    let hnsw = HNSWIndex::new(DistanceMetric::Euclidean, 16, 50, 16, 2);
    let mut segment = Segment::new(hnsw);

    let id = segment.insert(vecf(&[3.0, 3.0]), None).unwrap();
    segment.delete(id).unwrap();

    let results = segment.search_unfiltered(&vecf(&[3.0, 3.0]), 1).unwrap();
    assert!(
        results.is_empty() || results[0].id == id,
        "Expected deleted ID to possibly appear in unfiltered search"
    );
}

#[test]
fn test_segment_purge_actually_removes_deleted() {
    let hnsw = HNSWIndex::new(DistanceMetric::Euclidean, 16, 50, 16, 2);
    let mut segment = Segment::new(hnsw);

    let id1 = segment.insert(vecf(&[1.0, 1.0]), None).unwrap();
    let id2 = segment.insert(vecf(&[2.0, 2.0]), None).unwrap();
    segment.delete(id1).unwrap();
    segment.purge().unwrap();

    let results = segment.search_unfiltered(&vecf(&[1.0, 1.0]), 5).unwrap();
    assert!(!results.iter().any(|r| r.id == id1), "Deleted ID should be purged");
    assert!(results.iter().any(|r| r.id == id2));
}

#[test]
fn test_segment_post_filter_with_metadata() {
    let hnsw = HNSWIndex::new(DistanceMetric::Dot, 16, 50, 16, 2);
    let mut segment = Segment::new(hnsw);

    for i in 0..100 {
        let mut payload = Payload::default();
        let parity = if i % 2 == 0 { "even" } else { "odd" };
        payload.set("parity", PayloadValue::Str(parity.to_string()));
        segment.insert(vecf(&[i as f32, 0.0]), Some(payload)).unwrap();
    }

    let filter = Filter::Match {
        key: "parity".into(),
        value: PayloadValue::Str("even".into()),
    };

    let results = segment.post_filter(&vecf(&[0.0, 0.0]), 10, Some(&filter)).unwrap();
    assert!(results.len() <= 10);
    for res in results {
        let payload = segment.get_payload(res.id).unwrap();
        assert_eq!(payload.get("parity"), Some(&PayloadValue::Str("even".to_string())));
    }
}

#[test]
fn test_segment_search_with_nonexistent_filter_key() {
    let hnsw = HNSWIndex::new(DistanceMetric::Dot, 16, 50, 16, 2);
    let mut segment = Segment::new(hnsw);

    let mut payload = Payload::default();
    payload.set("color", PayloadValue::Str("red".to_string()));
    segment.insert(vecf(&[1.0, 2.0]), Some(payload)).unwrap();

    let nonexistent_filter = Filter::Match {
        key: "size".into(),
        value: PayloadValue::Str("large".into()),
    };

    let results = segment.post_filter(&vecf(&[1.0, 2.0]), 10, Some(&nonexistent_filter)).unwrap();
    assert!(results.is_empty(), "No match should be returned for nonexistent filter key");
}

