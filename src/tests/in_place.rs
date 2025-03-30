use crate::segment::segment::Segment;
use crate::utils::payload::{Payload, PayloadValue};
use crate::utils::types::{DistanceMetric, Vector};
use crate::vector::hnsw::HNSWIndex;

fn vecf(v: &[f32]) -> Vector {
    v.to_vec()
}

fn test_filter_aware_edges_preserve_reachability() {
    let hnsw = HNSWIndex::new(DistanceMetric::Euclidean, 16, 64, 16, 2);
    let mut segment = Segment::new(hnsw);

    for i in 0..100 {
        let mut payload = Payload::default();
        payload.set("tag", PayloadValue::Str("apple".into()));
        payload.set("weight", PayloadValue::Int(150 + i));
        payload.set("organic", PayloadValue::Bool(i % 2 == 0));
        segment.insert(vecf(&[1.0 + i as f32 * 0.01, 1.0 + i as f32 * 0.01]), Some(payload)).unwrap();
    }

    for i in 0..10 {
        let mut payload_far = Payload::default();
        payload_far.set("tag", PayloadValue::Str("banana".into()));
        payload_far.set("weight", PayloadValue::Int(200));
        segment.insert(vecf(&[10.0 + i as f32, 10.0 + i as f32]), Some(payload_far)).unwrap();
    }

    let res = segment.search(&vecf(&[2.0, 2.0]), 10).unwrap();
    for sp in &res {
        let tag = segment.get_payload(sp.id).unwrap().get("tag").unwrap();
        assert_eq!(tag, &PayloadValue::Str("apple".into()));
    }
}

fn test_shared_trait_connectivity() {
    let mut segment = Segment::new(HNSWIndex::new(DistanceMetric::Cosine, 16, 64, 8, 2));

    for i in 0..100 {
        let mut payload = Payload::default();
        payload.set("brand", PayloadValue::Str("Nike".into()));
        payload.set("release_year", PayloadValue::Int(2010 + i as i64 % 10));
        segment.insert(vecf(&[0.0, 1.0 + i as f32 * 0.01]), Some(payload)).unwrap();
    }

    let results = segment.search(&vecf(&[0.0, 1.5]), 20).unwrap();
    assert!(results.len() >= 20);
}

use crate::payload_storage::filters::Filter;

fn test_different_trait_isolation() {
    let mut segment = Segment::new(HNSWIndex::new(DistanceMetric::Cosine, 16, 64, 8, 2));

    for i in 0..50 {
        let mut payload_fruit = Payload::default();
        payload_fruit.set("category", PayloadValue::Str("fruit".into()));
        payload_fruit.set("organic", PayloadValue::Bool(i % 2 == 0));
        segment.insert(vecf(&[0.1 * i as f32, 1.0]), Some(payload_fruit)).unwrap();
    }

    for i in 0..50 {
        let mut payload_furniture = Payload::default();
        payload_furniture.set("category", PayloadValue::Str("furniture".into()));
        payload_furniture.set("material", PayloadValue::Str("wood".into()));
        segment.insert(vecf(&[5.0, 0.1 * i as f32]), Some(payload_furniture)).unwrap();
    }

    // Use explicit post-filtering to ensure only "fruit" category is returned
    let filter = Filter::Match {
        key: "category".into(),
        value: PayloadValue::Str("fruit".into()),
    };

    let result = segment.post_filter(&vecf(&[1.0, 1.0]), 10, Some(&filter)).unwrap();
    let categories: Vec<_> = result
        .iter()
        .filter_map(|sp| segment.get_payload(sp.id).and_then(|p| p.get("category")))
        .collect();

    assert!(
        categories.iter().all(|v| v == &&PayloadValue::Str("fruit".into())),
        "Expected all returned points to be from 'fruit' category"
    );
}



fn test_filtering_on_multiple_fields() {
    let mut segment = Segment::new(HNSWIndex::new(DistanceMetric::Euclidean, 16, 64, 8, 2));

    let types = ["shoe", "hat", "jacket"];
    let genders = ["male", "female", "unisex"];
    for t in types.iter() {
        for g in genders.iter() {
            for size in 5..15 {
                let mut p = Payload::default();
                p.set("type", PayloadValue::Str((*t).into()));
                p.set("gender", PayloadValue::Str((*g).into()));
                p.set("size", PayloadValue::Int(size));
                p.set("available", PayloadValue::Bool(size % 2 == 0));
                segment.insert(vecf(&[size as f32 * 0.1, 0.5]), Some(p)).unwrap();
            }
        }
    }

    let results = segment.search(&vecf(&[1.25, 0.5]), 30).unwrap();
    assert!(results.len() >= 20);
}

fn test_fallback_brute_force_on_small_traits() {
    let mut segment = Segment::new(HNSWIndex::new(DistanceMetric::Dot, 16, 64, 8, 2));

    let colors = ["blue", "green", "red", "yellow", "purple"];
    for c in colors.iter() {
        for i in 0..20 {
            let mut p = Payload::default();
            p.set("color", PayloadValue::Str((*c).into()));
            p.set("intensity", PayloadValue::Float((0.5 + i as f64 * 0.05).into()));
            segment.insert(vecf(&[1.0 - i as f32 * 0.01, 0.0]), Some(p)).unwrap();
        }
    }

    let results = segment.search(&vecf(&[0.8, 0.0]), 50).unwrap();
    assert!(results.len() >= 40);
}

fn test_filter_aware_edge_with_no_payload() {
    let mut segment = Segment::new(HNSWIndex::new(DistanceMetric::Cosine, 16, 64, 8, 2));

    for i in 0..100 {
        segment.insert(vecf(&[0.1, 0.9 + i as f32 * 0.005]), None).unwrap();
    }

    let results = segment.search(&vecf(&[0.1, 0.95]), 20).unwrap();
    assert!(results.len() >= 20);
}

pub fn run_in_place_tests() {
    test_filter_aware_edges_preserve_reachability();
    test_shared_trait_connectivity();
    test_different_trait_isolation();
    test_filtering_on_multiple_fields();
    test_fallback_brute_force_on_small_traits();
    test_filter_aware_edge_with_no_payload();
    println!("✅ In-place filtering tests passed");
}
