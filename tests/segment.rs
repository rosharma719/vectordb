use vectordb::segment::segment::Segment;
use vectordb::vector::hnsw::HNSWIndex;
use vectordb::utils::types::{DistanceMetric, Vector};
use vectordb::utils::payload::{Payload, PayloadValue, ScalarComparisonOp};
use vectordb::payload_storage::filters::Filter;

fn vecf(v: &[f32]) -> Vector {
    v.to_vec()
}


#[test]
fn test_large_scale_insert_and_search_all_metrics() {
    for metric in [DistanceMetric::Euclidean, DistanceMetric::Cosine, DistanceMetric::Dot] {
        let hnsw = HNSWIndex::new(metric, 16, 50, 16, 3);
        let mut segment = Segment::new(hnsw);

        let mut ids = Vec::new();
        let mut vectors = Vec::new();

        for i in 0..1000 {
            let mut payload = Payload::default();
            payload.set("index", PayloadValue::Int(i));

            let vec = vecf(&[
                (i as f32).sin() * 5.0,
                ((i * 3) as f32).cos() * 3.0,
                ((i % 7) as f32).sqrt(),
            ]);

            let id = segment.insert(vec.clone(), Some(payload)).unwrap();
            ids.push(id);
            vectors.push((id, vec));
        }

        for (expected_id, query) in vectors.iter().take(10) {
            let noisy_query: Vec<f32> = query.iter().map(|x| x + 0.001).collect();
            let results = segment.search(&noisy_query, 5).unwrap();

            if metric == DistanceMetric::Dot {
                // For Dot, we want to preserve magnitude.
                // Compute the best candidate among all inserted vectors
                let mut best_id = None;
                let mut best_dot = std::f32::NEG_INFINITY;
                for (id, vec) in vectors.iter() {
                    // Note: For dot, we do not normalize.
                    let dot: f32 = vec.iter().zip(&noisy_query).map(|(a, b)| a * b).sum();
                    if dot > best_dot {
                        best_dot = dot;
                        best_id = Some(*id);
                    }
                }
                let expected_dot_id = best_id.expect("at least one vector exists");
                assert!(
                    results[0].id == expected_dot_id,
                    "For Dot metric, expected top candidate id {:?}, but got {:?} for query {:?}",
                    expected_dot_id,
                    results[0].id,
                    noisy_query
                );
            } else {
                // For Euclidean and Cosine, the inserted vector should be among top results.
                let found = results.iter().any(|r| r.id == *expected_id);
                assert!(
                    found,
                    "Expected ID {:?} not found in top 5 for metric {:?}",
                    expected_id,
                    metric
                );
            }
        }
    }
}


#[test]
fn test_large_scale_filtered_queries_all_metrics() {
    for metric in [DistanceMetric::Euclidean, DistanceMetric::Cosine, DistanceMetric::Dot] {
        let hnsw = HNSWIndex::new(metric, 16, 50, 16, 3);
        let mut segment = Segment::new(hnsw);

        for i in 0..1000 {
            let mut payload = Payload::default();
            let animal = match i % 4 {
                0 => "dog",
                1 => "cat",
                2 => "bird",
                _ => "fish",
            };
            payload.set("animal", PayloadValue::Str(animal.to_string()));
            payload.set("age", PayloadValue::Int((i % 8 + 1) as i64));
            payload.set("score", PayloadValue::Float((60.0 + (i % 40) as f64).into()));

            let vec = vecf(&[
                ((i % 3) as f32).ln_1p(),
                ((i % 5) as f32).exp().fract(),
                ((i % 7) as f32).powf(1.5),
            ]);
            segment.insert(vec, Some(payload)).unwrap();
        }

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

        let results = segment.post_filter(&vecf(&[1.0, 0.0, 0.0]), 15, Some(&filter)).unwrap();
        for r in &results {
            let p = segment.get_payload(r.id).unwrap();
            assert_eq!(p.get("animal").unwrap(), &PayloadValue::Str("dog".into()));
            assert!(matches!(p.get("age").unwrap(), PayloadValue::Int(n) if *n >= 6));
            assert!(matches!(p.get("score").unwrap(), PayloadValue::Float(f) if *f < 90.0.into()));
        }
    }
}

#[test]
fn test_list_filters_with_larger_pool_all_metrics() {
    for metric in [DistanceMetric::Euclidean, DistanceMetric::Cosine, DistanceMetric::Dot] {
        let hnsw = HNSWIndex::new(metric, 16, 50, 16, 3);
        let mut segment = Segment::new(hnsw);

        for i in 0..1000 {
            let mut payload = Payload::default();
            let tags = if i % 2 == 0 {
                vec!["cheap".to_string(), "small".to_string()]
            } else {
                vec!["expensive".to_string(), "large".to_string()]
            };
            let active = i % 3 == 0;
            payload.set("tags", PayloadValue::ListStr(tags));
            payload.set("active", PayloadValue::Bool(active));
            let vec = vecf(&[
                (i as f32).cos(),
                (i as f32).sin(),
                (i as f32).tan().fract(),
            ]);
            segment.insert(vec, Some(payload)).unwrap();
        }

        let filter = Filter::Compare {
            key: "tags".into(),
            op: ScalarComparisonOp::Eq,
            value: PayloadValue::Str("cheap".into()),
        };
        let results = segment.post_filter(&vecf(&[0.0, 1.0, 0.0]), 10, Some(&filter)).unwrap();
        assert!(results.iter().all(|r| {
            let p = segment.get_payload(r.id).unwrap();
            match p.get("tags") {
                Some(PayloadValue::ListStr(tags)) => tags.contains(&"cheap".to_string()),
                _ => false,
            }
        }));
        assert!(results.len() >= 1);
    }
}

#[test]
fn test_deletion_and_purge_with_large_set_all_metrics() {
    for metric in [DistanceMetric::Euclidean, DistanceMetric::Cosine, DistanceMetric::Dot] {
        let hnsw = HNSWIndex::new(metric, 16, 50, 16, 3);
        let mut segment = Segment::new(hnsw);

        let mut ids = Vec::new();
        for i in 0..200 {
            let mut payload = Payload::default();
            payload.set("idx", PayloadValue::Int(i));
            let vec = vecf(&[i as f32, 0.0, 0.0]);
            let id = segment.insert(vec, Some(payload)).unwrap();
            ids.push(id);
        }

        // Delete every 7th point.
        for i in (0..200).step_by(7) {
            segment.delete(ids[i]).unwrap();
        }

        let results = segment.search(&vecf(&[10.0, 0.0, 0.0]), 30).unwrap();
        for r in &results {
            assert!(!segment.is_deleted(r.id));
        }

        // Delete all points. (After the previous purge, some points may already be gone;
        // our delete method now returns Ok in that case.)
        for id in ids.iter() {
            segment.delete(*id).unwrap();
        }

        for id in &ids {
            assert!(segment.get_vector(*id).is_none(), "❌ NOT purged: id = {}", id);
        }
    }
}
