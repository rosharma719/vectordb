use vectordb::utils::payload::ScalarComparisonOp;

use vectordb::segment::segment::Segment;
use vectordb::utils::payload::{Payload, PayloadValue};
use vectordb::utils::types::{DistanceMetric, Vector};
use vectordb::vector::hnsw::HNSWIndex;
use vectordb::payload_storage::filters::Filter;
use vectordb::vector::in_place::in_place_filtered_search;

fn vecf(v: &[f32]) -> Vector {
    v.to_vec()
}

fn make_segment(metric: DistanceMetric, dim: usize) -> Segment {
    Segment::new(HNSWIndex::new(metric, 16, 64, 8, dim))
}


#[test]
fn test_in_place_deep_logical_filtering_and_edge_cases() {
    let mut segment = make_segment(DistanceMetric::Cosine, 3);

    let labels = ["A", "B", "C"];
    let shapes = ["circle", "square", "triangle"];
    let categories = ["fruit", "animal", "furniture"];

    // Insert 200 points with varying payloads
    for i in 0..200 {
        let mut p = Payload::default();

        // Scalar traits
        p.set("label", PayloadValue::Str(labels[i % 3].into()));
        p.set("score", PayloadValue::Int((i % 100) as i64));
        p.set("verified", PayloadValue::Bool(i % 2 == 0));

        // Semi-shared trait
        if i % 4 == 0 {
            p.set("shape", PayloadValue::Str(shapes[i % 3].into()));
        }

        // List trait
        if i % 5 == 0 {
            p.set(
                "tags",
                PayloadValue::ListStr(vec!["hot".into(), "new".into(), "eco".into()]),
            );
        }

        // Category for filter separation
        p.set("category", PayloadValue::Str(categories[i % 3].into()));

        segment
            .insert(vecf(&[i as f32 * 0.01, (i % 10) as f32, (i % 5) as f32]), Some(p))
            .unwrap();
    }

    // Now test a complex filter:
    // (category == "fruit" OR shape == "circle") AND verified == true AND score <= 80
    let filter = Filter::And(vec![
        Filter::Or(vec![
            Filter::Match {
                key: "category".into(),
                value: PayloadValue::Str("fruit".into()),
            },
            Filter::Match {
                key: "shape".into(),
                value: PayloadValue::Str("circle".into()),
            },
        ]),
        Filter::Match {
            key: "verified".into(),
            value: PayloadValue::Bool(true),
        },
        Filter::Compare {
            key: "score".into(),
            op: ScalarComparisonOp::Lte,
            value: PayloadValue::Int(80),
        },
    ]);

    let results = in_place_filtered_search(
        &vecf(&[1.0, 1.0, 1.0]),
        30,
        segment.hnsw(),
        segment.payloads(),
        segment.payload_index(), // ✅ Required argument now added
        Some(&filter),
        &|id| segment.is_deleted(id),
    )
    .unwrap();

    for sp in results {
        let p = segment.get_payload(sp.id).unwrap();
        let verified = p.get("verified").unwrap() == &PayloadValue::Bool(true);
        let category = p.get("category").unwrap();
        let shape = p.get("shape");
        let score = p.get("score").and_then(|v| match v {
            PayloadValue::Int(i) => Some(*i),
            _ => None,
        }).unwrap_or_default();

        let is_ok =
            verified &&
            score <= 80 &&
            (category == &PayloadValue::Str("fruit".into())
                || shape == Some(&PayloadValue::Str("circle".into())));

        assert!(is_ok, "Point {:?} failed filter logic", sp.id);
    }
}