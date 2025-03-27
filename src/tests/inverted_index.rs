use std::collections::HashSet;

use crate::payload_storage::stores::PayloadIndex;
use crate::utils::payload::{Payload, PayloadValue};
use ordered_float::OrderedFloat;


fn test_index_insert_and_query() {
    let mut index = PayloadIndex::new();

    let mut payload = Payload::default();
    payload.set("category", PayloadValue::Str("fruit".into()));
    payload.set("rank", PayloadValue::Int(1));
    payload.set("confidence", PayloadValue::Float(OrderedFloat(0.95)));
    payload.set("active", PayloadValue::Bool(true));

    index.insert(42, &payload);
    index.insert(43, &payload);

    let q1 = index.query_exact("category", &PayloadValue::Str("fruit".into()));
    assert_eq!(q1.unwrap(), &HashSet::from([42, 43]));

    let q2 = index.query_exact("rank", &PayloadValue::Int(1));
    assert_eq!(q2.unwrap(), &HashSet::from([42, 43]));

    let q3 = index.query_exact("confidence", &PayloadValue::Float(OrderedFloat(0.95)));
    assert_eq!(q3.unwrap(), &HashSet::from([42, 43]));

    let q4 = index.query_exact("active", &PayloadValue::Bool(true));
    assert_eq!(q4.unwrap(), &HashSet::from([42, 43]));
}

fn test_index_removal() {
    let mut index = PayloadIndex::new();

    let mut payload = Payload::default();
    payload.set("rank", PayloadValue::Int(99));

    index.insert(1, &payload);
    index.insert(2, &payload);

    let before = index.query_exact("rank", &PayloadValue::Int(99));
    assert_eq!(before.unwrap().len(), 2);

    index.remove(1, &payload);
    let after = index.query_exact("rank", &PayloadValue::Int(99));
    assert_eq!(after.unwrap(), &HashSet::from([2]));

    index.remove(2, &payload);
    let gone = index.query_exact("rank", &PayloadValue::Int(99));
    assert!(gone.is_none());
}

fn test_non_indexed_types() {
    let mut index = PayloadIndex::new();

    let mut payload = Payload::default();
    payload.set("list", PayloadValue::ListStr(vec!["a".into(), "b".into()]));
    payload.set("numbers", PayloadValue::ListInt(vec![1, 2, 3]));

    index.insert(99, &payload);

    // These should not be indexed
    assert!(index.query_exact("list", &PayloadValue::Str("a".into())).is_none());
    assert!(index.query_exact("numbers", &PayloadValue::Int(1)).is_none());

    // Confirm that they aren't present even in `all_for_key`
    assert!(index.all_for_key("list").is_none());
    assert!(index.all_for_key("numbers").is_none());
}

fn test_all_for_key() {
    let mut index = PayloadIndex::new();

    let mut p1 = Payload::default();
    p1.set("color", PayloadValue::Str("red".into()));

    let mut p2 = Payload::default();
    p2.set("color", PayloadValue::Str("blue".into()));

    let mut p3 = Payload::default();
    p3.set("color", PayloadValue::Str("red".into()));

    index.insert(1, &p1);
    index.insert(2, &p2);
    index.insert(3, &p3);

    let all = index.all_for_key("color").unwrap();
    assert_eq!(all, HashSet::from([1, 2, 3]));
}


fn test_duplicate_inserts_are_idempotent() {
    let mut index = PayloadIndex::new();

    let mut payload = Payload::default();
    payload.set("kind", PayloadValue::Str("apple".into()));

    index.insert(1, &payload);
    index.insert(1, &payload); // Same point inserted again

    let result = index.query_exact("kind", &PayloadValue::Str("apple".into()));
    assert_eq!(result.unwrap(), &HashSet::from([1]));
}

fn test_insert_same_key_different_values() {
    let mut index = PayloadIndex::new();

    let mut p1 = Payload::default();
    p1.set("group", PayloadValue::Str("A".into()));

    let mut p2 = Payload::default();
    p2.set("group", PayloadValue::Str("B".into()));

    index.insert(1, &p1);
    index.insert(2, &p2);

    let a_ids = index.query_exact("group", &PayloadValue::Str("A".into()));
    let b_ids = index.query_exact("group", &PayloadValue::Str("B".into()));

    assert_eq!(a_ids.unwrap(), &HashSet::from([1]));
    assert_eq!(b_ids.unwrap(), &HashSet::from([2]));
}

fn test_query_nonexistent_key_or_value() {
    let mut index = PayloadIndex::new();

    let mut payload = Payload::default();
    payload.set("status", PayloadValue::Str("ok".into()));
    index.insert(1, &payload);

    assert!(index.query_exact("nonexistent", &PayloadValue::Str("nope".into())).is_none());
    assert!(index.query_exact("status", &PayloadValue::Str("error".into())).is_none());
}

pub fn run_inverted_index_tests() {
    println!("Running inverted index tests...");

    test_index_insert_and_query();
    test_index_removal();
    test_non_indexed_types();
    test_all_for_key();
    test_duplicate_inserts_are_idempotent();
    test_insert_same_key_different_values();
    test_query_nonexistent_key_or_value();

    println!("✅ All inverted index tests passed");
}