use crate::utils::payload::*;
use crate::utils::errors::DBError;

pub fn run_payload_tests() {
    println!("Running payload tests...");

    test_scalar_comparisons();
    test_list_contains();
    test_list_element_compare();
    test_list_length();
    test_list_equality();
    test_payload_set_and_get();
    test_payload_compare_field();
    test_payload_evaluate_list_field_errors();

    println!("✅ All payload tests passed");
}

fn test_scalar_comparisons() {
    // Int
    let a = PayloadValue::Int(10);
    let b = PayloadValue::Int(20);
    assert_eq!(a.compare_scalar(ScalarComparisonOp::Lt, &b), Some(true));

    // Float
    let a = PayloadValue::Float(1.0);
    let b = PayloadValue::Float(1.0);
    assert_eq!(a.compare_scalar(ScalarComparisonOp::Eq, &b), Some(true));

    // Str
    let a = PayloadValue::Str("abc".into());
    let b = PayloadValue::Str("xyz".into());
    assert_eq!(a.compare_scalar(ScalarComparisonOp::Lt, &b), Some(true));

    // Bool
    let a = PayloadValue::Bool(true);
    let b = PayloadValue::Bool(false);
    assert_eq!(a.compare_scalar(ScalarComparisonOp::Neq, &b), Some(true));

    // Mismatched types
    assert_eq!(a.compare_scalar(ScalarComparisonOp::Eq, &PayloadValue::Str("true".into())), None);
}

fn test_list_contains() {
    let ints = PayloadValue::ListInt(vec![1, 2, 3]);
    let floats = PayloadValue::ListFloat(vec![0.1, 0.2]);
    let strs = PayloadValue::ListStr(vec!["a".into(), "b".into()]);
    let bools = PayloadValue::ListBool(vec![true, false]);

    assert_eq!(
        ints.evaluate_list_query(ListQueryOp::Contains(&PayloadValue::Int(2))),
        Some(true)
    );
    assert_eq!(
        floats.evaluate_list_query(ListQueryOp::Contains(&PayloadValue::Float(0.3))),
        Some(false)
    );
    assert_eq!(
        strs.evaluate_list_query(ListQueryOp::Contains(&PayloadValue::Str("a".into()))),
        Some(true)
    );
    assert_eq!(
        bools.evaluate_list_query(ListQueryOp::Contains(&PayloadValue::Bool(false))),
        Some(true)
    );
}

fn test_list_element_compare() {
    let ints = PayloadValue::ListInt(vec![5, 10, 15]);
    let floats = PayloadValue::ListFloat(vec![0.1, 0.5]);
    let strs = PayloadValue::ListStr(vec!["a".into(), "b".into()]);
    let bools = PayloadValue::ListBool(vec![true, false]);

    assert_eq!(
        ints.evaluate_list_query(ListQueryOp::ElementCompare(2, ScalarComparisonOp::Gt, &PayloadValue::Int(10))),
        Some(true)
    );
    assert_eq!(
        floats.evaluate_list_query(ListQueryOp::ElementCompare(1, ScalarComparisonOp::Lte, &PayloadValue::Float(0.5))),
        Some(true)
    );
    assert_eq!(
        strs.evaluate_list_query(ListQueryOp::ElementCompare(0, ScalarComparisonOp::Eq, &PayloadValue::Str("a".into()))),
        Some(true)
    );
    assert_eq!(
        bools.evaluate_list_query(ListQueryOp::ElementCompare(1, ScalarComparisonOp::Eq, &PayloadValue::Bool(false))),
        Some(true)
    );
}

fn test_list_length() {
    let vec = PayloadValue::ListStr(vec!["a".into(), "b".into(), "c".into()]);

    assert_eq!(
        vec.evaluate_list_query(ListQueryOp::Length(ScalarComparisonOp::Eq, 3)),
        Some(true)
    );
    assert_eq!(
        vec.evaluate_list_query(ListQueryOp::Length(ScalarComparisonOp::Gt, 1)),
        Some(true)
    );
}

fn test_list_equality() {
    let a = PayloadValue::ListInt(vec![1, 2, 3]);
    let b = PayloadValue::ListInt(vec![1, 2, 3]);
    let c = PayloadValue::ListInt(vec![1, 2]);

    assert_eq!(a.evaluate_list_query(ListQueryOp::Equals(&b)), Some(true));
    assert_eq!(a.evaluate_list_query(ListQueryOp::Equals(&c)), Some(false));

    let wrong_type = PayloadValue::ListFloat(vec![1.0, 2.0, 3.0]);
    assert_eq!(a.evaluate_list_query(ListQueryOp::Equals(&wrong_type)), None);
}

fn test_payload_set_and_get() {
    let mut payload = Payload::default();
    payload.set("flag", PayloadValue::Bool(true));
    assert_eq!(payload.get("flag"), Some(&PayloadValue::Bool(true)));
}

fn test_payload_compare_field() {
    let mut payload = Payload::default();
    payload.set("x", PayloadValue::Int(42));

    let result = payload.compare_field("x", ScalarComparisonOp::Gte, &PayloadValue::Int(40));
    assert!(result.is_ok() && result.unwrap());  // Changed to assert the boolean value

    let missing = payload.compare_field("y", ScalarComparisonOp::Eq, &PayloadValue::Int(1));
    assert!(matches!(missing, Err(DBError::InvalidPayload(_))));

    let wrong_type = payload.compare_field("x", ScalarComparisonOp::Eq, &PayloadValue::Str("forty-two".into()));
    assert!(matches!(wrong_type, Err(DBError::InvalidPayload(_))));
}

fn test_payload_evaluate_list_field_errors() {
    let mut payload = Payload::default();
    payload.set("tags", PayloadValue::ListStr(vec!["a".into(), "b".into()]));

    let ok = payload.evaluate_list_field(
        "tags",
        ListQueryOp::Contains(&PayloadValue::Str("a".into())),
    );
    assert!(ok.is_ok() && ok.unwrap());  // Changed to assert the boolean value

    let missing = payload.evaluate_list_field(
        "nonexistent",
        ListQueryOp::Contains(&PayloadValue::Str("a".into())),
    );
    assert!(matches!(missing, Err(DBError::InvalidPayload(_))));

    let wrong_type = payload.evaluate_list_field(
        "tags",
        ListQueryOp::Contains(&PayloadValue::Float(1.0)),
    );
    assert!(matches!(wrong_type, Err(DBError::InvalidPayload(_))));
}