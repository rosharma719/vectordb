use crate::utils::payload::{Payload, PayloadValue, ScalarComparisonOp, ListQueryOp};
use crate::utils::errors::DBError;

/// The logical filter condition tree
#[derive(Debug, Clone)]
pub enum Filter {
    Match {
        key: String,
        value: PayloadValue,
    },
    Compare {
        key: String,
        op: ScalarComparisonOp,
        value: PayloadValue,
    },
    Contains {
        key: String,
        value: PayloadValue,
    },
    And(Vec<Filter>),
    Or(Vec<Filter>),
    Not(Box<Filter>),
}

/// Evaluates whether a given payload satisfies the filter condition.
pub fn evaluate_filter(filter: &Filter, payload: &Payload) -> Result<bool, DBError> {
    match filter {
        Filter::Match { key, value } => {
            match payload.get(key) {
                Some(actual) => Ok(actual == value),
                None => Ok(false),
            }
        }
        Filter::Compare { key, op, value } => {
            payload.compare_field(key, *op, value)
        }
        Filter::Contains { key, value } => {
            payload.evaluate_list_field(key, ListQueryOp::Contains(value))
        }
        Filter::And(conditions) => {
            for cond in conditions {
                if !evaluate_filter(cond, payload)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        Filter::Or(conditions) => {
            for cond in conditions {
                if evaluate_filter(cond, payload)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        Filter::Not(inner) => {
            Ok(!evaluate_filter(inner, payload)?)
        }
    }
}

pub fn test_not_filter() {
    let mut payload = Payload::default();
    payload.set("status", PayloadValue::Str("inactive".into()));

    let filter = Filter::Not(Box::new(Filter::Match {
        key: "status".into(),
        value: PayloadValue::Str("active".into()),
    }));

    let result = evaluate_filter(&filter, &payload).unwrap();
    assert!(result);
}


// Public test runner callable from main
pub fn run_filter_tests() {
    println!("Running filter tests...");

    test_compare_filter();
    test_contains_filter();
    test_not_filter();
    test_nested_and_or_filter();

    println!("✅ All filter tests passed");
}

// Promote test functions for use in run_filter_tests
pub fn test_compare_filter() {
    let mut payload = Payload::default();
    payload.set("score", PayloadValue::Int(42));

    let filter = Filter::Compare {
        key: "score".into(),
        op: ScalarComparisonOp::Gt,
        value: PayloadValue::Int(30),
    };

    let result = evaluate_filter(&filter, &payload).unwrap();
    assert!(result);
}

pub fn test_contains_filter() {
    let mut payload = Payload::default();
    payload.set("tags", PayloadValue::ListStr(vec!["science".into(), "ai".into()]));

    let filter = Filter::Contains {
        key: "tags".into(),
        value: PayloadValue::Str("ai".into()),
    };

    let result = evaluate_filter(&filter, &payload).unwrap();
    assert!(result);
}

pub fn test_nested_and_or_filter() {
    let mut payload = Payload::default();
    payload.set("kind", PayloadValue::Str("robot".into()));
    payload.set("active", PayloadValue::Bool(true));

    let filter = Filter::And(vec![
        Filter::Or(vec![
            Filter::Match {
                key: "kind".into(),
                value: PayloadValue::Str("robot".into()),
            },
            Filter::Match {
                key: "kind".into(),
                value: PayloadValue::Str("animal".into()),
            },
        ]),
        Filter::Match {
            key: "active".into(),
            value: PayloadValue::Bool(true),
        },
    ]);

    let result = evaluate_filter(&filter, &payload).unwrap();
    assert!(result);
}