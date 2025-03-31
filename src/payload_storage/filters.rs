use crate::utils::errors::DBError;
use crate::utils::payload::{Payload, PayloadValue, ScalarComparisonOp}; 


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
