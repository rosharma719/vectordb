use std::collections::{HashMap, HashSet};
use crate::payload_storage::filters::Filter;
use crate::utils::payload::{Payload, PayloadValue};
use crate::utils::types::PointId;
use serde::{Deserialize, Serialize};

/// Inverted index: field_name -> field_value -> set of PointIds
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct PayloadIndex {
    index: HashMap<String, HashMap<PayloadValue, HashSet<PointId>>>,
}

impl PayloadIndex {
    pub fn new() -> Self {
        Self {
            index: HashMap::new(),
        }
    }

    /// Indexes the payload of a given point.
    pub fn insert(&mut self, point_id: PointId, payload: &Payload) {
        for (key, value) in &payload.0 {
            if !Self::is_indexable(value) {
                continue;
            }

            self.index
                .entry(key.clone())
                .or_insert_with(HashMap::new)
                .entry(value.clone())
                .or_insert_with(HashSet::new)
                .insert(point_id);
        }
    }

    /// Removes a point's payload from the index.
    pub fn remove(&mut self, point_id: PointId, payload: &Payload) {
        for (key, value) in &payload.0 {
            if !Self::is_indexable(value) {
                continue;
            }

            if let Some(value_map) = self.index.get_mut(key) {
                if let Some(id_set) = value_map.get_mut(value) {
                    id_set.remove(&point_id);
                    if id_set.is_empty() {
                        value_map.remove(value);
                    }
                }
                if value_map.is_empty() {
                    self.index.remove(key);
                }
            }
        }
    }

    /// Returns a set of point IDs that match exactly this key-value pair.
    pub fn query_exact(&self, key: &str, value: &PayloadValue) -> Option<&HashSet<PointId>> {
        if !Self::is_indexable(value) {
            return None;
        }

        self.index.get(key)?.get(value)
    }

    /// Returns a candidate set for Match/And/Or filters based on the payload index.
    /// For And, it intersects all indexable clauses (ignoring non-indexable ones).
    /// For Or, all clauses must be indexable to return a safe candidate set.
    pub fn query_filter_ids(&self, filter: &Filter) -> Option<HashSet<PointId>> {
        match filter {
            Filter::Match { key, value } => self
                .query_exact(key, value)
                .map(|ids| ids.iter().copied().collect()),
            Filter::And(conds) => {
                let mut sets: Vec<HashSet<PointId>> = Vec::new();
                for cond in conds {
                    if let Some(set) = self.query_filter_ids(cond) {
                        sets.push(set);
                    }
                }
                if sets.is_empty() {
                    return None;
                }
                sets.sort_by_key(|s| s.len());
                let mut acc = sets.remove(0);
                for s in sets {
                    acc.retain(|id| s.contains(id));
                }
                Some(acc)
            }
            Filter::Or(conds) => {
                let mut acc = HashSet::new();
                for cond in conds {
                    let Some(set) = self.query_filter_ids(cond) else {
                        return None;
                    };
                    acc.extend(set);
                }
                Some(acc)
            }
            Filter::Compare { .. } | Filter::Not(_) => None,
        }
    }

    fn is_indexable(value: &PayloadValue) -> bool {
        matches!(
            value,
            PayloadValue::Int(_)
                | PayloadValue::Float(_)
                | PayloadValue::Str(_)
                | PayloadValue::Bool(_)
        )
    }

    /// Optional: Returns all point IDs that have any value for the given key.
    pub fn all_for_key(&self, key: &str) -> Option<HashSet<PointId>> {
        self.index.get(key).map(|map| {
            map.values()
                .fold(HashSet::new(), |mut acc, set| {
                    acc.extend(set.iter().copied());
                    acc
                })
        })
    }
}
