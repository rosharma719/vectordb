use std::collections::{BinaryHeap, HashMap, HashSet};
use rand::seq::IteratorRandom;

use crate::payload_storage::stores::PayloadIndex;
use crate::utils::payload::Payload;
use crate::utils::types::{PointId, Vector, DistanceMetric};
use crate::utils::errors::DBError;
use crate::vector::metric::score;
use crate::payload_storage::filters::{Filter, evaluate_filter};
use crate::vector::hnsw::{HNSWIndex, ScoredPoint};

/// Builds extra edges at level 0 to preserve in-place filtering connectivity.
pub fn build_filter_aware_edges(
    point_id: PointId,
    vector: &Vector,
    payload: &Payload,
    hnsw: &mut HNSWIndex,
    payload_index: &PayloadIndex,
    payloads: &HashMap<PointId, Payload>,
    filter_keys: &[String],
) -> Result<(), DBError> {
    let mut extra_neighbors = HashSet::new();

    for key in filter_keys {
        if let Some(value) = payload.get(key) {
            // Try post-filtering from HNSW candidates
            let post_filtered = hnsw
                .search(vector, 20)?
                .into_iter()
                .filter(|sp| {
                    sp.id != point_id &&
                    payloads
                        .get(&sp.id)
                        .and_then(|p| p.get(key))
                        .map_or(false, |v| v == value)
                })
                .take(4)
                .map(|sp| sp.id)
                .collect::<Vec<_>>();

            if post_filtered.len() >= 4 {
                extra_neighbors.extend(post_filtered);
                continue;
            }

            // Fallback: brute-force scoring over filtered set
            if let Some(id_set) = payload_index.query_exact(key, value) {
                let mut rng = rand::rng();
                let candidates = id_set
                    .iter()
                    .filter(|&&id| id != point_id && hnsw.get_vector(&id).is_some())
                    .copied()
                    .choose_multiple(&mut rng, 1000);

                let mut scored = candidates
                    .into_iter()
                    .filter_map(|id| {
                        hnsw.get_vector(&id).map(|vec| {
                            let score = score(vector, vec, hnsw.metric());
                            let sort_key = match hnsw.metric() {
                                DistanceMetric::Dot => -score,
                                _ => score,
                            };
                            ScoredPoint { id, raw_score: score, sort_key }
                        })
                    })
                    .collect::<Vec<_>>();

                scored.sort_by(|a, b| a.sort_key.partial_cmp(&b.sort_key).unwrap_or(std::cmp::Ordering::Equal));
                extra_neighbors.extend(scored.into_iter().take(4).map(|sp| sp.id));
            }
        }
    }

    for neighbor_id in extra_neighbors {
        hnsw.add_bidirectional_edge(0, point_id, neighbor_id);
    }

    Ok(())
}

/// Main filtered search entry point
pub fn in_place_filtered_search(
    query: &Vector,
    top_k: usize,
    hnsw: &HNSWIndex,
    payloads: &HashMap<PointId, Payload>,
    payload_index: &PayloadIndex,
    filter: Option<&Filter>,
    is_deleted: &dyn Fn(PointId) -> bool,
) -> Result<Vec<ScoredPoint>, DBError> {
    if query.len() != hnsw.dim() {
        return Err(DBError::VectorLengthMismatch {
            expected: hnsw.dim(),
            actual: query.len(),
        });
    }

    let mut current = match hnsw.get_entry_point() {
        Some(id) => {
            if let Some(f) = filter {
                if let Some(payload) = payloads.get(&id) {
                    if evaluate_filter(f, payload)? {
                        id
                    } else {
                        find_entry_point_matching_filter(f, payload_index, is_deleted, hnsw)
                            .unwrap_or(id)
                    }
                } else {
                    find_entry_point_matching_filter(f, payload_index, is_deleted, hnsw)
                        .unwrap_or(id)
                }
            } else {
                id
            }
        }
        None => {
            if let Some(f) = filter {
                match find_entry_point_matching_filter(f, payload_index, is_deleted, hnsw) {
                    Some(id) => id,
                    None => return Ok(vec![]),
                }
            } else {
                return Ok(vec![]);
            }
        }
    };

    for level in (1..=hnsw.current_max_level()).rev() {
        current = greedy_search_layer_with_filter(query, current, level, hnsw, payloads, filter, is_deleted)?;
    }

    let mut visited = HashSet::new();
    let mut candidates = BinaryHeap::new();
    let mut results = BinaryHeap::new();

    let dist = score(query, hnsw.get_vector(&current).unwrap(), hnsw.metric());
    let sort_key = match hnsw.metric() {
        DistanceMetric::Dot => -dist,
        _ => dist,
    };
    let first = ScoredPoint { id: current, raw_score: dist, sort_key };
    candidates.push(first.clone());
    results.push(first);
    visited.insert(current);

    while let Some(current) = candidates.pop() {
        if let Some(neighbors) = hnsw.layer_neighbors(0, current.id) {
            for &neighbor in neighbors {
                if !visited.insert(neighbor) || is_deleted(neighbor) {
                    continue;
                }

                if let Some(f) = filter {
                    let Some(payload) = payloads.get(&neighbor) else {
                        continue;
                    };
                    if !evaluate_filter(f, payload)? {
                        continue;
                    }
                }

                let d = score(query, hnsw.get_vector(&neighbor).unwrap(), hnsw.metric());
                let sort_key = match hnsw.metric() {
                    DistanceMetric::Dot => -d,
                    _ => d,
                };
                let sp = ScoredPoint { id: neighbor, raw_score: d, sort_key };
                candidates.push(sp.clone());
                results.push(sp);

                if results.len() > hnsw.ef() {
                    results.pop();
                }
            }
        }
    }

    let mut res = results.into_sorted_vec();
    res.truncate(top_k);
    Ok(res)
}

fn greedy_search_layer_with_filter(
    query: &Vector,
    entry: PointId,
    level: usize,
    hnsw: &HNSWIndex,
    payloads: &HashMap<PointId, Payload>,
    filter: Option<&Filter>,
    is_deleted: &dyn Fn(PointId) -> bool,
) -> Result<PointId, DBError> {
    let mut current = entry;
    let mut changed = true;

    while changed {
        changed = false;

        if let Some(neighbors) = hnsw.layer_neighbors(level, current) {
            for &neighbor in neighbors {
                if is_deleted(neighbor) {
                    continue;
                }

                if let Some(f) = filter {
                    let Some(payload) = payloads.get(&neighbor) else {
                        continue;
                    };
                    if !evaluate_filter(f, payload)? {
                        continue;
                    }
                }

                let d_current = score(query, hnsw.get_vector(&current).unwrap(), hnsw.metric());
                let d_new = score(query, hnsw.get_vector(&neighbor).unwrap(), hnsw.metric());

                let s_current = match hnsw.metric() {
                    DistanceMetric::Dot => -d_current,
                    _ => d_current,
                };
                let s_new = match hnsw.metric() {
                    DistanceMetric::Dot => -d_new,
                    _ => d_new,
                };

                if s_new < s_current {
                    current = neighbor;
                    changed = true;
                }
            }
        }
    }

    Ok(current)
}

fn find_entry_point_matching_filter(
    filter: &Filter,
    payload_index: &PayloadIndex,
    is_deleted: &dyn Fn(PointId) -> bool,
    hnsw: &HNSWIndex,
) -> Option<PointId> {
    match filter {
        Filter::Match { key, value } => {
            payload_index
                .query_exact(key, value)?
                .iter()
                .find(|&&id| !is_deleted(id) && hnsw.get_vector(&id).is_some())
                .copied()
        }
        Filter::And(conds) | Filter::Or(conds) => {
            for cond in conds {
                if let Some(id) = find_entry_point_matching_filter(cond, payload_index, is_deleted, hnsw) {
                    return Some(id);
                }
            }
            None
        }
        Filter::Not(inner) => find_entry_point_matching_filter(inner, payload_index, is_deleted, hnsw),
        Filter::Compare { .. } => None, // Fallback if not indexed
    }
}
