use std::collections::{HashMap, HashSet};
use rand::seq::IteratorRandom;

use crate::payload_storage::stores::PayloadIndex;
use crate::utils::payload::Payload;
use crate::utils::types::{PointId, Vector};
use crate::utils::errors::DBError;
use crate::vector::hnsw::HNSWIndex;
use crate::vector::metric::distance;

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
                            let score = distance(vector, vec, hnsw.metric());
                            (id, score)
                        })
                    })
                    .collect::<Vec<_>>();

                scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                extra_neighbors.extend(scored.into_iter().take(4).map(|(id, _)| id));
            }
        }
    }

    for neighbor_id in extra_neighbors {
        hnsw.add_bidirectional_edge(0, point_id, neighbor_id);
    }

    Ok(())
}
