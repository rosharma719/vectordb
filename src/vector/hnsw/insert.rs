use std::collections::HashSet;
use std::io::Write;
use std::time::Instant;

use rand::seq::IteratorRandom;
use serde::Serialize;

use crate::payload_storage::filters::{evaluate_filter, Filter};
use crate::payload_storage::stores::PayloadIndex;
use crate::utils::errors::DBError;
use crate::utils::payload::{Payload, PayloadValue};
use crate::utils::types::{PointId, Vector, DistanceMetric};

use super::config::{
    FILTER_EDGE_LOG_CHUNK,
    VERBOSE,
    diversity_alpha_for_level,
    diversity_prune_floor,
    insert_trace_logger,
    next_insert_trace_seq,
    trace_every,
};
use super::stats::{FILTER_EDGE_STATS, FILTER_EDGE_TOTAL_KEYS, FilterEdgeAgg};
use super::types::{NodeCandidate};
use super::HNSWIndex;

#[derive(Serialize)]
struct InsertTraceEntry {
    insert_id: u64,
    point_id: PointId,
    level: usize,
    current_max_level: usize,
    layer: usize,
    entry_point: Option<PointId>,
    current_entry: PointId,
    candidates: usize,
    neighbors: usize,
    best_neighbor: Option<PointId>,
    elapsed_ms: f64,
}

fn log_insert_trace(entry: &InsertTraceEntry) {
    let Some(logger) = insert_trace_logger() else {
        return;
    };
    if let Ok(mut guard) = logger.lock() {
        if serde_json::to_writer(&mut *guard, entry).is_ok() {
            let _ = guard.write_all(b"\n");
            let _ = guard.flush();
        }
    }
}

impl HNSWIndex {
    pub fn insert(&mut self, point_id: PointId, vector: Vector) -> Result<(), DBError> {
        if self.point_to_idx.contains_key(&point_id) {
            if VERBOSE {
                log::debug!(target: "vector::hnsw", "[INSERT] Point {} already exists. Skipping.", point_id);
            }
            return Ok(());
        }

        self.validate_dim(&vector)?;

        let trace_id = next_insert_trace_seq();
        let trace_mod = trace_every() as u64;
        let trace_enabled = insert_trace_logger().is_some() && trace_id % trace_mod == 0;
        let trace_start = if trace_enabled { Some(Instant::now()) } else { None };

        let level = self.assign_random_level();
        let vec = self.maybe_normalize(&vector);
        let idx = self.register_node(point_id, vec, level);
        let nodes_len = self.vectors.len();
        self.ensure_level_capacity(level, nodes_len);
        self.extend_layers_for_new_node(nodes_len);

        for l in 0..=level {
            self.layers[l][idx].push(idx);
        }

        if self.entry_point.is_none() {
            if VERBOSE {
                log::debug!(
                    target: "vector::hnsw",
                    "[INSERT] First point. Setting entry point to {} at level {}",
                    point_id,
                    level
                );
            }
            self.allocate_entry_point(idx, level);
            return Ok(());
        }

        let mut current_entry = if let Some(ep) = self.entry_point {
            if self.deleted.get(ep).copied().unwrap_or(false) {
                self.find_highest_level_entry_point().unwrap_or(idx)
            } else {
                ep
            }
        } else {
            self.find_highest_level_entry_point().unwrap_or(idx)
        };

        for l in ((level + 1)..=self.current_max_level).rev() {
            current_entry = self.greedy_search_layer_unfiltered(&self.vectors[idx], current_entry, l);
        }

        for l in (0..=level).rev() {
            let use_norm = self.metric == DistanceMetric::Cosine || self.metric == DistanceMetric::Dot;
            let candidates = self.search_layer_unfiltered(
                &self.vectors[idx],
                current_entry,
                l,
                self.ef_construct,
                use_norm,
                None,
                None,
            )?;
            let m_for_layer = if l == 0 { self.m0 } else { self.m };
            let neighbors: Vec<usize> = self.select_diverse_neighbors(&candidates, m_for_layer, use_norm, l);

            let layer = self.layers.get_mut(l).unwrap();
            let mut linked = neighbors.clone();
            if !linked.contains(&idx) {
                linked.push(idx);
            }
            layer[idx] = linked;

            for &n in &neighbors {
                let e = &mut layer[n];
                if !e.contains(&idx) {
                    e.push(idx);
                }
            }

            if let Some(&best) = neighbors.first() {
                current_entry = best;
            }

            if trace_enabled {
                let entry = InsertTraceEntry {
                    insert_id: trace_id,
                    point_id,
                    level,
                    current_max_level: self.current_max_level,
                    layer: l,
                    entry_point: self.entry_point.map(|ep| self.point_id(ep)),
                    current_entry: self.point_id(current_entry),
                    candidates: candidates.len(),
                    neighbors: neighbors.len(),
                    best_neighbor: neighbors.first().map(|&n| self.point_id(n)),
                    elapsed_ms: trace_start.map(|t| t.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0),
                };
                log_insert_trace(&entry);
            }
        }

        if level > self.current_max_level {
            if VERBOSE {
                log::debug!(
                    target: "vector::hnsw",
                    "[INSERT] Promoting {} to new entry point at level {}",
                    point_id,
                    level
                );
            }
            self.entry_point = Some(idx);
            self.current_max_level = level;
        }

        Ok(())
    }

    pub fn build_filter_aware_edges(
        &mut self,
        point_id: PointId,
        vector: &Vector,
        payload: &Payload,
        payload_index: &PayloadIndex,
        _payloads: &std::collections::HashMap<PointId, Payload>,
        filter_keys: &[String],
    ) -> Result<(), DBError> {
        if filter_keys.is_empty() {
            return Ok(());
        }
        let query_vector = if self.metric == DistanceMetric::Cosine {
            self.maybe_normalize(vector)
        } else {
            vector.clone()
        };

        let mut extra_neighbors = HashSet::new();
        let m = self.m0();
        let log_edges_agg = std::env::var("VECTORDB_LOG_FILTER_EDGES_AGG")
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(false);

        let sample_limit: usize = m.saturating_mul(2);

        for key in filter_keys {
            if let Some(value) = payload.get(key) {
                let key_start = if log_edges_agg { Some(std::time::Instant::now()) } else { None };
                let mut sample_len = 0usize;
                let mut scored_len = 0usize;
                let prev_len = extra_neighbors.len();

                if let Some(id_set) = payload_index.query_exact(key, value) {
                    let mut rng = rand::rng();
                    let candidates: Vec<_> = if id_set.len() <= sample_limit {
                        id_set
                            .iter()
                            .filter(|&&id| id != point_id && self.get_vector(&id).is_some())
                            .copied()
                            .collect()
                    } else {
                        id_set
                            .iter()
                            .filter(|&&id| id != point_id && self.get_vector(&id).is_some())
                            .copied()
                            .choose_multiple(&mut rng, sample_limit)
                    };

                    let mut scored: Vec<_> = candidates
                        .into_iter()
                        .filter_map(|id| {
                            self.get_vector(&id).map(|vec| {
                                let raw = self.fast_score(&query_vector, vec);
                                let sort_key = self.normalize_score(raw);
                                super::types::ScoredPoint { id, raw_score: raw, sort_key }
                            })
                        })
                        .collect();
                    sample_len = scored.len();

                    scored.sort_by(|a, b| {
                        if self.metric == DistanceMetric::Dot {
                            b.raw_score.partial_cmp(&a.raw_score).unwrap()
                        } else {
                            a.raw_score.partial_cmp(&b.raw_score).unwrap()
                        }
                    });
                    scored_len = scored.len();

                    for sp in scored.into_iter().take(m) {
                        extra_neighbors.insert(sp.id);
                    }

                    if extra_neighbors.len() >= m {
                        if let Some(start) = key_start {
                            let dur = start.elapsed();
                            if log_edges_agg && sample_len > 0 {
                                let bucket = match value {
                                    PayloadValue::Bool(_) => 0,
                                    PayloadValue::Str(_) => 1,
                                    PayloadValue::Int(_) => 2,
                                    PayloadValue::Float(_) => 3,
                                    _ => 3,
                                };
                                FILTER_EDGE_STATS.with(|cell| {
                                    let mut agg = cell.borrow_mut();
                                    agg.count += 1;
                                    agg.samples += sample_len;
                                    agg.scored += scored_len;
                                    agg.added += extra_neighbors.len().saturating_sub(prev_len);
                                    agg.ns_by_type[bucket] += dur.as_nanos();
                                    if agg.count % FILTER_EDGE_LOG_CHUNK == 0 {
                                        FILTER_EDGE_TOTAL_KEYS.with(|tk| *tk.borrow_mut() += agg.count);
                                        let cum = FILTER_EDGE_TOTAL_KEYS.with(|tk| *tk.borrow());
                                        let total_ns: u128 = agg.ns_by_type.iter().sum();
                                        let to_ms = |ns: u128| (ns as f64) / 1_000_000.0;
                                        println!(
                                            "[filter_edges_agg] n={} cum_n={} samples={} scored={} added={} total_ms={:.3} bool_ms={:.3} str_ms={:.3} int_ms={:.3} float_ms={:.3}",
                                            agg.count,
                                            cum,
                                            agg.samples,
                                            agg.scored,
                                            agg.added,
                                            to_ms(total_ns),
                                            to_ms(agg.ns_by_type[0]),
                                            to_ms(agg.ns_by_type[1]),
                                            to_ms(agg.ns_by_type[2]),
                                            to_ms(agg.ns_by_type[3]),
                                        );
                                        *agg = FilterEdgeAgg::default();
                                    }
                                });
                            }
                        }
                        continue;
                    }
                }

                if let Some(start) = key_start {
                    let dur = start.elapsed();
                    if log_edges_agg && sample_len > 0 {
                        let bucket = match value {
                            PayloadValue::Bool(_) => 0,
                            PayloadValue::Str(_) => 1,
                            PayloadValue::Int(_) => 2,
                            PayloadValue::Float(_) => 3,
                            _ => 3,
                        };
                        FILTER_EDGE_STATS.with(|cell| {
                            let mut agg = cell.borrow_mut();
                            agg.count += 1;
                            agg.samples += sample_len;
                            agg.scored += scored_len;
                            agg.added += extra_neighbors.len().saturating_sub(prev_len);
                            agg.ns_by_type[bucket] += dur.as_nanos();
                            if agg.count % FILTER_EDGE_LOG_CHUNK == 0 {
                                FILTER_EDGE_TOTAL_KEYS.with(|tk| *tk.borrow_mut() += agg.count);
                                let cum = FILTER_EDGE_TOTAL_KEYS.with(|tk| *tk.borrow());
                                let total_ns: u128 = agg.ns_by_type.iter().sum();
                                let to_ms = |ns: u128| (ns as f64) / 1_000_000.0;
                                println!(
                                    "[filter_edges_agg] n={} cum_n={} samples={} scored={} added={} total_ms={:.3} bool_ms={:.3} str_ms={:.3} int_ms={:.3} float_ms={:.3}",
                                    agg.count,
                                    cum,
                                    agg.samples,
                                    agg.scored,
                                    agg.added,
                                    to_ms(total_ns),
                                    to_ms(agg.ns_by_type[0]),
                                    to_ms(agg.ns_by_type[1]),
                                    to_ms(agg.ns_by_type[2]),
                                    to_ms(agg.ns_by_type[3]),
                                );
                                *agg = FilterEdgeAgg::default();
                            }
                        });
                    }
                }
            }
        }

        let cap = m.max(1);
        for neighbor_id in extra_neighbors.into_iter().filter(|id| *id != point_id).take(cap) {
            self.add_one_way_edge(0, point_id, neighbor_id);
        }

        Ok(())
    }

    pub fn add_bidirectional_edge(&mut self, level: usize, a: PointId, b: PointId) {
        let (Some(a_idx), Some(b_idx)) = (self.idx_of(a), self.idx_of(b)) else { return; };
        let nodes_len = self.vectors.len();
        self.ensure_level_capacity(level, nodes_len);
        self.extend_layers_for_new_node(nodes_len);
        Self::push_unique(&mut self.layers[level][a_idx], b_idx);
        Self::push_unique(&mut self.layers[level][b_idx], a_idx);
    }

    pub fn add_one_way_edge(&mut self, level: usize, from: PointId, to: PointId) {
        let (Some(from_idx), Some(to_idx)) = (self.idx_of(from), self.idx_of(to)) else { return; };
        let nodes_len = self.vectors.len();
        self.ensure_level_capacity(level, nodes_len);
        self.extend_layers_for_new_node(nodes_len);
        Self::push_unique(&mut self.layers[level][from_idx], to_idx);
    }

    #[inline]
    fn push_unique(vec: &mut Vec<usize>, val: usize) {
        if !vec.contains(&val) {
            vec.push(val);
        }
    }

    pub fn greedy_search_layer_unfiltered(&self, query: &Vector, entry: usize, level: usize) -> usize {
        let mut current = entry;
        let mut changed = true;
        let mut steps = 0;

        while changed && steps < 1000 {
            steps += 1;
            changed = false;
            if let Some(neighbors) = self.layers.get(level).and_then(|l| l.get(current)) {
                for &neighbor in neighbors {
                    if self.deleted.get(neighbor).copied().unwrap_or(false) {
                        continue;
                    }

                    let d_current = self.fast_score(query, &self.vectors[current]);
                    let d_new = self.fast_score(query, &self.vectors[neighbor]);
                    let s_current = self.normalize_score(d_current);
                    let s_new = self.normalize_score(d_new);

                    if s_new < s_current {
                        current = neighbor;
                        changed = true;
                        break;
                    }
                }
            }
        }

        if steps >= 1000 {
            log::warn!(target: "vector::hnsw", "[GREEDY] Reached max steps at level {}, current = {}", level, current);
        }

        current
    }

    /// Heuristic neighbor selector that enforces diversity (HNSW heuristic 2).
    pub(crate) fn select_diverse_neighbors(
        &self,
        candidates: &[NodeCandidate],
        m: usize,
        normalize_scores: bool,
        level: usize,
    ) -> Vec<usize> {
        let alpha = diversity_alpha_for_level(level);
        let prune_floor = diversity_prune_floor().min(m);
        let mut result = Vec::with_capacity(m);
        for cand in candidates {
            if result.len() >= m {
                break;
            }
            if result.contains(&cand.idx) {
                continue;
            }
            let Some(cand_vec) = self.get_vector_by_idx(cand.idx) else { continue; };
            let d_qc = cand.sort_key;
            if result.len() < prune_floor {
                result.push(cand.idx);
                continue;
            }
            let mut too_close = false;
            for &r_id in &result {
                let Some(r_vec) = self.get_vector_by_idx(r_id) else { continue; };
                let d_cr_raw = self.fast_score(cand_vec, r_vec);
                let d_cr = if normalize_scores { self.normalize_score(d_cr_raw) } else { d_cr_raw };
                if d_cr < d_qc * alpha {
                    too_close = true;
                    break;
                }
            }
            if !too_close {
                result.push(cand.idx);
            }
        }
        if result.len() < m {
            for cand in candidates {
                if result.len() >= m {
                    break;
                }
                if !result.contains(&cand.idx) {
                    result.push(cand.idx);
                }
            }
        }
        result
    }
}

impl HNSWIndex {
    pub fn greedy_search_layer_with_filter(
        &self,
        query: &Vector,
        entry: usize,
        level: usize,
        payloads: &std::collections::HashMap<PointId, Payload>,
        filter: Option<&Filter>,
    ) -> Result<usize, DBError> {
        let mut current = entry;
        let mut changed = true;

        while changed {
            changed = false;

            if let Some(neighbors) = self.layer_neighbors(level, current) {
                for &neighbor in neighbors {
                    if self.deleted.get(neighbor).copied().unwrap_or(false) {
                        continue;
                    }

                    if let Some(f) = filter {
                        let id = self.point_id(neighbor);
                        let Some(payload) = payloads.get(&id) else { continue; };
                        if !evaluate_filter(f, payload)? {
                            continue;
                        }
                    }

                    let d_current = self.fast_score(query, self.get_vector_by_idx(current).unwrap());
                    let d_new = self.fast_score(query, self.get_vector_by_idx(neighbor).unwrap());

                    let s_current = match self.metric {
                        DistanceMetric::Dot => -d_current,
                        _ => d_current,
                    };
                    let s_new = match self.metric {
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
}
