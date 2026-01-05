use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::time::Instant;

use serde::Serialize;

use crate::utils::errors::DBError;
use crate::utils::types::{DistanceMetric, Vector};

#[derive(Default, Debug, Copy, Clone)]
pub(crate) struct SearchCounters {
    adjacency_reads: usize,
    distance_computations: usize,
    cap_breaks: usize,
    patience_breaks: usize,
}

use super::HNSWIndex;
use super::config::{
    disable_early_exit, early_exit_patience, log_neighbor_scan_state, log_unfiltered_enabled,
    neighbor_scan_cap, neighbor_scan_patience, neighbor_scan_rotate_enabled,
    neighbor_scan_stride_enabled, next_search_trace_seq, search_expansion_cap_override,
    search_expansion_multiplier, search_trace_logger, trace_every,
};
use super::scratch::SEARCH_SCRATCH;
use super::stats::{SearchLayerStats, SearchStats, UNFILTERED_SEARCH_AGG, UnfilteredSample};
use super::types::{NodeCandidate, NodeResult, ScoredPoint, SearchRuntimeOptions};

#[derive(Serialize)]
struct SearchTraceEntry {
    search_id: u64,
    metric: String,
    ef_search: usize,
    exact_fallback_enabled: bool,
    exact_fallback_threshold: usize,
    collection_size: usize,
    exact_scan: bool,
    top_id: Option<u64>,
    top_raw: Option<f32>,
    top_sort_key: Option<f32>,
    step: usize,
    level: usize,
    current_idx: usize,
    current_point: u64,
    current_sort_key: f32,
    visited: usize,
    expanded: usize,
    candidate_len: usize,
    results_len: usize,
    worst_score: f32,
    stop_reason: Option<String>,
    elapsed_ms: f64,
}

pub(crate) struct SearchTraceCtx {
    id: u64,
    step: usize,
    every: usize,
    start: Instant,
}

fn log_search_trace(entry: &SearchTraceEntry) {
    let Some(logger) = search_trace_logger() else {
        return;
    };
    if let Ok(mut guard) = logger.lock() {
        if serde_json::to_writer(&mut *guard, entry).is_ok() {
            let _ = guard.write_all(b"\n");
            let _ = guard.flush();
        }
    }
}

fn hash_query(query: &Vector) -> u64 {
    let mut hasher = DefaultHasher::new();
    query.len().hash(&mut hasher);
    for &value in query {
        hasher.write_u32(value.to_bits());
    }
    hasher.finish()
}

fn stride_for_degree(degree: usize, seed: u64) -> usize {
    if degree <= 1 {
        return 1;
    }
    let mut stride = (((seed >> 32) as usize) % (degree - 1)) + 1;
    while gcd(stride, degree) != 1 {
        stride = (stride % (degree - 1)) + 1;
    }
    stride
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

impl HNSWIndex {
    fn exact_scan(&self, query: &Vector, normalize_scores: bool, top_k: usize) -> Vec<ScoredPoint> {
        let mut brute: Vec<ScoredPoint> = self
            .vectors
            .iter()
            .enumerate()
            .filter_map(|(idx, vec)| {
                if self.deleted.get(idx).copied().unwrap_or(false) {
                    return None;
                }
                let raw = self.fast_score(query, vec);
                let sort_key = if normalize_scores {
                    self.normalize_score(raw)
                } else {
                    raw
                };
                Some(ScoredPoint {
                    id: self.point_id(idx),
                    raw_score: raw,
                    sort_key,
                })
            })
            .collect();
        brute.sort_by(|a, b| {
            a.sort_key
                .partial_cmp(&b.sort_key)
                .unwrap()
                .then_with(|| a.id.cmp(&b.id))
        });
        brute.truncate(top_k);
        brute
    }

    pub(crate) fn search_layer_unfiltered(
        &self,
        query: &Vector,
        entry: usize,
        level: usize,
        ef: usize,
        opts: &SearchRuntimeOptions,
        normalize: bool,
        stats: Option<&mut SearchLayerStats>,
        trace: Option<&mut SearchTraceCtx>,
    ) -> Result<(Vec<NodeCandidate>, SearchCounters), DBError> {
        self.validate_dim(query)?;

        let mut trace = trace;
        let query_signature = hash_query(query);
        let rotate_neighbor_scans = neighbor_scan_rotate_enabled();
        let stride_enabled = neighbor_scan_stride_enabled();
        let expansion_mult = search_expansion_multiplier();
        let expansion_cap_override = search_expansion_cap_override();
        let expansion_cap_value =
            expansion_cap_override.or_else(|| Some(ef.saturating_mul(expansion_mult).max(ef)));
        log_neighbor_scan_state(expansion_mult, expansion_cap_value);
        SEARCH_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            scratch.next_epoch(self.vectors.len());
            scratch.candidate_queue.clear();
            scratch.result_set.clear();

            let mut visited_count = 0usize;
            let mut expanded = 0usize;
            let expansion_cap = expansion_cap_value;
            let mut adjacency_reads = 0usize;
            let mut distance_computations = 0usize;
            let mut cap_breaks = 0usize;
            let mut patience_breaks = 0usize;
            let neighbor_patience = opts
                .neighbor_scan_patience
                .unwrap_or_else(neighbor_scan_patience);

            let start_entry = if self.deleted.get(entry).copied().unwrap_or(false) {
                self.deleted
                    .iter()
                    .position(|deleted| !*deleted)
                    .unwrap_or(entry)
            } else {
                entry
            };

            let entry_distance = self.fast_score(query, &self.vectors[start_entry]);
            let entry_score = if normalize {
                self.normalize_score(entry_distance)
            } else {
                entry_distance
            };

            let initial = NodeCandidate {
                idx: start_entry,
                raw_score: entry_distance,
                sort_key: entry_score,
            };

            scratch.candidate_queue.push(initial.clone());
            scratch.result_set.push(NodeResult(initial.clone()));
            if scratch.mark_visited(start_entry) {
                visited_count += 1;
            }

            let mut worst_score = scratch.result_set.peek().unwrap().0.sort_key;
            let allow_early_exit = self.metric != DistanceMetric::Dot && !disable_early_exit();
            let patience_limit = if allow_early_exit {
                opts.early_exit_patience
                    .unwrap_or_else(early_exit_patience)
            } else {
                0
            };
            let mut no_improve_streak = 0usize;
            let mut stop_reason = "queue_empty";

            while let Some(current) = scratch.candidate_queue.peek() {
                if allow_early_exit && scratch.result_set.len() >= ef {
                    if current.sort_key > worst_score {
                        no_improve_streak += 1;
                    } else {
                        no_improve_streak = 0;
                    }
                    if no_improve_streak > patience_limit {
                        stop_reason = "early_exit_patience";
                        break;
                    }
                }

                let current = scratch.candidate_queue.pop().unwrap();
                expanded += 1;
                if let Some(neighbors) = self.layers.get(level).and_then(|l| l.get(current.idx)) {
                    const BATCH: usize = 16;
                    let mut batch = [0usize; BATCH];
                    let mut batch_len = 0usize;
                    let degree = neighbors.len();
                    if degree > 0 {
                        let cap = if level == 0 {
                            opts.neighbor_scan_cap_level0
                                .map(|v| if v == 0 { usize::MAX } else { v })
                                .unwrap_or_else(|| neighbor_scan_cap(level))
                        } else {
                            neighbor_scan_cap(level)
                        };
                        let window = degree.min(cap);
                        if window > 0 {
                            let need_seed = (rotate_neighbor_scans && window < degree)
                                || (stride_enabled && window < degree);
                            let seed = if need_seed {
                                query_signature
                                    .wrapping_add(current.idx as u64)
                                    .wrapping_mul(0x9e3779b97f4a7c15)
                            } else {
                                0
                            };
                            let start = if rotate_neighbor_scans && window < degree {
                                (seed % degree as u64) as usize
                            } else {
                                0
                            };
                            let stride = if stride_enabled && window < degree {
                                stride_for_degree(degree, seed)
                            } else {
                                1
                            };
                            let mut patience_triggered = false;
                            let mut neighbor_no_improve = 0usize;
                            let mut neighbors_examined = 0usize;
                            let mut offset = start;
                            let cap_hit = window >= cap;
                            'neighbor_scan: while neighbors_examined < window {
                                let neighbor = neighbors[offset];
                                neighbors_examined += 1;
                                offset = (offset + stride) % degree;
                                adjacency_reads += 1;
                                if self.deleted.get(neighbor).copied().unwrap_or(false)
                                    || !scratch.mark_visited(neighbor)
                                {
                                    continue;
                                }
                                visited_count += 1;

                                batch[batch_len] = neighbor;
                                batch_len += 1;
                                if batch_len == BATCH {
                                    for i in 0..batch_len {
                                        let idx = batch[i];
                                        distance_computations += 1;
                                        let raw = self.fast_score(query, &self.vectors[idx]);
                                        let score_val = if normalize {
                                            self.normalize_score(raw)
                                        } else {
                                            raw
                                        };

                                        let improves_result_set = scratch.result_set.len() < ef
                                            || score_val < worst_score;
                                        let push_candidate = self.metric == DistanceMetric::Dot
                                            || improves_result_set;

                                        if push_candidate {
                                            let sp = NodeCandidate {
                                                idx,
                                                raw_score: raw,
                                                sort_key: score_val,
                                            };
                                            scratch.candidate_queue.push(sp.clone());

                                            if improves_result_set {
                                                scratch.result_set.push(NodeResult(sp));
                                                if scratch.result_set.len() > ef {
                                                    scratch.result_set.pop();
                                                }
                                                if let Some(rp) = scratch.result_set.peek() {
                                                    worst_score = rp.0.sort_key;
                                                }
                                            }
                                        }
                                        if neighbor_patience > 0
                                            && self.metric != DistanceMetric::Dot
                                        {
                                            if improves_result_set {
                                                neighbor_no_improve = 0;
                                            } else {
                                                neighbor_no_improve += 1;
                                                if neighbor_no_improve >= neighbor_patience {
                                                    patience_triggered = true;
                                                    patience_breaks += 1;
                                                    break 'neighbor_scan;
                                                }
                                            }
                                        }
                                    }
                                    batch_len = 0;
                                }
                            }
                            if cap_hit && !patience_triggered && neighbors_examined >= window {
                                cap_breaks += 1;
                            }
                            if !patience_triggered && batch_len > 0 {
                                for i in 0..batch_len {
                                    let idx = batch[i];
                                    distance_computations += 1;
                                    let raw = self.fast_score(query, &self.vectors[idx]);
                                    let score_val = if normalize {
                                        self.normalize_score(raw)
                                    } else {
                                        raw
                                    };

                                    let improves_result_set =
                                        scratch.result_set.len() < ef || score_val < worst_score;
                                    let push_candidate =
                                        self.metric == DistanceMetric::Dot || improves_result_set;

                                    if push_candidate {
                                        let sp = NodeCandidate {
                                            idx,
                                            raw_score: raw,
                                            sort_key: score_val,
                                        };
                                        scratch.candidate_queue.push(sp.clone());

                                        if improves_result_set {
                                            scratch.result_set.push(NodeResult(sp));
                                            if scratch.result_set.len() > ef {
                                                scratch.result_set.pop();
                                            }
                                            if let Some(rp) = scratch.result_set.peek() {
                                                worst_score = rp.0.sort_key;
                                            }
                                        }
                                    }
                                    if neighbor_patience > 0 && self.metric != DistanceMetric::Dot {
                                        if improves_result_set {
                                            neighbor_no_improve = 0;
                                        } else {
                                            neighbor_no_improve += 1;
                                            if neighbor_no_improve >= neighbor_patience {
                                                patience_breaks += 1;
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if self.metric != DistanceMetric::Dot {
                    if let Some(cap) = expansion_cap {
                        if expanded >= cap {
                            stop_reason = "expansion_cap";
                            break;
                        }
                    }
                }

                if let Some(ctx) = trace.as_mut() {
                    let ctx = &mut **ctx;
                    ctx.step += 1;
                    if ctx.step % ctx.every == 0 {
                        let entry = SearchTraceEntry {
                            search_id: ctx.id,
                            metric: format!("{:?}", self.metric),
                            ef_search: ef,
                            exact_fallback_enabled: self.exact_fallback_enabled,
                            exact_fallback_threshold: self.exact_fallback_threshold,
                            collection_size: self.vectors.len(),
                            exact_scan: false,
                            top_id: None,
                            top_raw: None,
                            top_sort_key: None,
                            step: ctx.step,
                            level,
                            current_idx: current.idx,
                            current_point: self.point_id(current.idx),
                            current_sort_key: current.sort_key,
                            visited: visited_count,
                            expanded,
                            candidate_len: scratch.candidate_queue.len(),
                            results_len: scratch.result_set.len(),
                            worst_score,
                            stop_reason: None,
                            elapsed_ms: ctx.start.elapsed().as_secs_f64() * 1000.0,
                        };
                        log_search_trace(&entry);
                    }
                }
            }

            let mut results: Vec<NodeCandidate> = std::mem::take(&mut scratch.result_set)
                .into_iter()
                .map(|rp| rp.0)
                .collect();
            results.sort_by(|a, b| a.sort_key.partial_cmp(&b.sort_key).unwrap());

            if let Some(ctx) = trace.as_mut() {
                let ctx = &mut **ctx;
                let entry = SearchTraceEntry {
                    search_id: ctx.id,
                    metric: format!("{:?}", self.metric),
                    ef_search: ef,
                    exact_fallback_enabled: self.exact_fallback_enabled,
                    exact_fallback_threshold: self.exact_fallback_threshold,
                    collection_size: self.vectors.len(),
                    exact_scan: false,
                    top_id: None,
                    top_raw: None,
                    top_sort_key: None,
                    step: ctx.step,
                    level,
                    current_idx: entry,
                    current_point: self.point_id(entry),
                    current_sort_key: entry_score,
                    visited: visited_count,
                    expanded,
                    candidate_len: scratch.candidate_queue.len(),
                    results_len: results.len(),
                    worst_score,
                    stop_reason: Some(stop_reason.to_string()),
                    elapsed_ms: ctx.start.elapsed().as_secs_f64() * 1000.0,
                };
                log_search_trace(&entry);
            }

            let counters = SearchCounters {
                adjacency_reads,
                distance_computations,
                cap_breaks,
                patience_breaks,
            };
            if let Some(stats) = stats {
                stats.visited = visited_count;
                stats.expanded = expanded;
                stats.adjacency_reads = counters.adjacency_reads;
                stats.distance_computations = counters.distance_computations;
                stats.cap_breaks = counters.cap_breaks;
                stats.patience_breaks = counters.patience_breaks;
            }

            Ok((results, counters))
        })
    }

    pub fn search_with_stats(
        &self,
        query: &Vector,
        top_k: usize,
    ) -> Result<(Vec<ScoredPoint>, SearchStats), DBError> {
        self.search_with_stats_with_options(query, top_k, &SearchRuntimeOptions::default())
    }

    pub fn search_with_stats_with_options(
        &self,
        query: &Vector,
        top_k: usize,
        opts: &SearchRuntimeOptions,
    ) -> Result<(Vec<ScoredPoint>, SearchStats), DBError> {
        if self.entry_point.is_none() {
            return Ok((
                vec![],
                SearchStats {
                    ef_search: opts.ef_search.unwrap_or(top_k).max(top_k),
                    ..SearchStats::default()
                },
            ));
        }
        self.validate_dim(query)?;

        let (normalize_query, normalize_score_flag) = match self.metric {
            DistanceMetric::Cosine => (true, true),
            DistanceMetric::Dot => (false, true),
            DistanceMetric::Euclidean => (false, false),
        };

        let prepared_query = if normalize_query {
            self.maybe_normalize(query)
        } else {
            query.clone()
        };

        let mut trace_ctx = search_trace_logger().map(|_| SearchTraceCtx {
            id: next_search_trace_seq(),
            step: 0,
            every: trace_every(),
            start: Instant::now(),
        });

        let deleted_count = self.deleted.iter().filter(|d| **d).count();
        let collection_size = self.vectors.len().saturating_sub(deleted_count);
        let exact_scan_possible =
            self.exact_fallback_enabled && collection_size <= self.exact_fallback_threshold;
        if let Some(ctx) = trace_ctx.as_mut() {
            let entry = SearchTraceEntry {
                search_id: ctx.id,
                metric: format!("{:?}", self.metric),
                ef_search: top_k,
                exact_fallback_enabled: self.exact_fallback_enabled,
                exact_fallback_threshold: self.exact_fallback_threshold,
                collection_size,
                exact_scan: exact_scan_possible,
                top_id: None,
                top_raw: None,
                top_sort_key: None,
                step: 0,
                level: 0,
                current_idx: 0,
                current_point: 0,
                current_sort_key: 0.0,
                visited: 0,
                expanded: 0,
                candidate_len: 0,
                results_len: 0,
                worst_score: 0.0,
                stop_reason: Some("start".to_string()),
                elapsed_ms: ctx.start.elapsed().as_secs_f64() * 1000.0,
            };
            log_search_trace(&entry);
        }

        if exact_scan_possible {
            let scored = self.exact_scan(&prepared_query, normalize_score_flag, top_k);
            let best = scored.first().map(|r| r.sort_key).unwrap_or(0.0);
            let worst = scored.last().map(|r| r.sort_key).unwrap_or(0.0);
            let top = scored.first();
            if let Some(ctx) = trace_ctx.as_mut() {
                let entry = SearchTraceEntry {
                    search_id: ctx.id,
                    metric: format!("{:?}", self.metric),
                    ef_search: top_k,
                    exact_fallback_enabled: self.exact_fallback_enabled,
                    exact_fallback_threshold: self.exact_fallback_threshold,
                    collection_size,
                    exact_scan: true,
                    top_id: top.map(|r| r.id),
                    top_raw: top.map(|r| r.raw_score),
                    top_sort_key: top.map(|r| r.sort_key),
                    step: 0,
                    level: 0,
                    current_idx: 0,
                    current_point: 0,
                    current_sort_key: 0.0,
                    visited: collection_size,
                    expanded: collection_size,
                    candidate_len: 0,
                    results_len: scored.len(),
                    worst_score: worst,
                    stop_reason: Some("exact_scan".to_string()),
                    elapsed_ms: ctx.start.elapsed().as_secs_f64() * 1000.0,
                };
                log_search_trace(&entry);
            }
            return Ok((
                scored,
                SearchStats {
                    ef_search: top_k,
                    visited: collection_size,
                    expanded: collection_size,
                    best_score: best,
                    worst_score: worst,
                    exact: true,
                    adjacency_reads: 0,
                    distance_computations: 0,
                    cap_breaks: 0,
                    patience_breaks: 0,
                },
            ));
        }

        let query_for_greedy = &prepared_query;
        let mut current = self.entry_point.unwrap();
        for l in (1..=self.current_max_level).rev() {
            current = self.greedy_search_layer_unfiltered(query_for_greedy, current, l);
        }

        let final_query = &prepared_query;
        let ef_search = opts.ef_search.map(|v| v.max(top_k)).unwrap_or_else(|| {
            if self.metric == DistanceMetric::Dot {
                self.vectors.len().max(top_k)
            } else {
                self.ef.max(top_k)
            }
        });

        let mut layer_stats = SearchLayerStats::default();
        let (mut results, _counters) = self.search_layer_unfiltered(
            final_query,
            current,
            0,
            ef_search,
            opts,
            normalize_score_flag,
            Some(&mut layer_stats),
            trace_ctx.as_mut(),
        )?;
        results.sort_by(|a, b| {
            a.sort_key
                .partial_cmp(&b.sort_key)
                .unwrap()
                .then_with(|| self.point_id(a.idx).cmp(&self.point_id(b.idx)))
        });
        results.truncate(top_k);
        let mut scored = results
            .into_iter()
            .map(|cand| ScoredPoint {
                id: self.point_id(cand.idx),
                raw_score: cand.raw_score,
                sort_key: cand.sort_key,
            })
            .collect::<Vec<_>>();
        scored.sort_by(|a, b| {
            a.sort_key
                .partial_cmp(&b.sort_key)
                .unwrap()
                .then_with(|| a.id.cmp(&b.id))
        });
        scored.truncate(top_k);

        let best = scored.first().map(|r| r.sort_key).unwrap_or(0.0);
        let worst = scored.last().map(|r| r.sort_key).unwrap_or(0.0);
        let stats = SearchStats {
            ef_search,
            visited: layer_stats.visited,
            expanded: layer_stats.expanded,
            best_score: best,
            worst_score: worst,
            exact: false,
            adjacency_reads: layer_stats.adjacency_reads,
            distance_computations: layer_stats.distance_computations,
            cap_breaks: layer_stats.cap_breaks,
            patience_breaks: layer_stats.patience_breaks,
        };

        Ok((scored, stats))
    }

    pub fn search(&self, query: &Vector, top_k: usize) -> Result<Vec<ScoredPoint>, DBError> {
        self.search_with_options(query, top_k, &SearchRuntimeOptions::default())
    }

    pub fn search_with_options(
        &self,
        query: &Vector,
        top_k: usize,
        opts: &SearchRuntimeOptions,
    ) -> Result<Vec<ScoredPoint>, DBError> {
        if self.entry_point.is_none() {
            return Ok(vec![]);
        }
        self.validate_dim(query)?;

        let (normalize_query, normalize_score_flag) = match self.metric {
            DistanceMetric::Cosine => (true, true),
            DistanceMetric::Dot => (false, true),
            DistanceMetric::Euclidean => (false, false),
        };

        let prepared_query = if normalize_query {
            self.maybe_normalize(query)
        } else {
            query.clone()
        };

        let mut trace_ctx = search_trace_logger().map(|_| SearchTraceCtx {
            id: next_search_trace_seq(),
            step: 0,
            every: trace_every(),
            start: Instant::now(),
        });

        let deleted_count = self.deleted.iter().filter(|d| **d).count();
        let collection_size = self.vectors.len().saturating_sub(deleted_count);
        let exact_scan_possible =
            self.exact_fallback_enabled && collection_size <= self.exact_fallback_threshold;
        if let Some(ctx) = trace_ctx.as_mut() {
            let entry = SearchTraceEntry {
                search_id: ctx.id,
                metric: format!("{:?}", self.metric),
                ef_search: top_k,
                exact_fallback_enabled: self.exact_fallback_enabled,
                exact_fallback_threshold: self.exact_fallback_threshold,
                collection_size,
                exact_scan: exact_scan_possible,
                top_id: None,
                top_raw: None,
                top_sort_key: None,
                step: 0,
                level: 0,
                current_idx: 0,
                current_point: 0,
                current_sort_key: 0.0,
                visited: 0,
                expanded: 0,
                candidate_len: 0,
                results_len: 0,
                worst_score: 0.0,
                stop_reason: Some("start".to_string()),
                elapsed_ms: ctx.start.elapsed().as_secs_f64() * 1000.0,
            };
            log_search_trace(&entry);
        }

        if exact_scan_possible {
            let scored = self.exact_scan(&prepared_query, normalize_score_flag, top_k);
            let top = scored.first();
            if let Some(ctx) = trace_ctx.as_mut() {
                let worst = scored.last().map(|r| r.sort_key).unwrap_or(0.0);
                let entry = SearchTraceEntry {
                    search_id: ctx.id,
                    metric: format!("{:?}", self.metric),
                    ef_search: top_k,
                    exact_fallback_enabled: self.exact_fallback_enabled,
                    exact_fallback_threshold: self.exact_fallback_threshold,
                    collection_size,
                    exact_scan: true,
                    top_id: top.map(|r| r.id),
                    top_raw: top.map(|r| r.raw_score),
                    top_sort_key: top.map(|r| r.sort_key),
                    step: 0,
                    level: 0,
                    current_idx: 0,
                    current_point: 0,
                    current_sort_key: 0.0,
                    visited: collection_size,
                    expanded: collection_size,
                    candidate_len: 0,
                    results_len: scored.len(),
                    worst_score: worst,
                    stop_reason: Some("exact_scan".to_string()),
                    elapsed_ms: ctx.start.elapsed().as_secs_f64() * 1000.0,
                };
                log_search_trace(&entry);
            }
            return Ok(scored);
        }

        let query_for_greedy = &prepared_query;
        let mut current = self.entry_point.unwrap();
        for l in (1..=self.current_max_level).rev() {
            current = self.greedy_search_layer_unfiltered(query_for_greedy, current, l);
        }

        let final_query = &prepared_query;
        let ef_search = opts.ef_search.map(|v| v.max(top_k)).unwrap_or_else(|| {
            if self.metric == DistanceMetric::Dot {
                self.vectors.len().max(top_k)
            } else {
                self.ef.max(top_k)
            }
        });

        let log_enabled = log_unfiltered_enabled();
        let mut layer_stats = SearchLayerStats::default();
        let (mut results, _counters) = self.search_layer_unfiltered(
            final_query,
            current,
            0,
            ef_search,
            opts,
            normalize_score_flag,
            if log_enabled {
                Some(&mut layer_stats)
            } else {
                None
            },
            trace_ctx.as_mut(),
        )?;
        results.sort_by(|a, b| {
            a.sort_key
                .partial_cmp(&b.sort_key)
                .unwrap()
                .then_with(|| self.point_id(a.idx).cmp(&self.point_id(b.idx)))
        });
        results.truncate(top_k);
        let mut scored = results
            .into_iter()
            .map(|cand| ScoredPoint {
                id: self.point_id(cand.idx),
                raw_score: cand.raw_score,
                sort_key: cand.sort_key,
            })
            .collect::<Vec<_>>();
        scored.sort_by(|a, b| {
            a.sort_key
                .partial_cmp(&b.sort_key)
                .unwrap()
                .then_with(|| a.id.cmp(&b.id))
        });
        scored.truncate(top_k);

        if log_enabled {
            let best = scored.first().map(|r| r.sort_key).unwrap_or(0.0);
            let worst = scored.last().map(|r| r.sort_key).unwrap_or(0.0);
            UNFILTERED_SEARCH_AGG.with(|cell| {
                cell.borrow_mut().record(UnfilteredSample {
                    ef_search,
                    visited: layer_stats.visited,
                    expanded: layer_stats.expanded,
                    best_score: best,
                    worst_score: worst,
                });
            });
        }

        Ok(scored)
    }
}
