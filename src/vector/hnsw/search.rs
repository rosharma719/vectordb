use crate::utils::errors::DBError;
use crate::utils::types::{DistanceMetric, Vector};

use super::config::{disable_early_exit, early_exit_patience, log_unfiltered_enabled, search_expansion_cap_override, search_expansion_multiplier};
use super::scratch::SEARCH_SCRATCH;
use super::stats::{SearchLayerStats, SearchStats, UnfilteredSample, UNFILTERED_SEARCH_AGG};
use super::types::{NodeCandidate, NodeResult, ScoredPoint};
use super::HNSWIndex;

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
        normalize: bool,
        stats: Option<&mut SearchLayerStats>,
    ) -> Result<Vec<NodeCandidate>, DBError> {
        self.validate_dim(query)?;

        SEARCH_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            scratch.next_epoch(self.vectors.len());
            scratch.candidate_queue.clear();
            scratch.result_set.clear();

            let mut visited_count = 0usize;
            let mut expanded = 0usize;
            let expansion_cap = search_expansion_cap_override()
                .or_else(|| Some(ef.saturating_mul(search_expansion_multiplier()).max(ef)));

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
            let patience_limit = if allow_early_exit { early_exit_patience() } else { 0 };
            let mut no_improve_streak = 0usize;

            while let Some(current) = scratch.candidate_queue.peek() {
                if allow_early_exit && scratch.result_set.len() >= ef {
                    if current.sort_key > worst_score {
                        no_improve_streak += 1;
                    } else {
                        no_improve_streak = 0;
                    }
                    if no_improve_streak > patience_limit {
                        break;
                    }
                }

                let current = scratch.candidate_queue.pop().unwrap();
                expanded += 1;
                if let Some(neighbors) = self.layers.get(level).and_then(|l| l.get(current.idx)) {
                    const BATCH: usize = 16;
                    let mut batch = [0usize; BATCH];
                    let mut batch_len = 0usize;

                    for &neighbor in neighbors {
                        if self.deleted.get(neighbor).copied().unwrap_or(false) || !scratch.mark_visited(neighbor) {
                            continue;
                        }
                        visited_count += 1;

                        batch[batch_len] = neighbor;
                        batch_len += 1;
                        if batch_len == BATCH {
                            for i in 0..batch_len {
                                let idx = batch[i];
                                let raw = self.fast_score(query, &self.vectors[idx]);
                                let score_val = if normalize {
                                    self.normalize_score(raw)
                                } else {
                                    raw
                                };

                                let push_candidate = self.metric == DistanceMetric::Dot
                                    || scratch.result_set.len() < ef
                                    || score_val < worst_score;

                                if push_candidate {
                                    let sp = NodeCandidate {
                                        idx,
                                        raw_score: raw,
                                        sort_key: score_val,
                                    };
                                    scratch.candidate_queue.push(sp.clone());

                                    if scratch.result_set.len() < ef || score_val < worst_score {
                                        scratch.result_set.push(NodeResult(sp));
                                        if scratch.result_set.len() > ef {
                                            scratch.result_set.pop();
                                        }
                                        if let Some(rp) = scratch.result_set.peek() {
                                            worst_score = rp.0.sort_key;
                                        }
                                    }
                                }
                            }
                            batch_len = 0;
                        }
                    }
                    if batch_len > 0 {
                        for i in 0..batch_len {
                            let idx = batch[i];
                            let raw = self.fast_score(query, &self.vectors[idx]);
                            let score_val = if normalize {
                                self.normalize_score(raw)
                            } else {
                                raw
                            };

                            let push_candidate = self.metric == DistanceMetric::Dot
                                || scratch.result_set.len() < ef
                                || score_val < worst_score;

                            if push_candidate {
                                let sp = NodeCandidate {
                                    idx,
                                    raw_score: raw,
                                    sort_key: score_val,
                                };
                                scratch.candidate_queue.push(sp.clone());

                                if scratch.result_set.len() < ef || score_val < worst_score {
                                    scratch.result_set.push(NodeResult(sp));
                                    if scratch.result_set.len() > ef {
                                        scratch.result_set.pop();
                                    }
                                    if let Some(rp) = scratch.result_set.peek() {
                                        worst_score = rp.0.sort_key;
                                    }
                                }
                            }
                        }
                    }
                }

                if self.metric != DistanceMetric::Dot {
                    if let Some(cap) = expansion_cap {
                        if expanded >= cap {
                            break;
                        }
                    }
                }
            }

            let mut results: Vec<NodeCandidate> = std::mem::take(&mut scratch.result_set)
                .into_iter()
                .map(|rp| rp.0)
                .collect();
            results.sort_by(|a, b| a.sort_key.partial_cmp(&b.sort_key).unwrap());

            if let Some(stats) = stats {
                stats.visited = visited_count;
                stats.expanded = expanded;
            }

            Ok(results)
        })
    }

    pub fn search_with_stats(
        &self,
        query: &Vector,
        top_k: usize,
    ) -> Result<(Vec<ScoredPoint>, SearchStats), DBError> {
        if self.entry_point.is_none() {
            return Ok((vec![], SearchStats { ef_search: top_k, ..SearchStats::default() }));
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

        let deleted_count = self.deleted.iter().filter(|d| **d).count();
        let collection_size = self.vectors.len().saturating_sub(deleted_count);
        let exact_scan_possible = self.exact_fallback_enabled
            && collection_size <= self.exact_fallback_threshold;

        if exact_scan_possible {
            let scored = self.exact_scan(&prepared_query, normalize_score_flag, top_k);
            let best = scored.first().map(|r| r.sort_key).unwrap_or(0.0);
            let worst = scored.last().map(|r| r.sort_key).unwrap_or(0.0);
            return Ok((
                scored,
                SearchStats {
                    ef_search: top_k,
                    visited: collection_size,
                    expanded: collection_size,
                    best_score: best,
                    worst_score: worst,
                    exact: true,
                },
            ));
        }

        let query_for_greedy = &prepared_query;
        let mut current = self.entry_point.unwrap();
        for l in (1..=self.current_max_level).rev() {
            current = self.greedy_search_layer_unfiltered(query_for_greedy, current, l);
        }

        let final_query = &prepared_query;
        let ef_search = if self.metric == DistanceMetric::Dot {
            self.vectors.len().max(top_k)
        } else {
            self.ef.max(top_k)
        };

        let mut layer_stats = SearchLayerStats::default();
        let mut results = self.search_layer_unfiltered(
            final_query,
            current,
            0,
            ef_search,
            normalize_score_flag,
            Some(&mut layer_stats),
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
        };

        Ok((scored, stats))
    }

    pub fn search(&self, query: &Vector, top_k: usize) -> Result<Vec<ScoredPoint>, DBError> {
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

        let deleted_count = self.deleted.iter().filter(|d| **d).count();
        let collection_size = self.vectors.len().saturating_sub(deleted_count);
        let exact_scan_possible = self.exact_fallback_enabled
            && collection_size <= self.exact_fallback_threshold;

        if exact_scan_possible {
            return Ok(self.exact_scan(&prepared_query, normalize_score_flag, top_k));
        }

        let query_for_greedy = &prepared_query;
        let mut current = self.entry_point.unwrap();
        for l in (1..=self.current_max_level).rev() {
            current = self.greedy_search_layer_unfiltered(query_for_greedy, current, l);
        }

        let final_query = &prepared_query;
        let ef_search = if self.metric == DistanceMetric::Dot {
            self.vectors.len().max(top_k)
        } else {
            self.ef.max(top_k)
        };

        let log_enabled = log_unfiltered_enabled();
        let mut layer_stats = SearchLayerStats::default();
        let mut results = self.search_layer_unfiltered(
            final_query,
            current,
            0,
            ef_search,
            normalize_score_flag,
            if log_enabled { Some(&mut layer_stats) } else { None },
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
