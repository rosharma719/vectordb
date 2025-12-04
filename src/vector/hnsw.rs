use std::collections::{BinaryHeap, HashMap, HashSet};
use rand::seq::IteratorRandom;
use rand::Rng;
use crate::utils::types::{PointId, Vector, DistanceMetric, Score};
use crate::utils::errors::DBError;
use crate::payload_storage::stores::PayloadIndex;
use crate::utils::payload::Payload;
use crate::payload_storage::filters::{Filter, evaluate_filter};

const VERBOSE: bool = false;
const DEFAULT_EXACT_FALLBACK_THRESHOLD: usize = 256;

#[derive(Clone, Debug)]
pub struct ScoredPoint {
    pub id: PointId,
    pub raw_score: Score,
    pub sort_key: Score,
}

// This ordering is used for the candidate queue (we want the candidate with the lowest score to be popped first).
impl PartialEq for ScoredPoint {
    fn eq(&self, other: &Self) -> bool {
        self.sort_key == other.sort_key
    }
}

impl Eq for ScoredPoint {}

impl PartialOrd for ScoredPoint {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // Invert the ordering so that lower scores (better) are considered "greater" for the BinaryHeap.
        other.sort_key.partial_cmp(&self.sort_key)
    }
}

impl Ord for ScoredPoint {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

// A wrapper for the result set so that the worst candidate (largest score) is at the top.
#[derive(Clone, Debug, PartialEq)]
struct ResultPoint(ScoredPoint);

impl Eq for ResultPoint {}

impl PartialOrd for ResultPoint {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // Normal ordering: lower score is better, so when used in a max-heap the worst (largest score) will be at the top.
        self.0.sort_key.partial_cmp(&other.0.sort_key)
    }
}

impl Ord for ResultPoint {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.sort_key.partial_cmp(&other.0.sort_key).unwrap()
    }
}

pub struct HNSWIndex {
    layers: HashMap<usize, HashMap<PointId, Vec<PointId>>>,
    vectors: HashMap<PointId, Vector>,
    levels: HashMap<PointId, usize>,
    entry_point: Option<PointId>,
    metric: DistanceMetric,
    m: usize,
    ef: usize,
    ef_construct: usize,
    max_level_cap: usize,
    level_scale: f64,
    current_max_level: usize,
    dim: usize,
    // NEW: Maintain a set of deleted point IDs for lazy deletion
    deleted: HashSet<PointId>,
    exact_fallback_enabled: bool,
    exact_fallback_threshold: usize,
}


impl HNSWIndex {
    fn exact_scan(&self, query: &Vector, normalize_scores: bool, top_k: usize) -> Vec<ScoredPoint> {
        let mut brute: Vec<ScoredPoint> = self
            .iter_vectors()
            .filter_map(|(&id, vec)| {
                if self.deleted.contains(&id) {
                    return None;
                }
                let raw = self.fast_score(query, vec);
                let sort_key = if normalize_scores {
                    self.normalize_score(raw)
                } else {
                    raw
                };
                Some(ScoredPoint { id, raw_score: raw, sort_key })
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
    pub fn new(metric: DistanceMetric, m: usize, ef: usize, max_level_cap: usize, dim: usize) -> Self {
        let level_scale = 1.0 / (m as f64).ln();
        if VERBOSE {
            log::debug!(
                target: "vector::hnsw",
                "Creating new HNSWIndex with dim {}, M {}, ef {}, max_level_cap {}",
                dim,
                m,
                ef,
                max_level_cap
            );
        }
        Self {
            layers: HashMap::new(),
            vectors: HashMap::new(),
            levels: HashMap::new(),
            entry_point: None,
            metric,
            m,
            ef,
            ef_construct: ef,
            max_level_cap,
            level_scale,
            current_max_level: 0,
            dim,
            deleted: HashSet::new(),
            exact_fallback_enabled: true,
            exact_fallback_threshold: DEFAULT_EXACT_FALLBACK_THRESHOLD,
        }
    }

    fn assign_random_level(&self) -> usize {
        let r: f64 = rand::rng().random_range(0.0..1.0);
        let l = (-r.ln() * self.level_scale).floor() as usize;
        let level = l.min(self.max_level_cap);
        level
    }

    pub fn normalize_score(&self, raw: f32) -> f32 {
        match self.metric {
            DistanceMetric::Cosine | DistanceMetric::Euclidean => raw,
            DistanceMetric::Dot => -raw,  // So we can use a min-heap
        }
    }

    #[inline]
    fn fast_score(&self, query: &Vector, vec: &Vector) -> f32 {
        match self.metric {
            // Vectors/queries are normalized in HNSW for cosine; use dot directly.
            DistanceMetric::Cosine => {
                let dot: f32 = query.iter().zip(vec.iter()).map(|(x, y)| x * y).sum();
                1.0 - dot
            }
            DistanceMetric::Dot => query.iter().zip(vec.iter()).map(|(x, y)| x * y).sum(),
            DistanceMetric::Euclidean => {
                query
                    .iter()
                    .zip(vec.iter())
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<f32>()
                    .sqrt()
            }
        }
    }
    
    /// Mark a point as deleted and, if needed, update the entry point.
    pub fn mark_deleted(&mut self, point_id: PointId) {
        self.deleted.insert(point_id);
        // If the deleted point was the entry point, try to choose a new one.
        if Some(point_id) == self.entry_point {
            self.entry_point = self.find_highest_level_entry_point();
        }
        
    }

    pub fn find_highest_level_entry_point(&self) -> Option<PointId> {
        self.levels
            .iter()
            .filter(|(id, _)| !self.deleted.contains(id) && self.vectors.contains_key(id))
            .max_by_key(|(_, level)| *level)
            .map(|(id, _)| *id)
    }
    

    pub fn insert(&mut self, point_id: PointId, vector: Vector) -> Result<(), DBError> {
        //println!("\n[INSERT] Attempting to insert point: {}", point_id);
    
        if self.vectors.contains_key(&point_id) {
            if VERBOSE {
                log::debug!(target: "vector::hnsw", "[INSERT] Point {} already exists. Skipping.", point_id);
            }
            return Ok(());
        }

        if vector.len() != self.dim {
            log::warn!(
                target: "vector::hnsw",
                "[INSERT] Vector length mismatch. Expected {}, got {}.",
                self.dim,
                vector.len()
            );
            return Err(DBError::VectorLengthMismatch {
                expected: self.dim,
                actual: vector.len(),
            });
        }
    
        let level = self.assign_random_level();
        //println!("[INSERT] Assigned random level {} to point {}", level, point_id);
    
        let vec = self.maybe_normalize(&vector);
        self.vectors.insert(point_id, vec);
        self.levels.insert(point_id, level);
    
        // Initialize self-links
        for l in 0..=level {
            self.layers.entry(l).or_default()
                .entry(point_id).or_insert_with(Vec::new)
                .push(point_id);
            //println!("[INSERT] Initialized self-link at level {}", l);
        }
    
        if self.entry_point.is_none() {
            if VERBOSE {
                log::debug!(target: "vector::hnsw", "[INSERT] First point. Setting entry point to {} at level {}", point_id, level);
            }
            self.entry_point = Some(point_id);
            self.current_max_level = level;
            return Ok(());
        }
    
        // If the current entry point is deleted, pick a non-deleted candidate.
        let mut current_entry = if let Some(ep) = self.entry_point {
            if self.deleted.contains(&ep) {
                self.find_highest_level_entry_point().unwrap_or(point_id)
            } else {
                ep
            }
        } else {
            self.find_highest_level_entry_point().unwrap_or(point_id)
        };
        
        
        for l in ((level + 1)..=self.current_max_level).rev() {
            //println!("[INSERT] Greedy search for entry at level {} starting from {}", l, current_entry);
            current_entry = self.greedy_search_layer_unfiltered(&self.vectors[&point_id], current_entry, l);
            //println!("[INSERT] Entry point after greedy search at level {}: {}", l, current_entry);
        }
    
        for l in (0..=level).rev() {
            //println!("[INSERT] Performing search layer at level {}...", l);
            let use_norm = self.metric == DistanceMetric::Cosine || self.metric == DistanceMetric::Dot;
            let candidates = self.search_layer_unfiltered(&self.vectors[&point_id], current_entry, l, self.ef_construct, use_norm)?;
            // Diverse neighbor selection: skip a candidate if it is closer to any already-picked neighbor than to the query.
            let neighbors: Vec<PointId> = self.select_diverse_neighbors(&candidates, self.m, use_norm);
            //println!("[INSERT] Found neighbors at level {} for {}: {:?}", l, point_id, neighbors);

            let layer = self.layers.get_mut(&l).unwrap();
            let mut linked = neighbors.clone();
            if !linked.contains(&point_id) {
                linked.push(point_id);
            }
            layer.insert(point_id, linked.clone());
    
            for &n in &neighbors {
                let e = layer.entry(n).or_default();
                if !e.contains(&point_id) {
                    e.push(point_id);
                }
            }
    
            if let Some(&best) = neighbors.first() {
                current_entry = best;
            }
        }
    
        if level > self.current_max_level {
            if VERBOSE {
                log::debug!(target: "vector::hnsw", "[INSERT] Promoting {} to new entry point at level {}", point_id, level);
            }
            self.entry_point = Some(point_id);
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
        payloads: &HashMap<PointId, Payload>,
        filter_keys: &[String],
    ) -> Result<(), DBError> {
        // Skip if no payload keys were provided; avoids O(N) fallback when payloads are absent.
        if filter_keys.is_empty() {
            return Ok(());
        }
        let query_vector = if self.metric == DistanceMetric::Cosine {
            self.maybe_normalize(vector)
        } else {
            vector.clone()
        };
    
        let mut extra_neighbors = HashSet::new();
        let m = self.m();
        // Optional toggle: skip the fallback graph search to speed up inserts.
        let allow_fallback = std::env::var("VECTORDB_FILTER_EDGE_FALLBACK")
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(true);
    
        // Limit how many candidates we sample from the inverted index per key; tie to construction beam.
        let sample_limit: usize = self.ef_construct.max(self.m());

        for key in filter_keys {
            if let Some(value) = payload.get(key) {
                // ✅ Try fast exact match via payload index first
                if let Some(id_set) = payload_index.query_exact(key, value) {
                    let mut rng = rand::rng();
                    // If the posting list is small, score all of it; otherwise take a random sample.
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
                                ScoredPoint { id, raw_score: raw, sort_key }
                            })
                        })
                        .collect();

                    scored.sort_by(|a, b| {
                        if self.metric == DistanceMetric::Dot {
                            b.raw_score.partial_cmp(&a.raw_score).unwrap()
                        } else {
                            a.raw_score.partial_cmp(&b.raw_score).unwrap()
                        }
                    });

                    for sp in scored.into_iter().take(m) {
                        extra_neighbors.insert(sp.id);
                    }

                // If the inverted-index sample already filled the desired neighbor budget for this key,
                // skip the fallback search but keep checking other keys.
                if extra_neighbors.len() >= m {
                    continue;
                }
                }

                // ⛔ If the posting list was empty, fall back to unfiltered search to find nearby vectors
                // and then filter them by this key/value to establish some connectivity.
                if allow_fallback {
                    let mut candidates: Vec<ScoredPoint> = if self.current_max_level() > 0 {
                        let mut entry = self.get_entry_point().unwrap();
                        for l in (1..=self.current_max_level()).rev() {
                            entry = self.greedy_search_layer_unfiltered(&query_vector, entry, l);
                        }
                        self.search_layer_unfiltered(
                            &query_vector,
                            entry,
                            0,
                            self.ef(),
                            self.metric == DistanceMetric::Cosine || self.metric == DistanceMetric::Dot,
                        )?
                    } else {
                        self.iter_vectors()
                            .filter_map(|(&id, vec)| {
                                if id != point_id && !self.deleted.contains(&id) {
                                    let raw = self.fast_score(&query_vector, vec);
                                    Some(ScoredPoint {
                                        id,
                                        raw_score: raw,
                                        sort_key: self.normalize_score(raw),
                                    })
                                } else {
                                    None
                                }
                            })
                            .collect()
                    };

                    candidates.sort_by(|a, b| {
                        if self.metric == DistanceMetric::Dot {
                            b.raw_score.partial_cmp(&a.raw_score).unwrap()
                        } else {
                            a.raw_score.partial_cmp(&b.raw_score).unwrap()
                        }
                    });

                    let filtered: Vec<_> = candidates
                        .into_iter()
                        .filter(|sp| {
                            payloads
                                .get(&sp.id)
                                .and_then(|p| p.get(key))
                                .map_or(false, |v| v == value)
                        })
                        .take(m)
                        .map(|sp| sp.id)
                        .collect();

                    extra_neighbors.extend(filtered);
                }

                if extra_neighbors.len() >= 2 * m {
                    break;
                }
            }
        }
    
        extra_neighbors.insert(point_id); // self-loop
    
        for neighbor_id in extra_neighbors {
            self.add_bidirectional_edge(0, point_id, neighbor_id);
        }
    
        Ok(())
    }
    

    pub fn add_bidirectional_edge(&mut self, level: usize, a: PointId, b: PointId) {
        let layer = self.layers.entry(level).or_default();
        Self::push_unique(layer.entry(a).or_default(), b);
        Self::push_unique(layer.entry(b).or_default(), a);
    }

    #[inline]
    fn push_unique(vec: &mut Vec<PointId>, val: PointId) {
        if !vec.contains(&val) {
            vec.push(val);
        }
    }

    pub fn greedy_search_layer_unfiltered(&self, query: &Vector, entry: PointId, level: usize) -> PointId {
        //println!("[GREEDY] Start at level {}, from entry {}", level, entry);
        let mut current = entry;
        let mut changed = true;
        let mut steps = 0;
    
        while changed && steps < 1000 {
            steps += 1;
            changed = false;
            if let Some(neighbors) = self.layers.get(&level).and_then(|l| l.get(&current)) {
                for &neighbor in neighbors {
                    if self.deleted.contains(&neighbor) {
                        continue;
                    }
    
                    let d_current = self.fast_score(query, &self.vectors[&current]);
                    let d_new = self.fast_score(query, &self.vectors[&neighbor]);
                    let s_current = self.normalize_score(d_current);
                    let s_new = self.normalize_score(d_new);
    
                    if s_new < s_current {
                        current = neighbor;
                        changed = true;
                        break; // exit early if we move
                    }
                }
            }
        }
    
        if steps >= 1000 {
            log::warn!(target: "vector::hnsw", "[GREEDY] Reached max steps at level {}, current = {}", level, current);
        }
    
        //println!("[GREEDY] Finished at point {} at level {}", current, level);
        current
    }
        
    pub fn greedy_search_layer_with_filter(
        &self,
        query: &Vector,
        entry: PointId,
        level: usize,
        payloads: &HashMap<PointId, Payload>,
        filter: Option<&Filter>,
    ) -> Result<PointId, DBError> {
        let mut current = entry;
        let mut changed = true;

        while changed {
            changed = false;

            if let Some(neighbors) = self.layer_neighbors(level, current) {
                for &neighbor in neighbors {
                    if self.deleted.contains(&neighbor) {
                        continue;
                    }

                    if let Some(f) = filter {
                        let Some(payload) = payloads.get(&neighbor) else { continue; };
                        if !evaluate_filter(f, payload)? {
                            continue;
                        }
                    }

                    let d_current = self.fast_score(query, self.get_vector(&current).unwrap());
                    let d_new = self.fast_score(query, self.get_vector(&neighbor).unwrap());

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
    
    fn search_layer_unfiltered(
        &self,
        query: &Vector,
        entry: PointId,
        level: usize,
        ef: usize,
        normalize: bool,
    ) -> Result<Vec<ScoredPoint>, DBError> {
        if query.len() != self.dim {
            return Err(DBError::VectorLengthMismatch {
                expected: self.dim,
                actual: query.len(),
            });
        }
    
        let mut visited = HashSet::new();
        let mut candidate_queue = BinaryHeap::new();
        let mut result_set = BinaryHeap::new();
    
        // If the entry is deleted, skip it by choosing a non-deleted vector (if possible)
        let start_entry = if self.deleted.contains(&entry) {
            self.vectors.keys().find(|&&id| !self.deleted.contains(&id)).cloned().unwrap_or(entry)
        } else {
            entry
        };
    
        let entry_distance = self.fast_score(query, &self.vectors[&start_entry]);
        let entry_score = if normalize {
            self.normalize_score(entry_distance)
        } else {
            entry_distance
        };
    
        let initial = ScoredPoint {
            id: start_entry,
            raw_score: entry_distance,
            sort_key: entry_score,
        };
    
        candidate_queue.push(initial.clone());
        result_set.push(ResultPoint(initial.clone()));
        visited.insert(start_entry);
    
        //println!("[search_layer_unfiltered] Initial score at entry {}: {:.4}",start_entry, entry_score);
    
        let mut worst_score = result_set.peek().unwrap().0.sort_key;
        let allow_early_exit = self.metric != DistanceMetric::Dot;
    
        while let Some(current) = candidate_queue.peek() {
            if allow_early_exit && current.sort_key > worst_score {
                break;
            }
    
            let current = candidate_queue.pop().unwrap();
            if let Some(neighbors) = self.layers.get(&level).and_then(|l| l.get(&current.id)) {
                for &neighbor in neighbors {
                    if self.deleted.contains(&neighbor) || !visited.insert(neighbor) {
                        continue;
                    }
    
                    let raw = self.fast_score(query, &self.vectors[&neighbor]);
                    let score_val = if normalize {
                        self.normalize_score(raw)
                    } else {
                        raw
                    };

                    // For Dot we explore broadly (push all neighbors) to avoid getting stuck in local optima.
                    let push_candidate = self.metric == DistanceMetric::Dot
                        || result_set.len() < ef
                        || score_val < worst_score;

                    if push_candidate {
                        let sp = ScoredPoint {
                            id: neighbor,
                            raw_score: raw,
                            sort_key: score_val,
                        };
                        candidate_queue.push(sp.clone());

                        // Still keep result_set trimmed to ef best items.
                        if result_set.len() < ef || score_val < worst_score {
                            result_set.push(ResultPoint(sp));
                            if result_set.len() > ef {
                                result_set.pop();
                            }
                            if let Some(rp) = result_set.peek() {
                                worst_score = rp.0.sort_key;
                            }
                        }
                    }
                }
            }
        }
    
        let mut results: Vec<ScoredPoint> = result_set.into_iter().map(|rp| rp.0).collect();
        results.sort_by(|a, b| a.sort_key.partial_cmp(&b.sort_key).unwrap());
    
        //println!("[search_layer_unfiltered] Done. Returning top {} results: {:?}",results.len(),results.iter().map(|sp| sp.id).collect::<Vec<_>>());
    
        Ok(results)
    }
       
    pub fn search(&self, query: &Vector, top_k: usize) -> Result<Vec<ScoredPoint>, DBError> {
        //println!("Searching top_k = {}", top_k);
        if self.entry_point.is_none() {
            //println!("No entry point. Returning empty result.");
            return Ok(vec![]);
        }
        if query.len() != self.dim {
            return Err(DBError::VectorLengthMismatch {
                expected: self.dim,
                actual: query.len(),
            });
        }
        
        let (normalize_query, normalize_score_flag) = match self.metric {
            DistanceMetric::Cosine => (true, true),
            DistanceMetric::Dot => (false, true), // invert score but don’t normalize vec
            DistanceMetric::Euclidean => (false, false),
        };
        
        // Normalize once when needed and reuse to avoid duplicate work per query.
        let prepared_query = if normalize_query {
            self.maybe_normalize(query)
        } else {
            query.clone()
        };

        let collection_size = self.vectors.len() - self.deleted.len();
        let exact_scan_possible = self.exact_fallback_enabled
            && collection_size <= self.exact_fallback_threshold;

        // For small collections, optionally fall back to an exact scan for deterministic results.
        if exact_scan_possible {
            return Ok(self.exact_scan(&prepared_query, normalize_score_flag, top_k));
        }

        let query_for_greedy = &prepared_query;
        let mut current = self.entry_point.unwrap();
        for l in (1..=self.current_max_level).rev() {
            current = self.greedy_search_layer_unfiltered(query_for_greedy, current, l);
        }
        
        let final_query = &prepared_query;
        // Ensure beam width is at least top_k to avoid recall loss when top_k > ef.
        // Dot similarity benefits from a very wide beam to avoid local optima.
        let ef_search = if self.metric == DistanceMetric::Dot {
            self.vectors.len().max(top_k)
        } else {
            self.ef.max(top_k)
        };

        let mut results = self.search_layer_unfiltered(final_query, current, 0, ef_search, normalize_score_flag)?;
        results.sort_by(|a, b| {
            a.sort_key
                .partial_cmp(&b.sort_key)
                .unwrap()
                .then_with(|| a.id.cmp(&b.id))
        });
        results.truncate(top_k);
        Ok(results)
    }

    pub fn find_entry_point_matching_filter(
        &self,
        filter: &Filter,
        payload_index: &PayloadIndex,
        payloads: &HashMap<PointId, Payload>,
    ) -> Option<PointId> {
        fn max_level_point<'a>(
            ids: impl Iterator<Item = &'a PointId>,
            levels: &HashMap<PointId, usize>,
            deleted: &HashSet<PointId>,
            vectors: &HashMap<PointId, Vector>,
        ) -> Option<PointId> {
            ids.filter(|&&id| !deleted.contains(&id) && vectors.contains_key(&id))
                .max_by_key(|&&id| levels.get(&id).copied().unwrap_or(0))
                .copied()
        }
    
        match filter {
            Filter::Match { key, value } => {
                payload_index
                    .query_exact(key, value)
                    .and_then(|ids| {
                        max_level_point(ids.iter(), &self.levels, &self.deleted, &self.vectors)
                    })
            }
            Filter::And(conds) | Filter::Or(conds) => {
                let mut best: Option<(PointId, usize)> = None;
    
                for cond in conds {
                    if let Some(id) = self.find_entry_point_matching_filter(cond, payload_index, payloads) {
                        let level = self.levels.get(&id).copied().unwrap_or(0);
                        if best.map_or(true, |(_, l)| level > l) {
                            best = Some((id, level));
                        }
                    }
                }
    
                best.map(|(id, _)| id)
            }
            Filter::Not(inner) => self.find_entry_point_matching_filter(inner, payload_index, payloads),
            Filter::Compare { .. } => None,
        }
    }
    

    fn extract_match_filter(&self, orig: Option<&Filter>) -> Option<Filter> {
        match orig {
            None => None,
            Some(Filter::Match { key, value }) => {
                Some(Filter::Match { key: key.clone(), value: value.clone() })
            }
            Some(Filter::And(conds)) => {
                let mut keep = Vec::new();
                for c in conds {
                    if let Filter::Match { .. } = c {
                        keep.push(c.clone());
                    }
                }
                match keep.len() {
                    0 => None,
                    1 => Some(keep.into_iter().next().unwrap()),
                    _ => Some(Filter::And(keep)),
                }
            }
            // drop everything else
            _ => None,
        }
    }

    /// REPLACEMENT for your old in_place_filtered_search:
    pub fn in_place_filtered_search(
        &self,
        query: &Vector,
        top_k: usize,
        payloads: &HashMap<PointId, Payload>,
        payload_index: &PayloadIndex,
        full_filter: Option<&Filter>,
    ) -> Result<Vec<ScoredPoint>, DBError> {
        if query.len() != self.dim {
            return Err(DBError::VectorLengthMismatch {
                expected: self.dim,
                actual: query.len(),
            });
        }

        let (normalize_query, use_normalize_score) = match self.metric {
            DistanceMetric::Cosine => (true, true),
            DistanceMetric::Dot => (false, true), // invert score but don’t normalize vec
            DistanceMetric::Euclidean => (false, false),
        };

        // Normalize once when needed and reuse for both greedy descent and ef search.
        let prepared_query = if normalize_query {
            self.maybe_normalize(query)
        } else {
            query.clone()
        };

        let query_for_greedy = &prepared_query;
        let query_for_search = &prepared_query;

        // 1) carve out only the Match clauses for graph‐hopping
        let match_filter = self.extract_match_filter(full_filter);

        // 2) pick the entry point exactly as before
        let mut entry = match self.get_entry_point() {
            Some(id) => {
                if let Some(f) = full_filter {
                    if let Some(p) = payloads.get(&id) {
                        if evaluate_filter(f, p)? {
                            id
                        } else {
                            self.find_entry_point_matching_filter(f, payload_index, payloads)
                                .unwrap_or(id)
                        }
                    } else {
                        self.find_entry_point_matching_filter(f, payload_index, payloads)
                            .unwrap_or(id)
                    }
                } else {
                    id
                }
            }
            None => {
                if let Some(f) = full_filter {
                    match self.find_entry_point_matching_filter(f, payload_index, payloads) {
                        Some(id) => id,
                        None => return Ok(vec![]),
                    }
                } else {
                    return Ok(vec![]);
                }
            }
        };

        // 3) greedy‐search down from top level *using only* the equality edges
        for level in (1..=self.current_max_level()).rev() {
            entry = self.greedy_search_layer_with_filter(
                &query_for_greedy,
                entry,
                level,
                payloads,
                match_filter.as_ref(),
            )?;
        }

        // 4) now do the ef‐search on level 0, but *only* apply the full filter
        //    at the moment we push into the result‐heap:
        let mut visited = HashSet::new();
        let mut candidate_queue: BinaryHeap<ScoredPoint> = BinaryHeap::new();
        let mut result_set:  BinaryHeap<ResultPoint>    = BinaryHeap::new();

        let dist0 = self.fast_score(&query_for_search, self.get_vector(&entry).unwrap());
        let sk0   = if use_normalize_score { self.normalize_score(dist0) } else { dist0 };
        let first = ScoredPoint { id: entry, raw_score: dist0, sort_key: sk0 };

        candidate_queue.push(first.clone());
        // only push into result_set if it passes the *full* filter:
        if full_filter.map_or(true, |f| {
            payloads.get(&entry).map_or(false, |p| evaluate_filter(f, p).unwrap_or(false))
        }) {
            result_set.push(ResultPoint(first));
        }
        visited.insert(entry);

        // Seed the beam with a few IDs pulled from the inverted index that match equality filters.
        // This helps reach filtered regions even when the graph lacks filter-aware edges.
        let seed_limit = self.ef;
        let mut seed_ids = HashSet::new();
        fn collect_match_ids(filter: &Filter, idx: &PayloadIndex, out: &mut HashSet<PointId>) {
            match filter {
                Filter::Match { key, value } => {
                    if let Some(ids) = idx.query_exact(key, value) {
                        out.extend(ids.iter().copied());
                    }
                }
                Filter::And(parts) => {
                    // For AND, intersect the matches to keep the seed set tight.
                    let mut local: Option<HashSet<PointId>> = None;
                    for p in parts {
                        let mut subset = HashSet::new();
                        collect_match_ids(p, idx, &mut subset);
                        if let Some(acc) = &mut local {
                            acc.retain(|id| subset.contains(id));
                        } else {
                            local = Some(subset);
                        }
                    }
                    if let Some(s) = local {
                        out.extend(s);
                    }
                }
                Filter::Or(parts) => {
                    for p in parts {
                        collect_match_ids(p, idx, out);
                    }
                }
                Filter::Not(_) | Filter::Compare { .. } => {}
            }
        }
        if let Some(f) = full_filter {
            collect_match_ids(f, payload_index, &mut seed_ids);
        }
        if !seed_ids.is_empty() {
            let mut added = 0usize;
            for id in seed_ids {
                if added >= seed_limit {
                    break;
                }
                if self.deleted.contains(&id) || !visited.insert(id) {
                    continue;
                }
                let Some(vec) = self.get_vector(&id) else { continue; };
                let d = self.fast_score(&query_for_search, vec);
                let sk = if use_normalize_score { self.normalize_score(d) } else { d };
                let sp = ScoredPoint { id, raw_score: d, sort_key: sk };
                candidate_queue.push(sp.clone());
                if full_filter.map_or(true, |f| {
                    payloads.get(&id).map_or(false, |p| evaluate_filter(f, p).unwrap_or(false))
                }) {
                    result_set.push(ResultPoint(sp));
                }
                added += 1;
            }
        }

        let mut worst = result_set.peek().map(|rp| rp.0.sort_key).unwrap_or(f32::MAX);

        let max_expansions = self.ef.saturating_mul(4);
        let mut expansions = 0usize;

        while let Some(curr) = candidate_queue.pop() {
            expansions += 1;
            if expansions >= max_expansions {
                break;
            }
            // Once we already have top_k passing the filter, stop exploring candidates
            // that are worse than our current worst. This restores the usual early-exit
            // even when a filter is present.
            if result_set.len() >= top_k && curr.sort_key > worst {
                break;
            }

            if let Some(neighs) = self.layer_neighbors(0, curr.id) {
                for &nb in neighs {
                    if self.deleted.contains(&nb) || !visited.insert(nb) {
                        continue;
                    }

                    let d   = self.fast_score(&query_for_search, self.get_vector(&nb).unwrap());
                    let sk  = if use_normalize_score { self.normalize_score(d) } else { d };
                    let sp  = ScoredPoint { id: nb, raw_score: d, sort_key: sk };

                    candidate_queue.push(sp.clone());

                    // *now* apply the full filter before pushing to result_set
                    if full_filter.map_or(true, |f| {
                        payloads.get(&nb).map_or(false, |p| evaluate_filter(f, p).unwrap_or(false))
                    }) {
                        result_set.push(ResultPoint(sp));
                        if result_set.len() > self.ef {
                            result_set.pop();
                        }
                        if result_set.len() >= top_k {
                            worst = result_set.peek().unwrap().0.sort_key;
                        }
                    }
                }
            }
        }

        // unwrap the inner ScoredPoint and truncate to top_k
        let mut out = result_set
            .into_sorted_vec()
            .into_iter()
            .map(|rp| rp.0)
            .collect::<Vec<_>>();
        out.truncate(top_k);
        Ok(out)
    }

    /// Heuristic neighbor selector that enforces diversity (HNSW heuristic 2).
    fn select_diverse_neighbors(&self, candidates: &[ScoredPoint], m: usize, normalize_scores: bool) -> Vec<PointId> {
        let mut result = Vec::with_capacity(m);
        for cand in candidates {
            if result.len() >= m {
                break;
            }
            let Some(cand_vec) = self.get_vector(&cand.id) else { continue; };
            let d_qc = cand.sort_key; // already normalized when requested
            let mut too_close = false;
            for &r_id in &result {
                let Some(r_vec) = self.get_vector(&r_id) else { continue; };
                let d_cr_raw = self.fast_score(cand_vec, r_vec);
                let d_cr = if normalize_scores { self.normalize_score(d_cr_raw) } else { d_cr_raw };
                if d_cr < d_qc {
                    too_close = true;
                    break;
                }
            }
            if !too_close {
                result.push(cand.id);
            }
        }
        // If we selected fewer than m due to the diversity filter, backfill with remaining closest candidates.
        if result.len() < m {
            for cand in candidates {
                if result.len() >= m {
                    break;
                }
                if !result.contains(&cand.id) {
                    result.push(cand.id);
                }
            }
        }
        result
    }
    
    
    pub fn contains(&self, point_id: &PointId) -> bool {
        self.vectors.contains_key(point_id)
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn layer_neighbors(&self, level: usize, point_id: PointId) -> Option<&Vec<PointId>> {
        self.layers.get(&level)?.get(&point_id)
    }

    pub fn iter_vectors(&self) -> impl Iterator<Item = (&PointId, &Vector)> {
        self.vectors.iter()
    }

    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    pub fn m(&self) -> usize {
        self.m
    }

    pub fn ef(&self) -> usize {
        self.ef
    }

    pub fn set_ef_construct(&mut self, ef: usize) {
        self.ef_construct = ef;
    }

    pub fn set_ef_search(&mut self, ef: usize) {
        self.ef = ef;
    }

    pub fn max_level_cap(&self) -> usize {
        self.max_level_cap
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn get_vector(&self, point_id: &PointId) -> Option<&Vector> {
        // Optionally, one might return None for deleted points.
        if self.deleted.contains(point_id) {
            None
        } else {
            self.vectors.get(point_id)
        }
    }

    pub fn get_entry_point(&self) -> Option<u64> {
        self.entry_point
    }

    pub fn current_max_level(&self) -> usize {
        self.current_max_level
    }

    pub fn set_entry_point(&mut self, point_id: PointId) {
        self.entry_point = Some(point_id);
    }

    pub fn set_current_max_level(&mut self, level: usize) {
        self.current_max_level = level;
    }

    pub fn set_exact_fallback_enabled(&mut self, enabled: bool) {
        self.exact_fallback_enabled = enabled;
    }

    pub fn set_exact_fallback_threshold(&mut self, threshold: usize) {
        self.exact_fallback_threshold = threshold;
    }

    pub fn maybe_normalize(&self, vec: &Vector) -> Vector {
        match self.metric {
            DistanceMetric::Cosine => {
                let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm == 0.0 {
                    vec.clone()
                } else {
                    vec.iter().map(|x| x / norm).collect()
                }
            }
            _ => vec.clone(),
        }
    }
}
