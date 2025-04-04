use std::collections::{BinaryHeap, HashMap, HashSet};
use rand::seq::IteratorRandom;
use rand::Rng;
use crate::utils::types::{PointId, Vector, DistanceMetric, Score};
use crate::vector::metric::score;
use crate::utils::errors::DBError;
use crate::payload_storage::stores::PayloadIndex;
use crate::utils::payload::Payload;

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
    max_level_cap: usize,
    level_scale: f64,
    current_max_level: usize,
    dim: usize,
    // NEW: Maintain a set of deleted point IDs for lazy deletion
    deleted: HashSet<PointId>,
}


impl HNSWIndex {
    pub fn new(metric: DistanceMetric, m: usize, ef: usize, max_level_cap: usize, dim: usize) -> Self {
        let level_scale = 1.0 / (m as f64).ln();
        println!("Creating new HNSWIndex with dim {}, M {}, ef {}, max_level_cap {}", dim, m, ef, max_level_cap);
        Self {
            layers: HashMap::new(),
            vectors: HashMap::new(),
            levels: HashMap::new(),
            entry_point: None,
            metric,
            m,
            ef,
            max_level_cap,
            level_scale,
            current_max_level: 0,
            dim,
            deleted: HashSet::new(),
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
    
    /// Mark a point as deleted and, if needed, update the entry point.
    pub fn mark_deleted(&mut self, point_id: PointId) {
        self.deleted.insert(point_id);
        // If the deleted point was the entry point, try to choose a new one.
        if Some(point_id) == self.entry_point {
            self.entry_point = self.vectors.keys().find(|&&id| !self.deleted.contains(&id)).cloned();
        }
    }

    pub fn insert(&mut self, point_id: PointId, vector: Vector) -> Result<(), DBError> {
        //println!("\n[INSERT] Attempting to insert point: {}", point_id);
    
        if self.vectors.contains_key(&point_id) {
            println!("[INSERT] Point {} already exists. Skipping.", point_id);
            return Ok(());
        }
    
        if vector.len() != self.dim {
            println!("[INSERT] Vector length mismatch. Expected {}, got {}.", self.dim, vector.len());
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
            println!("[INSERT] First point. Setting entry point to {} at level {}", point_id, level);
            self.entry_point = Some(point_id);
            self.current_max_level = level;
            return Ok(());
        }
    
        // If the current entry point is deleted, pick a non-deleted candidate.
        let mut current_entry = if let Some(ep) = self.entry_point {
            if self.deleted.contains(&ep) {
                // Find a non-deleted entry in the index.
                self.vectors.keys().find(|&&id| !self.deleted.contains(&id)).cloned().unwrap_or(ep)
            } else {
                ep
            }
        } else {
            point_id
        };
        
        for l in ((level + 1)..=self.current_max_level).rev() {
            //println!("[INSERT] Greedy search for entry at level {} starting from {}", l, current_entry);
            current_entry = self.greedy_search_layer(&self.vectors[&point_id], current_entry, l);
            //println!("[INSERT] Entry point after greedy search at level {}: {}", l, current_entry);
        }
    
        for l in (0..=level).rev() {
            //println!("[INSERT] Performing search layer at level {}...", l);
            let use_norm = self.metric == DistanceMetric::Cosine || self.metric == DistanceMetric::Dot;
            let candidates = self.search_layer(&self.vectors[&point_id], current_entry, l, self.ef, use_norm)?;
            let neighbors: Vec<PointId> = candidates.iter().take(self.m).map(|sp| sp.id).collect();
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
            println!("[INSERT] Promoting {} to new entry point at level {}", point_id, level);
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
        let query_vector = if self.metric == DistanceMetric::Cosine {
            self.maybe_normalize(vector)
        } else {
            vector.clone()
        };
    
        let mut extra_neighbors = HashSet::new();
        let m = self.m();
    
        for key in filter_keys {
            if let Some(value) = payload.get(key) {
                // ✅ Try fast exact match via payload index first
                if let Some(id_set) = payload_index.query_exact(key, value) {
                    let mut rng = rand::rng();
                    let sample = id_set
                        .iter()
                        .filter(|&&id| id != point_id && self.get_vector(&id).is_some())
                        .copied()
                        .choose_multiple(&mut rng, 100); // sample limit
    
                    let mut scored: Vec<_> = sample
                        .into_iter()
                        .filter_map(|id| {
                            self.get_vector(&id).map(|vec| {
                                let raw = score(&query_vector, vec, self.metric);
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
    
                    if extra_neighbors.len() >= m {
                        break;
                    }
                }
    
                // ⛔ If fast path didn't yield enough, fallback to filtered vector search
                let mut candidates: Vec<ScoredPoint> = if self.current_max_level() > 0 {
                    let mut entry = self.get_entry_point().unwrap();
                    for l in (1..=self.current_max_level()).rev() {
                        entry = self.greedy_search_layer(&query_vector, entry, l);
                    }
                    self.search_layer(&query_vector, entry, 0, self.ef(), self.metric == DistanceMetric::Cosine || self.metric == DistanceMetric::Dot)?
                } else {
                    self.iter_vectors()
                        .filter_map(|(&id, vec)| {
                            if id != point_id && !self.deleted.contains(&id) {
                                let raw = score(&query_vector, vec, self.metric);
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
                        payloads.get(&sp.id)
                            .and_then(|p| p.get(key))
                            .map_or(false, |v| v == value)
                    })
                    .take(m)
                    .map(|sp| sp.id)
                    .collect();
    
                extra_neighbors.extend(filtered);
    
                if extra_neighbors.len() >= m {
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
        self.layers.entry(level).or_default().entry(a).or_default().push(b);
        self.layers.entry(level).or_default().entry(b).or_default().push(a);
    }

    pub fn greedy_search_layer(&self, query: &Vector, entry: PointId, level: usize) -> PointId {
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
    
                    let d_current = score(query, &self.vectors[&current], self.metric);
                    let d_new = score(query, &self.vectors[&neighbor], self.metric);
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
            println!("[GREEDY] WARNING: Reached max steps at level {}, current = {}", level, current);
        }
    
        //println!("[GREEDY] Finished at point {} at level {}", current, level);
        current
    }
        

    
    fn search_layer(
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
    
        let entry_distance = score(query, &self.vectors[&start_entry], self.metric);
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
    
        //println!("[SEARCH_LAYER] Initial score at entry {}: {:.4}",start_entry, entry_score);
    
        let mut worst_score = result_set.peek().unwrap().0.sort_key;
    
        while let Some(current) = candidate_queue.peek() {
            if current.sort_key > worst_score {
                break;
            }
    
            let current = candidate_queue.pop().unwrap();
            if let Some(neighbors) = self.layers.get(&level).and_then(|l| l.get(&current.id)) {
                for &neighbor in neighbors {
                    if self.deleted.contains(&neighbor) || !visited.insert(neighbor) {
                        continue;
                    }
    
                    let raw = score(query, &self.vectors[&neighbor], self.metric);
                    let score_val = if normalize {
                        self.normalize_score(raw)
                    } else {
                        raw
                    };
    
                    if result_set.len() < ef || score_val < worst_score {
                        let sp = ScoredPoint {
                            id: neighbor,
                            raw_score: raw,
                            sort_key: score_val,
                        };
                        candidate_queue.push(sp.clone());
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
    
        let mut results: Vec<ScoredPoint> = result_set.into_iter().map(|rp| rp.0).collect();
        results.sort_by(|a, b| a.sort_key.partial_cmp(&b.sort_key).unwrap());
    
        //println!("[SEARCH_LAYER] Done. Returning top {} results: {:?}",results.len(),results.iter().map(|sp| sp.id).collect::<Vec<_>>());
    
        Ok(results)
    }
       
    pub fn search(&self, query: &Vector, top_k: usize) -> Result<Vec<ScoredPoint>, DBError> {
        println!("Searching top_k = {}", top_k);
        if self.entry_point.is_none() {
            println!("No entry point. Returning empty result.");
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
        
        let query_for_greedy = if normalize_query {
            self.maybe_normalize(query)
        } else {
            query.clone()
        };
                
        let mut current = self.entry_point.unwrap();
        for l in (1..=self.current_max_level).rev() {
            current = self.greedy_search_layer(&query_for_greedy, current, l);
        }
        
        let final_query = if normalize_query {
            self.maybe_normalize(query)
        } else {
            query.clone()
        };
        
        let mut results = self.search_layer(&final_query, current, 0, self.ef, normalize_score_flag)?;
        results.sort_by(|a, b| a.sort_key.partial_cmp(&b.sort_key).unwrap());
        results.truncate(top_k);
        println!("Search complete. Returning {} results", results.len());
        Ok(results)
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
