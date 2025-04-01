use std::collections::{BinaryHeap, HashMap, HashSet};
use rand::Rng;
use crate::utils::types::{PointId, Vector, DistanceMetric, Score};
use crate::vector::metric::score;
use crate::utils::errors::DBError;

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
        }
    }

    fn assign_random_level(&self) -> usize {
        let r: f64 = rand::rng().random_range(0.0..1.0);
        let l = (-r.ln() * self.level_scale).floor() as usize;
        let level = l.min(self.max_level_cap);
        level
    }

    fn normalize_score(&self, raw: f32) -> f32 {
        match self.metric {
            DistanceMetric::Cosine | DistanceMetric::Euclidean => raw,
            DistanceMetric::Dot => -raw,
        }
    }

    pub fn insert(&mut self, point_id: PointId, vector: Vector) -> Result<(), DBError> {
        println!("Inserting point {}", point_id);
    
        if self.vectors.contains_key(&point_id) {
            println!("Point {} already exists, skipping", point_id);
            return Ok(());
        }
    
        if vector.len() != self.dim {
            println!("Vector length mismatch for point {}", point_id);
            return Err(DBError::VectorLengthMismatch {
                expected: self.dim,
                actual: vector.len(),
            });
        }
    
        let level = self.assign_random_level();
        let vec = self.maybe_normalize(&vector);
        self.vectors.insert(point_id, vec);
        self.levels.insert(point_id, level);
    
        // Initialize this point in all levels 0..=level with a self-loop.
        for l in 0..=level {
            let entry = self.layers.entry(l).or_default().entry(point_id).or_insert_with(Vec::new);
            entry.push(point_id); // Add self-loop
        }
    
        // If this is the first insertion, set the entry point.
        if self.entry_point.is_none() {
            println!("Setting entry point to {}", point_id);
            self.entry_point = Some(point_id);
            self.current_max_level = level;
            return Ok(());
        }
    
        // For levels above the new point's level, perform greedy search to find a good entry.
        let mut current_entry = self.entry_point.unwrap();
        for l in ((level + 1)..=self.current_max_level).rev() {
            println!("Greedy search from level {} for point {}", l, point_id);
            current_entry = self.greedy_search_layer(&self.vectors[&point_id], current_entry, l);
        }
    
        // For each level from new_point's level down to 0, do a candidate search and update links.
        for l in (0..=level).rev() {
            println!("Search layer at level {} for point {}", l, point_id);
            let candidates = self.search_layer(&self.vectors[&point_id], current_entry, l, self.ef)?;
            let neighbors: Vec<PointId> = candidates.into_iter().take(self.m).map(|sp| sp.id).collect();
            println!("Neighbors at level {} for point {}: {:?}", l, point_id, neighbors);
    
            // Link new point with its selected neighbors.
            let layer = self.layers.get_mut(&l).unwrap();
    
            // Insert point_id with neighbors + self-loop (if not already present)
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
    
            // Update current entry for next lower level (if available).
            if let Some(&best) = neighbors.first() {
                current_entry = best;
            }
        }
    
        // Update the global entry point if this point has a new highest level.
        if level > self.current_max_level {
            println!("Updating entry point to {} at new max level {}", point_id, level);
            self.entry_point = Some(point_id);
            self.current_max_level = level;
        }
    
        Ok(())
    }
    


    pub fn add_bidirectional_edge(&mut self, level: usize, a: PointId, b: PointId) {
        self.layers.entry(level).or_default().entry(a).or_default().push(b);
        self.layers.entry(level).or_default().entry(b).or_default().push(a);
    }

    fn greedy_search_layer(&self, query: &Vector, entry: PointId, level: usize) -> PointId {
        println!("Starting greedy search at level {} from entry {}", level, entry);
        let mut current = entry;
        let mut changed = true;
        while changed {
            changed = false;
            if let Some(neighbors) = self.layers.get(&level).and_then(|l| l.get(&current)) {
                for &neighbor in neighbors {
                    let d_current = score(query, &self.vectors[&current], self.metric);
                    let d_new = score(query, &self.vectors[&neighbor], self.metric);
                    let s_current = self.normalize_score(d_current);
                    let s_new = self.normalize_score(d_new);
                    if s_new < s_current {
                        println!("Found closer neighbor at level {}: {} -> {}", level, current, neighbor);
                        current = neighbor;
                        changed = true;
                    }
                }
            }
        }
        println!("Greedy search finished at {} on level {}", current, level);
        current
    }

    fn search_layer(&self, query: &Vector, entry: PointId, level: usize, ef: usize) -> Result<Vec<ScoredPoint>, DBError> {
        println!("Search layer: level={}, entry={}, ef={}", level, entry, ef);
        if query.len() != self.dim {
            return Err(DBError::VectorLengthMismatch {
                expected: self.dim,
                actual: query.len(),
            });
        }
        let mut visited = HashSet::new();
        let mut candidate_queue = BinaryHeap::new();
        let mut result_set = BinaryHeap::new();

        let entry_distance = score(query, &self.vectors[&entry], self.metric);
        let entry_score = self.normalize_score(entry_distance);
        let initial = ScoredPoint { id: entry, raw_score: entry_distance, sort_key: entry_score };
        candidate_queue.push(initial.clone());
        result_set.push(ResultPoint(initial.clone()));
        visited.insert(entry);

        let mut worst_score = result_set.peek().unwrap().0.sort_key;
        while let Some(current) = candidate_queue.peek() {
            if current.sort_key > worst_score {
                break;
            }
            let current = candidate_queue.pop().unwrap();
            if let Some(neighbors) = self.layers.get(&level).and_then(|l| l.get(&current.id)) {
                for &neighbor in neighbors {
                    if !visited.insert(neighbor) {
                        continue;
                    }
                    let raw = score(query, &self.vectors[&neighbor], self.metric);
                    let score = self.normalize_score(raw);
                    if result_set.len() < ef || score < worst_score {
                        let sp = ScoredPoint { id: neighbor, raw_score: raw, sort_key: score };
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
        let normalized_query = self.maybe_normalize(query);
        let mut current = self.entry_point.unwrap();
        for l in (1..=self.current_max_level).rev() {
            current = self.greedy_search_layer(&normalized_query, current, l);
        }
        let mut results = self.search_layer(query, current, 0, self.ef)?;
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
        self.vectors.get(point_id)
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

    fn maybe_normalize(&self, vec: &Vector) -> Vector {
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


