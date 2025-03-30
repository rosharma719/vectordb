use std::collections::{BinaryHeap, HashMap, HashSet};
use rand::Rng;
use crate::utils::types::{PointId, Vector, DistanceMetric, Score};
use crate::vector::metric::distance;
use crate::utils::errors::DBError;

#[derive(Clone, Debug)]
pub struct ScoredPoint {
    pub id: PointId,
    pub score: Score,
}

impl PartialEq for ScoredPoint {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}


impl Eq for ScoredPoint {}

impl PartialOrd for ScoredPoint {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        other.score.partial_cmp(&self.score)
    }
}
impl Ord for ScoredPoint {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
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
        l.min(self.max_level_cap)
    }

    pub fn insert(&mut self, point_id: PointId, vector: Vector) -> Result<(), DBError> {
        if self.vectors.contains_key(&point_id) {
            return Ok(());
        }

        if vector.len() != self.dim {
            return Err(DBError::VectorLengthMismatch {
                expected: self.dim,
                actual: vector.len(),
            });
        }

        let level = self.assign_random_level();
        self.vectors.insert(point_id, vector);
        self.levels.insert(point_id, level);

        for l in 0..=level {
            self.layers.entry(l).or_default().insert(point_id, Vec::new());
        }

        if self.entry_point.is_none() {
            self.entry_point = Some(point_id);
            self.current_max_level = level;
            return Ok(());
        }

        let mut current_entry = self.entry_point.unwrap();

        for l in ((level + 1)..=self.current_max_level).rev() {
            current_entry = self.greedy_search_layer(&self.vectors[&point_id], current_entry, l);
        }

        for l in (0..=level).rev() {
            let candidates = self.search_layer(&self.vectors[&point_id], current_entry, l, self.ef)?;
            let neighbors = candidates.into_iter().take(self.m).map(|sp| sp.id).collect::<Vec<_>>();

            let layer = self.layers.get_mut(&l).unwrap();
            layer.insert(point_id, neighbors.clone());
            for &n in &neighbors {
                layer.entry(n).or_default().push(point_id);
            }
        }

        if level > self.current_max_level {
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
        let mut current = entry;
        let mut changed = true;

        while changed {
            changed = false;
            if let Some(neighbors) = self.layers.get(&level).and_then(|l| l.get(&current)) {
                for &neighbor in neighbors {
                    let d_current = distance(query, &self.vectors[&current], self.metric);
                    let d_new = distance(query, &self.vectors[&neighbor], self.metric);
                    if d_new < d_current {
                        current = neighbor;
                        changed = true;
                    }
                }
            }
        }
        current
    }

    fn search_layer(&self, query: &Vector, entry: PointId, level: usize, ef: usize) -> Result<Vec<ScoredPoint>, DBError> {
        if query.len() != self.dim {
            return Err(DBError::VectorLengthMismatch {
                expected: self.dim,
                actual: query.len(),
            });
        }

        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut results = BinaryHeap::new();

        let dist = distance(query, &self.vectors[&entry], self.metric);
        candidates.push(ScoredPoint { id: entry, score: dist });
        results.push(ScoredPoint { id: entry, score: dist });
        visited.insert(entry);

        while let Some(current) = candidates.pop() {
            if let Some(neighbors) = self.layers.get(&level).and_then(|l| l.get(&current.id)) {
                for &neighbor in neighbors {
                    if !visited.insert(neighbor) {
                        continue;
                    }
                    let d = distance(query, &self.vectors[&neighbor], self.metric);
                    let sp = ScoredPoint { id: neighbor, score: d };
                    candidates.push(sp.clone());
                    results.push(sp);
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        let mut res = results.into_sorted_vec();
        res.truncate(ef);
        Ok(res)
    }

    pub fn search(&self, query: &Vector, top_k: usize) -> Result<Vec<ScoredPoint>, DBError> {
        if self.entry_point.is_none() {
            return Ok(vec![]);
        }

        if query.len() != self.dim {
            return Err(DBError::VectorLengthMismatch {
                expected: self.dim,
                actual: query.len(),
            });
        }

        let mut current = self.entry_point.unwrap();

        for l in (1..=self.current_max_level).rev() {
            current = self.greedy_search_layer(query, current, l);
        }

        let mut results = self.search_layer(query, current, 0, self.ef)?;
        results.sort_by(|a, b: &ScoredPoint| a.score.partial_cmp(&b.score).unwrap());
        results.truncate(top_k);
        Ok(results)
    }

    pub fn contains(&self, point_id: &PointId) -> bool {
        self.vectors.contains_key(point_id)
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }


    // Expose vector storage
    pub fn iter_vectors(&self) -> impl Iterator<Item = (&PointId, &Vector)> {
        self.vectors.iter()
    }

    // Expose metric and params
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
    

}
