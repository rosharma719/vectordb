use std::collections::{HashMap, HashSet};

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::utils::errors::DBError;
use crate::utils::types::{DistanceMetric, PointId, Vector};

use super::config::{DEFAULT_EXACT_FALLBACK_THRESHOLD, VERBOSE, log_unfiltered_enabled};
use super::stats::UNFILTERED_SEARCH_AGG;

#[derive(Serialize, Deserialize, Clone)]
pub struct HnswSnapshot {
    pub layers: HashMap<usize, HashMap<PointId, Vec<PointId>>>,
    pub vectors: HashMap<PointId, Vector>,
    pub levels: HashMap<PointId, usize>,
    pub entry_point: Option<PointId>,
    pub metric: DistanceMetric,
    pub m: usize,
    pub ef: usize,
    pub ef_construct: usize,
    pub max_level_cap: usize,
    pub level_scale: f64,
    pub current_max_level: usize,
    pub dim: usize,
    pub deleted: HashSet<PointId>,
    pub exact_fallback_enabled: bool,
    pub exact_fallback_threshold: usize,
}

pub struct HNSWIndex {
    pub(crate) layers: Vec<Vec<Vec<usize>>>,
    pub(crate) vectors: Vec<Vector>,
    pub(crate) levels: Vec<usize>,
    pub(crate) entry_point: Option<usize>,
    pub(crate) metric: DistanceMetric,
    pub(crate) m: usize,
    pub(crate) ef: usize,
    pub(crate) ef_construct: usize,
    pub(crate) max_level_cap: usize,
    pub(crate) level_scale: f64,
    pub(crate) current_max_level: usize,
    pub(crate) dim: usize,
    pub(crate) deleted: Vec<bool>,
    pub(crate) point_to_idx: HashMap<PointId, usize>,
    pub(crate) idx_to_point: Vec<PointId>,
    pub(crate) exact_fallback_enabled: bool,
    pub(crate) exact_fallback_threshold: usize,
}

impl HNSWIndex {
    pub fn new(metric: DistanceMetric, m: usize, ef: usize, max_level_cap: usize, dim: usize) -> Self {
        // Touch the flag early so the enable banner shows up before long inserts.
        let _ = log_unfiltered_enabled();
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
            layers: Vec::new(),
            vectors: Vec::new(),
            levels: Vec::new(),
            entry_point: None,
            metric,
            m,
            ef,
            ef_construct: ef,
            max_level_cap,
            level_scale,
            current_max_level: 0,
            dim,
            deleted: Vec::new(),
            point_to_idx: HashMap::new(),
            idx_to_point: Vec::new(),
            exact_fallback_enabled: true,
            exact_fallback_threshold: DEFAULT_EXACT_FALLBACK_THRESHOLD,
        }
    }

    pub(crate) fn assign_random_level(&self) -> usize {
        let r: f64 = rand::rng().random_range(0.0..1.0);
        let l = (-r.ln() * self.level_scale).floor() as usize;
        let level = l.min(self.max_level_cap);
        level
    }

    pub fn normalize_score(&self, raw: f32) -> f32 {
        match self.metric {
            DistanceMetric::Cosine | DistanceMetric::Euclidean => raw,
            DistanceMetric::Dot => -raw,
        }
    }

    #[inline]
    pub(crate) fn fast_score(&self, query: &Vector, vec: &Vector) -> f32 {
        match self.metric {
            DistanceMetric::Cosine => {
                let dot: f32 = query.iter().zip(vec.iter()).map(|(x, y)| x * y).sum();
                1.0 - dot
            }
            DistanceMetric::Dot => query.iter().zip(vec.iter()).map(|(x, y)| x * y).sum(),
            DistanceMetric::Euclidean => query
                .iter()
                .zip(vec.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt(),
        }
    }

    pub fn mark_deleted(&mut self, point_id: PointId) {
        let Some(idx) = self.idx_of(point_id) else { return; };
        if let Some(flag) = self.deleted.get_mut(idx) {
            *flag = true;
        }
        if Some(idx) == self.entry_point {
            self.entry_point = self.find_highest_level_entry_point();
        }
    }

    pub fn find_highest_level_entry_point(&self) -> Option<usize> {
        self.levels
            .iter()
            .enumerate()
            .filter(|(idx, _)| !self.deleted.get(*idx).copied().unwrap_or(false))
            .max_by_key(|(_, level)| *level)
            .map(|(idx, _)| idx)
    }

    #[inline]
    pub(crate) fn idx_of(&self, point_id: PointId) -> Option<usize> {
        self.point_to_idx.get(&point_id).copied()
    }

    #[inline]
    pub(crate) fn point_id(&self, idx: usize) -> PointId {
        self.idx_to_point[idx]
    }

    #[inline]
    pub(crate) fn get_vector_by_idx(&self, idx: usize) -> Option<&Vector> {
        if self.deleted.get(idx).copied().unwrap_or(false) {
            None
        } else {
            self.vectors.get(idx)
        }
    }

    #[inline]
    pub(crate) fn neighbor_list(&self) -> Vec<usize> {
        Vec::with_capacity(self.m + 1)
    }

    pub(crate) fn ensure_level_capacity(&mut self, level: usize, nodes_len: usize) {
        if self.layers.len() <= level {
            for _ in self.layers.len()..=level {
                let mut layer = Vec::with_capacity(nodes_len);
                for _ in 0..nodes_len {
                    layer.push(self.neighbor_list());
                }
                self.layers.push(layer);
            }
        }
    }

    pub(crate) fn extend_layers_for_new_node(&mut self, nodes_len: usize) {
        let cap = self.m + 1;
        for layer in &mut self.layers {
            if layer.len() < nodes_len {
                layer.push(Vec::with_capacity(cap));
            }
        }
    }

    pub fn contains(&self, point_id: &PointId) -> bool {
        self.point_to_idx.contains_key(point_id)
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn layer_neighbors(&self, level: usize, idx: usize) -> Option<&Vec<usize>> {
        self.layers.get(level)?.get(idx)
    }

    pub fn iter_vectors(&self) -> impl Iterator<Item = (&PointId, &Vector)> {
        self.vectors
            .iter()
            .enumerate()
            .map(|(idx, vec)| (&self.idx_to_point[idx], vec))
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
        if log_unfiltered_enabled() {
            UNFILTERED_SEARCH_AGG.with(|cell| cell.borrow_mut().flush());
        }
        self.ef = ef;
    }

    pub fn flush_unfiltered_search_stats(&self) {
        if log_unfiltered_enabled() {
            UNFILTERED_SEARCH_AGG.with(|cell| cell.borrow_mut().flush());
        }
    }

    pub fn max_level_cap(&self) -> usize {
        self.max_level_cap
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn get_vector(&self, point_id: &PointId) -> Option<&Vector> {
        let idx = self.idx_of(*point_id)?;
        if self.deleted.get(idx).copied().unwrap_or(false) {
            None
        } else {
            self.vectors.get(idx)
        }
    }

    pub fn get_entry_point(&self) -> Option<u64> {
        self.entry_point.map(|idx| self.point_id(idx))
    }

    pub fn current_max_level(&self) -> usize {
        self.current_max_level
    }

    pub fn set_entry_point(&mut self, point_id: PointId) {
        if let Some(idx) = self.idx_of(point_id) {
            self.entry_point = Some(idx);
        }
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

impl HNSWIndex {
    pub(crate) fn register_node(&mut self, point_id: PointId, vector: Vector, level: usize) -> usize {
        let idx = self.vectors.len();
        self.vectors.push(vector);
        self.levels.push(level);
        self.deleted.push(false);
        self.idx_to_point.push(point_id);
        self.point_to_idx.insert(point_id, idx);
        idx
    }
}

impl HNSWIndex {
    pub(crate) fn allocate_entry_point(&mut self, idx: usize, level: usize) {
        self.entry_point = Some(idx);
        self.current_max_level = level;
    }
}

impl HNSWIndex {
    pub(crate) fn validate_dim(&self, vec: &Vector) -> Result<(), DBError> {
        if vec.len() != self.dim {
            return Err(DBError::VectorLengthMismatch {
                expected: self.dim,
                actual: vec.len(),
            });
        }
        Ok(())
    }
}
