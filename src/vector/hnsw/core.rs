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

#[derive(Debug, Clone, Copy)]
pub struct HnswConfigSummary {
    pub metric: DistanceMetric,
    pub m: usize,
    pub ef: usize,
    pub ef_construct: usize,
    pub max_level_cap: usize,
    pub level_scale: f64,
    pub current_max_level: usize,
    pub dim: usize,
    pub exact_fallback_enabled: bool,
    pub exact_fallback_threshold: usize,
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
                let dot: f32 = dot_product(query, vec);
                let sim = if dot > 1.0 {
                    1.0
                } else if dot < -1.0 {
                    -1.0
                } else {
                    dot
                };
                1.0 - sim
            }
            DistanceMetric::Dot => dot_product(query, vec),
            DistanceMetric::Euclidean => l2_squared(query, vec),
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

    pub fn config_summary(&self) -> HnswConfigSummary {
        HnswConfigSummary {
            metric: self.metric,
            m: self.m,
            ef: self.ef,
            ef_construct: self.ef_construct,
            max_level_cap: self.max_level_cap,
            level_scale: self.level_scale,
            current_max_level: self.current_max_level,
            dim: self.dim,
            exact_fallback_enabled: self.exact_fallback_enabled,
            exact_fallback_threshold: self.exact_fallback_threshold,
        }
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
                let norm = vec
                    .iter()
                    .map(|x| (*x as f64) * (*x as f64))
                    .sum::<f64>()
                    .sqrt();
                if norm == 0.0 {
                    vec.clone()
                } else {
                    let inv = 1.0 / norm;
                    vec.iter().map(|x| (*x as f64 * inv) as f32).collect()
                }
            }
            _ => vec.clone(),
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn dot_product(query: &[f32], vec: &[f32]) -> f32 {
    unsafe { dot_neon(query, vec) }
}

#[cfg(all(not(target_arch = "aarch64"), target_arch = "x86_64"))]
#[inline]
fn dot_product(query: &[f32], vec: &[f32]) -> f32 {
    if std::arch::is_x86_feature_detected!("avx2") {
        unsafe { dot_avx2(query, vec) }
    } else {
        dot_scalar(query, vec)
    }
}

#[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
#[inline]
fn dot_product(query: &[f32], vec: &[f32]) -> f32 {
    dot_scalar(query, vec)
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn l2_squared(query: &[f32], vec: &[f32]) -> f32 {
    unsafe { l2_neon(query, vec) }
}

#[cfg(all(not(target_arch = "aarch64"), target_arch = "x86_64"))]
#[inline]
fn l2_squared(query: &[f32], vec: &[f32]) -> f32 {
    if std::arch::is_x86_feature_detected!("avx2") {
        unsafe { l2_avx2(query, vec) }
    } else {
        l2_scalar(query, vec)
    }
}

#[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
#[inline]
fn l2_squared(query: &[f32], vec: &[f32]) -> f32 {
    l2_scalar(query, vec)
}

#[allow(dead_code)]
#[inline]
fn dot_scalar(query: &[f32], vec: &[f32]) -> f32 {
    query.iter().zip(vec.iter()).map(|(x, y)| x * y).sum()
}

#[allow(dead_code)]
#[inline]
fn l2_scalar(query: &[f32], vec: &[f32]) -> f32 {
    query
        .iter()
        .zip(vec.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum::<f32>()
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
unsafe fn dot_avx2(query: &[f32], vec: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;
    let len = query.len().min(vec.len());
    while i + 8 <= len {
        let q = _mm256_loadu_ps(query.as_ptr().add(i));
        let v = _mm256_loadu_ps(vec.as_ptr().add(i));
        sum = _mm256_add_ps(sum, _mm256_mul_ps(q, v));
        i += 8;
    }
    let sum128 = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
    let sum128 = _mm_hadd_ps(sum128, sum128);
    let sum128 = _mm_hadd_ps(sum128, sum128);
    let mut acc = _mm_cvtss_f32(sum128);
    while i < len {
        acc += query.get_unchecked(i) * vec.get_unchecked(i);
        i += 1;
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
unsafe fn l2_avx2(query: &[f32], vec: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;
    let len = query.len().min(vec.len());
    while i + 8 <= len {
        let q = _mm256_loadu_ps(query.as_ptr().add(i));
        let v = _mm256_loadu_ps(vec.as_ptr().add(i));
        let diff = _mm256_sub_ps(q, v);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
        i += 8;
    }
    let sum128 = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
    let sum128 = _mm_hadd_ps(sum128, sum128);
    let sum128 = _mm_hadd_ps(sum128, sum128);
    let mut acc = _mm_cvtss_f32(sum128);
    while i < len {
        let diff = query.get_unchecked(i) - vec.get_unchecked(i);
        acc += diff * diff;
        i += 1;
    }
    acc
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
unsafe fn dot_neon(query: &[f32], vec: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let mut sum = vdupq_n_f32(0.0);
    let mut i = 0;
    let len = query.len().min(vec.len());
    while i + 4 <= len {
        let q = vld1q_f32(query.as_ptr().add(i));
        let v = vld1q_f32(vec.as_ptr().add(i));
        sum = vmlaq_f32(sum, q, v);
        i += 4;
    }
    let mut acc = vaddvq_f32(sum);
    while i < len {
        acc += query.get_unchecked(i) * vec.get_unchecked(i);
        i += 1;
    }
    acc
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
unsafe fn l2_neon(query: &[f32], vec: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let mut sum = vdupq_n_f32(0.0);
    let mut i = 0;
    let len = query.len().min(vec.len());
    while i + 4 <= len {
        let q = vld1q_f32(query.as_ptr().add(i));
        let v = vld1q_f32(vec.as_ptr().add(i));
        let diff = vsubq_f32(q, v);
        sum = vmlaq_f32(sum, diff, diff);
        i += 4;
    }
    let mut acc = vaddvq_f32(sum);
    while i < len {
        let diff = query.get_unchecked(i) - vec.get_unchecked(i);
        acc += diff * diff;
        i += 1;
    }
    acc
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::time::Instant;

    fn parse_metric() -> DistanceMetric {
        match env::var("VECTORDB_KERNEL_METRIC")
            .unwrap_or_else(|_| "cosine".to_string())
            .to_lowercase()
            .as_str()
        {
            "dot" => DistanceMetric::Dot,
            "euclidean" | "l2" => DistanceMetric::Euclidean,
            _ => DistanceMetric::Cosine,
        }
    }

    fn gen_vec(seed: u32, dim: usize) -> Vector {
        let mut v = Vec::with_capacity(dim);
        let mut x = seed as u64 + 1;
        for _ in 0..dim {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            let f = ((x >> 8) as u32) as f32 / (u32::MAX as f32);
            v.push(f * 2.0 - 1.0);
        }
        v
    }

    #[test]
    #[ignore]
    fn bench_fast_score_kernel() {
        let dim = env::var("VECTORDB_KERNEL_DIM")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1536);
        let vecs = env::var("VECTORDB_KERNEL_VECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1000);
        let iters = env::var("VECTORDB_KERNEL_ITERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(100);
        let metric = parse_metric();

        let hnsw = HNSWIndex::new(metric, 16, 64, 16, dim);
        let mut data: Vec<Vector> = (0..vecs as u32).map(|i| gen_vec(i, dim)).collect();
        let mut query = gen_vec(999_983, dim);
        if metric == DistanceMetric::Cosine {
            query = hnsw.maybe_normalize(&query);
            for v in &mut data {
                *v = hnsw.maybe_normalize(v);
            }
        }

        let start = Instant::now();
        let mut acc = 0.0f32;
        for _ in 0..iters {
            for v in &data {
                acc += hnsw.fast_score(&query, v);
            }
        }
        let elapsed = start.elapsed();
        let total = (iters as u64) * (vecs as u64);
        let ns_per = elapsed.as_secs_f64() * 1e9 / total as f64;
        let scores_per_sec = total as f64 / elapsed.as_secs_f64();
        println!(
            "kernel metric={:?} dim={} vecs={} iters={} total={} scores/s={:.2} ns/score={:.2} acc={:.4}",
            metric,
            dim,
            vecs,
            iters,
            total,
            scores_per_sec,
            ns_per,
            acc
        );
    }
}
