use std::cell::RefCell;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use rand::seq::IteratorRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};

use anyhow::anyhow;
use crate::utils::payload::PayloadValue;
use crate::utils::types::{PointId, Vector, DistanceMetric, Score};
use crate::utils::errors::DBError;
use crate::payload_storage::stores::PayloadIndex;
use crate::utils::payload::Payload;
use crate::payload_storage::filters::{Filter, evaluate_filter};

const VERBOSE: bool = false;
const DEFAULT_EXACT_FALLBACK_THRESHOLD: usize = 256;
const FILTER_EDGE_LOG_CHUNK: usize = 10_000;
thread_local! {
    static FILTER_EDGE_TOTAL_KEYS: RefCell<usize> = RefCell::new(0);
}
thread_local! {
    static UNFILTERED_SEARCH_AGG: RefCell<UnfilteredSearchAgg> = RefCell::new(UnfilteredSearchAgg::default());
}
thread_local! {
    static FILTER_SEED_COUNT: RefCell<usize> = RefCell::new(0);
}

static LOG_UNFILTERED_SEARCH: OnceLock<bool> = OnceLock::new();
static UNFILTERED_LOG_CHUNK: OnceLock<usize> = OnceLock::new();
static FILTER_SEED_LOG_CHUNK: OnceLock<usize> = OnceLock::new();
static DISABLE_EARLY_EXIT: OnceLock<bool> = OnceLock::new();
static SEARCH_EXPANSION_MULT: OnceLock<usize> = OnceLock::new();
static SEARCH_EXPANSION_CAP: OnceLock<Option<usize>> = OnceLock::new();
static EARLY_EXIT_PATIENCE: OnceLock<usize> = OnceLock::new();
static FILTER_SEARCH_LOG: OnceLock<Option<Mutex<BufWriter<File>>>> = OnceLock::new();
static FILTER_EXPANSION_CAP: OnceLock<Option<usize>> = OnceLock::new();
static FILTER_PASSING_BUDGET: OnceLock<Option<usize>> = OnceLock::new();
static FILTER_FAILING_BUDGET: OnceLock<Option<usize>> = OnceLock::new();
static FILTER_SEARCH_SEQ: OnceLock<Mutex<u64>> = OnceLock::new();

fn log_unfiltered_enabled() -> bool {
    *LOG_UNFILTERED_SEARCH.get_or_init(|| {
        let enabled = std::env::var("VECTORDB_LOG_UNFILTERED_SEARCH")
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(false);
        if enabled {
            let chunk = unfiltered_log_chunk();
            println!(
                "[unfiltered_search_stats] enabled (chunk={}, set VECTORDB_LOG_UNFILTERED_EVERY to change)",
                chunk
            );
        }
        enabled
    })
}

fn unfiltered_log_chunk() -> usize {
    *UNFILTERED_LOG_CHUNK.get_or_init(|| {
        std::env::var("VECTORDB_LOG_UNFILTERED_EVERY")
            .ok()
            .and_then(|v| v.replace('_', "").parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(1000)
    })
}

fn filter_seed_log_chunk() -> usize {
    *FILTER_SEED_LOG_CHUNK.get_or_init(|| {
        std::env::var("VECTORDB_LOG_FILTER_SEED_EVERY")
            .ok()
            .and_then(|v| v.replace('_', "").parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(100)
    })
}

fn disable_early_exit() -> bool {
    *DISABLE_EARLY_EXIT.get_or_init(|| {
        std::env::var("VECTORDB_DISABLE_EARLY_EXIT")
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(false)
    })
}

fn search_expansion_multiplier() -> usize {
    *SEARCH_EXPANSION_MULT.get_or_init(|| {
        std::env::var("VECTORDB_SEARCH_EXPANSION_MULT")
            .ok()
            .and_then(|v| v.replace('_', "").parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(4)
    })
}

fn search_expansion_cap_override() -> Option<usize> {
    SEARCH_EXPANSION_CAP
        .get_or_init(|| {
            std::env::var("VECTORDB_SEARCH_EXPANSION_CAP")
                .ok()
                .and_then(|v| v.replace('_', "").parse::<usize>().ok())
                // Treat 0 as "no cap" to allow unbounded expansion for experiments.
                .and_then(|v| if v == 0 { None } else { Some(v) })
        })
        .clone()
}

fn early_exit_patience() -> usize {
    *EARLY_EXIT_PATIENCE.get_or_init(|| {
        std::env::var("VECTORDB_EARLY_EXIT_PATIENCE")
            .ok()
            .and_then(|v| v.replace('_', "").parse::<usize>().ok())
            .unwrap_or(0)
    })
}

fn filter_expansion_cap() -> Option<usize> {
    // Specific cap for filtered search; 0 means unbounded.
    FILTER_EXPANSION_CAP
        .get_or_init(|| {
            std::env::var("VECTORDB_FILTER_EXPANSION_CAP")
                .ok()
                .and_then(|v| v.replace('_', "").parse::<usize>().ok())
                .map(|v| if v == 0 { usize::MAX } else { v })
        })
        .clone()
}

fn filter_passing_budget(m: usize) -> usize {
    FILTER_PASSING_BUDGET
        .get_or_init(|| {
            std::env::var("VECTORDB_FILTER_PASSING_BUDGET")
                .ok()
                .and_then(|v| v.replace('_', "").parse::<usize>().ok())
        })
        .clone()
        .unwrap_or_else(|| std::cmp::max(8, m.saturating_mul(2)))
}

fn filter_failing_budget(m: usize) -> usize {
    FILTER_FAILING_BUDGET
        .get_or_init(|| {
            std::env::var("VECTORDB_FILTER_FAILING_BUDGET")
                .ok()
                .and_then(|v| v.replace('_', "").parse::<usize>().ok())
        })
        .clone()
        .unwrap_or_else(|| std::cmp::max(1, m / 8))
}

fn filter_search_logger() -> Option<&'static Mutex<BufWriter<File>>> {
    FILTER_SEARCH_LOG
        .get_or_init(|| {
            std::env::var("VECTORDB_FILTER_SEARCH_LOG")
                .ok()
                .and_then(|path| File::create(path).ok())
                .map(|f| Mutex::new(BufWriter::new(f)))
        })
        .as_ref()
}

fn next_filter_search_seq() -> u64 {
    let lock = FILTER_SEARCH_SEQ.get_or_init(|| Mutex::new(0));
    let mut guard = lock.lock().unwrap();
    *guard += 1;
    *guard
}

#[derive(Default)]
struct UnfilteredSearchAgg {
    samples: Vec<UnfilteredSample>,
}

#[derive(Clone)]
struct UnfilteredSample {
    ef_search: usize,
    visited: usize,
    expanded: usize,
    best_score: f32,
    worst_score: f32,
}

#[derive(Default)]
struct SearchLayerStats {
    visited: usize,
    expanded: usize,
}

#[derive(Serialize)]
struct FilterSearchLogEntry {
    seq: u64,
    ef_search: usize,
    top_k: usize,
    filter_present: bool,
    patience_limit: usize,
    early_exit: bool,
    seeds_pool: usize,
    seeds_added: usize,
    seeds_accepted: usize,
    seeds_in_results: usize,
    seeds_popped: usize,
    filter_checked: usize,
    filter_passed: usize,
    visited: usize,
    expansions: usize,
    max_expansions: usize,
    results_len: usize,
    stop_reason: String,
    elapsed_ms: f64,
    routing_popped_total: usize,
    routing_popped_passing: usize,
    routing_popped_failing: usize,
    results_inserted: usize,
    results_pq_peek_dist: f32,
    best_routing_dist_at_exit: f32,
}

impl UnfilteredSearchAgg {
    fn record(&mut self, sample: UnfilteredSample) {
        self.samples.push(sample);
        let chunk = unfiltered_log_chunk();
        if self.samples.len() >= chunk {
            self.flush();
        }
    }

    fn flush(&mut self) {
        if self.samples.is_empty() {
            return;
        }

        let mut visited: Vec<f64> = self.samples.iter().map(|s| s.visited as f64).collect();
        let mut expanded: Vec<f64> = self.samples.iter().map(|s| s.expanded as f64).collect();
        let mut util_pct: Vec<f64> = self
            .samples
            .iter()
            .map(|s| (s.expanded as f64 / s.ef_search.max(1) as f64) * 100.0)
            .collect();
        let mut best: Vec<f64> = self.samples.iter().map(|s| s.best_score as f64).collect();
        let mut worst: Vec<f64> = self.samples.iter().map(|s| s.worst_score as f64).collect();
        let ef_min = self.samples.iter().map(|s| s.ef_search).min().unwrap_or(0);
        let ef_max = self.samples.iter().map(|s| s.ef_search).max().unwrap_or(0);

        let stats = |vals: &mut Vec<f64>| {
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p = |pct: f64| -> f64 {
                if vals.is_empty() {
                    return 0.0;
                }
                let idx = ((pct / 100.0) * (vals.len() as f64 - 1.0)).round() as usize;
                vals[idx]
            };
            (p(50.0), p(90.0), p(99.0), *vals.first().unwrap_or(&0.0), *vals.last().unwrap_or(&0.0))
        };

        let (vis_p50, vis_p90, vis_p99, vis_min, vis_max) = stats(&mut visited);
        let (exp_p50, exp_p90, exp_p99, exp_min, exp_max) = stats(&mut expanded);
        let (util_p50, util_p90, util_p99, util_min, util_max) = stats(&mut util_pct);
        let (best_p50, best_p90, best_p99, best_min, best_max) = stats(&mut best);
        let (worst_p50, worst_p90, worst_p99, worst_min, worst_max) = stats(&mut worst);

        println!(
            "[unfiltered_search_stats] n={} ef_search={}..{} visited(min/p50/p90/p99/max)={:.0}/{:.0}/{:.0}/{:.0}/{:.0} expanded={:.0}/{:.0}/{:.0}/{:.0}/{:.0} util%={:.1}/{:.1}/{:.1}/{:.1}/{:.1} best_score={:.4}/{:.4}/{:.4}/{:.4}/{:.4} worst_score={:.4}/{:.4}/{:.4}/{:.4}/{:.4}",
            self.samples.len(),
            ef_min,
            ef_max,
            vis_min,
            vis_p50,
            vis_p90,
            vis_p99,
            vis_max,
            exp_min,
            exp_p50,
            exp_p90,
            exp_p99,
            exp_max,
            util_min,
            util_p50,
            util_p90,
            util_p99,
            util_max,
            best_min,
            best_p50,
            best_p90,
            best_p99,
            best_max,
            worst_min,
            worst_p50,
            worst_p90,
            worst_p99,
            worst_max,
        );

        self.samples.clear();
    }
}

impl Drop for UnfilteredSearchAgg {
    fn drop(&mut self) {
        self.flush();
    }
}

#[derive(Default)]
struct FilterEdgeAgg {
    count: usize,
    // Buckets: Bool, Str, Int, Float
    ns_by_type: [u128; 4],
    samples: usize,
    scored: usize,
    added: usize,
}

thread_local! {
    static FILTER_EDGE_STATS: RefCell<FilterEdgeAgg> = RefCell::new(FilterEdgeAgg::default());
}

#[derive(Default)]
struct SearchScratch {
    visited_epoch: Vec<u32>,
    epoch: u32,
    candidate_queue: BinaryHeap<NodeCandidate>,
    result_set: BinaryHeap<NodeResult>,
    routing_pq: BinaryHeap<NodeRoutingEntry>,
    results_pq: BinaryHeap<NodeResult>,
    seed_epoch: Vec<u32>,
    seed_epoch_val: u32,
    seed_list: Vec<usize>,
    temp_epoch: Vec<u32>,
    temp_epoch_val: u32,
    temp_list: Vec<usize>,
}

impl SearchScratch {
    fn next_epoch(&mut self, len: usize) {
        if self.visited_epoch.len() < len {
            self.visited_epoch.resize(len, 0);
        }
        if self.epoch == u32::MAX {
            self.visited_epoch.fill(0);
            self.epoch = 1;
        } else {
            self.epoch = self.epoch.wrapping_add(1);
            if self.epoch == 0 {
                self.epoch = 1;
            }
        }
    }

    fn mark_visited(&mut self, idx: usize) -> bool {
        if self.visited_epoch[idx] == self.epoch {
            false
        } else {
            self.visited_epoch[idx] = self.epoch;
            true
        }
    }

    fn reset_seed(&mut self, len: usize) {
        if self.seed_epoch.len() < len {
            self.seed_epoch.resize(len, 0);
        }
        if self.seed_epoch_val == u32::MAX {
            self.seed_epoch.fill(0);
            self.seed_epoch_val = 1;
        } else {
            self.seed_epoch_val = self.seed_epoch_val.wrapping_add(1);
            if self.seed_epoch_val == 0 {
                self.seed_epoch_val = 1;
            }
        }
        self.seed_list.clear();
    }

    fn reset_temp(&mut self, len: usize) {
        if self.temp_epoch.len() < len {
            self.temp_epoch.resize(len, 0);
        }
        if self.temp_epoch_val == u32::MAX {
            self.temp_epoch.fill(0);
            self.temp_epoch_val = 1;
        } else {
            self.temp_epoch_val = self.temp_epoch_val.wrapping_add(1);
            if self.temp_epoch_val == 0 {
                self.temp_epoch_val = 1;
            }
        }
        self.temp_list.clear();
    }

    fn mark_seed(&mut self, idx: usize) -> bool {
        if self.seed_epoch[idx] == self.seed_epoch_val {
            false
        } else {
            self.seed_epoch[idx] = self.seed_epoch_val;
            self.seed_list.push(idx);
            true
        }
    }

    fn mark_temp(&mut self, idx: usize) -> bool {
        if self.temp_epoch[idx] == self.temp_epoch_val {
            false
        } else {
            self.temp_epoch[idx] = self.temp_epoch_val;
            self.temp_list.push(idx);
            true
        }
    }

    fn is_seed(&self, idx: usize) -> bool {
        self.seed_epoch.get(idx).copied().unwrap_or(0) == self.seed_epoch_val
    }

    fn is_temp(&self, idx: usize) -> bool {
        self.temp_epoch.get(idx).copied().unwrap_or(0) == self.temp_epoch_val
    }

}

thread_local! {
    static SEARCH_SCRATCH: RefCell<SearchScratch> = RefCell::new(SearchScratch::default());
}

#[derive(Clone, Debug)]
pub struct ScoredPoint {
    pub id: PointId,
    pub raw_score: Score,
    pub sort_key: Score,
}

#[derive(Clone, Debug)]
struct NodeCandidate {
    idx: usize,
    raw_score: Score,
    sort_key: Score,
}

#[derive(Clone, Debug)]
struct NodeRoutingEntry {
    node: NodeCandidate,
    passes_filter: bool,
    budget: usize,
}

impl PartialEq for NodeRoutingEntry {
    fn eq(&self, other: &Self) -> bool {
        self.node.sort_key == other.node.sort_key
    }
}

impl Eq for NodeRoutingEntry {}

impl PartialOrd for NodeRoutingEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // Lower scores are better; invert for max-heap behavior.
        other.node.sort_key.partial_cmp(&self.node.sort_key)
    }
}

impl Ord for NodeRoutingEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
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

impl PartialEq for NodeCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.sort_key == other.sort_key
    }
}

impl Eq for NodeCandidate {}

impl PartialOrd for NodeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // Invert the ordering so that lower scores (better) are considered "greater" for the BinaryHeap.
        other.sort_key.partial_cmp(&self.sort_key)
    }
}

impl Ord for NodeCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

#[derive(Clone, Debug, PartialEq)]
struct NodeResult(NodeCandidate);

impl Eq for NodeResult {}

impl PartialOrd for NodeResult {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // Normal ordering: lower score is better, so when used in a max-heap the worst (largest score) will be at the top.
        self.0.sort_key.partial_cmp(&other.0.sort_key)
    }
}

impl Ord for NodeResult {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.sort_key.partial_cmp(&other.0.sort_key).unwrap()
    }
}

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
    layers: Vec<Vec<Vec<usize>>>,
    vectors: Vec<Vector>,
    levels: Vec<usize>,
    entry_point: Option<usize>,
    metric: DistanceMetric,
    m: usize,
    ef: usize,
    ef_construct: usize,
    max_level_cap: usize,
    level_scale: f64,
    current_max_level: usize,
    dim: usize,
    // Maintain a deleted flag per internal node id for lazy deletion.
    deleted: Vec<bool>,
    point_to_idx: HashMap<PointId, usize>,
    idx_to_point: Vec<PointId>,
    exact_fallback_enabled: bool,
    exact_fallback_threshold: usize,
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

    fn assign_random_level(&self) -> usize {
        let r: f64 = rand::rng().random_range(0.0..1.0);
        let l = (-r.ln() * self.level_scale).floor() as usize;
        let level = l.min(self.max_level_cap);
        level
    }

    #[inline]
    fn idx_of(&self, point_id: PointId) -> Option<usize> {
        self.point_to_idx.get(&point_id).copied()
    }

    #[inline]
    fn point_id(&self, idx: usize) -> PointId {
        self.idx_to_point[idx]
    }

    #[inline]
    fn get_vector_by_idx(&self, idx: usize) -> Option<&Vector> {
        if self.deleted.get(idx).copied().unwrap_or(false) {
            None
        } else {
            self.vectors.get(idx)
        }
    }

    #[inline]
    fn neighbor_list(&self) -> Vec<usize> {
        Vec::with_capacity(self.m + 1)
    }

    fn ensure_level_capacity(&mut self, level: usize, nodes_len: usize) {
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

    fn extend_layers_for_new_node(&mut self, nodes_len: usize) {
        let cap = self.m + 1;
        for layer in &mut self.layers {
            if layer.len() < nodes_len {
                layer.push(Vec::with_capacity(cap));
            }
        }
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
        let Some(idx) = self.idx_of(point_id) else { return; };
        if let Some(flag) = self.deleted.get_mut(idx) {
            *flag = true;
        }
        // If the deleted point was the entry point, try to choose a new one.
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
    

    pub fn insert(&mut self, point_id: PointId, vector: Vector) -> Result<(), DBError> {
        //println!("\n[INSERT] Attempting to insert point: {}", point_id);
    
        if self.point_to_idx.contains_key(&point_id) {
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
        let idx = self.vectors.len();
        self.vectors.push(vec);
        self.levels.push(level);
        self.deleted.push(false);
        self.idx_to_point.push(point_id);
        self.point_to_idx.insert(point_id, idx);
        let nodes_len = self.vectors.len();
        self.ensure_level_capacity(level, nodes_len);
        self.extend_layers_for_new_node(nodes_len);
    
        // Initialize self-links
        for l in 0..=level {
            self.layers[l][idx].push(idx);
            //println!("[INSERT] Initialized self-link at level {}", l);
        }
    
        if self.entry_point.is_none() {
            if VERBOSE {
                log::debug!(target: "vector::hnsw", "[INSERT] First point. Setting entry point to {} at level {}", point_id, level);
            }
            self.entry_point = Some(idx);
            self.current_max_level = level;
            return Ok(());
        }
    
        // If the current entry point is deleted, pick a non-deleted candidate.
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
            //println!("[INSERT] Greedy search for entry at level {} starting from {}", l, current_entry);
            current_entry = self.greedy_search_layer_unfiltered(&self.vectors[idx], current_entry, l);
            //println!("[INSERT] Entry point after greedy search at level {}: {}", l, current_entry);
        }
    
        for l in (0..=level).rev() {
            //println!("[INSERT] Performing search layer at level {}...", l);
            let use_norm = self.metric == DistanceMetric::Cosine || self.metric == DistanceMetric::Dot;
            let candidates = self.search_layer_unfiltered(
                &self.vectors[idx],
                current_entry,
                l,
                self.ef_construct,
                use_norm,
                None,
            )?;
            // Diverse neighbor selection: skip a candidate if it is closer to any already-picked neighbor than to the query.
            let neighbors: Vec<usize> = self.select_diverse_neighbors(&candidates, self.m, use_norm);
            //println!("[INSERT] Found neighbors at level {} for {}: {:?}", l, point_id, neighbors);

            let layer = self.layers.get_mut(l).unwrap();
            let mut linked = neighbors.clone();
            if !linked.contains(&idx) {
                linked.push(idx);
            }
            layer[idx] = linked.clone();
    
            for &n in &neighbors {
                let e = &mut layer[n];
                if !e.contains(&idx) {
                    e.push(idx);
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
        _payloads: &HashMap<PointId, Payload>,
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
        let log_edges_agg = std::env::var("VECTORDB_LOG_FILTER_EDGES_AGG")
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(false);

        // Limit how many candidates we sample from the inverted index per key.
        // Keep it at least 2xM so we can still pick up to M good edges without scoring the whole posting list.
        let sample_limit: usize = self.m().saturating_mul(2);

        for key in filter_keys {
            if let Some(value) = payload.get(key) {
                let key_start = if log_edges_agg { Some(std::time::Instant::now()) } else { None };
                let mut sample_len = 0usize;
                let mut scored_len = 0usize;
                let prev_len = extra_neighbors.len();

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

                // If the inverted-index sample already filled the desired neighbor budget for this key,
                // skip the fallback search but keep checking other keys.
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

                // ⛔ If the posting list was empty, fall back to unfiltered search to find nearby vectors
                // and then filter them by this key/value to establish some connectivity.
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

        // Cap the total number of filter-aware neighbors to avoid blowing up degree.
        let cap = m.max(1);
        for neighbor_id in extra_neighbors.into_iter().filter(|id| *id != point_id).take(cap) {
            // One-way edges from the new point to matches to avoid inflating existing nodes' degree.
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
        //println!("[GREEDY] Start at level {}, from entry {}", level, entry);
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
        entry: usize,
        level: usize,
        payloads: &HashMap<PointId, Payload>,
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
    fn search_layer_unfiltered(
        &self,
        query: &Vector,
        entry: usize,
        level: usize,
        ef: usize,
        normalize: bool,
        stats: Option<&mut SearchLayerStats>,
    ) -> Result<Vec<NodeCandidate>, DBError> {
        if query.len() != self.dim {
            return Err(DBError::VectorLengthMismatch {
                expected: self.dim,
                actual: query.len(),
            });
        }
    
        SEARCH_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            scratch.next_epoch(self.vectors.len());
            scratch.candidate_queue.clear();
            scratch.result_set.clear();

            let mut visited_count = 0usize;
            let mut expanded = 0usize;
            // Allow a wider expansion budget than ef to avoid early cutoff; mirrors filtered search budget.
            let expansion_cap = search_expansion_cap_override()
                .or_else(|| Some(ef.saturating_mul(search_expansion_multiplier()).max(ef)));

            // If the entry is deleted, skip it by choosing a non-deleted vector (if possible)
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
        
            //println!("[search_layer_unfiltered] Initial score at entry {}: {:.4}",start_entry, entry_score);

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
                    for &neighbor in neighbors {
                        if self.deleted.get(neighbor).copied().unwrap_or(false) || !scratch.mark_visited(neighbor) {
                            continue;
                        }
                        visited_count += 1;
        
                        let raw = self.fast_score(query, &self.vectors[neighbor]);
                        let score_val = if normalize {
                            self.normalize_score(raw)
                        } else {
                            raw
                        };

                        // For Dot we explore broadly (push all neighbors) to avoid getting stuck in local optima.
                        let push_candidate = self.metric == DistanceMetric::Dot
                            || scratch.result_set.len() < ef
                            || score_val < worst_score;

                        if push_candidate {
                            let sp = NodeCandidate {
                                idx: neighbor,
                                raw_score: raw,
                                sort_key: score_val,
                            };
                            scratch.candidate_queue.push(sp.clone());

                            // Still keep result_set trimmed to ef best items.
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

                // Guard against runaway expansion; treat as a backstop, not the primary stop signal.
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
        
            //println!("[search_layer_unfiltered] Done. Returning top {} results: {:?}",results.len(),results.iter().map(|sp| sp.idx).collect::<Vec<_>>());
            if let Some(stats) = stats {
                stats.visited = visited_count;
                stats.expanded = expanded;
            }

            Ok(results)
        })
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

        let deleted_count = self.deleted.iter().filter(|d| **d).count();
        let collection_size = self.vectors.len().saturating_sub(deleted_count);
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

    pub fn find_entry_point_matching_filter(
        &self,
        filter: &Filter,
        payload_index: &PayloadIndex,
        payloads: &HashMap<PointId, Payload>,
    ) -> Option<usize> {
        fn max_level_point<'a>(
            ids: impl Iterator<Item = &'a PointId>,
            levels: &[usize],
            deleted: &[bool],
            map: &HashMap<PointId, usize>,
        ) -> Option<usize> {
            ids.filter_map(|id| map.get(id).copied())
                .filter(|&idx| !deleted.get(idx).copied().unwrap_or(false))
                .max_by_key(|&idx| levels.get(idx).copied().unwrap_or(0))
        }
    
        match filter {
            Filter::Match { key, value } => {
                payload_index
                    .query_exact(key, value)
                    .and_then(|ids| {
                        max_level_point(ids.iter(), &self.levels, &self.deleted, &self.point_to_idx)
                    })
            }
            Filter::And(conds) | Filter::Or(conds) => {
                let mut best: Option<(usize, usize)> = None;
    
                for cond in conds {
                    if let Some(id) = self.find_entry_point_matching_filter(cond, payload_index, payloads) {
                        let level = self.levels.get(id).copied().unwrap_or(0);
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
        let mut entry = match self.entry_point {
            Some(idx) => {
                if let Some(f) = full_filter {
                    let entry_id = self.point_id(idx);
                    if let Some(p) = payloads.get(&entry_id) {
                        if evaluate_filter(f, p)? {
                            idx
                        } else {
                            self.find_entry_point_matching_filter(f, payload_index, payloads)
                                .unwrap_or(idx)
                        }
                    } else {
                        self.find_entry_point_matching_filter(f, payload_index, payloads)
                            .unwrap_or(idx)
                    }
                } else {
                    idx
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
        let result = SEARCH_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            scratch.next_epoch(self.vectors.len());
            scratch.routing_pq.clear();
            scratch.results_pq.clear();
            let mut visited_count = 0usize;
        let log_seed = std::env::var("VECTORDB_LOG_FILTER_SEED")
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(false);
        let allow_early_exit = self.metric != DistanceMetric::Dot && !disable_early_exit();
        let patience_limit = if allow_early_exit { early_exit_patience() } else { 0 };
        let mut no_improve_streak = 0usize;
        let mut early_exit = false;

        let search_start = std::time::Instant::now();
        let dist0 = self.fast_score(&query_for_search, self.get_vector_by_idx(entry).unwrap());
        let sk0   = if use_normalize_score { self.normalize_score(dist0) } else { dist0 };
        let first = NodeCandidate { idx: entry, raw_score: dist0, sort_key: sk0 };
        let mut filter_checked = 0usize;
        let mut filter_passed = 0usize;
        let mut seeds_popped = 0usize;
        let max_expansions = filter_expansion_cap()
            .unwrap_or_else(|| self.ef.saturating_mul(search_expansion_multiplier()));
        let result_cap = self.ef.max(top_k);
        let mut routing_popped_total = 0usize;
        let mut routing_popped_passing = 0usize;
        let mut results_inserted = 0usize;
        let passing_budget = filter_passing_budget(self.m);
        let failing_budget = filter_failing_budget(self.m);

        let entry_passes = full_filter.map_or(true, |f| {
            filter_checked += 1;
            let entry_id = self.point_id(entry);
            let ok = payloads.get(&entry_id).map_or(false, |p| evaluate_filter(f, p).unwrap_or(false));
            if ok {
                filter_passed += 1;
            }
            ok
        });
        scratch.routing_pq.push(NodeRoutingEntry {
            node: first.clone(),
            passes_filter: entry_passes,
            budget: if entry_passes { passing_budget } else { failing_budget },
        });
        // only push into results_pq if it passes the *full* filter:
        if entry_passes {
            scratch.results_pq.push(NodeResult(first));
        }
        if scratch.mark_visited(entry) {
            visited_count += 1;
        }

        // Seed the beam with a few IDs pulled from the inverted index that match equality filters.
        // This helps reach filtered regions even when the graph lacks filter-aware edges.
        let seed_limit = self.ef;
        let seed_len = self.vectors.len();
        fn collect_match_ids(
            filter: &Filter,
            idx: &PayloadIndex,
            scratch: &mut SearchScratch,
            map: &HashMap<PointId, usize>,
            len: usize,
            primary: bool,
        ) {
            match filter {
                Filter::Match { key, value } => {
                    if let Some(ids) = idx.query_exact(key, value) {
                        for id in ids.iter().copied() {
                            if let Some(&node_idx) = map.get(&id) {
                                if primary {
                                    scratch.mark_seed(node_idx);
                                } else {
                                    scratch.mark_temp(node_idx);
                                }
                            }
                        }
                    }
                }
                Filter::Or(parts) => {
                    for p in parts {
                        if primary {
                            scratch.reset_temp(len);
                            collect_match_ids(p, idx, scratch, map, len, false);
                            let other = scratch.temp_list.clone();
                            for id in other {
                                scratch.mark_seed(id);
                            }
                        } else {
                            scratch.reset_seed(len);
                            collect_match_ids(p, idx, scratch, map, len, true);
                            let other = scratch.seed_list.clone();
                            for id in other {
                                scratch.mark_temp(id);
                            }
                        }
                    }
                }
                Filter::And(parts) => {
                    if parts.is_empty() {
                        return;
                    }
                    if primary {
                        scratch.reset_seed(len);
                        collect_match_ids(&parts[0], idx, scratch, map, len, true);
                        for p in parts.iter().skip(1) {
                            scratch.reset_temp(len);
                            collect_match_ids(p, idx, scratch, map, len, false);
                            let retained = std::mem::take(&mut scratch.seed_list);
                            scratch.reset_seed(len);
                            for id in retained {
                                if scratch.is_temp(id) {
                                    scratch.mark_seed(id);
                                }
                            }
                        }
                    } else {
                        scratch.reset_temp(len);
                        collect_match_ids(&parts[0], idx, scratch, map, len, false);
                        for p in parts.iter().skip(1) {
                            scratch.reset_seed(len);
                            collect_match_ids(p, idx, scratch, map, len, true);
                            let retained = std::mem::take(&mut scratch.temp_list);
                            scratch.reset_temp(len);
                            for id in retained {
                                if scratch.is_seed(id) {
                                    scratch.mark_temp(id);
                                }
                            }
                        }
                    }
                }
                Filter::Not(_) | Filter::Compare { .. } => {}
            }
        }
        let use_seeds = std::env::var("VECTORDB_DISABLE_FILTER_SEEDS")
            .map(|v| v == "0" || v.to_lowercase() == "false")
            .unwrap_or(true);
        if use_seeds {
            if let Some(f) = full_filter {
                scratch.reset_seed(seed_len);
                scratch.reset_temp(seed_len);
                collect_match_ids(f, payload_index, &mut scratch, &self.point_to_idx, seed_len, true);
            }
        }
        let seed_pool_size = scratch.seed_list.len();
        let mut seeds_added = 0usize;
        let mut seeds_accepted = 0usize;
        if use_seeds && !scratch.seed_list.is_empty() {
            let seed_ids = scratch.seed_list.clone();
            for idx in seed_ids {
                if seeds_added >= seed_limit {
                    break;
                }
                if self.deleted.get(idx).copied().unwrap_or(false) || !scratch.mark_visited(idx) {
                    continue;
                }
                visited_count += 1;
                let Some(vec) = self.get_vector_by_idx(idx) else { continue; };
                let d = self.fast_score(&query_for_search, vec);
                let sk = if use_normalize_score { self.normalize_score(d) } else { d };
                let sp = NodeCandidate { idx, raw_score: d, sort_key: sk };
                let passes = full_filter.map_or(true, |f| {
                    filter_checked += 1;
                    let id = self.point_id(idx);
                    let ok = payloads.get(&id).map_or(false, |p| evaluate_filter(f, p).unwrap_or(false));
                    if ok {
                        filter_passed += 1;
                    }
                    ok
                });
                let budget = if passes { passing_budget } else { failing_budget };
                scratch.routing_pq.push(NodeRoutingEntry {
                    node: sp.clone(),
                    passes_filter: passes,
                    budget,
                });
                if passes {
                    scratch.results_pq.push(NodeResult(sp));
                    seeds_accepted += 1;
                }
                seeds_added += 1;
            }
        }

        let mut worst = scratch.results_pq.peek().map(|rp| rp.0.sort_key).unwrap_or(f32::MAX);

        let mut expansions = 0usize;
        let mut stop_reason = "queue_exhausted".to_string();

        while let Some(curr) = scratch.routing_pq.pop() {
            expansions += 1;
            if scratch.is_seed(curr.node.idx) {
                seeds_popped += 1;
            }
            routing_popped_total += 1;
            if curr.passes_filter {
                routing_popped_passing += 1;
            }
            if allow_early_exit && patience_limit > 0 && scratch.results_pq.len() >= self.ef {
                if let Some(next) = scratch.routing_pq.peek() {
                    if next.node.sort_key > worst {
                        no_improve_streak += 1;
                    } else {
                        no_improve_streak = 0;
                    }
                } else {
                    no_improve_streak += 1;
                }
                if no_improve_streak > patience_limit {
                    early_exit = true;
                    stop_reason = "early_exit_patience".to_string();
                    break;
                }
            }

            if expansions >= max_expansions {
                stop_reason = "max_expansions".to_string();
                break;
            }
            // Once we already have top_k passing the filter, stop exploring candidates
            // that are worse than our current worst. This restores the usual early-exit
            // even when a filter is present.
            if scratch.results_pq.len() >= top_k && curr.node.sort_key > worst {
                stop_reason = "pruned_by_worst".to_string();
                break;
            }

            if curr.budget <= 1 {
                continue;
            }

            if let Some(neighs) = self.layer_neighbors(0, curr.node.idx) {
                for &nb in neighs {
                    if self.deleted.get(nb).copied().unwrap_or(false) || !scratch.mark_visited(nb) {
                        continue;
                    }
                    visited_count += 1;

                    let d   = self.fast_score(&query_for_search, self.get_vector_by_idx(nb).unwrap());
                    let sk  = if use_normalize_score { self.normalize_score(d) } else { d };
                    let sp  = NodeCandidate { idx: nb, raw_score: d, sort_key: sk };

                let passes = full_filter.map_or(true, |f| {
                    filter_checked += 1;
                    let id = self.point_id(nb);
                    let ok = payloads.get(&id).map_or(false, |p| evaluate_filter(f, p).unwrap_or(false));
                    if ok {
                        filter_passed += 1;
                    }
                    ok
                });
                let budget = if passes { passing_budget } else { failing_budget };
                if budget > 0 {
                scratch.routing_pq.push(NodeRoutingEntry {
                    node: sp.clone(),
                    passes_filter: passes,
                    budget,
                });
                }
                let passes_for_results = passes;
                if passes_for_results {
                    // *now* apply the full filter before pushing to result_set
                        scratch.results_pq.push(NodeResult(sp));
                        results_inserted += 1;
                        if scratch.results_pq.len() > result_cap {
                            scratch.results_pq.pop();
                        }
                        if scratch.results_pq.len() >= top_k {
                            worst = scratch.results_pq.peek().unwrap().0.sort_key;
                        }
                    }
                }
            }
        }

        // unwrap the inner ScoredPoint and truncate to top_k
        let mut out = std::mem::take(&mut scratch.results_pq)
            .into_sorted_vec()
            .into_iter()
            .map(|rp| {
                let cand = rp.0;
                ScoredPoint {
                    id: self.point_id(cand.idx),
                    raw_score: cand.raw_score,
                    sort_key: cand.sort_key,
                }
            })
            .collect::<Vec<_>>();
        out.truncate(top_k);

        let seeds_in_results = out
            .iter()
            .filter(|sp| self.idx_of(sp.id).map_or(false, |idx| scratch.is_seed(idx)))
            .count();

        if log_seed && full_filter.is_some() {
            let chunk = filter_seed_log_chunk();
            let should_emit = FILTER_SEED_COUNT.with(|c| {
                let mut v = c.borrow_mut();
                *v += 1;
                *v % chunk == 0
            });
            if should_emit {
                let msg = format!(
                    "[filter_search_seed] seeds_pool={} seeds_added={} seeds_accepted={} seeds_in_results={} final_results={}",
                    seed_pool_size,
                    seeds_added,
                    seeds_accepted,
                    seeds_in_results,
                    out.len()
                );
                log::info!(target: "hnsw", "{}", msg);
                println!("{}", msg);
            }
        }

        if let Some(log) = filter_search_logger() {
            let best_routing_dist_at_exit = scratch.routing_pq.peek().map(|sp| sp.node.sort_key).unwrap_or(f32::MAX);
            let results_pq_peek_dist = worst;
            let entry = FilterSearchLogEntry {
                seq: next_filter_search_seq(),
                ef_search: self.ef,
                top_k,
                filter_present: full_filter.is_some(),
                seeds_pool: seed_pool_size,
                seeds_added,
                seeds_accepted,
                seeds_in_results,
                seeds_popped,
                filter_checked,
                filter_passed,
                visited: visited_count,
                expansions,
                max_expansions,
                results_len: out.len(),
                stop_reason,
                patience_limit,
                early_exit,
                routing_popped_total,
                routing_popped_passing,
                routing_popped_failing: routing_popped_total.saturating_sub(routing_popped_passing),
                results_inserted,
                results_pq_peek_dist,
                best_routing_dist_at_exit,
                elapsed_ms: search_start.elapsed().as_secs_f64() * 1000.0,
            };
            if let Ok(mut writer) = log.lock() {
                if serde_json::to_writer(&mut *writer, &entry).is_ok() {
                    let _ = writer.write_all(b"\n");
                }
            }
        }

        Ok(out)
    });

        result
    }

    /// Heuristic neighbor selector that enforces diversity (HNSW heuristic 2).
    fn select_diverse_neighbors(&self, candidates: &[NodeCandidate], m: usize, normalize_scores: bool) -> Vec<usize> {
        let mut result = Vec::with_capacity(m);
        for cand in candidates {
            if result.len() >= m {
                break;
            }
            let Some(cand_vec) = self.get_vector_by_idx(cand.idx) else { continue; };
            let d_qc = cand.sort_key; // already normalized when requested
            let mut too_close = false;
            for &r_id in &result {
                let Some(r_vec) = self.get_vector_by_idx(r_id) else { continue; };
                let d_cr_raw = self.fast_score(cand_vec, r_vec);
                let d_cr = if normalize_scores { self.normalize_score(d_cr_raw) } else { d_cr_raw };
                if d_cr < d_qc {
                    too_close = true;
                    break;
                }
            }
            if !too_close {
                result.push(cand.idx);
            }
        }
        // If we selected fewer than m due to the diversity filter, backfill with remaining closest candidates.
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
        // Optionally, one might return None for deleted points.
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

    pub fn to_snapshot(&self) -> HnswSnapshot {
        let mut vectors = HashMap::with_capacity(self.vectors.len());
        let mut levels = HashMap::with_capacity(self.levels.len());
        let mut deleted = HashSet::new();
        for (idx, vec) in self.vectors.iter().enumerate() {
            let id = self.point_id(idx);
            vectors.insert(id, vec.clone());
            levels.insert(id, self.levels.get(idx).copied().unwrap_or(0));
            if self.deleted.get(idx).copied().unwrap_or(false) {
                deleted.insert(id);
            }
        }

        let mut layers = HashMap::new();
        for (level, layer) in self.layers.iter().enumerate() {
            let mut level_map: HashMap<PointId, Vec<PointId>> = HashMap::new();
            for (idx, neighbors) in layer.iter().enumerate() {
                if neighbors.is_empty() {
                    continue;
                }
                let id = self.point_id(idx);
                let mapped = neighbors.iter().map(|&n| self.point_id(n)).collect::<Vec<_>>();
                level_map.insert(id, mapped);
            }
            if !level_map.is_empty() {
                layers.insert(level, level_map);
            }
        }

        HnswSnapshot {
            layers,
            vectors,
            levels,
            entry_point: self.entry_point.map(|idx| self.point_id(idx)),
            metric: self.metric,
            m: self.m,
            ef: self.ef,
            ef_construct: self.ef_construct,
            max_level_cap: self.max_level_cap,
            level_scale: self.level_scale,
            current_max_level: self.current_max_level,
            dim: self.dim,
            deleted,
            exact_fallback_enabled: self.exact_fallback_enabled,
            exact_fallback_threshold: self.exact_fallback_threshold,
        }
    }

    pub fn from_snapshot(snapshot: HnswSnapshot) -> Self {
        let mut ids: Vec<PointId> = snapshot.vectors.keys().copied().collect();
        ids.sort_unstable();
        let mut point_to_idx = HashMap::with_capacity(ids.len());
        for (idx, id) in ids.iter().copied().enumerate() {
            point_to_idx.insert(id, idx);
        }
        let mut vectors = Vec::with_capacity(ids.len());
        let mut levels = Vec::with_capacity(ids.len());
        let mut deleted = vec![false; ids.len()];
        for (idx, id) in ids.iter().copied().enumerate() {
            if let Some(vec) = snapshot.vectors.get(&id) {
                vectors.push(vec.clone());
            } else {
                vectors.push(Vec::new());
            }
            levels.push(snapshot.levels.get(&id).copied().unwrap_or(0));
            if snapshot.deleted.contains(&id) {
                deleted[idx] = true;
            }
        }
        let num_levels = snapshot
            .layers
            .keys()
            .copied()
            .max()
            .unwrap_or(0)
            .max(snapshot.current_max_level)
            + 1;
        let mut layers = vec![vec![Vec::with_capacity(snapshot.m + 1); ids.len()]; num_levels];
        for (level, layer_map) in snapshot.layers.iter() {
            if *level >= layers.len() {
                continue;
            }
            for (id, neighbors) in layer_map {
                let Some(&idx) = point_to_idx.get(id) else { continue; };
                let mapped = neighbors
                    .iter()
                    .filter_map(|n| point_to_idx.get(n).copied())
                    .collect::<Vec<_>>();
                layers[*level][idx] = mapped;
            }
        }

        Self {
            layers,
            vectors,
            levels,
            entry_point: snapshot.entry_point.and_then(|id| point_to_idx.get(&id).copied()),
            metric: snapshot.metric,
            m: snapshot.m,
            ef: snapshot.ef,
            ef_construct: snapshot.ef_construct,
            max_level_cap: snapshot.max_level_cap,
            level_scale: snapshot.level_scale,
            current_max_level: snapshot.current_max_level,
            dim: snapshot.dim,
            deleted,
            point_to_idx,
            idx_to_point: ids,
            exact_fallback_enabled: snapshot.exact_fallback_enabled,
            exact_fallback_threshold: snapshot.exact_fallback_threshold,
        }
    }

    pub fn save_to_path<P: AsRef<Path>>(&self, path: P) -> Result<(), DBError> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        bincode::serialize_into(&mut writer, &self.to_snapshot())
            .map_err(|e| DBError::SerializationError(anyhow!(e)))?;
        Ok(())
    }

    pub fn load_from_path<P: AsRef<Path>>(path: P) -> Result<Self, DBError> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let snapshot: HnswSnapshot = bincode::deserialize_from(&mut reader)
            .map_err(|e| DBError::SerializationError(anyhow!(e)))?;
        Ok(Self::from_snapshot(snapshot))
    }
}
