use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::thread_local;
use std::time::{Duration, Instant};

use anyhow::anyhow;
use serde::{Deserialize, Serialize};

use crate::payload_storage::filters::Filter;
use crate::payload_storage::stores::PayloadIndex;
use crate::utils::errors::DBError;
use crate::utils::payload::{Payload, PayloadValue};
use crate::utils::types::{PointId, Vector};
use crate::vector::hnsw::{HNSWIndex, HnswSnapshot, ScoredPoint, SearchStats};

/// A segment is the core unit that wraps vector storage, indexing, payloads, and deletion.
pub struct Segment {
    hnsw: HNSWIndex,
    payload_index: PayloadIndex,
    payloads: HashMap<PointId, Payload>,
    // This set is maintained in parallel with the HNSW deletion set.
    deleted: HashSet<PointId>,
    next_id: PointId,
}

#[derive(Serialize, Deserialize)]
struct SegmentSnapshot {
    hnsw: HnswSnapshot,
    payload_index: PayloadIndex,
    payloads: HashMap<PointId, Payload>,
    deleted: HashSet<PointId>,
    next_id: PointId,
}

const SEGMENT_SNAPSHOT_MAGIC: [u8; 4] = *b"VDBS";
const SEGMENT_SNAPSHOT_VERSION: u32 = 2;

#[derive(Serialize, Deserialize)]
struct HnswSnapshotV1 {
    layers: HashMap<usize, HashMap<PointId, Vec<PointId>>>,
    vectors: HashMap<PointId, Vector>,
    levels: HashMap<PointId, usize>,
    entry_point: Option<PointId>,
    metric: crate::utils::types::DistanceMetric,
    m: usize,
    ef: usize,
    ef_construct: usize,
    max_level_cap: usize,
    level_scale: f64,
    current_max_level: usize,
    dim: usize,
    deleted: HashSet<PointId>,
    exact_fallback_enabled: bool,
    exact_fallback_threshold: usize,
}

impl From<HnswSnapshotV1> for HnswSnapshot {
    fn from(snapshot: HnswSnapshotV1) -> Self {
        Self {
            layers: snapshot.layers,
            vectors: snapshot.vectors,
            levels: snapshot.levels,
            entry_point: snapshot.entry_point,
            metric: snapshot.metric,
            m: snapshot.m,
            m0: snapshot.m,
            ef: snapshot.ef,
            ef_construct: snapshot.ef_construct,
            max_level_cap: snapshot.max_level_cap,
            level_scale: snapshot.level_scale,
            current_max_level: snapshot.current_max_level,
            dim: snapshot.dim,
            deleted: snapshot.deleted,
            exact_fallback_enabled: snapshot.exact_fallback_enabled,
            exact_fallback_threshold: snapshot.exact_fallback_threshold,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct SegmentSnapshotV1 {
    hnsw: HnswSnapshotV1,
    payload_index: PayloadIndex,
    payloads: HashMap<PointId, Payload>,
    deleted: HashSet<PointId>,
    next_id: PointId,
}

impl From<SegmentSnapshotV1> for SegmentSnapshot {
    fn from(snapshot: SegmentSnapshotV1) -> Self {
        Self {
            hnsw: snapshot.hnsw.into(),
            payload_index: snapshot.payload_index,
            payloads: snapshot.payloads,
            deleted: snapshot.deleted,
            next_id: snapshot.next_id,
        }
    }
}

#[derive(Default)]
struct InsertTiming {
    count: usize,
    hnsw: Duration,
    payload_idx: Duration,
    filter_edges: Duration,
    total: Duration,
}

thread_local! {
    static INSERT_TIMINGS: RefCell<InsertTiming> = RefCell::new(InsertTiming::default());
    static INSERT_TOTAL: RefCell<usize> = RefCell::new(0);
}

impl Segment {
    pub fn new(hnsw: HNSWIndex) -> Self {
        Self {
            hnsw,
            payload_index: PayloadIndex::new(),
            payloads: HashMap::new(),
            deleted: HashSet::new(),
            next_id: 1,
        }
    }

    /// Insert a new vector and optional payload. Auto-generates ID.
    pub fn insert(&mut self, vector: Vector, payload: Option<Payload>) -> Result<PointId, DBError> {
        let point_id = self.next_id;
        self.insert_with_id(point_id, vector, payload)
    }

    /// Insert a vector with a caller-provided ID. Fails if the ID already exists or was deleted.
    pub fn insert_with_id(
        &mut self,
        point_id: PointId,
        vector: Vector,
        payload: Option<Payload>,
    ) -> Result<PointId, DBError> {
        let log_timing = Self::log_insert_timing();
        let total_start = if log_timing { Some(Instant::now()) } else { None };
        let mut last = total_start;
        let mut chunk_start = total_start;
        let mut hnsw_dur = Duration::from_millis(0);
        let mut payload_idx_dur = Duration::from_millis(0);
        let mut filter_edges_dur = Duration::from_millis(0);

        if self.hnsw.contains(&point_id) || self.payloads.contains_key(&point_id) || self.deleted.contains(&point_id) {
            return Err(DBError::DuplicatePointId(point_id));
        }

        self.hnsw.insert(point_id, vector.clone())?;
        if let Some(t) = last.as_mut() {
            hnsw_dur = t.elapsed();
            *t = Instant::now();
        }

        if let Some(p) = payload {
            self.payload_index.insert(point_id, &p);
            self.payloads.insert(point_id, p.clone());
            if let Some(t) = last.as_mut() {
                payload_idx_dur = t.elapsed();
                *t = Instant::now();
            }

            let filter_keys = Self::filter_keys_for_payload(&p);

            if !filter_keys.is_empty() && Self::filter_edges_enabled() {
                self.hnsw.build_filter_aware_edges(
                    point_id,
                    &vector,
                    &p,
                    &self.payload_index,
                    &self.payloads,
                    &filter_keys,
                )?;
                if let Some(t) = last.as_mut() {
                    filter_edges_dur = t.elapsed();
                    *t = Instant::now();
                }
            }
        }

        if point_id >= self.next_id {
            self.next_id = point_id.saturating_add(1);
        }

        if let Some(start) = total_start {
            let total = start.elapsed();
            // Accumulate and log every CHUNK inserts to avoid log spam.
            const CHUNK: usize = 5000;
            INSERT_TIMINGS.with(|cell| {
                INSERT_TOTAL.with(|tc| {
                    let mut s = cell.borrow_mut();
                    let mut total_count = tc.borrow_mut();
                    s.count += 1;
                    s.hnsw += hnsw_dur;
                    s.payload_idx += payload_idx_dur;
                    s.filter_edges += filter_edges_dur;
                    s.total += total;
                    if s.count % CHUNK == 0 {
                        let c = s.count as u32;
                        *total_count += s.count;
                        let chunk_elapsed = chunk_start.map(|cs| cs.elapsed()).unwrap_or_default();
                        let msg = format!(
                            "[insert_timing_chunk] n={} cum_n={} avg_hnsw={:?} avg_payload_idx={:?} avg_filter_edges={:?} avg_total={:?} chunk_elapsed={:?}",
                            s.count,
                            *total_count,
                            s.hnsw / c,
                            s.payload_idx / c,
                            s.filter_edges / c,
                            s.total / c,
                            chunk_elapsed
                        );
                        log::info!(target: "segment", "{}", msg);
                        println!("{}", msg);
                        // Reset for next chunk
                        *s = InsertTiming::default();
                        chunk_start = Some(Instant::now());
                    }
                });
            });
        }

        Ok(point_id)
    }

    /// Get the vector for a given point ID, if it exists and is not deleted.
    pub fn get_vector(&self, point_id: PointId) -> Option<&Vector> {
        if self.deleted.contains(&point_id) {
            return None;
        }
        self.hnsw.get_vector(&point_id)
    }

    pub fn delete(&mut self, point_id: PointId) -> Result<(), DBError> {
        // If the point is already marked as deleted OR is no longer in the index,
        // treat it as already deleted.
        if self.deleted.contains(&point_id) || !self.hnsw.contains(&point_id) {
            return Ok(());
        }
    
        if let Some(p) = self.payloads.get(&point_id) {
            self.payload_index.remove(point_id, p);
        }
    
        self.deleted.insert(point_id);
        self.hnsw.mark_deleted(point_id);
    
        let deleted_count = self.deleted.len();
        let total_count = self.hnsw.len();

        const MIN_DELETIONS_BEFORE_PURGE: usize = 100;
        const MAX_DELETION_RATIO: f32 = 0.25;

        if deleted_count >= MIN_DELETIONS_BEFORE_PURGE &&
        (deleted_count as f32 / total_count as f32) >= MAX_DELETION_RATIO {
            log::info!(
                target: "segment",
                "[DELETE] Triggering purge: {}/{} ({:.2}%) deleted",
                deleted_count,
                total_count,
                100.0 * deleted_count as f32 / total_count as f32
            );
            self.purge()?;
        }

    
        Ok(())
    }
    


    pub fn search(&self, query: &Vector, top_k: usize) -> Result<Vec<ScoredPoint>, DBError> {
        let total_non_deleted = self.hnsw.len() - self.deleted.len();
        if total_non_deleted == 0 {
            return Err(DBError::SearchError("No active points available to search.".into()));
        }

        // HNSWIndex now internally skips deleted points.
        let candidates = self.hnsw.search(query, top_k)?;
        // (The following filter is kept as extra safety.)
        let filtered = candidates
            .into_iter()
            .filter(|sp| !self.deleted.contains(&sp.id))
            .take(top_k)
            .collect();

        Ok(filtered)
    }

    pub fn search_with_stats(
        &self,
        query: &Vector,
        top_k: usize,
    ) -> Result<(Vec<ScoredPoint>, SearchStats), DBError> {
        let total_non_deleted = self.hnsw.len() - self.deleted.len();
        if total_non_deleted == 0 {
            return Err(DBError::SearchError("No active points available to search.".into()));
        }

        let (candidates, stats) = self.hnsw.search_with_stats(query, top_k)?;
        let filtered = candidates
            .into_iter()
            .filter(|sp| !self.deleted.contains(&sp.id))
            .take(top_k)
            .collect();

        Ok((filtered, stats))
    }

    pub fn search_with_filter(
        &self,
        query: &Vector,
        top_k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<ScoredPoint>, DBError> {
        let total_non_deleted = self.hnsw.len() - self.deleted.len();
        if total_non_deleted == 0 {
            return Err(DBError::SearchError("No active points available to search.".into()));
        }
    
        let results = self.hnsw.in_place_filtered_search(
            query,
            top_k * 4,
            &self.payloads,
            &self.payload_index,
            filter,
        )?;
    
        let filtered: Vec<_> = results
            .into_iter()
            .filter(|sp| !self.deleted.contains(&sp.id))
            .take(top_k)
            .collect();
    
        Ok(filtered)
    }
    
    

    /// Internal unfiltered search (used for diagnostics or filtered versions).
    pub fn search_unfiltered(&self, query: &Vector, top_k: usize) -> Result<Vec<ScoredPoint>, DBError> {
        self.hnsw.search(query, top_k)
    }

    /// Get payload metadata for a point.
    pub fn get_payload(&self, point_id: PointId) -> Option<&Payload> {
        self.payloads.get(&point_id)
    }

    /// Check if a point is deleted.
    pub fn is_deleted(&self, point_id: PointId) -> bool {
        self.deleted.contains(&point_id)
    }

    pub fn purge(&mut self) -> Result<(), DBError> {
        let mut new_hnsw = HNSWIndex::new(
            self.hnsw.metric(),
            self.hnsw.m(),
            self.hnsw.ef(),
            self.hnsw.max_level_cap(),
            self.hnsw.dim(),
        );
    
        let mut new_payload_index = PayloadIndex::new();
        let mut new_payloads = HashMap::new();
    
        for (&id, vector) in self.hnsw.iter_vectors() {
            if self.deleted.contains(&id) {
                continue;
            }
    
            // Reinsert into HNSW
            new_hnsw.insert(id, vector.clone())?;
    
            if let Some(p) = self.payloads.get(&id) {
                // Reinsert into payload structures
                new_payload_index.insert(id, p);
                new_payloads.insert(id, p.clone());
    
                // Rebuild filter-aware edges
                let filter_keys = Self::filter_keys_for_payload(p);

                if Self::filter_edges_enabled() {
                    new_hnsw.build_filter_aware_edges(
                        id,
                        vector,
                        p,
                        &new_payload_index,
                        &new_payloads,
                        &filter_keys,
                    )?;
                }
            }
        }
    
        // Swap in the rebuilt structures
        self.hnsw = new_hnsw;
        self.payload_index = new_payload_index;
        self.payloads = new_payloads;
    
        self.deleted.clear();
    
        Ok(())
    }
     


    /// Immutable reference to underlying HNSW index
    pub fn hnsw(&self) -> &HNSWIndex {
        &self.hnsw
    }

    /// Immutable reference to point payloads
    pub fn payloads(&self) -> &HashMap<PointId, Payload> {
        &self.payloads
    }

    /// Mutable reference to underlying HNSW index (for tuning ef_construct in benches/tests).
    pub fn hnsw_mut(&mut self) -> &mut HNSWIndex {
        &mut self.hnsw
    }

    pub fn payload_index(&self) -> &PayloadIndex {
        &self.payload_index
    }

    /// Toggle filter-aware edge building via env var. Defaults to false.
    fn filter_edges_enabled() -> bool {
        env::var("VECTORDB_FILTER_EDGES")
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(false)
    }

    /// Toggle per-insert timing logs via env var. Defaults to false.
    fn log_insert_timing() -> bool {
        env::var("VECTORDB_LOG_INSERT_TIMING")
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(false)
    }

    /// Persist the entire segment (graph, vectors, payloads, and inverted index) to disk.
    pub fn save_to_path<P: AsRef<Path>>(&self, path: P) -> Result<(), DBError> {
        let snapshot = SegmentSnapshot {
            hnsw: self.hnsw.to_snapshot(),
            payload_index: self.payload_index.clone(),
            payloads: self.payloads.clone(),
            deleted: self.deleted.clone(),
            next_id: self.next_id,
        };
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(&SEGMENT_SNAPSHOT_MAGIC)?;
        writer.write_all(&SEGMENT_SNAPSHOT_VERSION.to_le_bytes())?;
        bincode::serialize_into(&mut writer, &snapshot)
            .map_err(|e| DBError::SerializationError(anyhow!(e)))?;
        Ok(())
    }

    /// Restore a segment that was previously persisted with `save_to_path`.
    pub fn load_from_path<P: AsRef<Path>>(path: P) -> Result<Self, DBError> {
        let bytes = std::fs::read(path)?;
        let snapshot: SegmentSnapshot = if bytes.len() >= 8 && bytes[..4] == SEGMENT_SNAPSHOT_MAGIC {
            let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
            match version {
                2 => bincode::deserialize(&bytes[8..])
                    .map_err(|e| DBError::SerializationError(anyhow!(e)))?,
                _ => {
                    return Err(DBError::SerializationError(anyhow!(
                        "unsupported segment snapshot version {}",
                        version
                    )))
                }
            }
        } else if let Ok(snapshot) = bincode::deserialize::<SegmentSnapshot>(&bytes) {
            snapshot
        } else {
            let legacy: SegmentSnapshotV1 = bincode::deserialize(&bytes)
                .map_err(|e| DBError::SerializationError(anyhow!(e)))?;
            legacy.into()
        };
        Ok(Self {
            hnsw: HNSWIndex::from_snapshot(snapshot.hnsw),
            payload_index: snapshot.payload_index,
            payloads: snapshot.payloads,
            deleted: snapshot.deleted,
            next_id: snapshot.next_id,
        })
    }

    /// Build the list of payload keys to use for filter-aware edges, honoring an optional allowlist,
    /// a max count, and a type preference (Bool -> Str -> Int -> Float).
    fn filter_keys_for_payload(payload: &Payload) -> Vec<String> {
        let allow: Option<HashSet<String>> = env::var("VECTORDB_FILTER_KEYS")
            .ok()
            .map(|v| {
                v.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            });
        let max_keys: Option<usize> = env::var("VECTORDB_FILTER_MAX_KEYS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|v| *v > 0);

        fn type_rank(v: &PayloadValue) -> usize {
            match v {
                PayloadValue::Bool(_) => 0,
                PayloadValue::Str(_) => 1,
                PayloadValue::Int(_) => 2,
                PayloadValue::Float(_) => 3,
                _ => 4,
            }
        }

        let mut keys_with_rank: Vec<(usize, String)> = payload
            .0
            .iter()
            .filter_map(|(k, v)| {
                if matches!(
                    v,
                    PayloadValue::Int(_) | PayloadValue::Float(_) | PayloadValue::Str(_) | PayloadValue::Bool(_)
                ) && allow.as_ref().map_or(true, |set| set.contains(k))
                {
                    Some((type_rank(v), k.clone()))
                } else {
                    None
                }
            })
            .collect();

        // Prefer cheaper/more selective types first, then deterministic key order.
        keys_with_rank.sort_by(|a, b| a.cmp(b));
        if let Some(cap) = max_keys {
            keys_with_rank.truncate(cap);
        }
        keys_with_rank.into_iter().map(|(_, k)| k).collect()
    }
}
