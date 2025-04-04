use std::collections::{HashMap, HashSet};

use crate::payload_storage::filters::{Filter, evaluate_filter};
use crate::payload_storage::stores::PayloadIndex;
use crate::utils::errors::DBError;
use crate::utils::payload::{Payload, PayloadValue};
use crate::utils::types::{PointId, Vector};
use crate::vector::hnsw::{HNSWIndex, ScoredPoint};

/// A segment is the core unit that wraps vector storage, indexing, payloads, and deletion.
pub struct Segment {
    hnsw: HNSWIndex,
    payload_index: PayloadIndex,
    payloads: HashMap<PointId, Payload>,
    // This set is maintained in parallel with the HNSW deletion set.
    deleted: HashSet<PointId>,
    next_id: PointId,
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
        self.hnsw.insert(point_id, vector.clone())?;

        if let Some(p) = payload {
            self.payload_index.insert(point_id, &p);
            self.payloads.insert(point_id, p.clone());

            let filter_keys: Vec<String> = p.0
                .iter()
                .filter(|(_, v)| matches!(v, PayloadValue::Int(_) | PayloadValue::Float(_) | PayloadValue::Str(_) | PayloadValue::Bool(_)))
                .map(|(k, _)| k.clone())
                .collect();

            if !filter_keys.is_empty() {
                self.hnsw.build_filter_aware_edges(
                    point_id,
                    &vector,
                    &p,
                    &self.payload_index,
                    &self.payloads,
                    &filter_keys,
                )?;
            }

        }

        self.next_id += 1;
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
            println!("[DELETE] Triggering purge: {}/{} ({:.2}%) deleted", deleted_count, total_count, 100.0 * deleted_count as f32 / total_count as f32);
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
        let candidates = self.hnsw.search(query, top_k * 2)?;
        // (The following filter is kept as extra safety.)
        let filtered = candidates
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
                let filter_keys: Vec<String> = p.0
                    .iter()
                    .filter(|(_, v)| matches!(v, PayloadValue::Int(_) | PayloadValue::Float(_) | PayloadValue::Str(_) | PayloadValue::Bool(_)))
                    .map(|(k, _)| k.clone())
                    .collect();
    
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
    
        // Swap in the rebuilt structures
        self.hnsw = new_hnsw;
        self.payload_index = new_payload_index;
        self.payloads = new_payloads;
    
        self.deleted.clear();
    
        Ok(())
    }
     

    /// Vector search with logical payload filtering
    pub fn post_filter(
        &self,
        query: &Vector,
        top_k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<ScoredPoint>, DBError> {
        let total_non_deleted = self.hnsw.len() - self.deleted.len();
        if total_non_deleted == 0 {
            return Err(DBError::SearchError("No active points available to search.".into()));
        }

        let candidates = self.hnsw.search(query, top_k * 4)?;

        let filtered = candidates
            .into_iter()
            .filter(|sp| {
                !self.deleted.contains(&sp.id)
                    && filter.map_or(true, |f| {
                        self.payloads
                            .get(&sp.id)
                            .map(|p| evaluate_filter(f, p).unwrap_or(false))
                            .unwrap_or(false)
                    })
            })
            .take(top_k)
            .collect();

        Ok(filtered)
    }

    /// Immutable reference to underlying HNSW index
    pub fn hnsw(&self) -> &HNSWIndex {
        &self.hnsw
    }

    /// Immutable reference to point payloads
    pub fn payloads(&self) -> &HashMap<PointId, Payload> {
        &self.payloads
    }

    pub fn payload_index(&self) -> &PayloadIndex {
        &self.payload_index
    }
}
