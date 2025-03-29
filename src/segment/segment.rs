use std::collections::{HashMap, HashSet};

use crate::payload_storage::stores::PayloadIndex;
use crate::utils::errors::DBError;
use crate::utils::payload::Payload;
use crate::utils::types::{PointId, Vector};
use crate::vector::hnsw::{HNSWIndex, ScoredPoint};

/// A segment is the core unit that wraps vector storage, indexing, payloads, and deletion.
pub struct Segment {
    hnsw: HNSWIndex,
    payload_index: PayloadIndex,
    payloads: HashMap<PointId, Payload>,
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
            next_id: 0,
        }
    }

    /// Insert a new vector and optional payload. Auto-generates ID.
    pub fn insert(&mut self, vector: Vector, payload: Option<Payload>) -> Result<PointId, DBError> {
        let point_id = self.next_id;
        self.hnsw.insert(point_id, vector)?;

        if let Some(p) = payload {
            self.payload_index.insert(point_id, &p);
            self.payloads.insert(point_id, p);
        }

        self.next_id += 1;
        Ok(point_id)
    }

    pub fn delete(&mut self, point_id: PointId) -> Result<(), DBError> {
        if self.deleted.contains(&point_id) {
            return Ok(());
        }
    
        if !self.hnsw.contains(&point_id) {
            return Err(DBError::NotFound(point_id));
        }
    
        if let Some(p) = self.payloads.get(&point_id) {
            self.payload_index.remove(point_id, p);
        }
    
        self.deleted.insert(point_id);
    
        // 💡 Auto-rebuild if too many deletions
        let deleted_count = self.deleted.len();
        let total_count = self.hnsw.len(); // You might want to expose this in HNSW
        if deleted_count > 20 || deleted_count as f32 / total_count as f32 > 0.1 {
            self.purge().unwrap(); // implement `purge()` next
        }
    
        Ok(())
    }
    
    pub fn search(&self, query: &Vector, top_k: usize) -> Result<Vec<ScoredPoint>, DBError> {
        let total_non_deleted = self.hnsw.len() - self.deleted.len();
        if total_non_deleted == 0 {
            return Err(DBError::SearchError("No active points available to search.".into()));
        }
    
        let candidates = self.hnsw.search(query, top_k * 2)?;
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
    
        for (&id, vector) in self.hnsw.iter_vectors() {
            if !self.deleted.contains(&id) {
                new_hnsw.insert(id, vector.clone())?;
            }
        }
    
        let mut new_payload_index = PayloadIndex::new();
        for (&id, payload) in &self.payloads {
            if !self.deleted.contains(&id) {
                new_payload_index.insert(id, payload);
            }
        }
    
        self.hnsw = new_hnsw;
        self.payload_index = new_payload_index;
        self.deleted.clear();
    
        Ok(())
    }
    
    
}
