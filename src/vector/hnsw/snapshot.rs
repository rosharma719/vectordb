use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use anyhow::anyhow;

use crate::utils::errors::DBError;
use crate::utils::types::PointId;

use super::config::{exact_fallback_enabled_override, exact_fallback_threshold_override};
use super::{HNSWIndex, HnswSnapshot};

impl HNSWIndex {
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
            exact_fallback_enabled: exact_fallback_enabled_override().unwrap_or(false),
            exact_fallback_threshold: exact_fallback_threshold_override().unwrap_or(snapshot.exact_fallback_threshold),
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
