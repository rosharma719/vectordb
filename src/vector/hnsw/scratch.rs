use std::cell::RefCell;
use std::collections::BinaryHeap;

use super::types::{NodeCandidate, NodeResult, NodeRoutingEntry};

#[derive(Default)]
pub(crate) struct SearchScratch {
    pub(crate) visited_epoch: Vec<u32>,
    pub(crate) epoch: u32,
    pub(crate) candidate_queue: BinaryHeap<NodeCandidate>,
    pub(crate) result_set: BinaryHeap<NodeResult>,
    pub(crate) routing_pq: BinaryHeap<NodeRoutingEntry>,
    pub(crate) results_pq: BinaryHeap<NodeResult>,
    pub(crate) seed_epoch: Vec<u32>,
    pub(crate) seed_epoch_val: u32,
    pub(crate) seed_list: Vec<usize>,
    pub(crate) temp_epoch: Vec<u32>,
    pub(crate) temp_epoch_val: u32,
    pub(crate) temp_list: Vec<usize>,
}

impl SearchScratch {
    pub(crate) fn next_epoch(&mut self, len: usize) {
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

    pub(crate) fn mark_visited(&mut self, idx: usize) -> bool {
        if self.visited_epoch[idx] == self.epoch {
            false
        } else {
            self.visited_epoch[idx] = self.epoch;
            true
        }
    }

    pub(crate) fn reset_seed(&mut self, len: usize) {
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

    pub(crate) fn reset_temp(&mut self, len: usize) {
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

    pub(crate) fn mark_seed(&mut self, idx: usize) {
        if self.seed_epoch[idx] != self.seed_epoch_val {
            self.seed_epoch[idx] = self.seed_epoch_val;
            self.seed_list.push(idx);
        }
    }

    pub(crate) fn mark_temp(&mut self, idx: usize) {
        if self.temp_epoch[idx] != self.temp_epoch_val {
            self.temp_epoch[idx] = self.temp_epoch_val;
            self.temp_list.push(idx);
        }
    }

    pub(crate) fn is_seed(&self, idx: usize) -> bool {
        self.seed_epoch.get(idx).copied().unwrap_or(0) == self.seed_epoch_val
    }

    pub(crate) fn is_temp(&self, idx: usize) -> bool {
        self.temp_epoch.get(idx).copied().unwrap_or(0) == self.temp_epoch_val
    }
}

thread_local! {
    pub(crate) static SEARCH_SCRATCH: RefCell<SearchScratch> = RefCell::new(SearchScratch::default());
}
