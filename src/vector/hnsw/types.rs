use crate::utils::types::{PointId, Score};

#[derive(Clone, Debug, PartialEq)]
pub struct ScoredPoint {
    pub id: PointId,
    pub raw_score: Score,
    pub sort_key: Score,
}

#[derive(Clone, Debug)]
pub(crate) struct NodeCandidate {
    pub(crate) idx: usize,
    pub(crate) raw_score: Score,
    pub(crate) sort_key: Score,
}

#[derive(Clone, Debug)]
pub(crate) struct NodeRoutingEntry {
    pub(crate) node: NodeCandidate,
    pub(crate) passes_filter: bool,
    pub(crate) budget: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct NodeResult(pub(crate) NodeCandidate);

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
