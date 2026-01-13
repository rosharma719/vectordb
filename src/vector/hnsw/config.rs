use std::fs::File;
use std::io::BufWriter;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

pub(crate) const VERBOSE: bool = false;
pub(crate) const DEFAULT_EXACT_FALLBACK_THRESHOLD: usize = 256;
pub(crate) const FILTER_EDGE_LOG_CHUNK: usize = 10_000;

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
static FILTER_SEARCH_SEQ: AtomicU64 = AtomicU64::new(0);
static SEARCH_TRACE_LOG: OnceLock<Option<Mutex<BufWriter<File>>>> = OnceLock::new();
static INSERT_TRACE_LOG: OnceLock<Option<Mutex<BufWriter<File>>>> = OnceLock::new();
static TRACE_EVERY: OnceLock<usize> = OnceLock::new();
static SEARCH_TRACE_SEQ: AtomicU64 = AtomicU64::new(0);
static INSERT_TRACE_SEQ: AtomicU64 = AtomicU64::new(0);
static EXACT_FALLBACK_ENABLED: OnceLock<Option<bool>> = OnceLock::new();
static EXACT_FALLBACK_THRESHOLD: OnceLock<Option<usize>> = OnceLock::new();
static FILTER_ENTRY_CANDIDATES: OnceLock<Option<usize>> = OnceLock::new();
static NEIGHBOR_SCAN_CAP_LEVEL0: OnceLock<Option<usize>> = OnceLock::new();
static NEIGHBOR_SCAN_ROTATE: OnceLock<Option<bool>> = OnceLock::new();
static NEIGHBOR_SCAN_STRIDE: OnceLock<Option<bool>> = OnceLock::new();
static NEIGHBOR_SCAN_PATIENCE: OnceLock<Option<usize>> = OnceLock::new();
static NEIGHBOR_SCAN_STATE_LOGGED: OnceLock<()> = OnceLock::new();
static DIVERSITY_ALPHA: OnceLock<Option<f32>> = OnceLock::new();
static DIVERSITY_ALPHA_LOW: OnceLock<Option<f32>> = OnceLock::new();
static DIVERSITY_ALPHA_HIGH: OnceLock<Option<f32>> = OnceLock::new();
static DIVERSITY_PRUNE_FLOOR: OnceLock<Option<usize>> = OnceLock::new();

pub(crate) fn log_unfiltered_enabled() -> bool {
    *LOG_UNFILTERED_SEARCH.get_or_init(|| {
        let enabled = std::env::var("VECTORDB_LOG_UNFILTERED_SEARCH")
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(false);
        if enabled {
            let chunk = unfiltered_log_chunk();
            log::info!(
                target: "vector::hnsw",
                "[unfiltered_search_stats] enabled (chunk={}, set VECTORDB_LOG_UNFILTERED_EVERY to change)",
                chunk
            );
        }
        enabled
    })
}

pub(crate) fn unfiltered_log_chunk() -> usize {
    *UNFILTERED_LOG_CHUNK.get_or_init(|| {
        std::env::var("VECTORDB_LOG_UNFILTERED_EVERY")
            .ok()
            .and_then(|v| v.replace('_', "").parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(1000)
    })
}

pub(crate) fn filter_seed_log_chunk() -> usize {
    *FILTER_SEED_LOG_CHUNK.get_or_init(|| {
        std::env::var("VECTORDB_LOG_FILTER_SEED_EVERY")
            .ok()
            .and_then(|v| v.replace('_', "").parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(100)
    })
}

pub(crate) fn disable_early_exit() -> bool {
    *DISABLE_EARLY_EXIT.get_or_init(|| {
        std::env::var("VECTORDB_DISABLE_EARLY_EXIT")
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(false)
    })
}

pub fn search_expansion_multiplier() -> usize {
    *SEARCH_EXPANSION_MULT.get_or_init(|| {
        std::env::var("VECTORDB_SEARCH_EXPANSION_MULT")
            .ok()
            .and_then(|v| v.replace('_', "").parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(1)
    })
}

pub(crate) fn search_expansion_cap_override() -> Option<usize> {
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

pub(crate) fn early_exit_patience() -> usize {
    *EARLY_EXIT_PATIENCE.get_or_init(|| {
        std::env::var("VECTORDB_EARLY_EXIT_PATIENCE")
            .ok()
            .and_then(|v| v.replace('_', "").parse::<usize>().ok())
            .unwrap_or(2)
    })
}

pub(crate) fn filter_expansion_cap() -> Option<usize> {
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

pub(crate) fn filter_passing_budget(m: usize) -> usize {
    FILTER_PASSING_BUDGET
        .get_or_init(|| {
            std::env::var("VECTORDB_FILTER_PASSING_BUDGET")
                .ok()
                .and_then(|v| v.replace('_', "").parse::<usize>().ok())
        })
        .clone()
        .unwrap_or_else(|| std::cmp::max(8, m.saturating_mul(2)))
}

pub(crate) fn filter_failing_budget(m: usize) -> usize {
    FILTER_FAILING_BUDGET
        .get_or_init(|| {
            std::env::var("VECTORDB_FILTER_FAILING_BUDGET")
                .ok()
                .and_then(|v| v.replace('_', "").parse::<usize>().ok())
        })
        .clone()
        .unwrap_or_else(|| std::cmp::max(1, m / 8))
}

pub(crate) fn filter_search_logger() -> Option<&'static Mutex<BufWriter<File>>> {
    FILTER_SEARCH_LOG
        .get_or_init(|| {
            std::env::var("VECTORDB_FILTER_SEARCH_LOG")
                .ok()
                .and_then(|path| File::create(path).ok())
                .map(|f| Mutex::new(BufWriter::new(f)))
        })
        .as_ref()
}

pub(crate) fn next_filter_search_seq() -> u64 {
    FILTER_SEARCH_SEQ.fetch_add(1, Ordering::Relaxed) + 1
}

pub(crate) fn search_trace_logger() -> Option<&'static Mutex<BufWriter<File>>> {
    SEARCH_TRACE_LOG
        .get_or_init(|| {
            std::env::var("VECTORDB_SEARCH_TRACE_LOG")
                .ok()
                .and_then(|path| File::create(path).ok())
                .map(|f| Mutex::new(BufWriter::new(f)))
        })
        .as_ref()
}

pub(crate) fn insert_trace_logger() -> Option<&'static Mutex<BufWriter<File>>> {
    INSERT_TRACE_LOG
        .get_or_init(|| {
            std::env::var("VECTORDB_INSERT_TRACE_LOG")
                .ok()
                .and_then(|path| File::create(path).ok())
                .map(|f| Mutex::new(BufWriter::new(f)))
        })
        .as_ref()
}

pub(crate) fn trace_every() -> usize {
    *TRACE_EVERY.get_or_init(|| {
        std::env::var("VECTORDB_TRACE_EVERY")
            .ok()
            .and_then(|v| v.replace('_', "").parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(100)
    })
}

pub(crate) fn next_search_trace_seq() -> u64 {
    SEARCH_TRACE_SEQ.fetch_add(1, Ordering::Relaxed) + 1
}

pub(crate) fn next_insert_trace_seq() -> u64 {
    INSERT_TRACE_SEQ.fetch_add(1, Ordering::Relaxed) + 1
}

pub(crate) fn exact_fallback_enabled_override() -> Option<bool> {
    *EXACT_FALLBACK_ENABLED.get_or_init(|| {
        std::env::var("VECTORDB_EXACT_FALLBACK_ENABLED")
            .ok()
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
    })
}

pub(crate) fn exact_fallback_threshold_override() -> Option<usize> {
    EXACT_FALLBACK_THRESHOLD
        .get_or_init(|| {
            std::env::var("VECTORDB_EXACT_FALLBACK_THRESHOLD")
                .ok()
                .and_then(|v| v.replace('_', "").parse::<usize>().ok())
        })
        .clone()
}

pub(crate) fn filter_entry_candidates() -> Option<usize> {
    FILTER_ENTRY_CANDIDATES
        .get_or_init(|| {
            std::env::var("VECTORDB_FILTER_ENTRY_CANDIDATES")
                .ok()
                .and_then(|v| v.replace('_', "").parse::<usize>().ok())
                .filter(|v| *v > 0)
        })
        .clone()
}

pub fn neighbor_scan_cap(level: usize) -> usize {
    if level == 0 {
        NEIGHBOR_SCAN_CAP_LEVEL0
            .get_or_init(|| {
                std::env::var("VECTORDB_NEIGHBOR_SCAN_CAP_LEVEL0")
                    .ok()
                    .and_then(|v| v.replace('_', "").parse::<usize>().ok())
                    .map(|v| if v == 0 { usize::MAX } else { v })
            })
            .unwrap_or(64)
    } else {
        usize::MAX
    }
}

pub fn neighbor_scan_patience() -> usize {
    NEIGHBOR_SCAN_PATIENCE
        .get_or_init(|| {
            std::env::var("VECTORDB_NEIGHBOR_SCAN_PATIENCE")
                .ok()
                .and_then(|v| v.replace('_', "").parse::<usize>().ok())
                .filter(|&v| v > 0)
        })
        .clone()
        .unwrap_or(0)
}

pub fn neighbor_scan_rotate_enabled() -> bool {
    NEIGHBOR_SCAN_ROTATE
        .get_or_init(|| {
            std::env::var("VECTORDB_NEIGHBOR_SCAN_ROTATE")
                .ok()
                .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
        })
        .unwrap_or(false)
}

pub fn neighbor_scan_stride_enabled() -> bool {
    NEIGHBOR_SCAN_STRIDE
        .get_or_init(|| {
            std::env::var("VECTORDB_NEIGHBOR_SCAN_STRIDE")
                .ok()
                .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
        })
        .unwrap_or(false)
}

pub(crate) fn log_neighbor_scan_state(expansion_mult: usize, expansion_cap: Option<usize>) {
    NEIGHBOR_SCAN_STATE_LOGGED.get_or_init(|| {
        let cap = neighbor_scan_cap(0);
        let rotate = neighbor_scan_rotate_enabled();
        let stride = neighbor_scan_stride_enabled();
        log::info!(
            target: "vector::hnsw",
            "Neighbor scan cap L0={} rotation={} rotation_stride={} expansion_mult={} expansion_cap={}",
            cap,
            if rotate { "on" } else { "off" },
            if stride { "enabled" } else { "off" },
            expansion_mult,
            expansion_cap.map(|v| v.to_string()).unwrap_or_else(|| "none".into())
        );
    });
}

pub(crate) fn diversity_alpha_for_level(level: usize) -> f32 {
    if let Some(alpha) = DIVERSITY_ALPHA.get_or_init(|| {
        std::env::var("VECTORDB_DIVERSITY_ALPHA")
            .ok()
            .and_then(|v| v.replace('_', "").parse::<f32>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
    }) {
        return *alpha;
    }
    let low = DIVERSITY_ALPHA_LOW
        .get_or_init(|| {
            std::env::var("VECTORDB_DIVERSITY_ALPHA_LOW")
                .ok()
                .and_then(|v| v.replace('_', "").parse::<f32>().ok())
                .filter(|v| v.is_finite() && *v > 0.0)
        })
        .unwrap_or(1.0);
    let high = DIVERSITY_ALPHA_HIGH
        .get_or_init(|| {
            std::env::var("VECTORDB_DIVERSITY_ALPHA_HIGH")
                .ok()
                .and_then(|v| v.replace('_', "").parse::<f32>().ok())
                .filter(|v| v.is_finite() && *v > 0.0)
        })
        .unwrap_or(1.2);
    if level == 0 { low } else { high }
}

pub(crate) fn diversity_prune_floor() -> usize {
    DIVERSITY_PRUNE_FLOOR
        .get_or_init(|| {
            std::env::var("VECTORDB_DIVERSITY_PRUNE_FLOOR")
                .ok()
                .and_then(|v| v.replace('_', "").parse::<usize>().ok())
                .filter(|v| *v > 0)
        })
        .unwrap_or(4)
}
