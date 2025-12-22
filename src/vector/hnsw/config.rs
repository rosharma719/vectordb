use std::fs::File;
use std::io::BufWriter;
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
static FILTER_SEARCH_SEQ: OnceLock<Mutex<u64>> = OnceLock::new();
static SEARCH_TRACE_LOG: OnceLock<Option<Mutex<BufWriter<File>>>> = OnceLock::new();
static INSERT_TRACE_LOG: OnceLock<Option<Mutex<BufWriter<File>>>> = OnceLock::new();
static TRACE_EVERY: OnceLock<usize> = OnceLock::new();
static SEARCH_TRACE_SEQ: OnceLock<Mutex<u64>> = OnceLock::new();
static INSERT_TRACE_SEQ: OnceLock<Mutex<u64>> = OnceLock::new();

pub(crate) fn log_unfiltered_enabled() -> bool {
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

pub(crate) fn search_expansion_multiplier() -> usize {
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
            .unwrap_or(0)
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
    let lock = FILTER_SEARCH_SEQ.get_or_init(|| Mutex::new(0));
    let mut guard = lock.lock().unwrap();
    *guard += 1;
    *guard
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
    let lock = SEARCH_TRACE_SEQ.get_or_init(|| Mutex::new(0));
    let mut guard = lock.lock().unwrap();
    *guard += 1;
    *guard
}

pub(crate) fn next_insert_trace_seq() -> u64 {
    let lock = INSERT_TRACE_SEQ.get_or_init(|| Mutex::new(0));
    let mut guard = lock.lock().unwrap();
    *guard += 1;
    *guard
}
