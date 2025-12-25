use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::Path;
use std::time::Instant;

use ndarray::Array2;
use ndarray_npy::read_npy;
use serde::Serialize;
use serde_json::from_slice;

mod common;
use common::{
    QueryLogWriter,
    QueryStatsAgg,
    SearchConfig,
    SnapshotConfig,
    TestLogConfig,
    env_usize_first,
    log_peak_rss,
    summarize_f64,
    summarize_usize,
    MissStats,
};

use vectordb::segment::segment::Segment;
use vectordb::utils::types::{DistanceMetric, Vector};
use vectordb::vector::hnsw::HNSWIndex;

fn load_vectors(path: &Path) -> Vec<Vector> {
    let arr: Array2<f32> = read_npy(path).expect("failed to read .npy");
    arr.rows().into_iter().map(|r| r.to_vec()).collect()
}

fn load_ground_truth(path: &Path) -> Vec<Vec<usize>> {
    let data = fs::read(path).expect("failed to read ground truth json");
    from_slice(&data).expect("failed to parse ground truth json")
}

fn build_nytimes_segment(segment: &mut Segment, base: &[Vector], logs: &TestLogConfig) {
    logs.log_info(&format!(
        "🚀 Inserting {} vectors with dataset-aligned IDs...",
        base.len()
    ));
    let start_insert = Instant::now();
    let mut last_log = start_insert;
    for (i, v) in base.iter().enumerate() {
        let dataset_id = i as u64;
        segment.insert_with_id(dataset_id, v.clone(), None).unwrap();
        if logs.level.allows_debug() && i != 0 && i % 1000 == 0 {
            let now = Instant::now();
            logs.log_debug(&format!(
                "Inserted {} vectors (+{:?}, chunk={:?})",
                i,
                now - start_insert,
                now - last_log
            ));
            last_log = now;
        }
    }
    let insert_dur = start_insert.elapsed();
    let insert_ms = insert_dur.as_secs_f64() * 1000.0 / base.len().max(1) as f64;
    logs.log_info(&format!(
        "✅ Inserted {} vectors in {:?} (~{:.3} ms/insert)",
        base.len(),
        insert_dur,
        insert_ms
    ));
}

fn persist_segment(segment: &Segment, path: &str, logs: &TestLogConfig) {
    logs.log_info(&format!("💾 Persisting NYTimes segment to {} ...", path));
    let start = Instant::now();
    segment
        .save_to_path(path)
        .expect("failed to persist NYTimes segment");
    let elapsed = start.elapsed();
    let size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    logs.log_info(&format!(
        "✅ Saved NYTimes segment to {} (size={} bytes, elapsed={:?})",
        path, size, elapsed
    ));
}

fn ensure_exists(path: &Path) {
    assert!(
        path.exists(),
        "missing required dataset file: {} (run the download script in README)",
        path.display()
    );
}

fn read_search_caps_from_env() -> (bool, usize, Option<usize>) {
    let disable_early_exit = env::var("VECTORDB_DISABLE_EARLY_EXIT")
        .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
        .unwrap_or(false);
    let expansion_mult = env::var("VECTORDB_SEARCH_EXPANSION_MULT")
        .ok()
        .and_then(|v| v.replace('_', "").parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(4);
    let expansion_cap = env::var("VECTORDB_SEARCH_EXPANSION_CAP")
        .ok()
        .and_then(|v| v.replace('_', "").parse::<usize>().ok())
        .and_then(|v| if v == 0 { None } else { Some(v) });
    (disable_early_exit, expansion_mult, expansion_cap)
}

fn log_search_caps(logs: &TestLogConfig) {
    let (disable_early_exit, expansion_mult, expansion_cap) = read_search_caps_from_env();
    let patience = env::var("VECTORDB_EARLY_EXIT_PATIENCE")
        .ok()
        .and_then(|v| v.replace('_', "").parse::<usize>().ok())
        .unwrap_or(0);
    let cap_display = expansion_cap
        .map(|v| v.to_string())
        .unwrap_or_else(|| "none".to_string());
    logs.log_info(&format!(
        "⚙️  Query-time knobs: early_exit_disabled={} early_exit_patience={} expansion_mult={} expansion_cap={}",
        disable_early_exit, patience, expansion_mult, cap_display
    ));
}

#[derive(Serialize)]
struct QueryLogEntry {
    dataset: &'static str,
    query_idx: usize,
    ef_search: usize,
    top_k: usize,
    elapsed_ms: f64,
    visited: Option<usize>,
    expanded: Option<usize>,
    results_len: usize,
    recall: f64,
    misses: usize,
}

/// Uses the ANN-Benchmarks NYTimes 256-d Angular dataset from Hugging Face.
/// Requires local files:
///   base.npy, queries.npy, ground_truth.json in data/nytimes-256-angular (by default).
#[test]
#[ignore]
fn nytimes_256_angular_perf_and_recall() {
    run_nytimes_perf_and_recall(TestMode::Default);
}

#[test]
#[ignore]
fn nytimes_build_and_persist_snapshot_only() {
    run_nytimes_perf_and_recall(TestMode::BuildOnly);
}

#[test]
#[ignore]
fn nytimes_recall_from_snapshot() {
    run_nytimes_recall_only();
}

#[test]
#[ignore]
fn nytimes_qps_latency_curve() {
    run_nytimes_qps_latency_curve();
}

#[derive(Clone, Copy)]
enum TestMode {
    Default,
    BuildOnly,
}

fn run_nytimes_perf_and_recall(mode: TestMode) {
    let t0 = Instant::now();
    let logs = TestLogConfig::from_env();
    let data_dir = env::var("VECTORDB_NYT_DATA_DIR")
        .unwrap_or_else(|_| "data/nytimes-256-angular".to_string());
    let search = SearchConfig::from_env("NYT", &[32, 64, 128, 256, 512], 1000);
    let mut snapshot = SnapshotConfig::from_env(
        "NYT",
        "data/nytimes-256-angular/index_m16_m0_32_efc100.bin",
    );
    if matches!(mode, TestMode::BuildOnly) {
        snapshot.use_snapshot = false;
        snapshot.allow_build = true;
        snapshot.save_snapshot = true;
    }
    let top_k = search.top_k.unwrap_or(20);

    let base_path = Path::new(&data_dir).join("base.npy");
    let queries_path = Path::new(&data_dir).join("queries.npy");
    let truth_path = Path::new(&data_dir).join("ground_truth.json");
    ensure_exists(&base_path);
    ensure_exists(&queries_path);
    ensure_exists(&truth_path);
    log_search_caps(&logs);

    logs.log_info(&format!(
        "\n📚 Loading NYTimes dataset from {} (top_k={}, ef_search_list={:?}, max_queries={}, base_cap={:?}, ef_construct={})",
        data_dir, top_k, search.ef_values, search.queries_cap, search.base_cap, search.ef_construct
    ));

    let mut base = Vec::new();
    if !snapshot.use_snapshot || snapshot.allow_build {
        base = load_vectors(&base_path);
        if let Some(cap) = search.base_cap {
            if cap < base.len() {
                base.truncate(cap);
            }
        }
    }
    let queries = load_vectors(&queries_path);
    let ground_truth = load_ground_truth(&truth_path);
    assert_eq!(ground_truth.len(), queries.len(), "ground truth length must match queries");
    logs.log_info(&format!("⏱️  Data loaded in {:?}", t0.elapsed()));

    let mut segment = if snapshot.use_snapshot {
        if !Path::new(&snapshot.persist_path).exists() {
            if snapshot.allow_build {
                logs.log_info("💾 Snapshot missing; building a fresh segment...");
                let dim = base.first().map(|v| v.len()).unwrap_or(0);
                assert_eq!(dim, 256, "expected 256-d vectors");
                let metric = DistanceMetric::Cosine;
                let m = env_usize_first(&["VECTORDB_M", "VECTORDB_NYT_M"]).unwrap_or(16);
                let m0 = env_usize_first(&["VECTORDB_M0", "VECTORDB_NYT_M0"]).unwrap_or(m);
                let max_level = env_usize_first(&["VECTORDB_MAX_LEVEL", "VECTORDB_NYT_MAX_LEVEL"]).unwrap_or(16);
                let mut segment = Segment::new(HNSWIndex::new(
                    metric,
                    m,
                    search.ef_values.iter().copied().max().unwrap_or(1).max(top_k),
                    max_level,
                    dim,
                ));
                segment.hnsw_mut().set_m0(m0);
                segment.hnsw_mut().set_ef_construct(search.ef_construct);
                build_nytimes_segment(&mut segment, &base, &logs);
                if snapshot.save_snapshot {
                    persist_segment(&segment, &snapshot.persist_path, &logs);
                }
                segment
            } else {
                panic!(
                    "missing snapshot at {} (set VECTORDB_NYT_ALLOW_BUILD=1 or VECTORDB_ALLOW_BUILD=1 to build)",
                    snapshot.persist_path
                );
            }
        } else {
            logs.log_info(&format!(
                "💾 Loading persisted NYTimes segment from {} ...",
                snapshot.persist_path
            ));
            let (segment, metadata) = Segment::load_from_path_with_metadata(&snapshot.persist_path)
                .expect("failed to load persisted NYTimes segment");
            if let Some(meta) = metadata {
                logs.log_info(&format!(
                    "🧾 Snapshot metadata: created_at_ms={} points={} payloads={}",
                    meta.created_at_ms, meta.points, meta.payloads
                ));
            }
            segment
        }
    } else {
        let dim = base.first().map(|v| v.len()).unwrap_or(0);
        assert_eq!(dim, 256, "expected 256-d vectors");
        let metric = DistanceMetric::Cosine;
        let m = env_usize_first(&["VECTORDB_M", "VECTORDB_NYT_M"]).unwrap_or(16);
        let m0 = env_usize_first(&["VECTORDB_M0", "VECTORDB_NYT_M0"]).unwrap_or(m);
        let max_level = env_usize_first(&["VECTORDB_MAX_LEVEL", "VECTORDB_NYT_MAX_LEVEL"]).unwrap_or(16);
        let mut segment = Segment::new(HNSWIndex::new(
            metric,
            m,
            search.ef_values.iter().copied().max().unwrap_or(1).max(top_k),
            max_level,
            dim,
        ));
        segment.hnsw_mut().set_m0(m0);
        segment.hnsw_mut().set_ef_construct(search.ef_construct);
        build_nytimes_segment(&mut segment, &base, &logs);
        if snapshot.save_snapshot {
            persist_segment(&segment, &snapshot.persist_path, &logs);
        }
        segment
    };

    let num_queries = queries.len().min(search.queries_cap);
    let mut query_logger = QueryLogWriter::new(logs.query_log_path.clone(), logs.query_log_every);
    logs.log_info(&format!(
        "🔍 Sweeping ef_search over {:?} for {} queries (top_k={})...",
        search.ef_values, num_queries, top_k
    ));

    let mut summary: Vec<(usize, f64, f64)> = Vec::new();
    for &ef in &search.ef_values {
        let ef_search = ef.max(top_k);
        segment.hnsw_mut().set_ef_search(ef_search);
        let mut hits = 0usize;
        let mut total_targets = 0usize;
        let mut stats = QueryStatsAgg::new(num_queries);
        let mut miss_stats = MissStats::default();
        let start_search = Instant::now();
        for (qi, q) in queries.iter().take(num_queries).enumerate() {
            let q_start = Instant::now();
            let (approx, qstats) = segment.search_with_stats(q, top_k).unwrap();
            let truth = &ground_truth[qi];
            let truth_k = truth.len().min(top_k);
            let truth_set: HashSet<_> = truth
                .iter()
                .take(truth_k)
                // Ground truth IDs are 0-based row indices; we insert with the same IDs.
                .filter(|&&id| (id as usize) < segment.hnsw().len())
                .map(|&id| id as u64)
                .collect();
            total_targets += truth_set.len();
            let query_hits = approx.iter().filter(|r| truth_set.contains(&r.id)).count();
            hits += query_hits;
            let recall = query_hits as f64 / truth_set.len().max(1) as f64;
            let misses = truth_set.len().saturating_sub(query_hits);
            let elapsed_ms = q_start.elapsed().as_secs_f64() * 1000.0;
            stats.record(elapsed_ms, Some(qstats.visited), Some(qstats.expanded), recall);
            for miss_id in truth_set.iter().filter(|id| !approx.iter().any(|r| r.id == **id)) {
                if let Some(level) = segment.hnsw().point_level(*miss_id) {
                    let degree = segment.hnsw().point_degree(*miss_id, 0);
                    miss_stats.record(level, degree);
                }
            }
            query_logger.write(
                qi,
                &QueryLogEntry {
                    dataset: "nytimes-256-angular",
                    query_idx: qi,
                    ef_search,
                    top_k,
                    elapsed_ms,
                    visited: Some(qstats.visited),
                    expanded: Some(qstats.expanded),
                    results_len: approx.len(),
                    recall,
                    misses,
                },
            );
            if logs.level.allows_debug() && ((qi + 1) % logs.progress_every == 0 || qi + 1 == num_queries) {
                let partial_recall = hits as f64 / total_targets.max(1) as f64;
                logs.log_debug(&format!(
                    "  progress: query {}/{} (ef_search={}) cumulative recall={:.3}",
                    qi + 1,
                    num_queries,
                    ef_search,
                    partial_recall
                ));
            }
        }
        let search_dur = start_search.elapsed();
        let avg_ms = search_dur.as_secs_f64() * 1000.0 / num_queries as f64;
        let recall = hits as f64 / total_targets.max(1) as f64;
        logs.log_info(&format!(
            "🎯 [ef_search={}] recall@{}: {:.3} over {} queries (hits {}/{}) | avg {:.3} ms/query",
            ef_search,
            top_k,
            recall,
            num_queries,
            hits,
            total_targets,
            avg_ms
        ));
        let latency = summarize_f64(&stats.elapsed_ms);
        let visited = summarize_usize(&stats.visited);
        let expanded = summarize_usize(&stats.expanded);
        logs.log_info(&format!(
            "[query_stats] ef_search={} ms(p50/p90/p99)={:.3}/{:.3}/{:.3} visited(p50/p90/p99)={:.0}/{:.0}/{:.0} expanded(p50/p90/p99)={:.0}/{:.0}/{:.0}",
            ef_search,
            latency.p50,
            latency.p90,
            latency.p99,
            visited.p50,
            visited.p90,
            visited.p99,
            expanded.p50,
            expanded.p90,
            expanded.p99
        ));
        if miss_stats.total > 0 {
            let degree = summarize_usize(&miss_stats.degree_samples);
            let levels: Vec<String> = miss_stats
                .level_counts
                .iter()
                .enumerate()
                .map(|(lvl, count)| format!("L{}={}", lvl, count))
                .collect();
            logs.log_info(&format!(
                "[miss_stats] ef_search={} total={} levels={} degree(p50/p90/p99)={:.0}/{:.0}/{:.0}",
                ef_search,
                miss_stats.total,
                levels.join(","),
                degree.p50,
                degree.p90,
                degree.p99
            ));
        }
        // Flush any unfiltered search stats for this sweep so logging is emitted promptly.
        segment.hnsw().flush_unfiltered_search_stats();
        summary.push((ef_search, recall, avg_ms));
    }

    logs.log_info("\nSummary (ef_search -> recall, ms/query):");
    for (ef, recall, ms) in summary {
        logs.log_info(&format!("  {} -> {:.3}, {:.3} ms/query", ef, recall, ms));
    }
}

fn run_nytimes_recall_only() {
    let t0 = Instant::now();
    let logs = TestLogConfig::from_env();
    let data_dir = env::var("VECTORDB_NYT_DATA_DIR")
        .unwrap_or_else(|_| "data/nytimes-256-angular".to_string());
    let search = SearchConfig::from_env("NYT", &[32, 64, 128, 256, 512], 1000);
    let snapshot = SnapshotConfig::from_env(
        "NYT",
        "data/nytimes-256-angular/index_m16_m0_32_efc100.bin",
    );
    let top_k = search.top_k.unwrap_or(20);

    let queries_path = Path::new(&data_dir).join("queries.npy");
    let truth_path = Path::new(&data_dir).join("ground_truth.json");
    ensure_exists(&queries_path);
    ensure_exists(&truth_path);
    log_search_caps(&logs);

    let queries = load_vectors(&queries_path);
    let ground_truth = load_ground_truth(&truth_path);
    assert_eq!(ground_truth.len(), queries.len(), "ground truth length must match queries");
    logs.log_info(&format!("⏱️  Queries and truth loaded in {:?}", t0.elapsed()));
    log_peak_rss("nytimes_qps_loaded_queries");
    log_peak_rss("nytimes_loaded_queries");

    if !snapshot.use_snapshot {
        panic!("nytimes_recall_from_snapshot requires VECTORDB_USE_SNAPSHOT=1");
    }
    if !Path::new(&snapshot.persist_path).exists() {
        panic!(
            "missing snapshot at {} (run nytimes_build_and_persist_snapshot_only first)",
            snapshot.persist_path
        );
    }
    logs.log_info(&format!(
        "💾 Loading persisted NYTimes segment from {} ...",
        snapshot.persist_path
    ));
    let (mut segment, metadata) = Segment::load_from_path_with_metadata(&snapshot.persist_path)
        .expect("failed to load persisted NYTimes segment");
    let cfg = segment.hnsw().config_summary();
    logs.log_info(&format!(
        "✅ Loaded persisted segment with {} vectors (payloads={})",
        segment.hnsw().len(),
        segment.payloads().len()
    ));
    log_peak_rss("nytimes_qps_loaded_snapshot");
    log_peak_rss("nytimes_loaded_snapshot");
    logs.log_info(&format!(
        "🧭 HNSW config: metric={:?} dim={} m={} m0={} ef={} ef_construct={} level_cap={} level_scale={:.3} max_level={} exact_fallback={} threshold={}",
        cfg.metric,
        cfg.dim,
        cfg.m,
        cfg.m0,
        cfg.ef,
        cfg.ef_construct,
        cfg.max_level_cap,
        cfg.level_scale,
        cfg.current_max_level,
        cfg.exact_fallback_enabled,
        cfg.exact_fallback_threshold
    ));
    if let Some(meta) = metadata {
        logs.log_info(&format!(
            "🧾 Snapshot metadata: created_at_ms={} points={} payloads={}",
            meta.created_at_ms, meta.points, meta.payloads
        ));
    }

    let num_queries = queries.len().min(search.queries_cap);
    let mut query_logger = QueryLogWriter::new(logs.query_log_path.clone(), logs.query_log_every);
    logs.log_info(&format!(
        "🔍 Sweeping ef_search over {:?} for {} queries (top_k={})...",
        search.ef_values, num_queries, top_k
    ));
    log_peak_rss("nytimes_qps_before_sweep");
    log_peak_rss("nytimes_before_sweep");

    let mut summary: Vec<(usize, f64, f64)> = Vec::new();
    for &ef in &search.ef_values {
        let ef_search = ef.max(top_k);
        segment.hnsw_mut().set_ef_search(ef_search);
        let mut hits = 0usize;
        let mut total_targets = 0usize;
        let mut stats = QueryStatsAgg::new(num_queries);
        let mut miss_stats = MissStats::default();
        let start_search = Instant::now();
        for (qi, q) in queries.iter().take(num_queries).enumerate() {
            let q_start = Instant::now();
            let (approx, qstats) = segment.search_with_stats(q, top_k).unwrap();
            let truth = &ground_truth[qi];
            let truth_k = truth.len().min(top_k);
            let truth_set: HashSet<_> = truth
                .iter()
                .take(truth_k)
                // Ground truth IDs are 0-based row indices.
                .map(|&id| id as u64)
                .collect();
            total_targets += truth_set.len();
            let query_hits = approx.iter().filter(|r| truth_set.contains(&r.id)).count();
            hits += query_hits;
            let recall = query_hits as f64 / truth_set.len().max(1) as f64;
            let misses = truth_set.len().saturating_sub(query_hits);
            let elapsed_ms = q_start.elapsed().as_secs_f64() * 1000.0;
            stats.record(elapsed_ms, Some(qstats.visited), Some(qstats.expanded), recall);
            for miss_id in truth_set.iter().filter(|id| !approx.iter().any(|r| r.id == **id)) {
                if let Some(level) = segment.hnsw().point_level(*miss_id) {
                    let degree = segment.hnsw().point_degree(*miss_id, 0);
                    miss_stats.record(level, degree);
                }
            }
            query_logger.write(
                qi,
                &QueryLogEntry {
                    dataset: "nytimes-256-angular",
                    query_idx: qi,
                    ef_search,
                    top_k,
                    elapsed_ms,
                    visited: Some(qstats.visited),
                    expanded: Some(qstats.expanded),
                    results_len: approx.len(),
                    recall,
                    misses,
                },
            );
            if logs.level.allows_debug() && ((qi + 1) % logs.progress_every == 0 || qi + 1 == num_queries) {
                let partial_recall = hits as f64 / total_targets.max(1) as f64;
                logs.log_debug(&format!(
                    "  progress: query {}/{} (ef_search={}) cumulative recall={:.3}",
                    qi + 1,
                    num_queries,
                    ef_search,
                    partial_recall
                ));
            }
        }
        let search_dur = start_search.elapsed();
        let avg_ms = search_dur.as_secs_f64() * 1000.0 / num_queries as f64;
        let recall = hits as f64 / total_targets.max(1) as f64;
        logs.log_info(&format!(
            "🎯 [ef_search={}] recall@{}: {:.3} over {} queries (hits {}/{}) | avg {:.3} ms/query",
            ef_search,
            top_k,
            recall,
            num_queries,
            hits,
            total_targets,
            avg_ms
        ));
        let latency = summarize_f64(&stats.elapsed_ms);
        let visited = summarize_usize(&stats.visited);
        let expanded = summarize_usize(&stats.expanded);
        logs.log_info(&format!(
            "[query_stats] ef_search={} ms(p50/p90/p99)={:.3}/{:.3}/{:.3} visited(p50/p90/p99)={:.0}/{:.0}/{:.0} expanded(p50/p90/p99)={:.0}/{:.0}/{:.0}",
            ef_search,
            latency.p50,
            latency.p90,
            latency.p99,
            visited.p50,
            visited.p90,
            visited.p99,
            expanded.p50,
            expanded.p90,
            expanded.p99
        ));
        if miss_stats.total > 0 {
            let degree = summarize_usize(&miss_stats.degree_samples);
            let levels: Vec<String> = miss_stats
                .level_counts
                .iter()
                .enumerate()
                .map(|(lvl, count)| format!("L{}={}", lvl, count))
                .collect();
            logs.log_info(&format!(
                "[miss_stats] ef_search={} total={} levels={} degree(p50/p90/p99)={:.0}/{:.0}/{:.0}",
                ef_search,
                miss_stats.total,
                levels.join(","),
                degree.p50,
                degree.p90,
                degree.p99
            ));
        }
        segment.hnsw().flush_unfiltered_search_stats();
        summary.push((ef_search, recall, avg_ms));
    }

    logs.log_info("\nSummary (ef_search -> recall, ms/query):");
    for (ef, recall, ms) in summary {
        logs.log_info(&format!("  {} -> {:.3}, {:.3} ms/query", ef, recall, ms));
    }
    log_peak_rss("nytimes_recall_complete");
}

fn run_nytimes_qps_latency_curve() {
    let t0 = Instant::now();
    let logs = TestLogConfig::from_env();
    let data_dir = env::var("VECTORDB_NYT_DATA_DIR")
        .unwrap_or_else(|_| "data/nytimes-256-angular".to_string());
    let search = SearchConfig::from_env("NYT", &[32, 64, 128, 256, 512], 1000);
    let snapshot = SnapshotConfig::from_env(
        "NYT",
        "data/nytimes-256-angular/index_m16_m0_32_efc100.bin",
    );
    let top_k = search.top_k.unwrap_or(20);

    let queries_path = Path::new(&data_dir).join("queries.npy");
    let truth_path = Path::new(&data_dir).join("ground_truth.json");
    ensure_exists(&queries_path);
    ensure_exists(&truth_path);
    log_search_caps(&logs);

    let queries = load_vectors(&queries_path);
    let ground_truth = load_ground_truth(&truth_path);
    assert_eq!(ground_truth.len(), queries.len(), "ground truth length must match queries");
    logs.log_info(&format!("⏱️  Queries and truth loaded in {:?}", t0.elapsed()));

    if !snapshot.use_snapshot {
        panic!("nytimes_qps_latency_curve requires VECTORDB_USE_SNAPSHOT=1");
    }
    if !Path::new(&snapshot.persist_path).exists() {
        panic!(
            "missing snapshot at {} (run nytimes_build_and_persist_snapshot_only first)",
            snapshot.persist_path
        );
    }
    logs.log_info(&format!(
        "💾 Loading persisted NYTimes segment from {} ...",
        snapshot.persist_path
    ));
    let (mut segment, metadata) = Segment::load_from_path_with_metadata(&snapshot.persist_path)
        .expect("failed to load persisted NYTimes segment");
    let cfg = segment.hnsw().config_summary();
    logs.log_info(&format!(
        "✅ Loaded persisted segment with {} vectors (payloads={})",
        segment.hnsw().len(),
        segment.payloads().len()
    ));
    logs.log_info(&format!(
        "🧭 HNSW config: metric={:?} dim={} m={} m0={} ef={} ef_construct={} level_cap={} level_scale={:.3} max_level={} exact_fallback={} threshold={}",
        cfg.metric,
        cfg.dim,
        cfg.m,
        cfg.m0,
        cfg.ef,
        cfg.ef_construct,
        cfg.max_level_cap,
        cfg.level_scale,
        cfg.current_max_level,
        cfg.exact_fallback_enabled,
        cfg.exact_fallback_threshold
    ));
    if let Some(meta) = metadata {
        logs.log_info(&format!(
            "🧾 Snapshot metadata: created_at_ms={} points={} payloads={}",
            meta.created_at_ms, meta.points, meta.payloads
        ));
    }

    let num_queries = queries.len().min(search.queries_cap);
    let mut query_logger = QueryLogWriter::new(logs.query_log_path.clone(), logs.query_log_every);
    logs.log_info(&format!(
        "🔍 Sweeping ef_search over {:?} for {} queries (top_k={})...",
        search.ef_values, num_queries, top_k
    ));

    let mut summary: Vec<(usize, f64, f64, f64)> = Vec::new();
    for &ef in &search.ef_values {
        let ef_search = ef.max(top_k);
        segment.hnsw_mut().set_ef_search(ef_search);
        let mut hits = 0usize;
        let mut total_targets = 0usize;
        let mut stats = QueryStatsAgg::new(num_queries);
        let start_search = Instant::now();
        for (qi, q) in queries.iter().take(num_queries).enumerate() {
            let q_start = Instant::now();
            let (approx, qstats) = segment.search_with_stats(q, top_k).unwrap();
            let truth = &ground_truth[qi];
            let truth_k = truth.len().min(top_k);
            let truth_set: HashSet<_> = truth
                .iter()
                .take(truth_k)
                .map(|&id| id as u64)
                .collect();
            total_targets += truth_set.len();
            let query_hits = approx.iter().filter(|r| truth_set.contains(&r.id)).count();
            hits += query_hits;
            let recall = query_hits as f64 / truth_set.len().max(1) as f64;
            let elapsed_ms = q_start.elapsed().as_secs_f64() * 1000.0;
            stats.record(elapsed_ms, Some(qstats.visited), Some(qstats.expanded), recall);
            query_logger.write(
                qi,
                &QueryLogEntry {
                    dataset: "nytimes-256-angular",
                    query_idx: qi,
                    ef_search,
                    top_k,
                    elapsed_ms,
                    visited: Some(qstats.visited),
                    expanded: Some(qstats.expanded),
                    results_len: approx.len(),
                    recall,
                    misses: truth_set.len().saturating_sub(query_hits),
                },
            );
        }
        let search_dur = start_search.elapsed();
        let avg_ms = search_dur.as_secs_f64() * 1000.0 / num_queries as f64;
        let qps = num_queries as f64 / search_dur.as_secs_f64();
        let recall = hits as f64 / total_targets.max(1) as f64;
        let latency = summarize_f64(&stats.elapsed_ms);
        logs.log_info(&format!(
            "📈 [ef_search={}] qps={:.1} avg_ms={:.3} p50/p90/p99={:.3}/{:.3}/{:.3} recall@{}={:.3}",
            ef_search,
            qps,
            avg_ms,
            latency.p50,
            latency.p90,
            latency.p99,
            top_k,
            recall
        ));
        summary.push((ef_search, qps, avg_ms, recall));
    }

    logs.log_info("\nSummary (ef_search -> qps, ms/query, recall):");
    for (ef, qps, ms, recall) in summary {
        logs.log_info(&format!(
            "  {} -> {:.1} qps, {:.3} ms/query, {:.3} recall",
            ef, qps, ms, recall
        ));
    }
    log_peak_rss("nytimes_qps_complete");
}
