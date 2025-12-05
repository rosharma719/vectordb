use std::collections::HashSet;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

use ndarray::Array2;
use ndarray_npy::read_npy;
use ordered_float::OrderedFloat;
use serde::Deserialize;
use serde_json::{from_slice, Value};

use vectordb::payload_storage::filters::Filter;
use vectordb::segment::segment::Segment;
use vectordb::utils::payload::{Payload, PayloadValue, ScalarComparisonOp};
use vectordb::utils::types::{DistanceMetric, Vector};
use vectordb::vector::hnsw::HNSWIndex;

fn load_vectors(path: &Path) -> Vec<Vector> {
    let arr: Array2<f32> = read_npy(path).expect("failed to read vectors.npy");
    arr.rows().into_iter().map(|r| r.to_vec()).collect()
}

fn json_to_payload_value(v: &Value) -> Option<PayloadValue> {
    match v {
        Value::Null => None,
        Value::Bool(b) => Some(PayloadValue::Bool(*b)),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(PayloadValue::Int(i))
            } else {
                n.as_f64().map(|f| PayloadValue::Float(OrderedFloat(f)))
            }
        }
        Value::String(s) => Some(PayloadValue::Str(s.clone())),
        Value::Array(arr) => {
            if arr.is_empty() {
                return Some(PayloadValue::ListStr(Vec::new()));
            }
            if arr.iter().all(|x| x.as_i64().is_some()) {
                return Some(PayloadValue::ListInt(arr.iter().map(|x| x.as_i64().unwrap()).collect()));
            }
            if arr.iter().all(|x| x.as_f64().is_some()) {
                return Some(PayloadValue::ListFloat(
                    arr.iter().map(|x| OrderedFloat(x.as_f64().unwrap())).collect(),
                ));
            }
            if arr.iter().all(|x| x.as_str().is_some()) {
                return Some(PayloadValue::ListStr(
                    arr.iter().map(|x| x.as_str().unwrap().to_string()).collect(),
                ));
            }
            if arr.iter().all(|x| x.as_bool().is_some()) {
                return Some(PayloadValue::ListBool(arr.iter().map(|x| x.as_bool().unwrap()).collect()));
            }
            None
        }
        Value::Object(_) => None,
    }
}

fn json_obj_to_payload(obj: &serde_json::Map<String, Value>) -> Payload {
    let mut payload = Payload::default();
    for (k, v) in obj {
        if let Some(pv) = json_to_payload_value(v) {
            payload.set(k, pv);
        }
    }
    payload
}

fn load_payloads(path: &Path) -> Vec<Payload> {
    let file = File::open(path).expect("failed to open payloads.jsonl");
    let reader = BufReader::new(file);
    reader
        .lines()
        .enumerate()
        .map(|(idx, line)| {
            let line = line.expect("failed to read payload line");
            let value: Value = serde_json::from_str(&line)
                .unwrap_or_else(|e| panic!("payload line {} failed to parse: {}", idx, e));
            let obj = value.as_object().unwrap_or_else(|| {
                panic!("payload line {} is not an object payload", idx);
            });
            json_obj_to_payload(obj)
        })
        .collect()
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct RawTestCase {
    query: Vec<f32>,
    conditions: Value,
    closest_ids: Vec<usize>,
    closest_scores: Vec<f32>,
}

fn parse_range_filter(key: &str, range: &serde_json::Map<String, Value>) -> Result<Filter, String> {
    let mut parts = Vec::new();
    for (k, v) in range {
        let pv = json_to_payload_value(v).ok_or_else(|| format!("unsupported range value for {key}:{k}"))?;
        let op = match k.as_str() {
            "gt" => ScalarComparisonOp::Gt,
            "gte" => ScalarComparisonOp::Gte,
            "lt" => ScalarComparisonOp::Lt,
            "lte" => ScalarComparisonOp::Lte,
            _ => return Err(format!("unsupported range operator: {}", k)),
        };
        parts.push(Filter::Compare {
            key: key.to_string(),
            op,
            value: pv,
        });
    }
    match parts.len() {
        0 => Err(format!("range filter for {} had no operators", key)),
        1 => Ok(parts.into_iter().next().unwrap()),
        _ => Ok(Filter::And(parts)),
    }
}

fn parse_condition(node: &Value) -> Result<Filter, String> {
    let obj = node
        .as_object()
        .ok_or_else(|| "condition node must be an object".to_string())?;

    if let Some(and) = obj.get("and") {
        let clauses = and
            .as_array()
            .ok_or_else(|| "`and` must be an array".to_string())?
            .iter()
            .map(parse_condition)
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(Filter::And(clauses));
    }

    if let Some(or) = obj.get("or") {
        let clauses = or
            .as_array()
            .ok_or_else(|| "`or` must be an array".to_string())?
            .iter()
            .map(parse_condition)
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(Filter::Or(clauses));
    }

    if let Some(not) = obj.get("not") {
        let inner = parse_condition(not)?;
        return Ok(Filter::Not(Box::new(inner)));
    }

    let (field, inner) = obj
        .iter()
        .next()
        .ok_or_else(|| "empty condition object".to_string())?;
    let inner_obj = inner
        .as_object()
        .ok_or_else(|| format!("condition for {} must be an object", field))?;

    if let Some(m) = inner_obj.get("match") {
        let mv = m
            .get("value")
            .ok_or_else(|| format!("match for {} missing `value`", field))?;
        let value = json_to_payload_value(mv)
            .ok_or_else(|| format!("unsupported match value for key {}", field))?;
        return Ok(Filter::Match {
            key: field.clone(),
            value,
        });
    }

    if let Some(rng) = inner_obj.get("range") {
        let range_obj = rng
            .as_object()
            .ok_or_else(|| format!("range clause for {} must be object", field))?;
        return parse_range_filter(field, range_obj);
    }

    if inner_obj.get("geo").is_some() {
        return Err(format!("geo filters are not supported (field {})", field));
    }

    Err(format!("unsupported condition format for key {}", field))
}

fn load_test_cases(path: &Path) -> Vec<RawTestCase> {
    let file = File::open(path).expect("failed to open tests.jsonl");
    let reader = BufReader::new(file);
    reader
        .lines()
        .enumerate()
        .map(|(idx, line)| {
            let line = line.expect("failed to read test case line");
            from_slice::<RawTestCase>(line.as_bytes())
                .unwrap_or_else(|e| panic!("tests.jsonl line {} failed to parse: {}", idx, e))
        })
        .collect()
}

fn parse_usize_list(env_key: &str, default: &[usize]) -> Vec<usize> {
    env::var(env_key)
        .ok()
        .map(|v| {
            v.split(',')
                .filter_map(|s| s.trim().replace('_', "").parse::<usize>().ok())
                .collect::<Vec<_>>()
        })
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| default.to_vec())
}

#[derive(Clone)]
struct PreparedCase {
    query: Vector,
    filter: Option<Filter>,
    truth_ids: Vec<u64>,
}

#[test]
#[ignore]
fn hnm_filtered_cosine_recall() {
    let t0 = Instant::now();
    let data_dir = env::var("VECTORDB_HNM_DATA_DIR").unwrap_or_else(|_| "data/hnm".to_string());
    let top_k = env::var("VECTORDB_HNM_TOPK")
        .ok()
        .and_then(|v| v.replace('_', "").parse().ok());
    let ef_values = parse_usize_list("VECTORDB_HNM_EF_SEARCH_LIST", &[32, 64, 128, 256, 512]);
    let queries_cap = env::var("VECTORDB_HNM_QUERIES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1000usize);
    let base_cap = env::var("VECTORDB_HNM_BASE_LIMIT")
        .ok()
        .and_then(|v| v.parse().ok());
    let ef_construct = env::var("VECTORDB_HNM_EF_CONSTRUCT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100usize);

    let vectors_path = Path::new(&data_dir).join("vectors.npy");
    let payloads_path = Path::new(&data_dir).join("payloads.jsonl");
    let tests_path = Path::new(&data_dir).join("tests.jsonl");
    assert!(vectors_path.exists(), "missing {}", vectors_path.display());
    assert!(payloads_path.exists(), "missing {}", payloads_path.display());
    assert!(tests_path.exists(), "missing {}", tests_path.display());

    println!(
        "\n🧪 H&M filtered recall: dir={}, ef_search={:?}, ef_construct={}, base_cap={:?}, queries_cap={}",
        data_dir, ef_values, ef_construct, base_cap, queries_cap
    );

    let mut base = load_vectors(&vectors_path);
    if let Some(cap) = base_cap {
        if cap < base.len() {
            base.truncate(cap);
        }
    }
    let mut payloads = load_payloads(&payloads_path);
    if payloads.len() > base.len() {
        payloads.truncate(base.len());
    }
    assert_eq!(
        base.len(),
        payloads.len(),
        "vectors and payloads must have the same length"
    );
    let raw_tests = load_test_cases(&tests_path);
    println!("⏱️  Data loaded in {:?}", t0.elapsed());

    let dim = base.first().map(|v| v.len()).unwrap_or(0);
    assert_eq!(dim, 2048, "expected 2048-d vectors");

    let default_topk = raw_tests
        .first()
        .map(|c| c.closest_ids.len())
        .unwrap_or(10);
    let top_k = top_k.unwrap_or(default_topk);

    let prepared_cases: Vec<PreparedCase> = raw_tests
        .into_iter()
        .take(queries_cap)
        .map(|raw| {
            let filter = if raw.conditions.is_null() {
                None
            } else {
                Some(parse_condition(&raw.conditions).expect("failed to parse filter conditions"))
            };
            let truth_ids = raw
                .closest_ids
                .into_iter()
                .take(top_k)
                .map(|id| id as u64)
                .collect();
            PreparedCase {
                query: raw.query,
                filter,
                truth_ids,
            }
        })
        .collect();

    let metric = DistanceMetric::Cosine;
    let m = 16;
    let max_ef = ef_values.iter().copied().max().unwrap_or(1).max(top_k);
    let mut segment = Segment::new(HNSWIndex::new(metric, m, max_ef, 16, dim));
    segment.hnsw_mut().set_ef_construct(ef_construct);

    println!("🚀 Inserting {} vectors with dataset-aligned IDs...", base.len());
    let start_insert = Instant::now();
    for (i, (v, p)) in base.iter().zip(payloads.iter()).enumerate() {
        let dataset_id = i as u64;
        segment
            .insert_with_id(dataset_id, v.clone(), Some(p.clone()))
            .unwrap();
        if i != 0 && i % 5000 == 0 {
            println!("Inserted {} vectors (+{:?})", i, start_insert.elapsed());
        }
    }
    let insert_dur = start_insert.elapsed();
    let insert_ms = insert_dur.as_secs_f64() * 1000.0 / base.len().max(1) as f64;
    println!(
        "✅ Inserted {} vectors in {:?} (~{:.3} ms/insert)",
        base.len(),
        insert_dur,
        insert_ms
    );

    let num_queries = prepared_cases.len();
    println!(
        "🔍 Sweeping ef_search over {:?} for {} queries (top_k={})...",
        ef_values, num_queries, top_k
    );

    let mut summary: Vec<(usize, f64, f64)> = Vec::new();
    for &ef in &ef_values {
        let ef_search = ef.max(top_k);
        segment.hnsw_mut().set_ef_search(ef_search);
        let mut hits = 0usize;
        let mut total_targets = 0usize;
        let mut per_query_recalls: Vec<f64> = Vec::with_capacity(num_queries);
        let start_search = Instant::now();
        for (qi, case) in prepared_cases.iter().enumerate() {
            let res = segment
                .search_with_filter(&case.query, top_k, case.filter.as_ref())
                .unwrap();
            let truth: HashSet<_> = case
                .truth_ids
                .iter()
                .filter(|id| (**id as usize) < base.len())
                .copied()
                .collect();
            total_targets += truth.len();
            let query_hits = res.iter().filter(|r| truth.contains(&r.id)).count();
            hits += query_hits;
            let recall = query_hits as f64 / truth.len().max(1) as f64;
            per_query_recalls.push(recall);
            if (qi + 1) % 50 == 0 || qi + 1 == num_queries {
                let partial = hits as f64 / total_targets.max(1) as f64;
                println!(
                    "  progress: query {}/{} (ef_search={}) cumulative recall={:.3}",
                    qi + 1,
                    num_queries,
                    ef_search,
                    partial
                );
            }
        }
        let search_dur = start_search.elapsed();
        let avg_ms = search_dur.as_secs_f64() * 1000.0 / num_queries.max(1) as f64;
        let recall = hits as f64 / total_targets.max(1) as f64;
        println!(
            "🎯 [ef_search={}] recall@{}: {:.3} over {} queries (hits {}/{}) | avg {:.3} ms/query",
            ef_search,
            top_k,
            recall,
            num_queries,
            hits,
            total_targets,
            avg_ms
        );

        // Per-query recall stats to understand distribution.
        per_query_recalls.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let stats = |p: f64| -> f64 {
            if per_query_recalls.is_empty() {
                0.0
            } else {
                let idx = ((p / 100.0) * (per_query_recalls.len() as f64 - 1.0)).round() as usize;
                per_query_recalls[idx]
            }
        };
        let mean = per_query_recalls.iter().copied().sum::<f64>() / per_query_recalls.len().max(1) as f64;
        let p50 = stats(50.0);
        let p90 = stats(90.0);
        let p99 = stats(99.0);
        let min = *per_query_recalls.first().unwrap_or(&0.0);
        let max = *per_query_recalls.last().unwrap_or(&0.0);
        let full = per_query_recalls.iter().filter(|&&r| (r - 1.0).abs() < 1e-6).count();
        let ge_08 = per_query_recalls.iter().filter(|&&r| r >= 0.8).count();
        let ge_05 = per_query_recalls.iter().filter(|&&r| r >= 0.5).count();
        println!(
            "[recall_stats] ef_search={} queries={} mean={:.3} p50={:.3} p90={:.3} p99={:.3} min={:.3} max={:.3} full={} ge_0.8={} ge_0.5={}",
            ef_search,
            num_queries,
            mean,
            p50,
            p90,
            p99,
            min,
            max,
            full,
            ge_08,
            ge_05
        );

        summary.push((ef_search, recall, avg_ms));
    }

    println!("\nSummary (ef_search -> recall, ms/query):");
    for (ef, recall, ms) in summary {
        println!("  {} -> {:.3}, {:.3} ms/query", ef, recall, ms);
    }
}
