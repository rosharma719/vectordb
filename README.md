# VectorDB: High-Performance In-Memory Vector Search Engine

## Overview

VectorDB is a lightweight, high-performance in-memory vector search engine implementing HNSW for approximate nearest neighbor search with payload-aware filtering.

## Logging Levels
- `error` / `warn`: unexpected conditions (e.g., length mismatch, invalid payload)
- `info`: lifecycle and maintenance events (purge triggers)
- `debug`: per-operation details (inserts, entry point changes, filter misses)
- `trace`: reserved for hot-loop diagnostics (currently off by default)
Logging uses `log` targets (e.g., `vector::hnsw`, `segment`, `payload`, `filter`). Control verbosity with your logger backend/env (e.g., `RUST_LOG=vector::hnsw=debug`).

## Features

### Current Capabilities
- **HNSW Indexing**
  - Approximate nearest neighbor search
  - Supports high-dimensional vector spaces
  - Tunable accuracy/performance via `ef_construct` and `ef_search`
  - Distance metrics: Cosine, Euclidean, Dot

- **Payload Storage**
  - Metadata attached to vectors (ints, floats, strings, homogeneous lists)
  - Integrated with vector search

## Roadmap
- [x] HNSW Indexing  
- [x] Payload Storage  
- [x] Inverse Indexing  
- [x] Vector Deletion  
- [x] In-Place Filtering  
- [x] Filtering and Query Schema  
- [ ] Python API  

---

## H&M (2048-D Cosine) Filtered Recall

- Download the dataset (`vectors.npy`, `payloads.jsonl`, `tests.jsonl`) — see `docs/data-download.md` for CLI commands.
- Place the files under `data/hnm` (or override with `VECTORDB_HNM_DATA_DIR`).
- Run the harness:
  ```
  cargo test --release hnm_filtered_cosine_recall -- --ignored --nocapture
  ```
- Knobs: `VECTORDB_HNM_TOPK` (defaults to truth length), `VECTORDB_HNM_EF_SEARCH_LIST` (comma list, default `64,128,256`), `VECTORDB_HNM_QUERIES` (default `200`), `VECTORDB_HNM_BASE_LIMIT` (cap inserts), `VECTORDB_HNM_EF_CONSTRUCT` (default `200`).

---

### Filtering & Payloads
- Schema: schema-agnostic; payloads are per-point key/value maps. Missing fields simply evaluate to false for match/compare checks.
- Types: ints, floats, strings, bools, and homogeneous lists of those.
- Filtering syntax (code):
  - Equality: `Filter::Match { key, value }`
  - Scalar compare: `Filter::Compare { key, op, value }` with `Eq, Neq, Lt, Lte, Gt, Gte`
  - Boolean composition: `Filter::And(Vec<Filter>)`, `Filter::Or(Vec<Filter>)`, `Filter::Not(Box<Filter>)`
- Evaluation: filters are applied against the payload of each candidate; a missing key fails the condition. List fields support `Eq`/`Neq` (whole-list) and `Eq`/`Neq` against a scalar string for containment.
- Indexing: simple inverted index on indexable scalar types (int/float/str/bool) for fast equality matching; lists are not indexed.
- Filter-aware search: `search_with_filter` uses in-graph filter-aware entry selection and in-place filtering during HNSW search (see `HNSWIndex::in_place_filtered_search`).


## NYTimes (256-D Angular) Results

### Download the dataset (from Hugging Face)
Requires an `HF_TOKEN` with repo read access for `open-vdb/nytimes-256-angular` (see `docs/data-download.md` for CLI commands).

### Run the harness
```
cargo test --release nytimes_256_angular_perf_and_recall -- --ignored --nocapture
```

### QPS/Latency curve (from snapshot)
```
VECTORDB_USE_SNAPSHOT=1 \
VECTORDB_NYT_PERSIST_PATH=data/nytimes-256-angular/index_m16_m0_32_efc100.bin \
cargo test --release nytimes_qps_latency_curve -- --ignored --nocapture
```

### Build a snapshot (with insert trace logging)
```
VECTORDB_NYT_PERSIST_PATH=data/nytimes-256-angular/index_m16_m0_32_efc100.bin \
VECTORDB_NYT_M=16 \
VECTORDB_NYT_M0=32 \
VECTORDB_NYT_EF_CONSTRUCT=100 \
VECTORDB_NYT_EF_SEARCH_LIST=100 \
VECTORDB_DIVERSITY_ALPHA=1 \
VECTORDB_INSERT_TRACE_LOG=logs/nytimes_insert_m16_m0_32_efc100.jsonl \
VECTORDB_NYT_ALLOW_BUILD=1 \
VECTORDB_NYT_SAVE_SNAPSHOT=1 \
cargo test --release nytimes_build_and_persist_snapshot_only -- --ignored --nocapture
```

### Snapshot notes
- Snapshots are written atomically and include a checksum footer for corruption detection.
- New snapshots load with `Segment::load_from_path`; older snapshots without a footer are still supported.
- Background snapshotting is opt-in (see `start_background_snapshots` in `src/segment/background.rs`).
- Inspect a snapshot with `cargo run --bin snapshot_info -- <path>` or `scripts/nyt_snapshot_info.sh <path>`.

### Autosave (background snapshots)
```
use std::sync::{Arc, RwLock};
use std::time::Duration;
use vectordb::segment::{Segment, SnapshotConfig, start_background_snapshots};
use vectordb::vector::hnsw::HNSWIndex;
use vectordb::utils::types::DistanceMetric;

let segment = Arc::new(RwLock::new(Segment::new(
    HNSWIndex::new(DistanceMetric::Euclidean, 16, 32, 8, 2),
)));

let mut cfg = SnapshotConfig::new("data/segment_autosave.bin");
cfg.interval = Duration::from_secs(30);
cfg.max_ops = 5_000;

let snapshotter = start_background_snapshots(segment.clone(), cfg);
// keep `snapshotter` alive; call snapshotter.stop() on shutdown
```

### WAL (write-ahead log)
Enable a WAL to make inserts/deletes/payload updates durable between snapshots. WAL replay is automatic on
`Segment::load_from_path` when `<snapshot>.wal` exists.

```
use vectordb::segment::Segment;

let mut segment = Segment::new(hnsw);
segment.enable_wal("data/segment.wal")?;

segment.insert_with_id(1, vec![1.0, 0.0], None)?;
segment.save_to_path("data/segment_snapshot.bin")?;

let restored = Segment::load_from_path("data/segment_snapshot.bin")?;
```

Notes:
- Auto-replay can be disabled with `VECTORDB_WAL_AUTO_REPLAY=0`.
- Use `save_to_path_and_checkpoint` to save a snapshot and truncate the WAL after a successful write.
- You can auto-enable WAL by setting `VECTORDB_WAL_PATH` or `VECTORDB_WAL_DIR` (uses `<dir>/segment.wal`).
- Fsync tuning: `VECTORDB_WAL_FSYNC` (0/1), `VECTORDB_WAL_FSYNC_EVERY` (N ops), `VECTORDB_WAL_FSYNC_MS` (time-based).

### Deletion purge
By default, deletions leave tombstones in place. To enable automatic purge/rebuild after a deletion threshold:
`VECTORDB_PURGE_DELETIONS=1`.

During a rebuild, queries return a `SearchError` ("Segment rebuild in progress. Retry later.").

### Unfiltered Recall/Latency Curve with ef_construct = 100 (M=16, M0=32, diversity_alpha=1)
**Dataset: NYT-256-Angular**
**EF/ Queries per Second / Latency (ms/query) / recall**
  32 -> 1894.6 qps, 0.528 ms/query, 0.816 recall
  64 -> 1307.5 qps, 0.765 ms/query, 0.860 recall
  128 -> 719.4 qps, 1.390 ms/query, 0.895 recall
  256 -> 383.7 qps, 2.606 ms/query, 0.926 recall
  512 -> 200.8 qps, 4.980 ms/query, 0.953 recall
---

### Filtered Recall/Latency Curve with ef_construct = 100 (M=16)
**Dataset: H&M 2048D Cosine**
**EF / Recall / Latency (ms/query)**
  32 -> 0.566, 0.752 ms/query
  64 -> 0.709, 0.773 ms/query
  128 -> 0.859, 1.270 ms/query
  256 -> 0.907, 1.994 ms/query
  512 -> 0.939, 2.952 ms/query
---

## Performance Benchmarks  
**(Euclidean, dim=1536, top_k=20, ef_construct=100, m=16, ef_search=64)**

### 20,000 vectors
- Insert: **5.24 s** (~3.8k vec/s)  
- Search: **0.198 ms/query**

### 100,000 vectors
- Insert: **33.24 s** (~3.1k vec/s)  
- Search: **0.310 ms/query**

### 1,000,000 vectors
- Insert: **475.81 s** (~2.1k vec/s)  
- Search: **0.495 ms/query**
