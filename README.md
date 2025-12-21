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

### Recall/Latency Curve with ef_construct = 100 
**Recall / Latency (ms/query):**
  32 -> 0.743, 0.301 ms/query
  64 -> 0.818, 0.461 ms/query
  128 -> 0.860, 0.820 ms/query
  256 -> 0.894, 1.514 ms/query
  512 -> 0.926, 2.843 ms/query
---

## Performance Benchmarks  
**(Euclidean, dim=1536, top_k=20, ef_construct=100, m=16, ef_search=64)**

### 20,000 vectors
- Insert: **16.33 s** (~1.2k vec/s)  
- Search: **0.644 ms/query**

### 100,000 vectors
- Insert: **102.33 s** (~0.9k vec/s)  
- Search: **0.919 ms/query**

### 1,000,000 vectors
- Insert: **1312.07 s** (~0.76k vec/s)  
- Search: **1.165 ms/query**
