# VectorDB: High-Performance In-Memory Vector Search Engine

## Overview

VectorDB (better name coming soon!) is a lightweight, high-performance in-memory vector search engine designed for efficient similarity search and payload storage. 

## Features

### Current Capabilities
- **HNSW (Hierarchical Navigable Small World) Indexing**
  - Optimized approximate nearest neighbor search
  - Supports arbitrarily high-dimensional vector spaces
  - Efficient search with configurable trade-offs between accuracy and performance (user-friendly interface in the works)
  - Supports common distance metrics (Cosine, Euclidean, Dot product similarity)

- **Payload Storage**
  - Store additional metadata alongside vector embeddings (ints, floats, strings, homogeneous lists)
  - Seamless integration with vector search operations

## Roadmap

- [x] HNSW Indexing
- [x] Payload Storage
- [x] Inverse Indexing
- [x] Vector Deletion
- [x] In-Place Filtering
- [X] Filtering and Query Schema
- [ ] Python API

## Other potential features

- [ ] Persistence
- [ ] Mutable/immutable segmentation
  - [ ] Compression and quantization for fast immutable segment search
- [ ] Graph functionality
- [ ] Generative AI query builder

## Performance

The current implementation focuses on single-threaded HNSW performance with an in-process API. All benchmarks below are run with `cargo test --release` and use Euclidean distance.

### 20k × 1536-d (realistic embedding workload)

To approximate a typical modern embedding use case, the engine was benchmarked on **20,000 vectors** with **1,536 dimensions**, no payloads, and `top_k = 40`:

- **Insert:** `20,000` vectors in **10.76 s**  
  - ≈ **1,860 inserts/sec**  
  - ≈ **0.54 ms/insert**

- **Unfiltered search:** `5` queries in **3.79 ms** total  
  - ≈ **0.76 ms/query**  
  - ≈ **1,320 QPS** (single thread)

This shows that for high-dimensional embeddings at 20k scale, the core HNSW search loop stays comfortably sub-millisecond per query.

### 1M × 3-d (million-scale sanity check)

To validate behavior at larger collection sizes, a separate benchmark inserts **1,000,000 vectors** of dimension **3** and runs unfiltered HNSW search with `top_k = 40`:

- **Insert:** `1,000,000` vectors in **883.62 s**  
  - ≈ **1,130 inserts/sec**  
  - ≈ **0.88 ms/insert**

- **Unfiltered search:** `3` queries in **5.53 ms** total  
  - ≈ **1.84 ms/query**  
  - ≈ **540 QPS** (single thread)

Even at million scale, unfiltered HNSW queries remain in the low-millisecond range while insert throughput stays around ~1k vectors/sec, confirming the expected logarithmic scaling with respect to the index size.

### Summary

- At **20k × 1536-d**, unfiltered search is **~0.76 ms/query** with **~1.9k inserts/sec**.
- At **1M × 3-d**, unfiltered search is **~1.84 ms/query** with **~1.1k inserts/sec**.

These benchmarks are single-threaded and in-process; there is still room for improvement (SIMD distance kernels, parameter tuning, and multi-threaded indexing/search), but they establish a solid baseline for both realistic embedding workloads and million-scale collections.

## Testing

- 20k benchmark (debug):
cargo test bench_unfiltered_large_scale -- --nocapture

- 20k benchmark (release):
cargo test --release bench_unfiltered_large_scale -- --nocapture

- 1M benchmark (ignored; debug):
cargo test bench_unfiltered_million_scale -- --ignored --nocapture

- 1M benchmark (ignored; release):
cargo test --release bench_unfiltered_million_scale -- --ignored --nocapture