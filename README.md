# VectorDB: High-Performance In-Memory Vector Search Engine

## Overview

VectorDB is a lightweight, high-performance in-memory vector search engine implementing HNSW for approximate nearest neighbor search with payload-aware filtering.

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

## NYTimes (256-D Angular) Results

### ef_construct = 200  
**Recall / Latency (ms/query):**
- 16 → 0.757, 0.352  
- 32 → 0.826, 0.582  
- 64 → 0.869, 0.874  
- 128 → 0.904, 1.493  
- 256 → 0.923, 2.760  

**Insert Performance:**  
Inserted **290,000 vectors** in **624.16 s** (~2.152 ms/insert)

---

### ef_construct = 100  
**Recall / Latency (ms/query):**
- 16 → 0.722, 0.368  
- 32 → 0.792, 0.547  
- 64 → 0.850, 0.881  
- 128 → 0.883, 1.498  
- 256 → 0.912, 2.732  

Test: `nytimes_256_angular_perf_and_recall ... ok`

---

## Performance Benchmarks  
**(Euclidean, dim=1536, top_k=20, ef_construct=100, m=16, ef_search=64)**

### 20,000 vectors
- Insert: **18.01 s**  
- Search: **0.723 ms/query**

### 100,000 vectors
- Insert: **112.82 s**  
- Search: **0.926 ms/query**

### 1,000,000 vectors
- Insert: **1312.07 s**  
- Search: **1.165 ms/query**
