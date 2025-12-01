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

The engine is optimized for **single-threaded, in-process HNSW** search. All benchmarks were run with `cargo test --release` using Euclidean distance and `top_k = 40`.

### **20k × 1536-d (realistic embedding workload)**

A representative high-dimensional scenario using 20,000 vectors of dimension 1,536:

- **Insert:** 20,000 vectors in **10.76 s**  
  - ≈ **1,860 inserts/sec**  
  - ≈ **0.54 ms/insert**

- **Unfiltered search:** 5 queries in **3.79 ms**  
  - ≈ **0.76 ms/query**  
  - ≈ **1,320 QPS**

This demonstrates that the core HNSW search loop remains sub-millisecond for typical embedding sizes at moderate scale.

---

### **100k × 1536-d (intermediate scale)**

To evaluate scaling between moderate and large collections, a parameterized benchmark inserts 100,000 vectors:

- **Insert:** 100,000 vectors in **63.33 s**  
  - ≈ **1,579 inserts/sec**  
  - ≈ **0.63 ms/insert**

- **Unfiltered search:** 3 queries in **2.58 ms**  
  - ≈ **0.86 ms/query**  
  - ≈ **1,160 QPS**

Latency and per-insert cost increase only slightly relative to the 20k benchmark.

---

### **1M × 1536-d (million-scale benchmark)**

A full-scale experiment inserting 1,000,000 high-dimensional vectors:

- **Insert:** 1,000,000 vectors in **883.62 s**  
  - ≈ **1,132 inserts/sec**  
  - ≈ **0.88 ms/insert**

- **Unfiltered search:** 3 queries in **5.53 ms**  
  - ≈ **1.84 ms/query**  
  - ≈ **540 QPS**

Even at one million vectors, unfiltered HNSW queries remain in the low-millisecond range, and insert throughput stays near ~1k vectors/sec, matching the expected logarithmic growth of HNSW.

---

### **Scaling Summary**

Across 20k → 100k → 1M (all at 1,536 dimensions):

| Collection Size | Insert Rate | Insert Cost | Search Latency | QPS |
|-----------------|-------------|-------------|----------------|-----|
| **20k**         | ~1,860/s    | 0.54 ms     | 0.76 ms        | ~1,320 |
| **100k**        | ~1,579/s    | 0.63 ms     | 0.86 ms        | ~1,160 |
| **1M**          | ~1,132/s    | 0.88 ms     | 1.84 ms        | ~540 |

- Insert cost rises moderately (0.54 → 0.88 ms) as N increases by **50×**.  
- Search latency grows from 0.76 to 1.84 ms, consistent with HNSW’s **log-N behavior**.  
- The engine delivers competitive unfiltered performance at realistic embedding dimensions across both moderate and million-scale workloads.

There is room for continued improvement—SIMD distance kernels, recall tuning, multi-threaded indexing/search—but the current results establish a strong baseline.

---

## Testing

- **20k benchmark (debug):**  
  `cargo test bench_unfiltered_large_scale -- --ignored --nocapture`

- **20k benchmark (release):**  
  `cargo test --release bench_unfiltered_large_scale -- --ignored --nocapture`

- **Parameterized benchmark (set size/dim via environment variables):**  
  `VECTORDB_BENCH_SIZE=100000 cargo test --release bench_unfiltered_param_scale -- --ignored --nocapture`  
  `VECTORDB_BENCH_DIM=1536 VECTORDB_BENCH_SIZE=50000 cargo test --release bench_unfiltered_param_scale -- --ignored --nocapture`

- **1M benchmark (ignored tests):**  
  `cargo test bench_unfiltered_million_scale -- --ignored --nocapture`  
  `cargo test --release bench_unfiltered_million_scale -- --ignored --nocapture`

## Recall

The recall harness computes exact neighbors with the same distance implementation (`metric::score`) and compares them to HNSW results. It now uses a deterministic random dataset plus noisy in-dataset queries (to avoid trivial self-matches) and enforces **ef_search ≥ top_k**.

Key env vars:
- Graph: Euclidean, dim 1536, `top_k=20`, `m=16`, `ef_construct=200`, `ef_search` sweep `32,64,128`.
- `VECTORDB_RECALL_SIZE` (default `20000`)
- `VECTORDB_RECALL_DIM` (default `1536`)
- `VECTORDB_RECALL_TOPK` (default `20`)
- `VECTORDB_RECALL_QUERIES` (default `20`)
- `VECTORDB_RECALL_EF_SEARCH` (comma list; default `32,64,128`)
- `VECTORDB_RECALL_SEED` (default `42`)
- `VECTORDB_RECALL_NOISE` (default `0.002`; set to `0` for parity sweeps)
- `VECTORDB_RECALL_MIN` (informational only; no assertion)
- `VECTORDB_RECALL_RANDOM` (default `true`; toggle between random/noisy dataset vs. sinusoidal deterministic dataset)

Example runs:
- 20k × 1536-d sweep (random/noisy, default):  
  `VECTORDB_RECALL_SIZE=20000 VECTORDB_RECALL_DIM=1536 VECTORDB_RECALL_NOISE=0 cargo test --release recall_unfiltered_euclidean -- --ignored --nocapture`
- 20k × 1536-d sweep (sinusoidal):  
  `VECTORDB_RECALL_SIZE=20000 VECTORDB_RECALL_DIM=1536 VECTORDB_RECALL_RANDOM=false VECTORDB_RECALL_NOISE=0 cargo test --release recall_unfiltered_euclidean -- --ignored --nocapture`
- 100k × 1536-d sweep (random/noisy, longer):  
  `VECTORDB_RECALL_SIZE=100000 VECTORDB_RECALL_DIM=1536 VECTORDB_RECALL_NOISE=0 cargo test --release recall_unfiltered_euclidean -- --ignored --nocapture`
- 100k × 1536-d sweep (sinusoidal):  
  `VECTORDB_RECALL_SIZE=100000 VECTORDB_RECALL_DIM=1536 VECTORDB_RECALL_RANDOM=false VECTORDB_RECALL_NOISE=0 cargo test --release recall_unfiltered_euclidean -- --ignored --nocapture`

Expect recall to increase with ef_search; lower ef settings will show non-perfect recall because of the noisy queries and the absence of search-result doubling.

See `docs/test-config.md` for a concise list of default parameters across benchmarks and recall tests.

## Recall Results

### 20,000 vectors (dim = 1536, Euclidean, top_k = 20, m = 16, ef_construct = 200)

#### Sinusoidal Dataset
- ef_search = 32  → **1.000**
- ef_search = 64  → **1.000**
- ef_search = 128 → **1.000**

#### Random Dataset (uniform [-1,1], noise = 0.002)
- ef_search = 32  → **0.740**
- ef_search = 64  → **0.782**
- ef_search = 128 → **0.832**

---

### 100,000 vectors (dim = 1536, sinusoidal)

- ef_search = 32  → **0.960**
- ef_search = 64  → **0.960**
- ef_search = 128 → **0.988**

## NYTimes (256-d Angular) via Hugging Face

Run an opt-in ignored test that loads the ANN-Benchmarks NYTimes dataset from Hugging Face (dataset repo `open-vdb/nytimes-256-angular`), inserts it, times unfiltered search, and reports recall against the provided ground truth.

1) Download and materialize the dataset (writes to `data/nytimes-256-angular`). Requires a Hugging Face token (`HF_TOKEN`) with repo read access:
```
export HF_TOKEN=your_hf_token
python - <<'PY'
import os, json, numpy as np, pathlib
from datasets import load_dataset

token = os.environ["HF_TOKEN"]
train = load_dataset("open-vdb/nytimes-256-angular", name="train", split="train", token=token)
test = load_dataset("open-vdb/nytimes-256-angular", name="test", split="test", token=token)
nbrs = load_dataset("open-vdb/nytimes-256-angular", name="neighbors", split="neighbors", token=token)

def first_list_col(ds):
    for name in ds.column_names:
        if isinstance(ds[0][name], (list, tuple)):
            return name
    raise RuntimeError(f"no list-like column in {ds.column_names}")

emb_col = first_list_col(train)   # 256-d vectors
q_col = first_list_col(test)      # 256-d queries
nbr_col = first_list_col(nbrs)    # ground-truth neighbor ids

out = pathlib.Path("data/nytimes-256-angular"); out.mkdir(parents=True, exist_ok=True)
np.save(out/"base.npy", np.stack(train[emb_col]).astype("float32"))
np.save(out/"queries.npy", np.stack(test[q_col]).astype("float32"))
neighbors_list = nbrs[nbr_col].to_pylist() if hasattr(nbrs[nbr_col], "to_pylist") else list(nbrs[nbr_col])
with open(out/"ground_truth.json","w") as f:
    json.dump(neighbors_list, f)
print("wrote", out, "cols:", emb_col, q_col, nbr_col)
PY
```

2) Run the harness (ignored by default); you can sweep ef_search in one run using `VECTORDB_NYT_EF_SEARCH_LIST` to avoid rebuilding:
```
cargo test --release nytimes_256_angular_perf_and_recall -- --ignored --nocapture
```

 Env knobs:
- `VECTORDB_NYT_DATA_DIR` (default `data/nytimes-256-angular`)
- `VECTORDB_NYT_TOPK` (default `20`)
- `VECTORDB_NYT_EF_SEARCH` (default `128`) or `VECTORDB_NYT_EF_SEARCH_LIST` (e.g., `16,32,64,128,256`) to sweep without rebuilding the index
- `VECTORDB_NYT_QUERIES` (default `1000`)
- `VECTORDB_NYT_EF_CONSTRUCT` (default `100`) to tune build beam width
- `VECTORDB_NYT_BASE_LIMIT` to cap how many base vectors are inserted (useful for quick iteration)
