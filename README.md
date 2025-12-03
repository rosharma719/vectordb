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

## NYTimes (256-D Angular) Results

### Download the dataset (from Hugging Face)
Requires an `HF_TOKEN` with repo read access for `open-vdb/nytimes-256-angular`:
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

emb_col = first_list_col(train)
q_col = first_list_col(test)
nbr_col = first_list_col(nbrs)

out = pathlib.Path("data/nytimes-256-angular"); out.mkdir(parents=True, exist_ok=True)
np.save(out/"base.npy", np.stack(train[emb_col]).astype("float32"))
np.save(out/"queries.npy", np.stack(test[q_col]).astype("float32"))
neighbors_list = nbrs[nbr_col].to_pylist() if hasattr(nbrs[nbr_col], "to_pylist") else list(nbrs[nbr_col])
with open(out/"ground_truth.json","w") as f:
    json.dump(neighbors_list, f)
print("wrote", out, "cols:", emb_col, q_col, nbr_col)
PY
```

### Run the harness
```
cargo test --release nytimes_256_angular_perf_and_recall -- --ignored --nocapture
```

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

### Recall with ef_construct = 200  
**Recall / Latency (ms/query):**
- 16 → 0.757, 0.352  
- 32 → 0.826, 0.582  
- 64 → 0.869, 0.874  
- 128 → 0.904, 1.493  
- 256 → 0.923, 2.760  

**Insert Performance:**  
Inserted **290,000 vectors** in **624.16 s** (~2.152 ms/insert)

---

### Recall with ef_construct = 100  
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
