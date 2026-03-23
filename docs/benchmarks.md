# Benchmarks And Harnesses

This document keeps the dataset-specific commands and recorded benchmark numbers out of the root `README`.

## NYTimes (256-D Angular)

Download instructions live in [data-download.md](./data-download.md).

Run the main harness:

```bash
cargo test --release nytimes_256_angular_perf_and_recall -- --ignored --nocapture
```

Run the QPS/latency curve from a persisted snapshot:

```bash
VECTORDB_USE_SNAPSHOT=1 \
VECTORDB_NYT_PERSIST_PATH=data/nytimes-256-angular/index_m16_m0_32_efc100.bin \
cargo test --release nytimes_qps_latency_curve -- --ignored --nocapture
```

Build a snapshot with trace logging:

```bash
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

Analyze a snapshot:

```bash
cargo run --bin index_analyzer -- \
  --snapshot data/nytimes-256-angular/index_m16_m0_32_efc100.bin \
  --base data/nytimes-256-angular/base.npy \
  --queries data/nytimes-256-angular/queries.npy \
  --top-k 10 \
  --num-queries 100 \
  --sample-size 500
```

Sweep multiple snapshots:

```bash
cargo run --bin snapshot_sweeper -- \
  --snapshots data/nytimes-256-angular/index_m16_m0_16_efc100.bin,\
data/nytimes-256-angular/index_m16_m0_32_efc100.bin \
  --output logs/nyt_snapshot_sweep.jsonl \
  --top-k 10 \
  --num-queries 100 \
  --sample-size 500 \
  --neighbor-scan-cap 128
```

For canonical naming and build manifests, see [index-construction.md](./index-construction.md).

### Recorded unfiltered recall/latency curve

Dataset: NYT-256-Angular, `ef_construct=100`, `M=16`, `M0=32`, `diversity_alpha=1`

- `EF=32`: `1894.6 qps`, `0.528 ms/query`, `0.816 recall`
- `EF=64`: `1307.5 qps`, `0.765 ms/query`, `0.860 recall`
- `EF=128`: `719.4 qps`, `1.390 ms/query`, `0.895 recall`
- `EF=256`: `383.7 qps`, `2.606 ms/query`, `0.926 recall`
- `EF=512`: `200.8 qps`, `4.980 ms/query`, `0.953 recall`

## H&M (2048-D Cosine)

Download instructions live in [data-download.md](./data-download.md).

Run the filtered recall harness:

```bash
cargo test --release hnm_filtered_cosine_recall -- --ignored --nocapture
```

Useful runtime knobs:

- `VECTORDB_HNM_TOPK`
- `VECTORDB_HNM_EF_SEARCH_LIST`
- `VECTORDB_HNM_QUERIES`
- `VECTORDB_HNM_BASE_LIMIT`
- `VECTORDB_HNM_EF_CONSTRUCT`

### Recorded filtered recall/latency curve

Dataset: H&M 2048D cosine, `ef_construct=100`, `M=16`

- `EF=32`: `1329.8 qps`, `0.752 ms/query`, `0.566 recall`
- `EF=64`: `1293.8 qps`, `0.773 ms/query`, `0.709 recall`
- `EF=128`: `787.4 qps`, `1.270 ms/query`, `0.859 recall`
- `EF=256`: `501.5 qps`, `1.994 ms/query`, `0.907 recall`
- `EF=512`: `338.8 qps`, `2.952 ms/query`, `0.939 recall`

## General performance notes

Euclidean, `dim=1536`, `top_k=20`, `ef_construct=100`, `m=16`, `ef_search=64`

- `20,000` vectors: insert `5.24 s` (~`3.8k vec/s`), search `0.198 ms/query`
- `100,000` vectors: insert `33.24 s` (~`3.1k vec/s`), search `0.310 ms/query`
- `1,000,000` vectors: insert `475.81 s` (~`2.1k vec/s`), search `0.495 ms/query`
