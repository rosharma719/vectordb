# VectorDB

`vectordb` is an in-memory vector search engine in Rust. It implements HNSW-based approximate nearest-neighbor search, in-place payload filtering, snapshot persistence, and WAL-backed recovery.

## What it does

- HNSW indexing for approximate nearest-neighbor search
- Distance metrics: cosine, euclidean, dot
- Payload storage with schema-agnostic key/value metadata
- Equality and comparison filters with boolean composition
- Inverted indexing for filter acceleration on scalar payload fields
- Snapshot persistence, background snapshotting, and WAL replay

## Status

- Implemented: HNSW indexing, payload storage, inverted indexing, deletion, in-place filtering, snapshot/WAL support
- Not yet implemented: Python API

## Quickstart

Build the repo:

```bash
cargo build
```

Run the fast local test suite:

```bash
cargo test
```

Run a specific benchmark-style ignored test:

```bash
cargo test --release nytimes_256_angular_perf_and_recall -- --ignored --nocapture
```

## Filtering model

- Payloads are per-point key/value maps.
- Missing fields evaluate to false for match/compare checks.
- Supported scalar types are ints, floats, strings, and bools.
- Homogeneous lists are supported in payloads.
- Filters are expressed as `Filter::Match`, `Filter::Compare`, `Filter::And`, `Filter::Or`, and `Filter::Not`.
- Scalar payload fields can use the inverted index for fast exact-match filtering.

## Project layout

- `src/vector/hnsw/`: HNSW implementation
- `src/segment/`: segment lifecycle, persistence, WAL, background snapshotting
- `src/payload_storage/`: payload evaluation and inverted index
- `src/bin/`: analysis and snapshot utilities
- `tests/`: correctness, persistence, recall, and dataset-driven harnesses
- `scripts/`: experiment pipeline
- `docs/`: operational notes, benchmark workflows, and dataset setup

## Docs

- Dataset setup: [docs/data-download.md](docs/data-download.md)
- Test and env-var matrix: [docs/test-config.md](docs/test-config.md)
- Snapshot naming and build manifests: [docs/index-construction.md](docs/index-construction.md)
- Runtime persistence and operations: [docs/operations.md](docs/operations.md)
- Benchmark commands and recorded results: [docs/benchmarks.md](docs/benchmarks.md)
- Recall frontier pipeline: [docs/recall_frontier_pipeline.md](docs/recall_frontier_pipeline.md)

## Utilities

This repo ships a few binaries for snapshot inspection and analysis:

```bash
cargo run --bin snapshot_info -- <path>
cargo run --bin index_analyzer -- --help
cargo run --bin snapshot_sweeper -- --help
cargo run --bin index_stats -- --help
```

The default `main` binary is intentionally just a stub; operational workflows currently live in tests, scripts, and the dedicated analysis binaries above.

## Logging

Logging uses `log` targets such as `vector::hnsw`, `segment`, `payload`, and `filter`. Control verbosity with your logger backend and `RUST_LOG`.
