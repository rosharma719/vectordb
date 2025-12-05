# Test & Benchmark Parameters

## Unfiltered Benchmarks
- `bench_unfiltered_large_scale`: size=20,000, dim=1,536, metric=Euclidean, top_k=20, ef_construct=100, m=16, ef_search=64.
- `bench_unfiltered_param_scale`: defaults size=100,000, dim=1,536, metric=Euclidean, top_k=20, ef_construct=100, m=16, ef_search=64; override via env: `VECTORDB_BENCH_SIZE`, `VECTORDB_BENCH_DIM`.
- `bench_unfiltered_million_scale` (ignored by default): size=1,000,000, dim=1,536, metric=Euclidean, top_k=20, ef_construct=100, m=16, ef_search=64.

## Recall Harness (`recall_unfiltered_euclidean`)
- Graph: metric=Euclidean, m=16, ef_construct=200, max_level_cap=16.
- Defaults: `VECTORDB_RECALL_SIZE=20000`, `VECTORDB_RECALL_DIM=1536`, `VECTORDB_RECALL_TOPK=20`, `VECTORDB_RECALL_QUERIES=20`.
- Beam sweep: `VECTORDB_RECALL_EF_SEARCH=32,64,128` (comma list; ef clamps to ≥ top_k). No assertion is enforced; `VECTORDB_RECALL_MIN` is informational.
- Dataset toggle: `VECTORDB_RECALL_RANDOM=true` (seeded random data with in-dataset anchored queries) or `false` for sinusoidal deterministic data.
- Noise/seed: `VECTORDB_RECALL_NOISE=0.002` (defaults to a light perturbation), `VECTORDB_RECALL_SEED=42`. Set `VECTORDB_RECALL_NOISE=0` for parity runs.
- Assertion: `VECTORDB_RECALL_MIN` (default `0.0`) is informational only; the test no longer fails on recall.
- Query generation:
  - `VECTORDB_RECALL_RANDOM=true`: dataset vectors are random uniform in [-1,1]^dim and stored; queries are sampled from those stored vectors and optionally noised by `VECTORDB_RECALL_NOISE`.
  - `VECTORDB_RECALL_RANDOM=false`: dataset and queries use `generate_vector_dim(i, dim)`, i.e., `[(i+d).sin() for d in 0..dim]`; queries are sampled from stored vectors on the same sinusoidal manifold, with optional noise.
- Key knobs summary:
  - Scale: `VECTORDB_RECALL_SIZE`, `VECTORDB_RECALL_DIM`
  - Difficulty: `VECTORDB_RECALL_NOISE` (higher = harder), dataset toggle (random vs sinusoidal)
  - Beam: `VECTORDB_RECALL_EF_SEARCH`
  - Target: `VECTORDB_RECALL_TOPK`
- Averaging: `VECTORDB_RECALL_QUERIES`
- Quality threshold: `VECTORDB_RECALL_MIN`
- Reproducibility: `VECTORDB_RECALL_SEED`

## H&M 2048-d Cosine (Filtered Recall)
- Data source: EfficientNet-encoded H&M clothes (≈105k vectors, dim=2048), cosine metric. Tarball: https://storage.googleapis.com/ann-filtered-benchmark/datasets/hnm.tgz.
- Files expected under `VECTORDB_HNM_DATA_DIR` (default `data/hnm`): `vectors.npy`, `payloads.jsonl`, `tests.jsonl`.
- Test: `hnm_filtered_cosine_recall` (ignored). Run with `cargo test --release hnm_filtered_cosine_recall -- --ignored --nocapture`.
- Knobs:
  - `VECTORDB_HNM_TOPK` (defaults to dataset truth length)
  - `VECTORDB_HNM_EF_SEARCH_LIST` (comma list; default `32,64,128,256,512`)
  - `VECTORDB_HNM_QUERIES` (default `1000`)
  - `VECTORDB_HNM_BASE_LIMIT` to cap inserts
  - `VECTORDB_HNM_EF_CONSTRUCT` (default `200`)
  - `VECTORDB_FILTER_EDGES` to toggle filter-aware edge construction (`0`/`false` to disable; default disabled)
- `VECTORDB_FILTER_KEYS` to allowlist payload keys (comma list) for filter-aware edges; defaults to all indexable scalars when edges are enabled
- `VECTORDB_FILTER_MAX_KEYS` to cap how many payload keys per point get filter-aware edges; keys are ordered Bool → Str → Int → Float, then name, before truncation (default unlimited)
  - `VECTORDB_DISABLE_FILTER_SEEDS` to turn off seeding the beam from the inverted index (default: seeds enabled)
- `VECTORDB_LOG_FILTER_SEED` to log per-query seeding stats (seed pool/added/accepted/in-results); `VECTORDB_LOG_FILTER_EDGES_AGG` for aggregated per-1000 key stats (ms by type)
  - `VECTORDB_LOG_INSERT_TIMING` to emit per-5k insert timing chunks (see `segment` target logs/STDOUT)
  - Filtered search behavior:
    - Beam seeds are pulled from inverted-index matches for equality filters (AND -> intersection, OR -> union), capped at `ef_search`.
    - Expansion budget is `ef_search * 4` even when a filter is present to avoid unbounded scans.
    - Filter-aware insertion adds one-way edges from the new point to at most `m` matches (no bidirectional update) to keep degree bounded.
    - If filter-aware edges are disabled, recall depends on seeding plus the unfiltered graph; enable edges for higher recall at the cost of slower inserts.

## NYTimes 256-d Angular (Hugging Face)
- Data source: `open-vdb/nytimes-256-angular` dataset; load configs `train` (base vectors), `test` (queries), `neighbors` (ground truth).
- Files expected under `VECTORDB_NYT_DATA_DIR` (default `data/nytimes-256-angular`): `base.npy`, `queries.npy`, `ground_truth.json`. See README for the download script (requires `HF_TOKEN` with repo read access).
- Test: `nytimes_256_angular_perf_and_recall` (ignored). Run with `cargo test --release nytimes_256_angular_perf_and_recall -- --ignored --nocapture`.
- Knobs: `VECTORDB_NYT_TOPK` (default `20`), `VECTORDB_NYT_EF_SEARCH` (default `128`) or `VECTORDB_NYT_EF_SEARCH_LIST` for sweeps (default `32,64,128,256,512`), `VECTORDB_NYT_QUERIES` (default `1000`), `VECTORDB_NYT_EF_CONSTRUCT` (default `100`), `VECTORDB_NYT_BASE_LIMIT` to cap inserts for faster iteration.
