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

## NYTimes 256-d Angular (Hugging Face)
- Data source: `open-vdb/nytimes-256-angular` dataset; load configs `train` (base vectors), `test` (queries), `neighbors` (ground truth).
- Files expected under `VECTORDB_NYT_DATA_DIR` (default `data/nytimes-256-angular`): `base.npy`, `queries.npy`, `ground_truth.json`. See README for the download script (requires `HF_TOKEN` with repo read access).
- Test: `nytimes_256_angular_perf_and_recall` (ignored). Run with `cargo test --release nytimes_256_angular_perf_and_recall -- --ignored --nocapture`.
- Knobs: `VECTORDB_NYT_TOPK` (default `20`), `VECTORDB_NYT_EF_SEARCH` (default `128`) or `VECTORDB_NYT_EF_SEARCH_LIST` for sweeps (e.g., `16,32,64,128,256`), `VECTORDB_NYT_QUERIES` (default `1000`), `VECTORDB_NYT_EF_CONSTRUCT` (default `100`), `VECTORDB_NYT_BASE_LIMIT` to cap inserts for faster iteration.
