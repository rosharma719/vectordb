# Recall/Filtering test config cheat sheet

Tests are `#[ignore]` and driven entirely by env vars. Defaults favor loading prebuilt snapshots.

Snapshot defaults (shared):
- `VECTORDB_PERSIST_PATH` (or dataset-specific overrides below)
- `VECTORDB_USE_SNAPSHOT` (default `1`)
- `VECTORDB_ALLOW_BUILD` (default `0` — set to `1` to build if snapshot is missing)
- `VECTORDB_SAVE_SNAPSHOT` (default `0`)

- Core recall controls (shared):
  - `VECTORDB_M` / `VECTORDB_M0` (graph connectivity; M0 defaults to M)
  - `VECTORDB_EF_SEARCH_LIST` (or dataset-specific overrides below)
  - `VECTORDB_EARLY_EXIT_PATIENCE`
  - `VECTORDB_DISABLE_EARLY_EXIT` (set to `1` to turn it off)
  - `VECTORDB_SEARCH_EXPANSION_MULT` (default 4)
  - `VECTORDB_SEARCH_EXPANSION_CAP` (0 => uncapped for unfiltered search)
  - `VECTORDB_FILTER_EXPANSION_CAP` (0 => uncapped for filtered search; otherwise max expansions)
  - `VECTORDB_DISABLE_FILTER_SEEDS` (set to `1` to disable seeding; default seeds enabled)
  - `VECTORDB_FILTER_SEARCH_LOG` path to JSONL for per-query traversal stats
  - `VECTORDB_FILTER_PASSING_BUDGET` / `VECTORDB_FILTER_FAILING_BUDGET` (override routing budgets; defaults derive from M: passing≈max(8, 2*M), failing=1)

- Logging/analysis:
  - `VECTORDB_TEST_LOG` -> `quiet|info|debug` (default `info`)
  - `VECTORDB_PROGRESS_EVERY` -> progress interval for debug logs (default `100`)
  - `VECTORDB_QUERY_LOG` -> JSONL path for per-query stats
  - `VECTORDB_QUERY_LOG_EVERY` -> sample interval for per-query stats (default `1`)
  - `VECTORDB_SEARCH_TRACE_LOG` -> JSONL path for unfiltered traversal trace
  - `VECTORDB_INSERT_TRACE_LOG` -> JSONL path for insert traversal trace
  - `VECTORDB_TRACE_EVERY` -> sample interval for trace logs (default `100`)
  - `VECTORDB_FILTER_SEARCH_LOG` -> JSONL with per-query expansions/visited/filter hit rates/seeds/stop_reason.
  - Parse filter logs with `python scripts/parse_filter_search_log.py <log.jsonl>`.

- NYTimes (unfiltered):
  - `VECTORDB_NYT_DATA_DIR` (default `data/nytimes-256-angular`)
  - `VECTORDB_NYT_PERSIST_PATH` (default `data/nytimes-256-angular/index_m16_m0_32_efc100.bin`)
  - `VECTORDB_NYT_USE_SNAPSHOT` (override `VECTORDB_USE_SNAPSHOT`)
  - `VECTORDB_NYT_ALLOW_BUILD` / `VECTORDB_NYT_SAVE_SNAPSHOT`
  - `VECTORDB_NYT_M` / `VECTORDB_NYT_M0`
  - `VECTORDB_NYT_EF_SEARCH_LIST` (falls back to `VECTORDB_EF_SEARCH_LIST`)
  - `VECTORDB_EF_CONSTRUCT` / `VECTORDB_NYT_EF_CONSTRUCT`

- H&M (filtered):
  - `VECTORDB_HNM_DATA_DIR` (default `data/hnm`)
  - `VECTORDB_HNM_PERSIST_PATH` (default `data/hnm/index_filtered.bin`)
  - `VECTORDB_HNM_USE_SNAPSHOT` (override `VECTORDB_USE_SNAPSHOT`)
  - `VECTORDB_HNM_ALLOW_BUILD` / `VECTORDB_HNM_SAVE_SNAPSHOT`
  - `VECTORDB_HNM_M` / `VECTORDB_HNM_M0` (override HNSW M/M0 for rebuilds; M0 defaults to M)
  - `VECTORDB_HNM_EF_SEARCH_LIST` (falls back to `VECTORDB_EF_SEARCH_LIST`)
  - `VECTORDB_HNM_EF_CONSTRUCT` (build-time)
  - `VECTORDB_HNM_TOPK` (default derived from test cases)
  - `VECTORDB_DISABLE_FILTER_SEEDS` (set to `1` for unseeded runs)

Example commands:

- H&M seeded, EF=64, patience=3, uncapped filtered expansions:
```
VECTORDB_HNM_PERSIST_PATH=data/hnm/index_filtered.bin \
VECTORDB_HNM_EF_SEARCH_LIST=64 \
VECTORDB_EARLY_EXIT_PATIENCE=3 \
VECTORDB_SEARCH_EXPANSION_MULT=4 \
VECTORDB_FILTER_EXPANSION_CAP=0 \
VECTORDB_FILTER_SEARCH_LOG=logs/hnm_seeded_ef64.jsonl \
cargo test --release hnm_recall_from_snapshot -- --ignored --nocapture
```

- H&M unseeded, EF=64, patience=3, uncapped filtered expansions:
```
VECTORDB_HNM_PERSIST_PATH=data/hnm/index_filtered.bin \
VECTORDB_HNM_EF_SEARCH_LIST=64 \
VECTORDB_EARLY_EXIT_PATIENCE=3 \
VECTORDB_SEARCH_EXPANSION_MULT=4 \
VECTORDB_FILTER_EXPANSION_CAP=0 \
VECTORDB_DISABLE_FILTER_SEEDS=1 \
VECTORDB_FILTER_SEARCH_LOG=logs/hnm_noseed_ef64.jsonl \
cargo test --release hnm_recall_from_snapshot -- --ignored --nocapture
```

- NYT unfiltered, EF list example:
```
VECTORDB_NYT_PERSIST_PATH=data/nytimes-256-angular/index.bin \
VECTORDB_EF_SEARCH_LIST=64,128,256,512 \
VECTORDB_EARLY_EXIT_PATIENCE=3 \
VECTORDB_SEARCH_EXPANSION_MULT=4 \
VECTORDB_SEARCH_EXPANSION_CAP=0 \
cargo test --release nytimes_recall_from_snapshot -- --ignored --nocapture
```
