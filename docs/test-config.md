# Recall/Filtering test config cheat sheet

Tests are `#[ignore]` and driven entirely by env vars. Use these knobs consistently across NYT + H&M runs:

- Core recall controls (shared):
  - `VECTORDB_EF_SEARCH_LIST` (or dataset-specific overrides below)
  - `VECTORDB_EARLY_EXIT_PATIENCE`
  - `VECTORDB_DISABLE_EARLY_EXIT` (set to `1` to turn it off)
  - `VECTORDB_SEARCH_EXPANSION_MULT` (default 4)
  - `VECTORDB_SEARCH_EXPANSION_CAP` (0 => uncapped for unfiltered search)
  - `VECTORDB_FILTER_EXPANSION_CAP` (0 => uncapped for filtered search; otherwise max expansions)
  - `VECTORDB_DISABLE_FILTER_SEEDS` (set to `1` to disable seeding; default seeds enabled)
  - `VECTORDB_FILTER_SEARCH_LOG` path to JSONL for per-query traversal stats

- NYTimes (unfiltered):
  - `VECTORDB_NYT_DATA_DIR` (default `data/nytimes-256-angular`)
  - `VECTORDB_NYT_PERSIST_PATH` snapshot path
  - `VECTORDB_NYT_EF_SEARCH_LIST` (falls back to `VECTORDB_EF_SEARCH_LIST`)
  - `VECTORDB_EF_CONSTRUCT` / `VECTORDB_NYT_EF_CONSTRUCT`

- H&M (filtered):
  - `VECTORDB_HNM_DATA_DIR` (default `data/hnm`)
  - `VECTORDB_HNM_PERSIST_PATH` snapshot path (e.g. `data/hnm/index_filtered.bin`)
  - `VECTORDB_HNM_EF_SEARCH_LIST` (falls back to `VECTORDB_EF_SEARCH_LIST`)
  - `VECTORDB_HNM_EF_CONSTRUCT` (build-time)
  - `VECTORDB_HNM_TOPK` (default derived from test cases)
  - `VECTORDB_DISABLE_FILTER_SEEDS` (set to `1` for unseeded runs)

- Logging/analysis:
  - `VECTORDB_FILTER_SEARCH_LOG` -> JSONL with per-query expansions/visited/filter hit rates/seeds/stop_reason.
  - Parse with `python scripts/parse_filter_search_log.py <log.jsonl>`.

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
