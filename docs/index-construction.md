# Index construction, naming, and ML tuning workflow

This project already ships tools to analyze existing snapshots (`analysis::analyze_snapshot`, `snapshot_sweeper`, `index_analyzer`, etc.), but the build/naming surface has been ad-hoc. The new manifest + helper script standardize how we name snapshots and how we collect labeled tuning data before rebuilding.

## 1. Canonical snapshot naming

Every persisted HNSW graph should be identified by the build-time knobs that generated it. We use the format:

```
{dataset_dir}/index_{metric}_{tokens}.bin
```

Tokens are appended in this order when the value is non-`null`:

- `m{M}` – the HNSW M parameter
- `m0_{M0}` – explicit M0 when it differs from M
- `efc{EF_CONSTRUCT}` – the `ef_construct` used for insert routing
- `a{ALPHA}` – `diversity_alpha` (dots become `p`, negatives become `n`)
- `l{LEVEL_SCALE}` – a custom level scale multiplier
- `maxlevel{MAX_LEVEL_CAP}` – `VECTORDB_MAX_LEVEL_CAP`
- `cap{NEIGHBOR_SCAN_CAP}` – the `VECTORDB_NEIGHBOR_SCAN_CAP_LEVEL0` window

All floating-point tokens are flattened (`1.3 → a1p3`, `0.5 → l0p5`), so names stay filesystem-friendly. The script in section 2 turns a manifest of knob combinations into these canonical names and records any legacy snapshot files that pre-date the schema.

## 2. Build manifest + standardize script

Edit `docs/nyt_snapshot_builds.json` (or create a new manifest for another dataset) with the desired build-time parameters. Each entry can also list `legacy_names` to document older filenames.

Run the helper to print canonical names and optionally write them to a manifest that downstream tools can consume:

```sh
python scripts/standardize_snapshot_name.py \
  --manifest docs/nyt_snapshot_builds.json \
  --output logs/nyt_snapshot_manifest.txt
```

The helper prints each canonical name with its `label`/`legacy_names` annotations and writes newline-separated paths to `logs/nyt_snapshot_manifest.txt`.

Use that manifest as input to `snapshot_sweeper` so every sweep run is tied to a well-defined config:

```sh
cargo run --bin snapshot_sweeper -- \
  --manifest logs/nyt_snapshot_manifest.txt \
  --output logs/nyt_snapshot_sweep.jsonl \
  --neighbor-scan-cap 128 \
  --top-k 10 \
  --num-queries 100 \
  --sample-size 500
```

You can reuse the same manifest to track builds (the canonical name tells you which knobs to pass via `VECTORDB_*` env vars) and to annotate analysis logs for ML training.

## 3. Sweeping build-time knobs

When you have a grid of values to explore, declare it in the manifest under a `grids`
section (see the new entry at the bottom of `docs/nyt_snapshot_builds.json`). Each
grid entry specifies a `label_prefix`, an ordered `axis_order`, and a map of `axes` to
value lists. The standardization script now expands those Cartesian products for you,
tacking a descriptive label such as `nyt_grid_m16_m0_16_diversity_alpha_low1p0_`
onto every combo before emitting canonical paths. The builder automatically skips
combos where `m > m0` since those graphs violate the usual HNSW construction rules.

The current grid iterates over `{m=8,16,32}`, `{m0=8,16,32}`, `{diversity_alpha_low=1,1.2}`,
`{diversity_alpha_high=1,1.2}`, `{diversity_prune_floor=0,2}`, and `{ef_construct=100,200,300}`
so you can sweep all requested build-time knobs without manually enumerating 216 files.
Run the helper as usual and pipe the output manifest into `snapshot_sweeper` so each
line is tied to a known configuration. Because the axis values map directly to environment
variables such as `VECTORDB_NYT_M`, `VECTORDB_DIVERSITY_ALPHA_LOW`, and so on, you can
also script builds that set the matching `VECTORDB_*` knobs and persist the snapshots
under those canonical names.

## 4. Automating snapshot builds

Use the new helper to translate every manifest entry into a concrete build command:

```sh
python scripts/build_snapshots_from_manifest.py \
  --manifest docs/nyt_snapshot_builds.json \
  --dataset-dir data/nytimes-256-angular \
  --skip-existing \
  --limit 20
```

By default the script prints an env-prefixed command for every configuration (already honoring the canonical path) so you can inspect or tweak it. Add `--execute` to actually run them sequentially, `--skip-existing` to avoid rebuilding paths you already have, and `--limit`/`--cargo-extra` to sample a subset or append extra args. The annotated output shows the label and knob-specific env vars so you know exactly which `VECTORDB_*` bindings match each canonical name.

## 5. Building with knobs

Snapshots are built via the existing test harnesses (`nytimes_build_and_persist_snapshot_only`, etc.). Export the appropriate `VECTORDB_*` env vars before invoking `cargo test --release` so the graph is built with the knobs encoded in the canonical name. For example:

```sh
VECTORDB_NYT_PERSIST_PATH=data/nytimes-256-angular/index_cosine_m16_m0_32_efc_100.bin \
VECTORDB_NYT_M=16 \
VECTORDB_NYT_M0=32 \
VECTORDB_NYT_EF_CONSTRUCT=100 \
VECTORDB_DIVERSITY_ALPHA=1 \
VECTORDB_NEIGHBOR_SCAN_CAP_LEVEL0=128 \
VECTORDB_NYT_ALLOW_BUILD=1 \
VECTORDB_NYT_SAVE_SNAPSHOT=1 \
cargo test --release nytimes_build_and_persist_snapshot_only -- --ignored --nocapture
```

Make sure the `VECTORDB_NYT_PERSIST_PATH` matches the canonical name generated by the script, and keep the manifest updated with any new knob combinations you plan to analyze.

## 6. Machine-learning tuning loop

With canonical naming in place you can now treat each manifest entry as a labeled sample for a surrogate model:

1. Generate canonical names + manifest (step 2) and build the snapshots (step 3).
2. Run the sweeper with your manifest to emit `logs/nyt_snapshot_sweep.jsonl`; each line already captures the config, per-level degrees, visit/expansion counts, neighbor scan stats, and sampled dataset distances.
3. Use `logs/nyt_snapshot_sweep.jsonl` as your training corpus. Extract features such as `m`, `m0`, `ef_construct`, `level_scale`, `diversity_alpha`, plus summary stats (degree p50/p90, visit-to-expansion ratio, distance percentiles).
4. Fit a surrogate regressor (tree/GP/ensemble) that predicts latency and recall for a new combination `C`. Run Bayesian optimization or multi-armed bandit search over the surrogate to triage the next snapshots to build.
5. Optionally, augment your controller with per-query knobs (`VECTORDB_EF_SEARCH_LIST`, `VECTORDB_FILTER_EXPANSION_CAP`, `VECTORDB_NEIGHBOR_SCAN_CAP_LEVEL0`, `VECTORDB_DIVERSITY_ALPHA`) and re-score snapshots whenever runtime logs drift.

`analysis::AnalyzerConfig` and `index_analyzer` already expose APIs to pull more detailed stats for a single snapshot, so you can gather richer features after the initial sweep.

## 7. Runtime knob adaptation

The analyzer utilities (`logs/nyt_snapshot_sweep.jsonl`, `logs/nyt_query_log.jsonl`, or filtered `VECTORDB_FILTER_SEARCH_LOG`) also include per-query visit/expanded/patience counts. Feed these signals into a lightweight controller that adjusts `ef_search` or `neighbor_scan_cap_level0` on the fly to hit latency targets, falling back to a new snapshot only when the surrogate predicts a meaningful recall uplift.

When you retrain the surrogate, append the new canonical names + measured stats to your dataset so the optimizer remains aware of past builds and their true performance curves.
