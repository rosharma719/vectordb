#!/usr/bin/env python3
"""Grid runner that exhaustively sweeps NYT snapshots over query-time knobs.

This script runs the canonical NYTimes recall harness (`nytimes_recall_from_snapshot`) so the
output includes:
- canonical recall@k per query (from dataset ground truth)
- latency per query in milliseconds (`elapsed_ms`)
- visited/expanded counters per query
- misses per query

It writes everything into a single JSONL:
- `type="query"`: one entry per query
- `type="summary"`: one entry per (snapshot × knobs) run

Use `--skip-existing` to resume: it skips any run that already has a `type="summary"` entry in the
output JSONL.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshots-dir",
        type=Path,
        default=Path("data/nyt-grid-snapshots"),
        help="Directory containing the snapshot binaries to score.",
    )
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=Path("logs/nyt_query_suite.jsonl"),
        help="Single JSONL to append per-query traces and per-run summaries.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("logs/nyt_query_suite/work"),
        help="Working directory for temporary per-run logs (safe to delete).",
    )
    parser.add_argument(
        "--ef-search",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256, 512],
        help="EF search values to sweep (per run, one value is treated as the list).",
    )
    parser.add_argument(
        "--neighbor-cap",
        type=int,
        nargs="+",
        default=[0, 32, 64],
        help="neighbor scan cap level0 options; 0 means off.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        nargs="+",
        default=[0, 2, 4],
        help="Early exit patience values (0 disables).",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--num-queries", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1, help="Not used (API compatibility).")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--keep-work",
        action="store_true",
        help="Keep per-run work files in --work-dir (default deletes them).",
    )
    return parser.parse_args()


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    idx = int(round((p / 100.0) * (len(values) - 1)))
    idx = max(0, min(idx, len(values) - 1))
    return float(values[idx])


def summarize_query_log(path: Path) -> tuple[dict, list[dict]]:
    elapsed_ms: list[float] = []
    recalls: list[float] = []
    visited: list[float] = []
    expanded: list[float] = []
    misses: list[float] = []
    entries: list[dict] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            entries.append(entry)
            elapsed_ms.append(float(entry["elapsed_ms"]))
            recalls.append(float(entry["recall"]))
            if entry.get("visited") is not None:
                visited.append(float(entry["visited"]))
            if entry.get("expanded") is not None:
                expanded.append(float(entry["expanded"]))
            misses.append(float(entry.get("misses", 0)))

    ms_avg = sum(elapsed_ms) / len(elapsed_ms) if elapsed_ms else float("nan")
    recall_avg = sum(recalls) / len(recalls) if recalls else float("nan")
    visited_avg = sum(visited) / len(visited) if visited else float("nan")
    expanded_avg = sum(expanded) / len(expanded) if expanded else float("nan")
    misses_avg = sum(misses) / len(misses) if misses else float("nan")
    return (
        {
            "recall_avg": recall_avg,
            "recall_p50": percentile(recalls, 50.0),
            "recall_p90": percentile(recalls, 90.0),
            "recall_p99": percentile(recalls, 99.0),
            "ms_avg": ms_avg,
            "ms_p50": percentile(elapsed_ms, 50.0),
            "ms_p90": percentile(elapsed_ms, 90.0),
            "ms_p99": percentile(elapsed_ms, 99.0),
            "visited_avg": visited_avg,
            "visited_p50": percentile(visited, 50.0),
            "visited_p90": percentile(visited, 90.0),
            "visited_p99": percentile(visited, 99.0),
            "expanded_avg": expanded_avg,
            "expanded_p50": percentile(expanded, 50.0),
            "expanded_p90": percentile(expanded, 90.0),
            "expanded_p99": percentile(expanded, 99.0),
            "misses_avg": misses_avg,
            "query_count": len(entries),
        },
        entries,
    )


def load_completed_runs(path: Path) -> set[tuple[str, int, int, int]]:
    if not path.exists():
        return set()
    completed: set[tuple[str, int, int, int]] = set()
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") != "summary":
                continue
            completed.add(
                (
                    str(obj.get("snapshot")),
                    int(obj.get("ef_search")),
                    int(obj.get("neighbor_cap")),
                    int(obj.get("patience")),
                )
            )
    return completed


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(obj) + "\n")


def run_sweep(
    snapshot: Path,
    ef: int,
    cap: int,
    patience: int,
    args: argparse.Namespace,
) -> dict:
    args.work_dir.mkdir(parents=True, exist_ok=True)
    query_log = args.work_dir / f"{snapshot.stem}_ef{ef}_cap{cap}_pat{patience}_query_log.jsonl"

    env = os.environ.copy()
    env["VECTORDB_EF_SEARCH_LIST"] = str(ef)
    env["VECTORDB_NEIGHBOR_SCAN_CAP_LEVEL0"] = str(cap)
    env["VECTORDB_EARLY_EXIT_PATIENCE"] = str(patience)
    env["VECTORDB_TEST_LOG"] = env.get("VECTORDB_TEST_LOG", "info")
    env["VECTORDB_QUERY_LOG"] = str(query_log)
    env["VECTORDB_QUERY_LOG_EVERY"] = "1"
    env["VECTORDB_TOPK"] = str(args.top_k)
    env["VECTORDB_QUERIES"] = str(args.num_queries)
    env["VECTORDB_NYT_PERSIST_PATH"] = str(snapshot)
    env["VECTORDB_NYT_USE_SNAPSHOT"] = "1"
    env["VECTORDB_NYT_ALLOW_BUILD"] = "0"

    cmd = [
        "cargo",
        "test",
        "--release",
        "nytimes_recall_from_snapshot",
        "--",
        "--ignored",
        "--nocapture",
    ]

    if args.skip_existing and query_log.exists():
        print(f"⏭️  skipping {snapshot.name} ef{ef} cap{cap} pat{patience} (work exists)")
        return {"query_log": query_log}

    print(f"▶️  snapshot={snapshot.name} ef={ef} cap={cap} pat={patience}")
    subprocess.run(cmd, check=True, env=env)
    return {"query_log": query_log}


def main() -> None:
    args = parse_args()
    snapshots = sorted(args.snapshots_dir.glob("*.bin"))
    if not snapshots:
        raise SystemExit(f"no snapshots found in {args.snapshots_dir}")
    completed = load_completed_runs(args.out_jsonl) if args.skip_existing else set()
    for snapshot in snapshots:
        for ef in args.ef_search:
            for cap in args.neighbor_cap:
                for patience in args.patience:
                    key = (snapshot.name, ef, cap, patience)
                    if args.skip_existing and key in completed:
                        print(f"⏭️  skipping {snapshot.name} ef{ef} cap{cap} pat{patience} (already recorded)")
                        continue
                    result = run_sweep(snapshot, ef, cap, patience, args)
                    run_meta = {
                        "snapshot": snapshot.name,
                        "ef_search": ef,
                        "neighbor_cap": cap,
                        "patience": patience,
                        "top_k": args.top_k,
                        "num_queries": args.num_queries,
                    }
                    summary, entries = summarize_query_log(result["query_log"])
                    for entry in entries:
                        append_jsonl(args.out_jsonl, {"type": "query", **run_meta, **entry})
                    append_jsonl(args.out_jsonl, {"type": "summary", **run_meta, **summary})
                    if not args.keep_work and result["query_log"].exists():
                        result["query_log"].unlink()
                    completed.add(key)


if __name__ == "__main__":
    main()
