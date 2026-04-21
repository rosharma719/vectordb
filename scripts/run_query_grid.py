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

RUNTIME_KNOB_COLS = [
    "search_expansion_mult",
    "search_expansion_cap",
    "disable_early_exit",
    "neighbor_rotate",
    "neighbor_stride",
    "neighbor_scan_patience",
]


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value.replace("_", ""))
    except ValueError as exc:
        raise SystemExit(f"invalid integer for {name}: {value}") from exc


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value != "0" and value.lower() != "false"


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def git_dirty() -> bool:
    result = subprocess.run(
        ["git", "diff", "--quiet", "--ignore-submodules", "HEAD", "--"],
        check=False,
    )
    return result.returncode != 0


def run_key(snapshot_name: str, ef: int, cap: int, patience: int, args: argparse.Namespace) -> tuple:
    return (
        snapshot_name,
        ef,
        cap,
        patience,
        args.search_expansion_mult,
        args.search_expansion_cap,
        args.disable_early_exit,
        args.neighbor_rotate,
        args.neighbor_stride,
        args.neighbor_scan_patience,
        args.top_k,
        args.num_queries,
    )


def run_meta(snapshot: Path, ef: int, cap: int, patience: int, args: argparse.Namespace) -> dict:
    return {
        "snapshot": snapshot.name,
        "ef_search": ef,
        "neighbor_cap": cap,
        "patience": patience,
        "search_expansion_mult": args.search_expansion_mult,
        "search_expansion_cap": args.search_expansion_cap,
        "disable_early_exit": args.disable_early_exit,
        "neighbor_rotate": args.neighbor_rotate,
        "neighbor_stride": args.neighbor_stride,
        "neighbor_scan_patience": args.neighbor_scan_patience,
        "top_k": args.top_k,
        "num_queries": args.num_queries,
        "git_commit": args.git_commit,
        "git_dirty": args.git_dirty,
    }


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
    parser.add_argument(
        "--search-expansion-mult",
        type=int,
        default=env_int("VECTORDB_SEARCH_EXPANSION_MULT", 1),
        help="Pins VECTORDB_SEARCH_EXPANSION_MULT for every run.",
    )
    parser.add_argument(
        "--search-expansion-cap",
        type=int,
        default=env_int("VECTORDB_SEARCH_EXPANSION_CAP", 0),
        help="Pins VECTORDB_SEARCH_EXPANSION_CAP; 0 means unbounded/no override.",
    )
    parser.add_argument(
        "--disable-early-exit",
        type=int,
        choices=[0, 1],
        default=1 if env_bool("VECTORDB_DISABLE_EARLY_EXIT", False) else 0,
        help="Pins VECTORDB_DISABLE_EARLY_EXIT for every run.",
    )
    parser.add_argument(
        "--neighbor-rotate",
        type=int,
        choices=[0, 1],
        default=1 if env_bool("VECTORDB_NEIGHBOR_SCAN_ROTATE", False) else 0,
        help="Pins VECTORDB_NEIGHBOR_SCAN_ROTATE for every run.",
    )
    parser.add_argument(
        "--neighbor-stride",
        type=int,
        choices=[0, 1],
        default=1 if env_bool("VECTORDB_NEIGHBOR_SCAN_STRIDE", False) else 0,
        help="Pins VECTORDB_NEIGHBOR_SCAN_STRIDE for every run.",
    )
    parser.add_argument(
        "--neighbor-scan-patience",
        type=int,
        default=env_int("VECTORDB_NEIGHBOR_SCAN_PATIENCE", 0),
        help="Pins VECTORDB_NEIGHBOR_SCAN_PATIENCE for every run.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--keep-work",
        action="store_true",
        help="Keep per-run work files in --work-dir (default deletes them).",
    )
    args = parser.parse_args()
    args.git_commit = git_commit()
    args.git_dirty = git_dirty()
    return args


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


def load_completed_runs(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    completed: set[tuple] = set()
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
                    obj.get("search_expansion_mult"),
                    obj.get("search_expansion_cap"),
                    obj.get("disable_early_exit"),
                    obj.get("neighbor_rotate"),
                    obj.get("neighbor_stride"),
                    obj.get("neighbor_scan_patience"),
                    int(obj.get("top_k")),
                    int(obj.get("num_queries")),
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
    query_log = args.work_dir / (
        f"{snapshot.stem}_ef{ef}_cap{cap}_pat{patience}"
        f"_em{args.search_expansion_mult}_ec{args.search_expansion_cap}"
        f"_dee{args.disable_early_exit}_rot{args.neighbor_rotate}"
        f"_str{args.neighbor_stride}_nsp{args.neighbor_scan_patience}"
        f"_topk{args.top_k}_q{args.num_queries}_query_log.jsonl"
    )

    env = os.environ.copy()
    env["VECTORDB_EF_SEARCH_LIST"] = str(ef)
    env["VECTORDB_NEIGHBOR_SCAN_CAP_LEVEL0"] = str(cap)
    env["VECTORDB_EARLY_EXIT_PATIENCE"] = str(patience)
    env["VECTORDB_SEARCH_EXPANSION_MULT"] = str(args.search_expansion_mult)
    env["VECTORDB_SEARCH_EXPANSION_CAP"] = str(args.search_expansion_cap)
    env["VECTORDB_DISABLE_EARLY_EXIT"] = str(args.disable_early_exit)
    env["VECTORDB_NEIGHBOR_SCAN_ROTATE"] = str(args.neighbor_rotate)
    env["VECTORDB_NEIGHBOR_SCAN_STRIDE"] = str(args.neighbor_stride)
    env["VECTORDB_NEIGHBOR_SCAN_PATIENCE"] = str(args.neighbor_scan_patience)
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

    print(
        "▶️  "
        f"snapshot={snapshot.name} ef={ef} cap={cap} pat={patience} "
        f"em={args.search_expansion_mult} ec={args.search_expansion_cap} "
        f"dee={args.disable_early_exit} rot={args.neighbor_rotate} "
        f"str={args.neighbor_stride} nsp={args.neighbor_scan_patience}"
    )
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
                    key = run_key(snapshot.name, ef, cap, patience, args)
                    if args.skip_existing and key in completed:
                        print(
                            "⏭️  skipping "
                            f"{snapshot.name} ef{ef} cap{cap} pat{patience} "
                            "(already recorded)"
                        )
                        continue
                    result = run_sweep(snapshot, ef, cap, patience, args)
                    meta = run_meta(snapshot, ef, cap, patience, args)
                    summary, entries = summarize_query_log(result["query_log"])
                    for entry in entries:
                        append_jsonl(args.out_jsonl, {"type": "query", **meta, **entry})
                    append_jsonl(args.out_jsonl, {"type": "summary", **meta, **summary})
                    if not args.keep_work and result["query_log"].exists():
                        result["query_log"].unlink()
                    completed.add(key)


if __name__ == "__main__":
    main()
