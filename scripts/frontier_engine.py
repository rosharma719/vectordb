#!/usr/bin/env python3
"""Build/query evaluation + frontier pipeline for recall sweeps."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

SNAPSHOT_FEATURES = re.compile(
    r"(?:m(?P<m>\d+))|(?:m0_(?P<m0>\d+))|(?:efc(?P<efc>\d+))|(?:alow(?P<alow>[\d\.p]+))"
    r"|(?:ahigh(?P<ahigh>[\d\.p]+))|(?:prunefloor(?P<pf>\d+))"
)


def parse_snapshot(snapshot_name: str) -> Mapping[str, float]:
    params: dict[str, float] = {}
    for token in snapshot_name.replace(".bin", "").split("_"):
        match = SNAPSHOT_FEATURES.fullmatch(token)
        if not match:
            continue
        for key, value in match.groupdict().items():
            if value is None:
                continue
            if key in {"alow", "ahigh"}:
                params[key] = float(value.replace("p", "."))
            elif key == "pf":
                params["prunefloor"] = float(value)
            else:
                params[key] = float(value)
    return params


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_eval_table(json_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(json_dir.glob("*.jsonl")):
        for record in read_jsonl(path):
            payload: dict = {}
            payload.update(record["config"])
            payload["visited_avg"] = record["query_stats"]["visited"]["avg"]
            payload["expanded_avg"] = record["query_stats"]["expanded"]["avg"]
            payload["best_score_avg"] = record["query_stats"]["best_score"]["avg"]
            payload["worst_score_avg"] = record["query_stats"]["worst_score"]["avg"]
            payload["adjacency_reads_avg"] = record["query_stats"]["neighbor_scan"]["adjacency_reads"]["avg"]
            payload["distance_computations_avg"] = record["query_stats"]["neighbor_scan"]["distance_computations"]["avg"]
            payload["cap_breaks_avg"] = record["query_stats"]["neighbor_scan"]["cap_breaks"]["avg"]
            payload["patience_breaks_avg"] = record["query_stats"]["neighbor_scan"]["patience_breaks"]["avg"]
            payload["dist_p50"] = record["distance_stats"]["p50"]
            payload["dist_p90"] = record["distance_stats"]["p90"]
            payload["snapshot"] = Path(record["snapshot"]).name
            payload["metric"] = record.get("metric", "cosine")
            # neighbor scan state encoded in filename, but we also accept fields in config
            rows.append(payload)
    if not rows:
        raise SystemExit(f"no JSONL files found in {json_dir}")
    df = pd.DataFrame(rows)
    parsed = pd.DataFrame([parse_snapshot(s) for s in df["snapshot"].astype(str)])
    return pd.concat([df, parsed], axis=1)


def per_snapshot_frontier(df: pd.DataFrame, thresholds: Iterable[float]) -> pd.DataFrame:
    records = []
    for snapshot, chunk in df.groupby("snapshot"):
        bests = compute_frontier(chunk, thresholds)
        for _, row in bests.iterrows():
            row = row.copy()
            row["snapshot"] = snapshot
            records.append(row)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def compute_frontier(df: pd.DataFrame, thresholds: Iterable[float]) -> pd.DataFrame:
    threshold_rows = []
    for r in thresholds:
        subset = df[df["best_score_avg"] >= r]
        if subset.empty:
            continue
        best = subset.loc[subset["visited_avg"].idxmin()]
        threshold_rows.append(
            {
                "recall_threshold": r,
                "visited_avg": best["visited_avg"],
                "snapshot": best["snapshot"],
                "neighbor_cap": best["neighbor_cap"],
                "patience": best["patience"],
                "rotate": best["rotate"],
            }
        )
    return pd.DataFrame(threshold_rows)


def print_frontier(frontier: pd.DataFrame) -> None:
    if frontier.empty:
        print("No frontier entries (make sure thresholds are reachable).")
        return
    print("Recall threshold  best visited avg  snapshot                       query knobs")
    for _, row in frontier.iterrows():
        print(
            f"{row['recall_threshold']:.2f}             "
            f"{row['visited_avg']:.1f}             "
            f"{row['snapshot']:<40} "
            f"cap={row['neighbor_cap']} pat={row['patience']} rot={row['rotate']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute recall/latency frontier.")
    parser.add_argument(
        "--json-dir",
        default="logs/recall_suite",
        help="Directory with per-config JSONL outputs.",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.8,0.85,0.9,0.95,0.97,0.99",
        help="Comma-separated best_score thresholds representing recall bands.",
    )
    parser.add_argument(
        "--out",
        default="logs/frontier.csv",
        help="Path to write the frontier summary.",
    )
    parser.add_argument(
        "--per-snapshot",
        action="store_true",
        help="Emit one frontier row per snapshot (threshold table per build).",
    )
    args = parser.parse_args()

    df = build_eval_table(Path(args.json_dir))
    thresholds = [float(t.strip()) for t in args.thresholds.split(",") if t.strip()]
    frontier = (
        per_snapshot_frontier(df, thresholds) if args.per_snapshot else compute_frontier(df, thresholds)
    )
    frontier.to_csv(args.out, index=False)
    print_frontier(frontier)


if __name__ == "__main__":
    main()
