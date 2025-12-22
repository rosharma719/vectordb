#!/usr/bin/env python3
"""
Parse VECTORDB_SEARCH_TRACE_LOG JSONL output and summarize traversal.
Usage: python scripts/parse_search_trace_log.py /path/to/search_trace.jsonl
"""
import json
import sys
from collections import defaultdict
from pathlib import Path


def pct(vals, p):
    if not vals:
        return 0.0
    idx = int(round((p / 100.0) * (len(vals) - 1)))
    return vals[idx]


def summarize(path: Path) -> None:
    by_search = defaultdict(list)
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            by_search[entry.get("search_id")].append(entry)

    if not by_search:
        print("no entries parsed")
        return

    def summarize_group(items, label):
        steps = []
        visited = []
        expanded = []
        stop_reasons = defaultdict(int)
        for _, entries in items:
            entries.sort(key=lambda e: e.get("step", 0))
            last = entries[-1]
            steps.append(last.get("step", 0))
            visited.append(last.get("visited", 0))
            expanded.append(last.get("expanded", 0))
            reason = last.get("stop_reason") or "unknown"
            stop_reasons[reason] += 1

        steps.sort()
        visited.sort()
        expanded.sort()

        def fmt(vals):
            return (
                f"min={vals[0]} p50={pct(vals,50)} p90={pct(vals,90)} "
                f"p99={pct(vals,99)} max={vals[-1]}"
            )

        print(f"\n{label} searches={len(items)}")
        print(f"steps:   {fmt(steps)}")
        print(f"visited: {fmt(visited)}")
        print(f"expanded:{fmt(expanded)}")
        print("stop_reasons:")
        for reason, count in sorted(stop_reasons.items(), key=lambda x: x[1], reverse=True):
            pct_val = (count / len(items)) * 100.0
            print(f"  {reason}: {count} ({pct_val:.1f}%)")

    summarize_group(list(by_search.items()), "all")

    by_metric = defaultdict(list)
    for key, entries in by_search.items():
        metric = entries[-1].get("metric", "unknown")
        by_metric[metric].append((key, entries))

    for metric, items in sorted(by_metric.items()):
        summarize_group(items, f"metric={metric}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    summarize(Path(sys.argv[1]))
