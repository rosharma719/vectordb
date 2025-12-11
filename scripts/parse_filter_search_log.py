#!/usr/bin/env python3
"""
Parse VECTORDB_FILTER_SEARCH_LOG output (JSONL) and print a few quick stats.
Usage: python scripts/parse_filter_search_log.py /path/to/log.jsonl
"""
import json
import math
import sys
from pathlib import Path


def pct(vals, p):
    if not vals:
        return 0.0
    idx = int(round((p / 100.0) * (len(vals) - 1)))
    return vals[idx]


def summarize(path: Path) -> None:
    entries = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not entries:
        print("no entries parsed")
        return

    expansions = sorted(e["expansions"] for e in entries)
    visited = sorted(e["visited"] for e in entries)
    ms = sorted(e["elapsed_ms"] for e in entries)
    chk = sorted(e.get("filter_checked", 0) for e in entries)
    pas = sorted(e.get("filter_passed", 0) for e in entries)
    stop_reasons = {}
    for e in entries:
        stop_reasons[e["stop_reason"]] = stop_reasons.get(e["stop_reason"], 0) + 1

    seeds_used = sum(1 for e in entries if e["seeds_added"] > 0)
    seeds_helped = sum(1 for e in entries if e["seeds_in_results"] > 0)
    pass_rates = sorted(
        (e.get("filter_passed", 0) / e.get("filter_checked", 1)) for e in entries if e.get("filter_checked", 0) > 0
    )
    seeds_in_results = sorted(e.get("seeds_in_results", 0) for e in entries)
    seeds_overlap = sorted(
        (e.get("seeds_in_results", 0) / max(e.get("results_len", 1), 1))
        for e in entries
        if e.get("results_len", 0) > 0
    )
    seeds_popped = sorted(e.get("seeds_popped", 0) for e in entries)

    def fmt_dist(vals):
        return (
            f"min={vals[0]:.1f} p50={pct(vals,50):.1f} "
            f"p90={pct(vals,90):.1f} p99={pct(vals,99):.1f} max={vals[-1]:.1f}"
        )

    print(f"entries={len(entries)} filter_present={entries[0].get('filter_present', False)}")
    print(f"expansions: {fmt_dist(expansions)}")
    print(f"visited:    {fmt_dist(visited)}")
    print(f"elapsed_ms: {fmt_dist(ms)}")
    if chk:
        print(f"filter_checked: {fmt_dist(chk)}")
        print(f"filter_passed:  {fmt_dist(pas)}")
    if pass_rates:
        pr = pass_rates
        print(
            f"filter_pass_rate: min={pr[0]:.3f} p50={pct(pr,50):.3f} "
            f"p90={pct(pr,90):.3f} p99={pct(pr,99):.3f} max={pr[-1]:.3f}"
        )
    print("stop_reasons:")
    for reason, count in sorted(stop_reasons.items(), key=lambda x: x[1], reverse=True):
        pct_val = (count / len(entries)) * 100.0
        print(f"  {reason}: {count} ({pct_val:.1f}%)")
    print(f"seeds_added>0: {seeds_used}/{len(entries)} ({(seeds_used/len(entries))*100:.1f}%)")
    print(f"seeds_in_results>0: {seeds_helped}/{len(entries)} ({(seeds_helped/len(entries))*100:.1f}%)")
    if seeds_in_results:
        print(f"seeds_in_results_count: {fmt_dist(seeds_in_results)}")
    if seeds_overlap:
        print(
            f"seeds_overlap_ratio: min={seeds_overlap[0]:.3f} p50={pct(seeds_overlap,50):.3f} "
            f"p90={pct(seeds_overlap,90):.3f} p99={pct(seeds_overlap,99):.3f} max={seeds_overlap[-1]:.3f}"
        )
    if seeds_popped:
        print(f"seeds_popped_during_walk: {fmt_dist(seeds_popped)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    summarize(Path(sys.argv[1]))
