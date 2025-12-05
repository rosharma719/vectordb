#!/usr/bin/env python3
"""
Parse benchmark logs and emit compact tables.

Usage:
  python scripts/parse_logs.py path/to/logfile.log
"""
import re
import sys
from collections import defaultdict
from pathlib import Path


INSERT_RE = re.compile(
    r"\[insert_timing_chunk\] n=(\d+) cum_n=(\d+) avg_hnsw=([\d\.]+)ms avg_payload_idx=([\d\.]+)\u00b5s avg_filter_edges=([\d\.]+)ms avg_total=([\d\.]+)ms chunk_elapsed=([\d\.\w]+)"
)
INSERTED_RE = re.compile(r"Inserted (\d+) vectors \(\+([\d\.]+)s\)")
SEED_RE = re.compile(
    r"\[filter_search_seed\] seeds_pool=(\d+) seeds_added=(\d+) seeds_accepted=(\d+) seeds_in_results=(\d+) final_results=(\d+)"
)
EF_RE = re.compile(r"ef_search=(\d+)")
RECALL_STATS_RE = re.compile(
    r"\[recall_stats\] ef_search=(\d+) queries=(\d+) mean=([\d\.]+) p50=([\d\.]+) p90=([\d\.]+) p99=([\d\.]+) min=([\d\.]+) max=([\d\.]+) full=(\d+) ge_0\.8=(\d+) ge_0\.5=(\d+)"
)
EDGE_AGG_RE = re.compile(
    r"\[filter_edges_agg\] n=(\d+) cum_n=(\d+) samples=(\d+) scored=(\d+) added=(\d+) total_ms=([\d\.]+) bool_ms=([\d\.]+) str_ms=([\d\.]+) int_ms=([\d\.]+) float_ms=([\d\.]+)"
)


def parse_log(path: Path):
    if not path.is_file():
        return [], defaultdict(list), {}, []
    inserts = []
    seeds_by_ef = defaultdict(list)
    recall_by_ef = {}
    edge_aggs = []
    last_inserted_time = None
    current_ef = None
    insert_chunk_idx = 0

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if m := EF_RE.search(line):
            current_ef = int(m.group(1))
        if m := INSERT_RE.search(line):
            inserts.append(
                {
                    "idx": len(inserts),
                    "n": int(m.group(1)),
                    "cum_n": int(m.group(2)),
                    "hnsw_ms": float(m.group(3)),
                    "payload_us": float(m.group(4)),
                    "filter_ms": float(m.group(5)),
                    "total_ms": float(m.group(6)),
                    "chunk_elapsed": m.group(7),
                    "delta_s": None,  # filled from INSERTED_RE
                }
            )
        elif m := INSERTED_RE.search(line):
            cumulative = float(m.group(2))
            if last_inserted_time is None:
                delta = cumulative
            else:
                delta = cumulative - last_inserted_time
            last_inserted_time = cumulative
            # attach delta to the next insert chunk in order, if present
            if insert_chunk_idx < len(inserts):
                inserts[insert_chunk_idx]["delta_s"] = round(delta, 3)
                insert_chunk_idx += 1
        elif m := SEED_RE.search(line):
            record = {
                "pool": int(m.group(1)),
                "added": int(m.group(2)),
                "accepted": int(m.group(3)),
                "in_results": int(m.group(4)),
                "final_results": int(m.group(5)),
            }
            seeds_by_ef[current_ef].append(record)
        elif m := RECALL_STATS_RE.search(line):
            recall_by_ef[int(m.group(1))] = {
                "queries": int(m.group(2)),
                "mean": float(m.group(3)),
                "p50": float(m.group(4)),
                "p90": float(m.group(5)),
                "p99": float(m.group(6)),
                "min": float(m.group(7)),
                "max": float(m.group(8)),
                "full": int(m.group(9)),
                "ge_08": int(m.group(10)),
                "ge_05": int(m.group(11)),
            }
        elif m := EDGE_AGG_RE.search(line):
            edge_aggs.append(
                {
                    "n": int(m.group(1)),
                    "cum_n": int(m.group(2)),
                    "samples": int(m.group(3)),
                    "scored": int(m.group(4)),
                    "added": int(m.group(5)),
                    "total_ms": float(m.group(6)),
                    "bool_ms": float(m.group(7)),
                    "str_ms": float(m.group(8)),
                    "int_ms": float(m.group(9)),
                    "float_ms": float(m.group(10)),
                }
            )

    inserts = sorted(inserts, key=lambda r: r["idx"])
    return inserts, seeds_by_ef, recall_by_ef, edge_aggs


def render_table(title: str, headers, rows) -> str:
    if not rows:
        return ""
    lines = [
        f"\n{title}",
        " | ".join(headers),
        " | ".join("---" for _ in headers),
    ]
    for row in rows:
        lines.append(" | ".join(str(row.get(h, "")) for h in headers))
    return "\n".join(lines)


def summarize_seeds(seeds_by_ef):
    rows = []
    for ef, records in sorted(
        ((k, v) for k, v in seeds_by_ef.items() if k is not None),
        key=lambda kv: kv[0],
    ):
        if not records:
            continue
        cnt = len(records)
        avg = lambda key: sum(r[key] for r in records) / cnt
        pool = avg("pool")
        added = avg("added")
        accepted = avg("accepted")
        in_results = avg("in_results")
        final_results = avg("final_results")
        rows.append(
            {
                "ef": ef,
                "samples": cnt,
                "pool": f"{pool:.1f}",
                "added": f"{added:.1f} ({(added/pool*100 if pool else 0):.1f}%)",
                "accepted": f"{accepted:.1f} ({(accepted/pool*100 if pool else 0):.1f}%)",
                "in_results": f"{in_results:.1f} ({(in_results/final_results*100 if final_results else 0):.1f}%)",
                "final_results": f"{final_results:.1f}",
            }
        )
    return rows


def summarize_recall(recall_by_ef):
    rows = []
    for ef, stats in sorted(recall_by_ef.items()):
        q = stats["queries"]
        rows.append(
            {
                "ef": ef,
                "mean": f"{stats['mean']:.3f}",
                "p50": f"{stats['p50']:.3f}",
                "p90": f"{stats['p90']:.3f}",
                "p99": f"{stats['p99']:.3f}",
                "min": f"{stats['min']:.3f}",
                "max": f"{stats['max']:.3f}",
                "full": f"{stats['full']}/{q}",
                ">=0.8": f"{stats['ge_08']}/{q}",
                ">=0.5": f"{stats['ge_05']}/{q}",
            }
        )
    return rows


def summarize_edges(edge_aggs):
    rows = []
    for e in edge_aggs:
        per_key_ms = e["total_ms"] / e["n"] if e["n"] else 0.0
        rows.append(
            {
                "n_keys": e["n"],
                "cum_n": e.get("cum_n", ""),
                "samples": e["samples"],
                "scored": e["scored"],
                "added": e["added"],
                "total_ms": f"{e['total_ms']:.3f}",
                "per_key_ms": f"{per_key_ms:.3f}",
                "bool_ms": f"{e['bool_ms']:.3f}",
                "str_ms": f"{e['str_ms']:.3f}",
                "int_ms": f"{e['int_ms']:.3f}",
                "float_ms": f"{e['float_ms']:.3f}",
            }
        )
    return rows


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"Log not found: {path}")
        sys.exit(1)

    inserts, seeds_by_ef, recall_by_ef, edge_aggs = parse_log(path)
    sections = [
        render_table(
            "Insert Timing (per chunk)",
            ["n", "cum_n", "hnsw_ms", "filter_ms", "total_ms", "delta_s", "chunk_elapsed"],
            inserts,
        ),
        render_table(
            "Seed Stats by ef_search (averages)",
            ["ef", "samples", "pool", "added", "accepted", "in_results", "final_results"],
            summarize_seeds(seeds_by_ef),
        ),
        render_table(
            "Recall Stats by ef_search",
            ["ef", "mean", "p50", "p90", "p99", "min", "max", "full", ">=0.8", ">=0.5"],
            summarize_recall(recall_by_ef),
        ),
        render_table(
            "Filter Edge Aggregates",
            ["n_keys", "cum_n", "samples", "scored", "added", "total_ms", "per_key_ms", "bool_ms", "str_ms", "int_ms", "float_ms"],
            summarize_edges(edge_aggs),
        ),
    ]

    summary_text = "\n\n".join(s for s in sections if s)
    if summary_text:
        print(summary_text)
        append_block = "\n\n=== parsed summary ===\n" + summary_text + "\n"
        # Append instead of rewriting the whole file to avoid copying large logs.
        with path.open("a", encoding="utf-8") as f:
            f.write(append_block)


if __name__ == "__main__":
    main()
