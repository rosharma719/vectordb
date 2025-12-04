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
    r"\[insert_timing_chunk\] n=(\d+) avg_hnsw=([\d\.]+)ms avg_payload_idx=([\d\.]+)\u00b5s avg_filter_edges=([\d\.]+)ms avg_total=([\d\.]+)ms"
)
INSERTED_RE = re.compile(r"Inserted (\d+) vectors \(\+([\d\.]+)s\)")
SEED_RE = re.compile(
    r"\[filter_search_seed\] seeds_pool=(\d+) seeds_added=(\d+) seeds_accepted=(\d+) seeds_in_results=(\d+) final_results=(\d+)"
)
EF_RE = re.compile(r"ef_search=(\d+)")


def parse_log(path: Path):
    inserts = []
    seeds_by_ef = defaultdict(list)
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
                    "hnsw_ms": float(m.group(2)),
                    "payload_us": float(m.group(3)),
                    "filter_ms": float(m.group(4)),
                    "total_ms": float(m.group(5)),
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

    inserts = sorted(inserts, key=lambda r: r["idx"])
    return inserts, seeds_by_ef


def print_table(title: str, headers, rows):
    if not rows:
        return
    print(f"\n{title}")
    print(" | ".join(headers))
    print(" | ".join("---" for _ in headers))
    for row in rows:
        print(" | ".join(str(row.get(h, "")) for h in headers))


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


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"Log not found: {path}")
        sys.exit(1)

    inserts, seeds_by_ef = parse_log(path)
    print_table(
        "Insert Timing (per chunk)",
        ["n", "hnsw_ms", "filter_ms", "total_ms", "delta_s"],
        inserts,
    )

    seed_rows = summarize_seeds(seeds_by_ef)
    print_table(
        "Seed Stats by ef_search (averages)",
        ["ef", "samples", "pool", "added", "accepted", "in_results", "final_results"],
        seed_rows,
    )


if __name__ == "__main__":
    main()
