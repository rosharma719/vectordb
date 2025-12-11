#!/usr/bin/env python3
"""Parse NYT patience/mult sweep logs and summarize results."""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional


RUN_RE = re.compile(r"^=== RUN patience=(?P<patience>\d+) mult=(?P<mult>\d+) ef_search_list=(?P<ef_list>[0-9,]+)")
DONE_RE = re.compile(r"^=== (?P<status>DONE|FAIL) patience=(?P<patience>\d+) mult=(?P<mult>\d+)")
RECALL_RE = re.compile(
    r"\[ef_search=(?P<ef>\d+)\] recall@(?P<k>\d+): (?P<recall>[0-9.]+) .* avg (?P<ms>[0-9.]+) ms/query"
)


class RunResult:
    def __init__(self, patience: int, mult: int, ef_list: str):
        self.patience = patience
        self.mult = mult
        self.ef_list = ef_list
        self.status: Optional[str] = None
        self.results: Dict[int, Dict[str, float]] = {}

    def key(self):
        return f"patience={self.patience} mult={self.mult}"


def parse_log(path: Path) -> List[RunResult]:
    runs: List[RunResult] = []
    current: Optional[RunResult] = None

    with path.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            m_run = RUN_RE.match(line)
            if m_run:
                if current is not None:
                    runs.append(current)
                current = RunResult(
                    patience=int(m_run.group("patience")),
                    mult=int(m_run.group("mult")),
                    ef_list=m_run.group("ef_list"),
                )
                continue

            m_done = DONE_RE.match(line)
            if m_done and current is not None:
                current.status = m_done.group("status")
                runs.append(current)
                current = None
                continue

            m_recall = RECALL_RE.search(line)
            if m_recall and current is not None:
                ef = int(m_recall.group("ef"))
                current.results[ef] = {
                    "k": float(m_recall.group("k")),
                    "recall": float(m_recall.group("recall")),
                    "ms": float(m_recall.group("ms")),
                }

    if current is not None:
        runs.append(current)

    return runs


def summarize_best(runs: List[RunResult]):
    by_ef: Dict[int, List[RunResult]] = {}
    for run in runs:
        if run.status != "DONE":
            continue
        for ef in run.results:
            by_ef.setdefault(ef, []).append(run)

    best_latency = {}
    best_recall = {}
    for ef, ef_runs in by_ef.items():
        # Best latency at this ef.
        lat_best = min(ef_runs, key=lambda r: r.results[ef]["ms"])
        best_latency[ef] = (lat_best.results[ef]["ms"], lat_best.results[ef]["recall"], lat_best)
        # Best recall at this ef.
        rec_best = max(ef_runs, key=lambda r: r.results[ef]["recall"])
        best_recall[ef] = (rec_best.results[ef]["recall"], rec_best.results[ef]["ms"], rec_best)
    return best_latency, best_recall


def main():
    parser = argparse.ArgumentParser(description="Summarize NYT patience/mult sweep log")
    parser.add_argument("log", nargs="?", default="logs/nyt_patience_mult_sweep.log", help="Path to log file")
    args = parser.parse_args()

    path = Path(args.log)
    if not path.exists():
        raise SystemExit(f"Log file not found: {path}")

    runs = parse_log(path)
    if not runs:
        raise SystemExit("No runs parsed.")

    print(f"Parsed {len(runs)} runs from {path}")
    for run in runs:
        status = run.status or "INCOMPLETE"
        print(f"\n{run.key()} status={status} ef_list={run.ef_list}")
        for ef in sorted(run.results):
            res = run.results[ef]
            print(
                f"  ef={ef}: recall@{int(res['k'])}={res['recall']:.3f} avg_ms={res['ms']:.3f}"
            )

    best_latency, best_recall = summarize_best(runs)

    if best_latency:
        print("\nBest latency per ef:")
        for ef in sorted(best_latency):
            ms, recall, run = best_latency[ef]
            print(
                f"  ef={ef}: {ms:.3f} ms (recall {recall:.3f}) from {run.key()}"
            )

    if best_recall:
        print("\nBest recall per ef:")
        for ef in sorted(best_recall):
            recall, ms, run = best_recall[ef]
            print(
                f"  ef={ef}: recall {recall:.3f} (avg_ms {ms:.3f}) from {run.key()}"
            )


if __name__ == "__main__":
    main()
