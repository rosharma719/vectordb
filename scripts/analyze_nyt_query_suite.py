#!/usr/bin/env python3
"""Analyze `run_query_grid.py` output JSONL into tables + knob-effect reports.

Inputs
  - A single JSONL produced by `scripts/run_query_grid.py` containing:
      * type="summary" rows (one per snapshot × query config)
      * type="query" rows (optional; ignored by default)

Outputs (by default under logs/nyt_query_suite_analysis/)
  - policy_table.csv: one row per snapshot × policy (query knobs + build knobs + metrics)
  - index_stats.csv: one row per snapshot (structural stats: degrees/levels + light geometry proxies)
  - policy_table_enriched.csv: policy_table.csv joined with index_stats.csv on snapshot (when --snapshot-dir is set)
  - run_summaries.csv: one row per snapshot × policy (recomputed from per-query rows when available)
  - frontier_table.csv: one row per snapshot × recall threshold (best latency under recall>=r)
  - global_frontier.csv: global best latency per recall threshold
  - effects_report.md: weighted linear & quadratic (degree-2) coefficient summaries + permutation importance
  - bottleneck_report.md: bottleneck diagnostics derived from query-suite metrics (no extra logging needed)
  - paired_knob_effects.csv: paired deltas within snapshots for EF/cap/patience changes
  - frontier_winners.csv: which knob settings win per threshold across snapshots
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd

QUERY_KNOB_COLS = ["ef_search", "neighbor_cap", "patience"]
RUNTIME_KNOB_COLS = [
    "search_expansion_mult",
    "search_expansion_cap",
    "disable_early_exit",
    "neighbor_rotate",
    "neighbor_stride",
    "neighbor_scan_patience",
]
RUN_ID_COLS = ["snapshot", *QUERY_KNOB_COLS, *RUNTIME_KNOB_COLS, "top_k", "num_queries"]

SNAPSHOT_PATTERN = re.compile(
    r"(?:m(?P<m>\d+))|(?:m0_(?P<m0>\d+))|(?:efc(?P<efc>\d+))|(?:alow(?P<alow>[\d\.p]+))"
    r"|(?:ahigh(?P<ahigh>[\d\.p]+))|(?:prunefloor(?P<pf>\d+))"
)


def parse_snapshot(snapshot_name: str) -> dict[str, float]:
    params: dict[str, float] = {}
    tokens = Path(snapshot_name).name.replace(".bin", "").split("_")
    for token in tokens:
        match = SNAPSHOT_PATTERN.fullmatch(token)
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


def compute_index_stats_csv(snapshot_paths: list[Path], pair_samples: int) -> pd.DataFrame:
    """Compute per-snapshot structural + geometry stats by calling the Rust `index_stats` binary."""

    if not snapshot_paths:
        return pd.DataFrame()

    repo_root = Path(__file__).resolve().parents[1]
    with tempfile.NamedTemporaryFile(mode="w", delete=True) as tmp:
        for p in snapshot_paths:
            tmp.write(str(p) + "\n")
        tmp.flush()

        cmd = [
            "cargo",
            "run",
            "-q",
            "--release",
            "--bin",
            "index_stats",
            "--",
            "--snapshots-file",
            tmp.name,
            "--pair-samples",
            str(int(pair_samples)),
        ]
        res = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
        if res.returncode != 0:
            raise SystemExit(
                "index_stats failed:\n"
                + (res.stdout or "")
                + ("\n" if res.stdout else "")
                + (res.stderr or "")
            )
        return pd.read_csv(io.StringIO(res.stdout))


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


@dataclass(frozen=True)
class RunKey:
    snapshot: str
    ef_search: int
    neighbor_cap: int
    patience: int
    search_expansion_mult: int | None
    search_expansion_cap: int | None
    disable_early_exit: int | None
    neighbor_rotate: int | None
    neighbor_stride: int | None
    neighbor_scan_patience: int | None
    top_k: int
    num_queries: int


def maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def runtime_knob_values(obj: dict[str, Any]) -> dict[str, int | None]:
    return {col: maybe_int(obj.get(col)) for col in RUNTIME_KNOB_COLS}


def present_runtime_knob_cols(df: pd.DataFrame) -> list[str]:
    return [col for col in RUNTIME_KNOB_COLS if col in df.columns and not df[col].isna().all()]


def present_run_id_cols(df: pd.DataFrame) -> list[str]:
    return [col for col in RUN_ID_COLS if col in df.columns]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    idx = int(round((p / 100.0) * (len(values) - 1)))
    idx = max(0, min(idx, len(values) - 1))
    return float(values[idx])


def aggregate_runs_from_queries(path: Path) -> pd.DataFrame:
    buckets: dict[RunKey, dict[str, list[float]]] = {}
    for obj in read_jsonl(path):
        if obj.get("type") != "query":
            continue
        key = RunKey(
            snapshot=str(obj["snapshot"]),
            ef_search=int(obj["ef_search"]),
            neighbor_cap=int(obj["neighbor_cap"]),
            patience=int(obj["patience"]),
            search_expansion_mult=maybe_int(obj.get("search_expansion_mult")),
            search_expansion_cap=maybe_int(obj.get("search_expansion_cap")),
            disable_early_exit=maybe_int(obj.get("disable_early_exit")),
            neighbor_rotate=maybe_int(obj.get("neighbor_rotate")),
            neighbor_stride=maybe_int(obj.get("neighbor_stride")),
            neighbor_scan_patience=maybe_int(obj.get("neighbor_scan_patience")),
            top_k=int(obj["top_k"]),
            num_queries=int(obj["num_queries"]),
        )
        bucket = buckets.setdefault(
            key,
            {
                "elapsed_ms": [],
                "recall": [],
                "visited": [],
                "expanded": [],
                "misses": [],
            },
        )
        bucket["elapsed_ms"].append(float(obj["elapsed_ms"]))
        bucket["recall"].append(float(obj["recall"]))
        if obj.get("visited") is not None:
            bucket["visited"].append(float(obj["visited"]))
        if obj.get("expanded") is not None:
            bucket["expanded"].append(float(obj["expanded"]))
        bucket["misses"].append(float(obj.get("misses", 0)))

    rows: list[dict[str, Any]] = []
    for key, bucket in buckets.items():
        elapsed = bucket["elapsed_ms"]
        recall = bucket["recall"]
        visited = bucket["visited"]
        expanded = bucket["expanded"]
        misses = bucket["misses"]
        rows.append(
            {
                "snapshot": key.snapshot,
                "ef_search": key.ef_search,
                "neighbor_cap": key.neighbor_cap,
                "patience": key.patience,
                "search_expansion_mult": key.search_expansion_mult,
                "search_expansion_cap": key.search_expansion_cap,
                "disable_early_exit": key.disable_early_exit,
                "neighbor_rotate": key.neighbor_rotate,
                "neighbor_stride": key.neighbor_stride,
                "neighbor_scan_patience": key.neighbor_scan_patience,
                "top_k": key.top_k,
                "num_queries": key.num_queries,
                "query_count": len(elapsed),
                "recall_avg": float(np.mean(recall)) if recall else float("nan"),
                "recall_p50": percentile(recall, 50.0),
                "recall_p90": percentile(recall, 90.0),
                "recall_p95": percentile(recall, 95.0),
                "recall_p99": percentile(recall, 99.0),
                "ms_avg": float(np.mean(elapsed)) if elapsed else float("nan"),
                "ms_p50": percentile(elapsed, 50.0),
                "ms_p90": percentile(elapsed, 90.0),
                "ms_p95": percentile(elapsed, 95.0),
                "ms_p99": percentile(elapsed, 99.0),
                "visited_avg": float(np.mean(visited)) if visited else float("nan"),
                "visited_p50": percentile(visited, 50.0),
                "visited_p90": percentile(visited, 90.0),
                "visited_p95": percentile(visited, 95.0),
                "visited_p99": percentile(visited, 99.0),
                "expanded_avg": float(np.mean(expanded)) if expanded else float("nan"),
                "expanded_p50": percentile(expanded, 50.0),
                "expanded_p90": percentile(expanded, 90.0),
                "expanded_p95": percentile(expanded, 95.0),
                "expanded_p99": percentile(expanded, 99.0),
                "misses_avg": float(np.mean(misses)) if misses else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def load_query_rows(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for obj in read_jsonl(path):
        if obj.get("type") != "query":
            continue
        rows.append(obj)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).reset_index(drop=True)
    # normalize types
    for col in [*QUERY_KNOB_COLS, *RUNTIME_KNOB_COLS, "top_k", "num_queries", "query_idx"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ["elapsed_ms", "recall", "visited", "expanded", "misses"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    parsed = pd.DataFrame([parse_snapshot(s) for s in df["snapshot"].astype(str)])
    df = pd.concat([df, parsed], axis=1)
    return df


def query_hardness_table(query_df: pd.DataFrame) -> pd.DataFrame:
    if query_df.empty or "query_idx" not in query_df.columns:
        return pd.DataFrame()
    df = query_df.copy()
    df["recall"] = pd.to_numeric(df["recall"], errors="coerce")
    df["elapsed_ms"] = pd.to_numeric(df["elapsed_ms"], errors="coerce")
    grouped = df.groupby("query_idx", sort=True)

    out = pd.DataFrame(
        {
            "query_idx": grouped.size().index.astype(int),
            "n": grouped.size().to_numpy(dtype=int),
            "recall_avg": grouped["recall"].mean().to_numpy(dtype=float),
            "recall_p10": grouped["recall"].quantile(0.10).to_numpy(dtype=float),
            "recall_p50": grouped["recall"].quantile(0.50).to_numpy(dtype=float),
            "recall_p90": grouped["recall"].quantile(0.90).to_numpy(dtype=float),
            "ms_avg": grouped["elapsed_ms"].mean().to_numpy(dtype=float),
            "ms_p50": grouped["elapsed_ms"].quantile(0.50).to_numpy(dtype=float),
            "ms_p90": grouped["elapsed_ms"].quantile(0.90).to_numpy(dtype=float),
            "ms_p99": grouped["elapsed_ms"].quantile(0.99).to_numpy(dtype=float),
        }
    )

    if "visited" in df.columns:
        out["visited_avg"] = grouped["visited"].mean().to_numpy(dtype=float)
        out["visited_p90"] = grouped["visited"].quantile(0.90).to_numpy(dtype=float)
    if "expanded" in df.columns:
        out["expanded_avg"] = grouped["expanded"].mean().to_numpy(dtype=float)
        out["expanded_p90"] = grouped["expanded"].quantile(0.90).to_numpy(dtype=float)

    return out.sort_values(["recall_p50", "ms_p90"], ascending=[True, False])


def query_knob_effects(
    query_df: pd.DataFrame,
    weights_scheme: str,
    ridge: float,
) -> pd.DataFrame:
    if query_df.empty or "query_idx" not in query_df.columns:
        return pd.DataFrame()

    # Focus on query-time knobs only; build knobs create huge interaction space and we already model them in policy_table.
    feature_cols = QUERY_KNOB_COLS + present_runtime_knob_cols(query_df)
    for col in feature_cols:
        if col not in query_df.columns:
            return pd.DataFrame()
    query_df = query_df.dropna(subset=feature_cols)
    weights = choose_weights(query_df, weights_scheme)

    rows: list[dict[str, Any]] = []
    for qidx, group in query_df.groupby("query_idx", sort=True):
        group = group.copy()
        for col in feature_cols:
            group[col] = pd.to_numeric(group[col], errors="coerce")
        # standardize features (per-query) so coefficients are comparable.
        x_raw = group[feature_cols].to_numpy(dtype=float)
        ok_x = np.all(np.isfinite(x_raw), axis=1)
        if ok_x.sum() < 25:
            continue
        w = weights[group.index.to_numpy()]
        x_std, _, _ = weighted_standardize(x_raw[ok_x], w[ok_x])
        group_std = group.iloc[np.where(ok_x)[0]].copy()
        for i, col in enumerate(feature_cols):
            group_std[col] = x_std[:, i]
        w2 = w[ok_x]

        for target in ["recall", "elapsed_ms", "visited", "expanded"]:
            if target not in group_std.columns:
                continue
            y = pd.to_numeric(group_std[target], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(y) & np.all(np.isfinite(x_std), axis=1)
            if ok.sum() < 25:
                continue
            sub = group_std.iloc[np.where(ok)[0]]
            x_lin, names = build_design(sub, feature_cols)
            beta = weighted_ridge_fit(x_lin, y[ok], w2[ok], ridge=ridge)
            yhat = x_lin @ beta
            rows.append(
                {
                    "query_idx": int(qidx),
                    "target": target,
                    "weighted_r2": weighted_r2(y[ok], yhat, w2[ok]),
                    "coef_ef_search": float(beta[names.index("ef_search")]),
                    "coef_neighbor_cap": float(beta[names.index("neighbor_cap")]),
                    "coef_patience": float(beta[names.index("patience")]),
                    "n": int(ok.sum()),
                }
            )
    return pd.DataFrame(rows)


def build_policy_table(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for obj in read_jsonl(path):
        if obj.get("type") != "summary":
            continue
        rows.append(obj)
    if not rows:
        raise SystemExit(f"no type=summary rows found in {path}")
    df = pd.DataFrame(rows)
    parsed = pd.DataFrame([parse_snapshot(s) for s in df["snapshot"].astype(str)])
    df = pd.concat([df, parsed], axis=1)
    # normalize types
    for col in [*QUERY_KNOB_COLS, *RUNTIME_KNOB_COLS, "top_k", "num_queries"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ["m", "m0", "efc", "alow", "ahigh", "prunefloor"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def compute_frontiers(
    policy_df: pd.DataFrame,
    thresholds: list[float],
    latency_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    global_rows = []
    for snapshot, group in policy_df.groupby("snapshot", sort=False):
        for r in thresholds:
            ok = group[group["recall_avg"] >= r]
            if ok.empty:
                continue
            best_row = ok.loc[ok[latency_column].idxmin()]
            rows.append(
                {
                    "snapshot": snapshot,
                    "recall_threshold": r,
                    "latency_metric": latency_column,
                    "best_latency": float(best_row[latency_column]),
                    "ef_search": int(best_row["ef_search"]),
                    "neighbor_cap": int(best_row["neighbor_cap"]),
                    "patience": int(best_row["patience"]),
                    **{
                        col: int(best_row[col])
                        for col in present_runtime_knob_cols(group)
                        if pd.notna(best_row[col])
                    },
                }
            )
    frontier_df = pd.DataFrame(rows)
    if frontier_df.empty:
        return frontier_df, pd.DataFrame()
    for r in thresholds:
        ok = frontier_df[frontier_df["recall_threshold"] == r]
        if ok.empty:
            continue
        best = ok.loc[ok["best_latency"].idxmin()]
        global_rows.append(best.to_dict())
    return frontier_df, pd.DataFrame(global_rows)


def safe_div(numer: float, denom: float) -> float:
    if denom is None or not np.isfinite(denom) or denom == 0:
        return float("nan")
    if numer is None or not np.isfinite(numer):
        return float("nan")
    return float(numer) / float(denom)


def weighted_r2(y: np.ndarray, yhat: np.ndarray, w: np.ndarray) -> float:
    w = w / w.sum()
    ybar = float((w * y).sum())
    ss_tot = float((w * (y - ybar) ** 2).sum())
    ss_res = float((w * (y - yhat) ** 2).sum())
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - (ss_res / ss_tot)


def add_derived_cost_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add ratio-style bottleneck diagnostics (ms per visit/expand) when inputs exist."""

    df = df.copy()

    def add_ratio(ms_col: str, work_col: str, out_col: str) -> None:
        ms = pd.to_numeric(df[ms_col], errors="coerce").to_numpy(dtype=float)
        work = pd.to_numeric(df[work_col], errors="coerce").to_numpy(dtype=float)
        df[out_col] = [safe_div(a, b) for a, b in zip(ms, work)]

    for ms_col in ["ms_avg", "ms_p50", "ms_p90", "ms_p95", "ms_p99"]:
        if ms_col not in df.columns:
            continue
        for work_col, prefix in [
            ("visited_avg", "visit_avg"),
            ("visited_p50", "visit_p50"),
            ("visited_p90", "visit_p90"),
            ("visited_p95", "visit_p95"),
            ("visited_p99", "visit_p99"),
            ("expanded_avg", "expand_avg"),
            ("expanded_p50", "expand_p50"),
            ("expanded_p90", "expand_p90"),
            ("expanded_p95", "expand_p95"),
            ("expanded_p99", "expand_p99"),
        ]:
            if work_col not in df.columns:
                continue
            add_ratio(ms_col, work_col, f"{ms_col}_per_{prefix}")
    return df


def weighted_standardize(x: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = w / w.sum()
    mean = (w[:, None] * x).sum(axis=0)
    var = (w[:, None] * (x - mean) ** 2).sum(axis=0)
    std = np.sqrt(np.maximum(var, 1e-12))
    return (x - mean) / std, mean, std


def weighted_ridge_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray, ridge: float) -> np.ndarray:
    w = w / w.sum()
    xw = x * np.sqrt(w[:, None])
    yw = y * np.sqrt(w)
    xtx = xw.T @ xw
    xty = xw.T @ yw
    reg = ridge * np.eye(xtx.shape[0], dtype=xtx.dtype)
    reg[0, 0] = 0.0  # don't penalize intercept
    beta = np.linalg.solve(xtx + reg, xty)
    return beta


def build_design(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    x = df[feature_cols].to_numpy(dtype=float)
    design = np.concatenate([np.ones((x.shape[0], 1), dtype=float), x], axis=1)
    names = ["intercept"] + feature_cols
    return design, names


def build_quadratic_features(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    base = df[feature_cols].to_numpy(dtype=float)
    feats = [base]
    names = list(feature_cols)
    # squared terms
    feats.append(base ** 2)
    names += [f"{c}^2" for c in feature_cols]
    # pairwise interactions (upper triangle)
    for i, ci in enumerate(feature_cols):
        for j in range(i + 1, len(feature_cols)):
            cj = feature_cols[j]
            feats.append((base[:, i] * base[:, j]).reshape(-1, 1))
            names.append(f"{ci}*{cj}")
    x = np.concatenate(feats, axis=1)
    design = np.concatenate([np.ones((x.shape[0], 1), dtype=float), x], axis=1)
    return design, ["intercept"] + names


def choose_weights(df: pd.DataFrame, scheme: str) -> np.ndarray:
    if scheme == "uniform":
        return np.ones(len(df), dtype=float)
    if scheme == "per_snapshot":
        counts = df.groupby("snapshot")["snapshot"].transform("count").to_numpy(dtype=float)
        return 1.0 / np.maximum(counts, 1.0)
    if scheme == "per_m":
        if "m" not in df.columns or df["m"].isna().all():
            return np.ones(len(df), dtype=float)
        counts = df.groupby("m")["m"].transform("count").to_numpy(dtype=float)
        return 1.0 / np.maximum(counts, 1.0)
    raise SystemExit(f"unknown weight scheme {scheme}")


def weighted_mse(y: np.ndarray, yhat: np.ndarray, w: np.ndarray) -> float:
    w = w / w.sum()
    return float((w * (y - yhat) ** 2).sum())


def permutation_importance(
    design: np.ndarray,
    names: list[str],
    y: np.ndarray,
    w: np.ndarray,
    beta: np.ndarray,
    rng_seed: int = 0,
) -> list[tuple[str, float]]:
    rng = np.random.default_rng(rng_seed)
    base_pred = design @ beta
    base = weighted_mse(y, base_pred, w)
    importances: list[tuple[str, float]] = []
    # skip intercept (col 0)
    for j in range(1, design.shape[1]):
        perm = design.copy()
        perm[:, j] = rng.permutation(perm[:, j])
        pred = perm @ beta
        score = weighted_mse(y, pred, w)
        importances.append((names[j], score - base))
    importances.sort(key=lambda p: p[1], reverse=True)
    return importances


def fit_weighted_linear(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    weights: np.ndarray,
    ridge: float = 1e-6,
) -> dict[str, Any]:
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    x = df[x_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    if ok.sum() < max(25, len(x_cols) + 3):
        return {"ok": False, "reason": f"not enough finite rows for {y_col}"}
    w = weights[ok]
    sub = df.iloc[np.where(ok)[0]]
    design, names = build_design(sub, x_cols)
    beta = weighted_ridge_fit(design, y[ok], w, ridge=ridge)
    yhat = design @ beta
    return {
        "ok": True,
        "y_col": y_col,
        "x_cols": list(x_cols),
        "names": names,
        "beta": beta,
        "r2": weighted_r2(y[ok], yhat, w),
        "mse": weighted_mse(y[ok], yhat, w),
        "n": int(ok.sum()),
    }


def write_bottleneck_report(
    out_path: Path,
    df: pd.DataFrame,
    latency_metric: str,
    weights: np.ndarray,
) -> None:
    lines: list[str] = []
    lines.append("# Bottleneck report\n\n")
    lines.append("Goal: identify where latency comes from using already-captured counters.\n\n")
    lines.append(f"- rows: {len(df)}\n")
    lines.append(f"- latency metric: `{latency_metric}`\n")
    if "snapshot" in df.columns:
        lines.append(f"- snapshots: {df['snapshot'].nunique()}\n")
    lines.append("\n")

    if "m" in df.columns and not df["m"].isna().all():
        counts = df.groupby("m")["snapshot"].count().sort_index()
        lines.append("## Coverage (by `m`)\n")
        for m, c in counts.items():
            lines.append(f"- m={int(m)}: {int(c)} rows\n")
        lines.append("\n")

    df = add_derived_cost_columns(df)
    cost_cols = [c for c in df.columns if c.startswith(latency_metric) and "_per_" in c]
    if cost_cols:
        lines.append("## Derived costs (ratios)\n")
        for c in sorted(cost_cols):
            series = pd.to_numeric(df[c], errors="coerce")
            finite = series[np.isfinite(series)]
            if finite.empty:
                continue
            lines.append(
                f"- `{c}`: median={float(finite.median()):.6f}, p90={float(finite.quantile(0.9)):.6f}\n"
            )
        lines.append("\n")

    work_cols = [c for c in ["visited_avg", "expanded_avg", "misses_avg"] if c in df.columns]
    if latency_metric in df.columns and work_cols:
        fit = fit_weighted_linear(df, latency_metric, work_cols, weights, ridge=1e-6)
        if fit["ok"]:
            lines.append("## Latency explained by work counters\n")
            lines.append(f"- model: `{latency_metric}` ~ {', '.join(work_cols)}\n")
            lines.append(f"- weighted R²: {fit['r2']:.4f} (n={fit['n']})\n")
            for name, val in zip(fit["names"], fit["beta"]):
                if name == "intercept":
                    continue
                lines.append(f"- slope `{name}`: {float(val):+.6f} ms per unit\n")
            lines.append("\n")

    if latency_metric in df.columns and "visited_avg" in df.columns and "expanded_avg" in df.columns:
        log_df = df.copy()
        for c in [latency_metric, "visited_avg", "expanded_avg"]:
            series = pd.to_numeric(log_df[c], errors="coerce").to_numpy(dtype=float)
            log_df[c] = np.log1p(np.maximum(series, 0.0))
        fit = fit_weighted_linear(
            log_df,
            latency_metric,
            ["visited_avg", "expanded_avg"],
            weights,
            ridge=1e-6,
        )
        if fit["ok"]:
            lines.append("## Elasticities (log-log)\n")
            lines.append("- model: log(1+ms) ~ log(1+visited) + log(1+expanded)\n")
            lines.append(f"- weighted R²: {fit['r2']:.4f} (n={fit['n']})\n")
            for name, val in zip(fit["names"], fit["beta"]):
                if name == "intercept":
                    continue
                lines.append(f"- `{name}` elasticity: {float(val):+.4f}\n")
            lines.append("\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines))


def paired_knob_effects(
    df: pd.DataFrame,
    latency_metric: str,
    recall_metric: str = "recall_avg",
) -> pd.DataFrame:
    need = {"snapshot", *QUERY_KNOB_COLS, latency_metric, recall_metric}
    missing = [c for c in need if c not in df.columns]
    if missing:
        return pd.DataFrame()

    df = df.copy()
    df[latency_metric] = pd.to_numeric(df[latency_metric], errors="coerce")
    df[recall_metric] = pd.to_numeric(df[recall_metric], errors="coerce")

    rows: list[dict[str, Any]] = []

    runtime_cols = present_runtime_knob_cols(df)

    def emit_pairs(knob: str, fixed_cols: list[str]) -> None:
        group_cols = ["snapshot"] + fixed_cols + runtime_cols
        for (snapshot, *fixed), group in df.groupby(group_cols, sort=False):
            group = group.dropna(subset=[knob, latency_metric, recall_metric])
            if group.empty:
                continue
            group = group.sort_values(knob)
            vals = group[knob].astype(int).tolist()
            for a, b in zip(vals, vals[1:]):
                ra = group[group[knob] == a].iloc[0]
                rb = group[group[knob] == b].iloc[0]
                rows.append(
                    {
                        "snapshot": snapshot,
                        "knob": knob,
                        "fixed": ",".join(f"{c}={int(ra[c])}" for c in fixed_cols + runtime_cols),
                        "from": int(a),
                        "to": int(b),
                        "latency_metric": latency_metric,
                        "recall_metric": recall_metric,
                        "latency_from": float(ra[latency_metric]),
                        "latency_to": float(rb[latency_metric]),
                        "latency_delta": float(rb[latency_metric]) - float(ra[latency_metric]),
                        "recall_from": float(ra[recall_metric]),
                        "recall_to": float(rb[recall_metric]),
                        "recall_delta": float(rb[recall_metric]) - float(ra[recall_metric]),
                        "visited_from": float(ra["visited_avg"]) if "visited_avg" in ra else float("nan"),
                        "visited_to": float(rb["visited_avg"]) if "visited_avg" in rb else float("nan"),
                        "expanded_from": float(ra["expanded_avg"]) if "expanded_avg" in ra else float("nan"),
                        "expanded_to": float(rb["expanded_avg"]) if "expanded_avg" in rb else float("nan"),
                        **{
                            col: int(ra[col])
                            for col in runtime_cols
                            if pd.notna(ra[col])
                        },
                    }
                )

    emit_pairs("ef_search", fixed_cols=["neighbor_cap", "patience"])
    emit_pairs("neighbor_cap", fixed_cols=["ef_search", "patience"])
    emit_pairs("patience", fixed_cols=["ef_search", "neighbor_cap"])

    return pd.DataFrame(rows)


def frontier_winners(frontier_df: pd.DataFrame) -> pd.DataFrame:
    if frontier_df.empty:
        return pd.DataFrame()
    cols = ["recall_threshold", *QUERY_KNOB_COLS, *present_runtime_knob_cols(frontier_df)]
    for c in cols:
        if c not in frontier_df.columns:
            return pd.DataFrame()
    grouped = (
        frontier_df.groupby(cols, dropna=False)
        .size()
        .reset_index(name="win_count")
        .sort_values(["recall_threshold", "win_count"], ascending=[True, False])
    )
    return grouped


def marginal_returns_table(policy_df: pd.DataFrame, latency_metric: str) -> pd.DataFrame:
    need = {"snapshot", *QUERY_KNOB_COLS, "recall_avg", latency_metric}
    missing = [c for c in need if c not in policy_df.columns]
    if missing:
        return pd.DataFrame()
    df = policy_df.copy()
    df["ef_search"] = pd.to_numeric(df["ef_search"], errors="coerce")
    df["neighbor_cap"] = pd.to_numeric(df["neighbor_cap"], errors="coerce")
    df["patience"] = pd.to_numeric(df["patience"], errors="coerce")
    for col in present_runtime_knob_cols(df):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["recall_avg"] = pd.to_numeric(df["recall_avg"], errors="coerce")
    df[latency_metric] = pd.to_numeric(df[latency_metric], errors="coerce")
    df = df.dropna(subset=["ef_search", "neighbor_cap", "patience", "recall_avg", latency_metric])

    rows: list[dict[str, Any]] = []
    runtime_cols = present_runtime_knob_cols(df)
    group_cols = ["snapshot", "neighbor_cap", "patience", *runtime_cols]
    for keys, group in df.groupby(group_cols, sort=False):
        snapshot = keys[0]
        cap = keys[1]
        pat = keys[2]
        group = group.sort_values("ef_search")
        efs = group["ef_search"].astype(int).tolist()
        for a, b in zip(efs, efs[1:]):
            ra = group[group["ef_search"] == a].iloc[0]
            rb = group[group["ef_search"] == b].iloc[0]
            d_recall = float(rb["recall_avg"]) - float(ra["recall_avg"])
            d_lat = float(rb[latency_metric]) - float(ra[latency_metric])
            d_vis = float(rb.get("visited_avg", float("nan"))) - float(ra.get("visited_avg", float("nan")))
            d_exp = float(rb.get("expanded_avg", float("nan"))) - float(ra.get("expanded_avg", float("nan")))
            rows.append(
                {
                    "snapshot": snapshot,
                    "neighbor_cap": int(cap),
                    "patience": int(pat),
                    **{
                        col: int(ra[col])
                        for col in runtime_cols
                        if pd.notna(ra[col])
                    },
                    "ef_from": int(a),
                    "ef_to": int(b),
                    "recall_delta": d_recall,
                    "latency_metric": latency_metric,
                    "latency_delta": d_lat,
                    "recall_per_ms": safe_div(d_recall, d_lat),
                    "visited_delta": d_vis,
                    "expanded_delta": d_exp,
                    "recall_per_visit": safe_div(d_recall, d_vis),
                    "recall_per_expand": safe_div(d_recall, d_exp),
                }
            )
    return pd.DataFrame(rows)


def write_effects_report(
    out_path: Path,
    df: pd.DataFrame,
    feature_cols: list[str],
    targets: list[str],
    weights: np.ndarray,
    ridge: float,
) -> None:
    lines: list[str] = []
    lines.append("# Knob effect report\n")
    lines.append(f"- rows: {len(df)}\n")
    lines.append(f"- ridge: {ridge}\n")
    lines.append(f"- features: {', '.join(feature_cols)}\n")
    lines.append("")

    # standardize features for interpretable coefficients
    x_raw = df[feature_cols].to_numpy(dtype=float)
    x_std, _, _ = weighted_standardize(x_raw, weights)
    df_std = df.copy()
    for idx, col in enumerate(feature_cols):
        df_std[col] = x_std[:, idx]

    for target in targets:
        if target not in df.columns:
            continue
        y = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(y) & np.all(np.isfinite(x_std), axis=1)
        if ok.sum() < 25:
            continue
        w = weights[ok]
        sub = df_std.iloc[np.where(ok)[0]]

        x_lin, lin_names = build_design(sub, feature_cols)
        beta_lin = weighted_ridge_fit(x_lin, y[ok], w, ridge=ridge)
        yhat_lin = x_lin @ beta_lin

        x_quad, quad_names = build_quadratic_features(sub, feature_cols)
        beta_quad = weighted_ridge_fit(x_quad, y[ok], w, ridge=ridge)
        yhat_quad = x_quad @ beta_quad

        def top_coeffs(beta: np.ndarray, names: list[str], k: int = 12) -> list[tuple[str, float]]:
            pairs = list(zip(names, beta))
            pairs = [p for p in pairs if p[0] != "intercept"]
            pairs.sort(key=lambda p: abs(p[1]), reverse=True)
            return pairs[:k]

        lines.append(f"## Target: `{target}`\n")
        lines.append("### Fit quality\n")
        lines.append(f"- linear weighted R²: {weighted_r2(y[ok], yhat_lin, w):.4f}\n")
        lines.append(f"- quadratic weighted R²: {weighted_r2(y[ok], yhat_quad, w):.4f}\n")

        if (target.startswith("ms_") or target in {"visited_avg", "expanded_avg", "misses_avg"}) and np.all(
            y[ok] >= 0
        ):
            y_log = np.log1p(y[ok])
            beta_log = weighted_ridge_fit(x_lin, y_log, w, ridge=ridge)
            yhat_log = x_lin @ beta_log
            lines.append(f"- log1p(target) linear weighted R²: {weighted_r2(y_log, yhat_log, w):.4f}\n")
        lines.append("\n")

        lines.append("### Linear (standardized)\n")
        for name, val in top_coeffs(beta_lin, lin_names, k=10):
            lines.append(f"- {name}: {val:+.4f}\n")
        lines.append("\n### Linear permutation importance (Δ weighted MSE)\n")
        for name, delta in permutation_importance(x_lin, lin_names, y[ok], w, beta_lin)[:10]:
            lines.append(f"- {name}: +{delta:.6f}\n")

        lines.append("\n### Quadratic (standardized)\n")
        for name, val in top_coeffs(beta_quad, quad_names, k=12):
            lines.append(f"- {name}: {val:+.4f}\n")
        lines.append("\n### Quadratic permutation importance (Δ weighted MSE)\n")
        for name, delta in permutation_importance(x_quad, quad_names, y[ok], w, beta_quad)[:12]:
            lines.append(f"- {name}: +{delta:.6f}\n")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-jsonl", type=Path, default=Path("logs/nyt_query_suite.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("logs/nyt_query_suite_analysis"))
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=None,
        help="If set, compute per-snapshot structural stats by loading snapshots from this directory.",
    )
    parser.add_argument(
        "--index-stats-pair-samples",
        type=int,
        default=5_000,
        help="Random vector-pair samples per snapshot for geometry stats.",
    )
    parser.add_argument(
        "--thresholds",
        default="0.8,0.85,0.9,0.95,0.97,0.99",
        help="Comma-separated recall thresholds for frontier.",
    )
    parser.add_argument(
        "--latency-metric",
        default="ms_p90",
        choices=["ms_avg", "ms_p50", "ms_p90", "ms_p95", "ms_p99"],
        help="Latency metric minimized when constructing the frontier.",
    )
    parser.add_argument(
        "--recompute-from-queries",
        action="store_true",
        help="Recompute per-run summaries from type=query rows when present (adds p95 metrics).",
    )
    parser.add_argument(
        "--weights",
        default="per_m",
        choices=["uniform", "per_snapshot", "per_m"],
        help="Reweight samples to account for build sampling imbalance.",
    )
    parser.add_argument("--ridge", type=float, default=1.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    policy = build_policy_table(args.in_jsonl)
    query_df = pd.DataFrame()
    if args.recompute_from_queries:
        query_df = load_query_rows(args.in_jsonl)
        runs = aggregate_runs_from_queries(args.in_jsonl)
        if not runs.empty:
            runs_out = args.out_dir / "run_summaries.csv"
            runs.to_csv(runs_out, index=False)
            # Prefer recomputed run summaries for analysis if they cover this run.
            key_cols = [col for col in RUN_ID_COLS if col in runs.columns and col in policy.columns]
            policy = policy.drop(
                columns=[c for c in runs.columns if c in policy.columns and c not in key_cols],
                errors="ignore",
            )
            policy = policy.merge(runs, on=key_cols, how="left")
    policy_out = args.out_dir / "policy_table.csv"
    policy.to_csv(policy_out, index=False)

    enriched_out = None
    if args.snapshot_dir is not None:
        snapshots = sorted(set(policy["snapshot"].astype(str)))
        existing = []
        missing = 0
        for name in snapshots:
            path = args.snapshot_dir / name
            if path.exists():
                existing.append(path)
            else:
                missing += 1
        if missing:
            print(f"[index_stats] missing {missing} snapshot files under {args.snapshot_dir}")
        if existing:
            stats_df = compute_index_stats_csv(existing, args.index_stats_pair_samples)
            stats_out = args.out_dir / "index_stats.csv"
            stats_df.to_csv(stats_out, index=False)
            enriched = policy.merge(stats_df.drop(columns=["snapshot_path"], errors="ignore"), on="snapshot", how="left")
            enriched_out = args.out_dir / "policy_table_enriched.csv"
            enriched.to_csv(enriched_out, index=False)

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    frontier, global_frontier = compute_frontiers(policy, thresholds, args.latency_metric)
    frontier.to_csv(args.out_dir / "frontier_table.csv", index=False)
    global_frontier.to_csv(args.out_dir / "global_frontier.csv", index=False)

    # Effect modeling: build knobs + query knobs -> dependent variables
    feature_cols = [
        # query knobs
        *QUERY_KNOB_COLS,
        *present_runtime_knob_cols(policy),
        # build knobs (parsed from snapshot name)
        "m",
        "m0",
        "efc",
        "alow",
        "ahigh",
        "prunefloor",
    ]
    for col in feature_cols:
        if col not in policy.columns:
            policy[col] = np.nan
    # drop rows missing key query knobs
    policy = policy.dropna(subset=QUERY_KNOB_COLS)
    weights = choose_weights(policy, args.weights)
    targets = [
        "recall_avg",
        "ms_avg",
        "ms_p90",
        "ms_p99",
        "visited_avg",
        "expanded_avg",
        "misses_avg",
    ]
    write_effects_report(
        args.out_dir / "effects_report.md",
        policy,
        feature_cols,
        targets,
        weights,
        ridge=args.ridge,
    )

    write_bottleneck_report(
        args.out_dir / "bottleneck_report.md",
        policy,
        latency_metric=args.latency_metric,
        weights=weights,
    )
    paired = paired_knob_effects(policy, latency_metric=args.latency_metric)
    if not paired.empty:
        paired.to_csv(args.out_dir / "paired_knob_effects.csv", index=False)
    winners = frontier_winners(frontier)
    if not winners.empty:
        winners.to_csv(args.out_dir / "frontier_winners.csv", index=False)

    marginal = marginal_returns_table(policy, latency_metric=args.latency_metric)
    if not marginal.empty:
        marginal.to_csv(args.out_dir / "marginal_returns.csv", index=False)
    if not query_df.empty:
        hardness = query_hardness_table(query_df)
        if not hardness.empty:
            hardness.to_csv(args.out_dir / "query_hardness.csv", index=False)
        qeffects = query_knob_effects(query_df, weights_scheme=args.weights, ridge=args.ridge)
        if not qeffects.empty:
            qeffects.to_csv(args.out_dir / "query_knob_effects.csv", index=False)

    print(f"wrote {policy_out}")
    if enriched_out is not None:
        print(f"wrote {args.out_dir / 'index_stats.csv'}")
        print(f"wrote {enriched_out}")
    print(f"wrote {args.out_dir / 'frontier_table.csv'}")
    print(f"wrote {args.out_dir / 'global_frontier.csv'}")
    print(f"wrote {args.out_dir / 'effects_report.md'}")
    print(f"wrote {args.out_dir / 'bottleneck_report.md'}")
    if not paired.empty:
        print(f"wrote {args.out_dir / 'paired_knob_effects.csv'}")
    if not winners.empty:
        print(f"wrote {args.out_dir / 'frontier_winners.csv'}")
    if not marginal.empty:
        print(f"wrote {args.out_dir / 'marginal_returns.csv'}")
    if not query_df.empty:
        if not hardness.empty:
            print(f"wrote {args.out_dir / 'query_hardness.csv'}")
        if not qeffects.empty:
            print(f"wrote {args.out_dir / 'query_knob_effects.csv'}")


if __name__ == "__main__":
    main()
