#!/usr/bin/env python3
"""
Prediction Market Accuracy Experiment

Fetches resolved Polymarket and Kalshi binary markets on crypto/stock/ETF
price movements, then tests whether market probabilities are well-calibrated
and whether mispricings produce tradeable edge.

Key questions:
  1. Calibration — when the market says 70%, does it actually happen 70% of the time?
  2. Bias — are markets systematically over- or underconfident?
  3. Edge — which strategies (follow / fade / extremes) produce positive expected value?
  4. Timing — how far in advance are markets accurate?

Usage:
  pip install -r requirements.txt
  python run_experiment.py                    # full run (needs live API access)
  python run_experiment.py --synthetic        # run on generated data (no network needed)
  python run_experiment.py --skip-fetch       # reuse cached API responses
  python run_experiment.py --no-clob          # skip CLOB history (faster, less accurate)
  python run_experiment.py --top-n 100        # limit CLOB enrichment to top 100 markets
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ── Directories ──────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
CACHE_DIR   = BASE_DIR / "cache"

# ── API bases ─────────────────────────────────────────────────────────────────
GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE  = "https://clob.polymarket.com"
KALSHI_BASE = "https://trading-api.kalshi.com/trade-api/v2"

# ── Market-classification keywords ───────────────────────────────────────────
ASSET_KW = [
    "bitcoin", "btc", "ethereum", "eth", "solana", "sol", "bnb", "binance",
    "xrp", "ripple", "cardano", "ada", "dogecoin", "doge", "polygon", "matic",
    "avalanche", "avax", "chainlink", "link", "litecoin", "ltc",
    "s&p", "sp500", "nasdaq", "dow", "spx", "ndx", "spy", "qqq",
    "gold", "silver", "oil", "crude",
]
DIRECTION_KW = [
    "above", "below", "reach", "hit", "over", "under", "exceed",
    "higher", "lower", "close above", "close below",
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_price_prediction(question: str) -> bool:
    q = question.lower()
    has_asset = any(kw in q for kw in ASSET_KW)
    has_direction = any(kw in q for kw in DIRECTION_KW)
    has_dollar = "$" in q and any(c.isdigit() for c in q)
    return has_asset and (has_direction or has_dollar)


def _parse_token_ids(raw) -> list:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return []
    return []


def _parse_resolution(m: dict) -> float | None:
    res = str(m.get("resolution", "")).strip().lower()
    if res in ("yes", "1", "1.0", "true"):
        return 1.0
    if res in ("no", "0", "0.0", "false"):
        return 0.0

    # Fallback via outcome prices (post-settlement they become 1 / 0)
    prices = _parse_token_ids(m.get("outcomePrices"))
    if prices:
        try:
            p = float(prices[0])
            if p >= 0.99:
                return 1.0
            if p <= 0.01:
                return 0.0
        except (TypeError, ValueError):
            pass
    return None


def _parse_dt(s: str) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Data fetching
# ─────────────────────────────────────────────────────────────────────────────

def fetch_polymarket_resolved(limit: int = 500) -> list[dict]:
    markets: list[dict] = []
    offset = 0
    page = 100

    print(f"Fetching up to {limit} resolved Polymarket markets (ordered by volume)…")
    while len(markets) < limit:
        batch_size = min(page, limit - len(markets))
        try:
            r = requests.get(
                f"{GAMMA_BASE}/markets",
                params={
                    "closed": "true",
                    "limit": batch_size,
                    "offset": offset,
                    "order": "volume",
                    "ascending": "false",
                },
                timeout=30,
            )
            r.raise_for_status()
            batch = r.json()
        except Exception as exc:
            print(f"  ✗ Polymarket Gamma error: {exc}")
            break

        if not batch:
            break
        markets.extend(batch)
        offset += len(batch)
        print(f"  {len(markets):>5} markets …", end="\r")
        if len(batch) < batch_size:
            break
        time.sleep(0.35)

    print(f"\n  Done: {len(markets)} markets")
    return markets


def fetch_kalshi_settled(limit: int = 300) -> list[dict]:
    markets: list[dict] = []
    cursor = None

    print(f"Fetching up to {limit} settled Kalshi markets…")
    for _ in range(6):
        params: dict = {"status": "settled", "limit": 100}
        if cursor:
            params["cursor"] = cursor
        try:
            r = requests.get(
                f"{KALSHI_BASE}/markets",
                params=params,
                headers={"accept": "application/json"},
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            print(f"  ✗ Kalshi error: {exc}")
            break

        batch = data.get("markets", [])
        markets.extend(batch)
        cursor = data.get("cursor")
        print(f"  {len(markets):>5} markets …", end="\r")
        if not batch or not cursor or len(markets) >= limit:
            break
        time.sleep(1.0)

    print(f"\n  Done: {len(markets)} markets")
    return markets


def build_polymarket_df(raw: list[dict]) -> pd.DataFrame:
    rows = []
    for m in raw:
        q = m.get("question", "")
        if not is_price_prediction(q):
            continue
        resolution = _parse_resolution(m)
        if resolution is None:
            continue
        token_ids = _parse_token_ids(m.get("clobTokenIds"))
        yes_token = token_ids[0] if token_ids else None
        try:
            vol = float(m.get("volume") or 0)
        except (TypeError, ValueError):
            vol = 0.0
        rows.append({
            "source":      "polymarket",
            "id":          m.get("id", ""),
            "yes_token":   yes_token,
            "question":    q,
            "end_date":    m.get("endDate", ""),
            "resolution":  resolution,
            "volume_usd":  vol,
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def build_kalshi_df(raw: list[dict]) -> pd.DataFrame:
    rows = []
    for m in raw:
        title    = m.get("title", "") or ""
        subtitle = m.get("subtitle", "") or ""
        q = f"{title} {subtitle}".strip()
        if not is_price_prediction(q):
            continue
        result = str(m.get("result", "")).lower()
        if result == "yes":
            resolution = 1.0
        elif result == "no":
            resolution = 0.0
        else:
            continue
        last_price_cents = m.get("last_price") or m.get("yes_bid") or 0
        try:
            prob = float(last_price_cents) / 100.0
        except (TypeError, ValueError):
            continue
        # Only keep genuinely uncertain pre-resolution prices
        if not (0.01 < prob < 0.99):
            continue
        try:
            vol = float(m.get("volume") or 0)
        except (TypeError, ValueError):
            vol = 0.0
        rows.append({
            "source":     "kalshi",
            "id":         m.get("ticker", ""),
            "yes_token":  None,
            "question":   q,
            "end_date":   m.get("close_time", ""),
            "resolution": resolution,
            "volume_usd": vol,
            # Kalshi's last_price may already be post-settlement, so treat with care
            "price_last": prob,
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — CLOB price history (Polymarket only)
# ─────────────────────────────────────────────────────────────────────────────

def _clob_history(token_id: str) -> list[dict]:
    try:
        r = requests.get(
            f"{CLOB_BASE}/prices-history",
            params={"market": token_id, "interval": "max", "fidelity": 200},
            timeout=30,
        )
        r.raise_for_status()
        return r.json().get("history", [])
    except Exception:
        return []


def _price_at_or_before(history: list[dict], target_ts: int) -> float | None:
    best_p, best_t = None, -1
    for pt in history:
        t = int(pt.get("t", 0))
        if t <= target_ts and t > best_t:
            best_t = t
            best_p = float(pt["p"])
    return best_p


def enrich_with_clob(df: pd.DataFrame, offsets_days=(1, 3, 7)) -> pd.DataFrame:
    rows = df.to_dict("records")
    n = len(rows)
    print(f"\nFetching CLOB price history for {n} markets…")

    for i, rec in enumerate(rows):
        print(f"  [{i+1}/{n}] {rec['question'][:70]}", end="\r")
        token = rec.get("yes_token")
        end_str = rec.get("end_date", "")
        if not token or not end_str:
            continue
        end_dt = _parse_dt(end_str)
        if end_dt is None:
            continue
        end_ts = int(end_dt.timestamp())

        history = _clob_history(token)
        for d in offsets_days:
            target_ts = end_ts - d * 86_400
            price = _price_at_or_before(history, target_ts)
            rec[f"price_t_minus_{d}d"] = price

        time.sleep(0.25)

    print()
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Analysis
# ─────────────────────────────────────────────────────────────────────────────

def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((probs - outcomes) ** 2))


def brier_skill_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    bs  = brier_score(probs, outcomes)
    ref = brier_score(np.full_like(probs, outcomes.mean()), outcomes)
    return float(1 - bs / ref) if ref > 0 else float("nan")


def calibration_table(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        p_mean = probs[mask].mean()
        o_mean = outcomes[mask].mean()
        rows.append({
            "bin":        f"{lo:.0%}–{hi:.0%}",
            "n":          int(mask.sum()),
            "pred_mean":  round(p_mean, 4),
            "actual_rate": round(o_mean, 4),
            "error":      round(p_mean - o_mean, 4),   # positive = overconfident
        })
    return pd.DataFrame(rows)


def simulate_strategy(
    probs: np.ndarray,
    outcomes: np.ndarray,
    strategy: str,
    conf_threshold: float = 0.70,
) -> dict:
    """
    Bet $1 per market.  Returns are per-dollar staked.

    Payoff:
      YES bet wins → (1/p - 1)   if outcome == 1, else -1
      NO  bet wins → (1/(1-p)-1) if outcome == 0, else -1
    """
    match strategy:
        case "follow_market":
            mask   = np.ones(len(probs), bool)
            bet_yes = probs > 0.5
        case "fade_market":
            mask   = np.ones(len(probs), bool)
            bet_yes = probs <= 0.5
        case "follow_extremes":
            mask   = (probs > conf_threshold) | (probs < (1 - conf_threshold))
            bet_yes = probs[mask] > 0.5
        case "fade_extremes":
            mask   = (probs > conf_threshold) | (probs < (1 - conf_threshold))
            bet_yes = probs[mask] <= 0.5
        case "always_yes":
            mask   = np.ones(len(probs), bool)
            bet_yes = np.ones(len(probs), bool)
        case _:
            raise ValueError(f"Unknown strategy: {strategy}")

    p = probs[mask]
    o = outcomes[mask]
    by = bet_yes if isinstance(bet_yes, np.ndarray) else np.full(len(p), bet_yes)

    if len(p) == 0:
        return {"n": 0, "mean_return": float("nan"), "sharpe": float("nan"),
                "win_rate": float("nan"), "total_return": float("nan")}

    yes_ret = np.where(o == 1, 1.0 / p - 1.0, -1.0)
    no_ret  = np.where(o == 0, 1.0 / (1.0 - p) - 1.0, -1.0)
    ret     = np.where(by, yes_ret, no_ret)
    wins    = np.where(by, o == 1, o == 0)

    return {
        "n":            int(len(ret)),
        "mean_return":  float(ret.mean()),
        "total_return": float(ret.sum()),
        "win_rate":     float(wins.mean()),
        "sharpe":       float(ret.mean() / (ret.std() + 1e-9)),
        "std":          float(ret.std()),
    }


def run_analysis(df: pd.DataFrame, price_col: str, label: str) -> dict | None:
    sub = df.dropna(subset=[price_col, "resolution"]).copy()
    sub = sub[(sub[price_col] > 0.01) & (sub[price_col] < 0.99)]

    if len(sub) < 15:
        print(f"\n  [{label}] only {len(sub)} valid markets — skipping")
        return None

    probs    = sub[price_col].to_numpy(dtype=float)
    outcomes = sub["resolution"].to_numpy(dtype=float)

    yes_rate = outcomes.mean()
    bs  = brier_score(probs, outcomes)
    bss = brier_skill_score(probs, outcomes)

    cal = calibration_table(probs, outcomes)

    # Mean Absolute Calibration Error
    mace = float(np.abs(cal["pred_mean"].values - cal["actual_rate"].values).mean())

    print(f"\n  {'─'*66}")
    print(f"  {label}")
    print(f"  n={len(sub)}  YES rate={yes_rate:.1%}  "
          f"Brier={bs:.4f}  Skill={bss:+.3f}  MACE={mace:.3f}")
    print(f"  {'─'*66}")
    print(f"  {'Bin':<12} {'N':>5}  {'Predicted':>10}  {'Actual':>8}  {'Error':>8}  {'Bias'}")
    print(f"  {'─'*66}")
    for _, row in cal.iterrows():
        bias = "↓ under" if row["error"] > 0.03 else ("↑ over" if row["error"] < -0.03 else "ok")
        print(f"  {row['bin']:<12} {row['n']:>5}  {row['pred_mean']:>9.1%}  "
              f"{row['actual_rate']:>7.1%}  {row['error']:>+7.1%}  {bias}")

    strategies = ["follow_market", "fade_market", "follow_extremes", "fade_extremes"]
    print(f"\n  Trading strategies (no transaction costs assumed):")
    print(f"  {'Strategy':<22} {'N':>5}  {'Ret/bet':>8}  {'Win%':>6}  {'Sharpe':>8}")
    print(f"  {'─'*56}")
    trading_results = {}
    for s in strategies:
        res = simulate_strategy(probs, outcomes, s)
        trading_results[s] = res
        print(f"  {s:<22} {res['n']:>5}  {res['mean_return']:>+7.1%}  "
              f"{res['win_rate']:>5.1%}  {res['sharpe']:>+7.3f}")

    return {
        "label":    label,
        "n":        len(sub),
        "yes_rate": yes_rate,
        "brier":    bs,
        "skill":    bss,
        "mace":     mace,
        "calibration": cal.to_dict("records"),
        "strategies": trading_results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data (for testing without live API access)
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_data(
    n: int = 600,
    seed: int = 42,
    overconfidence: float = 1.35,
) -> pd.DataFrame:
    """
    Simulate prediction markets with realistic miscalibration.

    Methodology:
      - true_prob ~ Beta(2,2)  (uniform-ish over [0,1])
      - stated_prob = sigmoid(logit(true_prob) * overconfidence)
        overconfidence > 1 → markets push extreme probs too far extreme
      - outcome ~ Bernoulli(true_prob)
      - Pre-resolution prices add Gaussian noise that shrinks as we approach end_date:
          T−7d: σ=0.09   T−3d: σ=0.05   T−1d: σ=0.025

    With overconfidence=1.35, this produces a calibration curve where:
      - 80–90% markets actually resolve YES only ~70% of the time
      - 10–20% markets resolve YES ~28% of the time
      → "fade_extremes" should outperform "follow_extremes"
    """
    rng = np.random.default_rng(seed)

    true_probs = rng.beta(2, 2, n)
    logit_true = np.log(true_probs / (1 - true_probs))
    stated_probs = 1.0 / (1.0 + np.exp(-logit_true * overconfidence))

    outcomes = (rng.uniform(size=n) < true_probs).astype(float)

    def noisy_price(sigma):
        return np.clip(stated_probs + rng.normal(0, sigma, n), 0.01, 0.99)

    assets = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "SPY", "QQQ", "Gold", "NDX"]
    directions = ["above", "below"]
    thresholds = ["$1,000", "$2,500", "$5,000", "$10,000", "$50,000", "$100,000"]

    questions = [
        f"Will {rng.choice(assets)} be {rng.choice(directions)} "
        f"{rng.choice(thresholds)} by end of {rng.integers(1,29)} {rng.choice(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])} 2025?"
        for _ in range(n)
    ]

    return pd.DataFrame({
        "source":           "synthetic",
        "question":         questions,
        "resolution":       outcomes,
        "price_t_minus_7d": noisy_price(0.09),
        "price_t_minus_3d": noisy_price(0.05),
        "price_t_minus_1d": noisy_price(0.025),
        "true_prob":        true_probs,    # oracle — only available in synthetic mode
        "stated_prob":      stated_probs,  # market's ideal price (without noise)
        "volume_usd":       rng.exponential(80_000, n),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prediction market accuracy experiment")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use generated data instead of live APIs (no network needed)")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip API calls and use cached JSON")
    parser.add_argument("--top-n", type=int, default=200,
                        help="Max markets to enrich with CLOB history")
    parser.add_argument("--no-clob", action="store_true",
                        help="Skip CLOB enrichment (only use Gamma/Kalshi last prices)")
    parser.add_argument("--no-kalshi", action="store_true",
                        help="Skip Kalshi fetching")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []

    # ── Synthetic mode ────────────────────────────────────────────────────────
    if args.synthetic:
        print("=" * 68)
        print("  SYNTHETIC MODE — simulated markets (overconfidence=1.35)")
        print("  Real data: run without --synthetic (needs Polymarket/Kalshi API)")
        print("=" * 68)

        df_syn = generate_synthetic_data(n=600, seed=42, overconfidence=1.35)
        df_syn.to_csv(RESULTS_DIR / "synthetic_markets.csv", index=False)

        print(f"\nGenerated {len(df_syn)} synthetic markets")
        print(f"YES rate: {df_syn['resolution'].mean():.1%}")

        # Show oracle calibration (stated_prob vs true_prob) as the ideal baseline
        print(f"\n{'═'*68}")
        print("  ORACLE BASELINE (stated_prob vs true_prob — only knowable in sim)")
        print(f"{'═'*68}")
        oracle = run_analysis(df_syn, "stated_prob", "Oracle (market's ideal price)")
        if oracle:
            all_results.append(oracle)

        # Real analysis: what we'd actually see from market data
        print(f"\n{'═'*68}")
        print("  OBSERVED ANALYSIS (what you'd compute from actual market data)")
        print(f"{'═'*68}")
        for d in (7, 3, 1):
            col = f"price_t_minus_{d}d"
            res = run_analysis(df_syn, col, f"Market price @ T−{d}d (with noise)")
            if res:
                all_results.append(res)

        # Extra: show the miscalibration the overconfidence creates
        print(f"\n{'─'*68}")
        print("  CALIBRATION INSIGHT (stated_prob vs actual outcome):")
        p = df_syn["stated_prob"].values
        o = df_syn["resolution"].values
        cal = calibration_table(p, o)
        over = cal[cal["error"] > 0.03]
        under = cal[cal["error"] < -0.03]
        if len(over):
            print(f"  Overconfident bins  (pred > actual): {over['bin'].tolist()}")
        if len(under):
            print(f"  Underconfident bins (pred < actual): {under['bin'].tolist()}")
        if not len(over) and not len(under):
            print("  Markets are well-calibrated!")

        # Summarise and exit
        print(f"\n{'═'*68}")
        print("  SUMMARY")
        print(f"{'═'*68}")
        for res in all_results:
            best_s = max(res["strategies"], key=lambda s: res["strategies"][s]["mean_return"])
            best_r = res["strategies"][best_s]["mean_return"]
            print(f"  {res['label']:<42} skill={res['skill']:>+.3f}  MACE={res['mace']:.3f}  "
                  f"best={best_s}({best_r:>+.1%})")

        (RESULTS_DIR / "experiment_results.json").write_text(
            json.dumps(all_results, indent=2, default=str)
        )
        print(f"\nResults saved to {RESULTS_DIR}/")
        print("\nKEY INTERPRETATION GUIDE:")
        print("  Brier Skill Score > 0  → markets add info beyond base rate")
        print("  MACE < 0.05            → well-calibrated (tight calibration curve)")
        print("  Strategy mean_return > 0.02–0.04 (vig) → potentially tradeable edge")
        print("  Typical Polymarket vig: ~2–3% per side (4–6% round-trip)")
        return

    # ── Polymarket ────────────────────────────────────────────────────────────
    poly_raw_cache  = CACHE_DIR / "polymarket_raw.json"
    poly_clob_cache = CACHE_DIR / "polymarket_clob.csv"

    if args.skip_fetch and poly_raw_cache.exists():
        print("Loading cached Polymarket raw data…")
        raw_poly = json.loads(poly_raw_cache.read_text())
    else:
        raw_poly = fetch_polymarket_resolved(limit=500)
        poly_raw_cache.write_text(json.dumps(raw_poly))

    df_poly = build_polymarket_df(raw_poly)
    print(f"\nPolymarket price-prediction markets with known resolution: {len(df_poly)}")

    if not df_poly.empty:
        df_poly.to_csv(RESULTS_DIR / "polymarket_markets.csv", index=False)

        # Show a few examples
        print("\nSample Polymarket markets:")
        for _, r in df_poly.sort_values("volume_usd", ascending=False).head(8).iterrows():
            tag = "YES ✓" if r["resolution"] == 1.0 else "NO  ✗"
            print(f"  [{tag}]  ${r['volume_usd']:>10,.0f}  {r['question'][:70]}")

        if not args.no_clob:
            if args.skip_fetch and poly_clob_cache.exists():
                print("\nLoading cached CLOB data…")
                df_clob = pd.read_csv(poly_clob_cache)
            else:
                top = df_poly.sort_values("volume_usd", ascending=False).head(args.top_n)
                df_clob = enrich_with_clob(top)
                df_clob.to_csv(poly_clob_cache, index=False)

            df_clob.to_csv(RESULTS_DIR / "polymarket_with_history.csv", index=False)

            print(f"\n{'═'*68}")
            print("  POLYMARKET — CALIBRATION & TRADING ANALYSIS")
            print(f"{'═'*68}")

            for d in (1, 3, 7):
                col = f"price_t_minus_{d}d"
                if col in df_clob.columns:
                    res = run_analysis(df_clob, col, f"Polymarket @ T−{d}d")
                    if res:
                        all_results.append(res)

    # ── Kalshi ────────────────────────────────────────────────────────────────
    if not args.no_kalshi:
        kalshi_raw_cache = CACHE_DIR / "kalshi_raw.json"

        if args.skip_fetch and kalshi_raw_cache.exists():
            print("\nLoading cached Kalshi raw data…")
            raw_kalshi = json.loads(kalshi_raw_cache.read_text())
        else:
            raw_kalshi = fetch_kalshi_settled(limit=300)
            kalshi_raw_cache.write_text(json.dumps(raw_kalshi))

        df_kalshi = build_kalshi_df(raw_kalshi)
        print(f"\nKalshi price-prediction markets with known resolution: {len(df_kalshi)}")

        if not df_kalshi.empty:
            df_kalshi.to_csv(RESULTS_DIR / "kalshi_markets.csv", index=False)

            print("\nSample Kalshi markets:")
            for _, r in df_kalshi.sort_values("volume_usd", ascending=False).head(8).iterrows():
                tag = "YES ✓" if r["resolution"] == 1.0 else "NO  ✗"
                print(f"  [{tag}]  ${r['volume_usd']:>10,.0f}  {r['question'][:70]}")

            if "price_last" in df_kalshi.columns:
                print(f"\n{'═'*68}")
                print("  KALSHI — CALIBRATION & TRADING ANALYSIS")
                print(f"{'═'*68}")
                res = run_analysis(df_kalshi, "price_last", "Kalshi (last price before close)")
                if res:
                    all_results.append(res)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*68}")
    print("  SUMMARY")
    print(f"{'═'*68}")
    if not all_results:
        print("  No analysis results — try without --no-clob or --skip-fetch")
    else:
        print(f"  {'Label':<42} {'Skill':>6}  {'MACE':>6}  {'Best strategy'}")
        print(f"  {'─'*68}")
        for res in all_results:
            best_s = max(res["strategies"], key=lambda s: res["strategies"][s]["mean_return"])
            best_r = res["strategies"][best_s]["mean_return"]
            print(f"  {res['label']:<42} {res['skill']:>+5.3f}  {res['mace']:>5.3f}  "
                  f"{best_s} ({best_r:+.1%}/bet)")

    # Save full results
    results_path = RESULTS_DIR / "experiment_results.json"
    results_path.write_text(json.dumps(all_results, indent=2, default=str))

    print(f"\nResults saved to {RESULTS_DIR}/")
    print("  polymarket_markets.csv          — all Polymarket markets")
    print("  polymarket_with_history.csv     — with pre-resolution CLOB prices")
    print("  kalshi_markets.csv              — all Kalshi markets")
    print("  experiment_results.json         — full calibration + trading stats")
    print()
    print("KEY INTERPRETATION GUIDE:")
    print("  Brier Skill Score > 0  → markets add info beyond base rate")
    print("  MACE < 0.05            → well-calibrated (tight calibration curve)")
    print("  Strategy mean_return > 0.02 after vig → potentially tradeable edge")
    print("  Typical Polymarket vig: ~2% per side (i.e. 4% round-trip)")


if __name__ == "__main__":
    main()
