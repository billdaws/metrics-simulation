#!/usr/bin/env python3
"""
Measure production metric characteristics needed to calibrate the simulation.

Usage:
    python calibrate.py <metric_path> [--url URL] [--days DAYS]

Example:
    python calibrate.py servers.web.*.errors.rate --url http://graphite:8080

Paste the printed report into chat for analysis.
"""

import argparse
import sys
import textwrap
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests


# ---------------------------------------------------------------------------
# Data fetch
# ---------------------------------------------------------------------------

def fetch(metric: str, graphite_url: str, days: int) -> pd.Series:
    until_ts = int(datetime.now(timezone.utc).timestamp())
    from_ts = until_ts - days * 86400
    resp = requests.get(
        f"{graphite_url}/render",
        params={
            "target": metric,
            "from": from_ts,
            "until": until_ts,
            "format": "json",
            "maxDataPoints": days * 1440,  # request 1-min resolution
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    if not data:
        print(f"ERROR: Graphite returned no series for target '{metric}'", file=sys.stderr)
        sys.exit(1)
    if len(data) > 1:
        print(
            f"WARNING: {len(data)} series matched. Using first: {data[0]['target']}",
            file=sys.stderr,
        )

    raw = data[0]["datapoints"]
    s = pd.Series(
        {pd.Timestamp(ts, unit="s", tz="UTC"): v for v, ts in raw if v is not None},
        dtype=float,
    )
    s.sort_index(inplace=True)
    return s


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def infer_resolution_seconds(s: pd.Series) -> float:
    diffs = s.index.to_series().diff().dropna().dt.total_seconds()
    return float(diffs.mode().iloc[0])


def autocorrelation_profile(residuals: pd.Series, max_lag: int = 10) -> dict[int, float]:
    return {lag: float(residuals.autocorr(lag=lag)) for lag in range(1, max_lag + 1)}


def seasonality_ratios(s: pd.Series) -> dict[str, float]:
    hourly = s.groupby(s.index.hour).mean()
    daily = s.groupby(s.index.day_of_week).mean()
    return {
        "hour_of_day_peak_to_trough": float(hourly.max() / hourly.min()) if hourly.min() > 0 else float("nan"),
        "day_of_week_peak_to_trough": float(daily.max() / daily.min()) if daily.min() > 0 else float("nan"),
    }


def distribution_stats(s: pd.Series) -> dict[str, float]:
    from scipy.stats import kurtosis, skew  # type: ignore[import-untyped]
    return {
        "mean": float(s.mean()),
        "std": float(s.std()),
        "cv": float(s.std() / s.mean()) if s.mean() > 0 else float("nan"),
        "p50": float(s.quantile(0.50)),
        "p90": float(s.quantile(0.90)),
        "p99": float(s.quantile(0.99)),
        "min": float(s.min()),
        "max": float(s.max()),
        "zero_fraction": float((s == 0).mean()),
        "skewness": float(skew(s.dropna())),
        "excess_kurtosis": float(kurtosis(s.dropna())),  # Fisher: 0 = Normal
    }


def breach_analysis(
    s: pd.Series, short_window: int = 5, long_window_days: int = 7, sigma: float = 3.0
) -> dict:
    """Run the z-score rule over the full series and characterise breach windows.

    Uses a 7-day baseline window (half the alert's 14-day window) so the evaluation
    period is non-trivial even when only 2–3 weeks of data are available.
    """
    vals = s.values
    sr = pd.Series(vals, dtype=float)
    long_window = long_window_days * 1440

    x = sr.rolling(short_window, min_periods=1).mean()
    mean = sr.rolling(long_window, min_periods=1).mean()
    std = sr.rolling(long_window, min_periods=2).std().fillna(0.0)
    z = ((x - mean) / std.clip(lower=1e-10)).values

    # Only evaluate after the burn-in window is populated.
    burn = min(long_window, len(z) - 1)
    z_eval = z[burn:]
    ts_eval = s.index[burn:]

    fires = np.abs(z_eval) > sigma
    breach_durations: list[int] = []
    peak_zscores: list[float] = []
    directions: list[str] = []
    consecutive = 0
    in_win = False
    win_start = 0
    win_peak = 0.0

    for i, f in enumerate(fires):
        if f:
            if not in_win:
                in_win = True
                win_start = i
                win_peak = abs(z_eval[i])
            else:
                win_peak = max(win_peak, abs(z_eval[i]))
        else:
            if in_win:
                duration = i - win_start
                breach_durations.append(duration)
                peak_zscores.append(win_peak)
                directions.append("up" if z_eval[win_start] > 0 else "down")
                in_win = False
                win_peak = 0.0
    if in_win:
        duration = len(fires) - win_start
        breach_durations.append(duration)
        peak_zscores.append(win_peak)
        directions.append("up" if z_eval[win_start] > 0 else "down")

    n_breaches = len(breach_durations)
    eval_days = (ts_eval[-1] - ts_eval[0]).total_seconds() / 86400 if len(ts_eval) > 1 else 0.0

    return {
        "eval_days": round(eval_days, 1),
        "n_breaches": n_breaches,
        "breaches_per_day": round(n_breaches / eval_days, 2) if eval_days > 0 else float("nan"),
        "median_breach_duration_min": float(np.median(breach_durations)) if breach_durations else None,
        "p90_breach_duration_min": float(np.percentile(breach_durations, 90)) if breach_durations else None,
        "max_breach_duration_min": float(np.max(breach_durations)) if breach_durations else None,
        "median_peak_z": float(np.median(peak_zscores)) if peak_zscores else None,
        "p90_peak_z": float(np.percentile(peak_zscores, 90)) if peak_zscores else None,
        "pct_upward": (
            round(100 * directions.count("up") / n_breaches) if n_breaches else None
        ),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def fmt(val, fmt_str=".4f") -> str:
    if val is None:
        return "n/a"
    if isinstance(val, float) and np.isnan(val):
        return "nan"
    return format(val, fmt_str)


def print_report(metric: str, graphite_url: str, days: int, s: pd.Series) -> None:
    res_sec = infer_resolution_seconds(s)
    total_possible = days * 86400 / res_sec
    null_frac = 1.0 - len(s) / total_possible

    dist = distribution_stats(s)

    # Compute residuals for autocorrelation: detrend with a 60-point rolling mean.
    rolling_window = max(5, int(3600 / res_sec))  # ~1 hour
    residuals = s - s.rolling(rolling_window, min_periods=1).mean()
    acf = autocorrelation_profile(residuals)

    seas = seasonality_ratios(s)
    breaches = breach_analysis(s)

    lines = [
        "=" * 60,
        "PRODUCTION METRIC CALIBRATION REPORT",
        "=" * 60,
        "",
        f"Metric  : {metric}",
        f"Source  : {graphite_url}",
        f"Window  : last {days} days",
        f"Fetched : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Points  : {len(s):,}  (resolution ≈ {res_sec:.0f}s, null fraction {null_frac:.1%})",
        "",
        "── DISTRIBUTION ─────────────────────────────────────",
        f"  mean              : {fmt(dist['mean'])}",
        f"  std               : {fmt(dist['std'])}",
        f"  coeff of variation: {fmt(dist['cv'])}   (std / mean; target for noise_std/rate)",
        f"  p50               : {fmt(dist['p50'])}",
        f"  p90               : {fmt(dist['p90'])}",
        f"  p99               : {fmt(dist['p99'])}",
        f"  min               : {fmt(dist['min'])}",
        f"  max               : {fmt(dist['max'])}",
        f"  zero fraction     : {fmt(dist['zero_fraction'], '.1%')}",
        f"  skewness          : {fmt(dist['skewness'])}   (0 = symmetric)",
        f"  excess kurtosis   : {fmt(dist['excess_kurtosis'])}   (0 = Gaussian)",
        "",
        "── AUTOCORRELATION (residuals after rolling detrend) ─",
    ]
    for lag, rho in acf.items():
        bar = "█" * int(abs(rho) * 20)
        lines.append(f"  lag {lag:2d}  : {fmt(rho, '+.3f')}  {bar}")
    lines += [
        "",
        "── SEASONALITY ──────────────────────────────────────",
        f"  hour-of-day peak/trough : {fmt(seas['hour_of_day_peak_to_trough'], '.2f')}x",
        f"  day-of-week peak/trough : {fmt(seas['day_of_week_peak_to_trough'], '.2f')}x",
        "  (ratios > 2x suggest seasonality matters for z-score σ inflation)",
        "",
        "── HISTORICAL Z-SCORE BREACHES (|z| > 3, 14-day window) ─",
        f"  evaluation window  : {breaches['eval_days']} days  (7-day baseline; actual alert uses 14d)",
        f"  total breaches     : {breaches['n_breaches']}",
        f"  breaches / day     : {fmt(breaches['breaches_per_day'], '.2f')}",
        f"  median duration    : {fmt(breaches['median_breach_duration_min'], '.1f')} min",
        f"  p90 duration       : {fmt(breaches['p90_breach_duration_min'], '.1f')} min",
        f"  max duration       : {fmt(breaches['max_breach_duration_min'], '.1f')} min",
        f"  median peak |z|    : {fmt(breaches['median_peak_z'], '.2f')}",
        f"  p90 peak |z|       : {fmt(breaches['p90_peak_z'], '.2f')}",
        f"  % upward breaches  : {fmt(breaches['pct_upward'], '.0f')}%",
        "",
        "── SIMULATION PARAMETER TARGETS ─────────────────────",
        f"  rate       ≈ {fmt(dist['mean'])}   (mean value)",
        f"  noise_std  ≈ {fmt(dist['std'])}   (std of raw values; refine after checking residuals)",
        f"  phi        ≈ {fmt(acf[1], '.3f')}    (lag-1 autocorrelation of residuals)",
        "  seasonality: " + (
            "PRESENT — consider adding diurnal scenario"
            if seas["hour_of_day_peak_to_trough"] > 2.0
            else "mild — rolling baseline should absorb it"
        ),
        "  distribution: " + (
            "non-Gaussian (|skew| > 1 or |kurtosis| > 2) — clipping model may be inadequate"
            if abs(dist["skewness"]) > 1.0 or abs(dist["excess_kurtosis"]) > 2.0
            else "roughly Gaussian — AR(1) model is reasonable"
        ),
        "",
        "=" * 60,
    ]
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure production metric characteristics for simulation calibration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python calibrate.py servers.web.*.errors.rate
              python calibrate.py my.metric --url http://graphite.internal:8080 --days 21
        """),
    )
    parser.add_argument("metric", help="Graphite metric path (wildcards OK if they resolve to one series)")
    parser.add_argument("--url", default="http://localhost:8080", help="Graphite base URL")
    parser.add_argument("--days", type=int, default=28, help="How many days of history to fetch (default: 28)")
    args = parser.parse_args()

    print(f"Fetching {args.days} days of '{args.metric}' from {args.url} ...", file=sys.stderr)
    s = fetch(args.metric, args.url, args.days)
    print(f"Got {len(s):,} non-null points.", file=sys.stderr)
    print_report(args.metric, args.url, args.days, s)


if __name__ == "__main__":
    main()
