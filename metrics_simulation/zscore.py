"""Z-score based alert simulation (pure Python, no Graphite required)."""
import math
from collections.abc import Callable
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from metrics_simulation.scenarios import Scenario


@dataclass
class ZScoreAlertRule:
    short_window: int = 5            # rolling mean window in minutes (x)
    long_window: int = 14 * 24 * 60  # rolling mean/std window in minutes (baseline μ and σ)
    sigma_threshold: float = 3.0
    burn_in_min: int = 14 * 24 * 60  # only evaluate alert after this many minutes of warm-up


@dataclass
class ZScoreResult:
    scenario: Scenario
    test_minutes: np.ndarray               # 0-based minutes from start of test window
    test_raw: np.ndarray                   # raw metric values in the test window
    test_zscore: np.ndarray                # z-scores in the test window
    fired: bool
    first_fire_test_min: float | None      # minutes from test window start
    firing_windows: list[tuple[float, float]]  # (start_min, end_min), test-window-relative


@dataclass
class ZScoreMCRun:
    test_minutes: list[float]
    zscore_values: list[float]
    fired: bool
    first_fire_test_min: float | None


@dataclass
class ZScoreMCResult:
    scenario_name: str
    description: str
    n_runs: int
    fire_rate: float
    first_fire_offsets: list[float]  # test-window-relative minutes, one per firing run
    runs: list[ZScoreMCRun]


def _zscore_array(values: np.ndarray, rule: ZScoreAlertRule) -> np.ndarray:
    s = pd.Series(values, dtype=float)
    x = s.rolling(rule.short_window, min_periods=1).mean()
    mean = s.rolling(rule.long_window, min_periods=1).mean()
    std = s.rolling(rule.long_window, min_periods=2).std().fillna(0.0)
    return ((x - mean) / std.clip(lower=1e-10)).values


def _detect_windows(z_test: np.ndarray, threshold: float) -> list[tuple[float, float]]:
    fires = np.abs(z_test) > threshold
    windows: list[tuple[float, float]] = []
    in_win = False
    start = 0.0
    for i, f in enumerate(fires):
        if f and not in_win:
            in_win, start = True, float(i)
        elif not f and in_win:
            in_win = False
            windows.append((start, float(i - 1)))
    if in_win:
        windows.append((start, float(len(fires) - 1)))
    return windows


def _eval_scenario(
    scenario: Scenario, rule: ZScoreAlertRule
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, float | None, list[tuple[float, float]]]:
    ts = scenario.timestamps.astype(float)
    values = scenario.values
    burn_idx = min(rule.burn_in_min, len(values) - 1)

    z_all = _zscore_array(values, rule)

    test_ts = ts[burn_idx:]
    t0_test = test_ts[0]
    test_mins = (test_ts - t0_test) / 60.0
    test_z = z_all[burn_idx:]

    windows = _detect_windows(test_z, rule.sigma_threshold)
    fired = bool(windows)
    first_fire = windows[0][0] if fired else None
    return test_mins, values[burn_idx:], test_z, fired, first_fire, windows


def simulate_zscore(rule: ZScoreAlertRule, scenario_list: list[Scenario]) -> list[ZScoreResult]:
    results = []
    for scenario in scenario_list:
        test_mins, test_raw, test_z, fired, first_fire, windows = _eval_scenario(scenario, rule)
        results.append(ZScoreResult(
            scenario=scenario,
            test_minutes=test_mins,
            test_raw=test_raw,
            test_zscore=test_z,
            fired=fired,
            first_fire_test_min=first_fire,
            firing_windows=windows,
        ))
    return results


def zscore_summary(results: list[ZScoreResult]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "scenario": r.scenario.name,
            "description": r.scenario.description,
            "fired": r.fired,
            "first_fire_test_min": r.first_fire_test_min,
        }
        for r in results
    ])


def plot_zscore_results(results: list[ZScoreResult], rule: ZScoreAlertRule) -> None:
    n = len(results)
    ncols = min(2, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows * 2, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)
    fig.suptitle(
        f"Z-Score Alert  |  x = rolling {rule.short_window}-min avg  "
        f"μ = rolling {rule.long_window}-min avg  threshold = ±{rule.sigma_threshold}σ",
        fontsize=9,
    )

    for i, result in enumerate(results):
        row = (i // ncols) * 2
        col = i % ncols
        ax_raw = axes[row][col]
        ax_z = axes[row + 1][col]

        mins = result.test_minutes
        ax_raw.plot(mins, result.test_raw, color="steelblue", linewidth=1, label="raw")
        ax_z.plot(mins, result.test_zscore, color="purple", linewidth=1, label="z-score")
        ax_z.axhline(rule.sigma_threshold, color="orange", linestyle="--", linewidth=1,
                     label=f"+{rule.sigma_threshold}σ")
        ax_z.axhline(-rule.sigma_threshold, color="darkorange", linestyle=":", linewidth=1,
                     label=f"-{rule.sigma_threshold}σ")
        ax_z.axhline(0, color="gray", linewidth=0.5, alpha=0.4)

        for ws, we in result.firing_windows:
            ax_raw.axvspan(ws, we + 1, color="red", alpha=0.2)
            ax_z.axvspan(ws, we + 1, color="red", alpha=0.2)

        status = "FIRED" if result.fired else "OK"
        title_color = "darkred" if result.fired else "darkgreen"
        ax_raw.set_title(
            f"[{status}] {result.scenario.name}\n{result.scenario.description}",
            color=title_color, fontsize=8,
        )
        ax_raw.set_ylabel("metric value", fontsize=7)
        ax_raw.legend(fontsize=7)
        ax_raw.tick_params(axis="x", labelsize=7)
        ax_z.set_ylabel("z-score", fontsize=7)
        ax_z.set_xlabel("minutes into test window", fontsize=8)
        ax_z.legend(fontsize=7)
        ax_z.tick_params(axis="x", labelsize=7)

    for j in range(n, nrows * ncols):
        axes[(j // ncols) * 2][j % ncols].set_visible(False)
        axes[(j // ncols) * 2 + 1][j % ncols].set_visible(False)

    plt.tight_layout()
    plt.show()


def monte_carlo_zscore(
    rule: ZScoreAlertRule,
    scenario_factories: list[Callable[[np.random.Generator], Scenario]],
    n: int = 100,
    seed: int | None = None,
) -> list[ZScoreMCResult]:
    master_rng = np.random.default_rng(seed)
    results = []

    for factory in scenario_factories:
        fire_offsets: list[float] = []
        runs: list[ZScoreMCRun] = []
        scenario_name = ""
        description = ""

        for run_i in range(n):
            rng = np.random.default_rng(master_rng.integers(0, 2**32))
            scenario = factory(rng)
            if run_i == 0:
                scenario_name = scenario.name
                description = scenario.description

            test_mins, _, test_z, fired, first_fire, _ = _eval_scenario(scenario, rule)
            if first_fire is not None:
                fire_offsets.append(first_fire)
            runs.append(ZScoreMCRun(
                test_minutes=test_mins.tolist(),
                zscore_values=test_z.tolist(),
                fired=fired,
                first_fire_test_min=first_fire,
            ))

        results.append(ZScoreMCResult(
            scenario_name=scenario_name,
            description=description,
            n_runs=n,
            fire_rate=len(fire_offsets) / n if n > 0 else 0.0,
            first_fire_offsets=fire_offsets,
            runs=runs,
        ))

    return results


def zscore_mc_summary(results: list[ZScoreMCResult]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "scenario": r.scenario_name,
            "n_runs": r.n_runs,
            "fire_rate": r.fire_rate,
            "median_first_fire_test_min": (
                float(np.median(r.first_fire_offsets)) if r.first_fire_offsets else None
            ),
        }
        for r in results
    ])


def plot_zscore_mc(results: list[ZScoreMCResult], rule: ZScoreAlertRule) -> None:
    n = len(results)
    ncols = min(2, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle(
        f"Z-Score Monte Carlo  |  x = rolling {rule.short_window}-min avg  "
        f"μ = rolling {rule.long_window}-min avg  threshold = ±{rule.sigma_threshold}σ",
        fontsize=9,
    )

    for i, result in enumerate(results):
        ax = axes[i // ncols][i % ncols]
        n_fired = sum(r.fired for r in result.runs)

        for run in result.runs:
            color = "tomato" if run.fired else "steelblue"
            ax.plot(
                run.test_minutes, run.zscore_values,
                color=color,
                alpha=0.35 if run.fired else 0.15,
                linewidth=0.8,
            )
            if run.fired and run.first_fire_test_min is not None:
                ax.axvline(run.first_fire_test_min, color="darkred", alpha=0.12, linewidth=0.6)

        ax.axhline(rule.sigma_threshold, color="orange", linestyle="--", linewidth=1.2)
        ax.axhline(-rule.sigma_threshold, color="darkorange", linestyle=":", linewidth=1.2)
        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.4)

        handles = [
            Line2D([0], [0], color="steelblue", alpha=0.7, label=f"not fired ({result.n_runs - n_fired})"),
            Line2D([0], [0], color="tomato", alpha=0.7, label=f"fired ({n_fired})"),
            Line2D([0], [0], color="darkred", alpha=0.5, linewidth=1, label="first fire"),
            Line2D([0], [0], color="orange", linestyle="--", label=f"+{rule.sigma_threshold}σ"),
            Line2D([0], [0], color="darkorange", linestyle=":", label=f"-{rule.sigma_threshold}σ"),
        ]
        ax.legend(handles=handles, fontsize=7)
        ax.set_title(
            f"{result.scenario_name}  —  fire rate: {result.fire_rate:.0%}  ({result.n_runs} runs)",
            fontsize=8,
        )
        ax.set_xlabel("minutes into test window", fontsize=8)
        ax.set_ylabel("z-score", fontsize=7)
        ax.tick_params(axis="x", labelsize=7)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    plt.tight_layout()
    plt.show()
