import math
from datetime import datetime

import matplotlib.pyplot as plt

from metrics_simulation.alert import AlertRule, MonteCarloResult, ScenarioResult


def plot_results(results: list[ScenarioResult], rule: AlertRule) -> None:
    n = len(results)
    ncols = min(2, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle(
        f"query: {rule.query}  |  threshold: {rule.comparator} {rule.threshold}",
        fontsize=10,
    )

    for i, result in enumerate(results):
        ax = axes[i // ncols][i % ncols]

        raw_times = [datetime.fromtimestamp(int(t)) for t in result.scenario.timestamps]
        ax.plot(raw_times, result.scenario.values, color="steelblue", alpha=0.35, linewidth=1, label="raw")

        if result.series.datapoints:
            valid = [(dp.timestamp, dp.value) for dp in result.series.datapoints if dp.value is not None]
            if valid:
                q_times = [datetime.fromtimestamp(t) for t, _ in valid]
                q_vals = [v for _, v in valid]
                ax.plot(q_times, q_vals, color="steelblue", linewidth=1.5, label="query output")

        ax.axhline(rule.threshold, color="orange", linestyle="--", linewidth=1, label=f"threshold ({rule.threshold})")

        for start_ts, end_ts in result.firing_windows:
            ax.axvspan(datetime.fromtimestamp(start_ts), datetime.fromtimestamp(end_ts), color="red", alpha=0.2)

        status = "FIRED" if result.fired else "OK"
        title_color = "darkred" if result.fired else "darkgreen"
        ax.set_title(
            f"[{status}] {result.scenario.name}\n{result.scenario.description}",
            color=title_color,
            fontsize=8,
        )
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
        ax.legend(fontsize=7)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_monte_carlo(results: list[MonteCarloResult]) -> None:
    n = len(results)
    ncols = min(2, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle("Monte Carlo: alert firing distribution", fontsize=10)

    for i, result in enumerate(results):
        ax = axes[i // ncols][i % ncols]

        if result.first_fire_offsets:
            ax.hist(result.first_fire_offsets, bins=20, color="steelblue", alpha=0.7, edgecolor="white")
            ax.set_xlabel("minutes to first fire")
            ax.set_ylabel("runs")
        else:
            ax.text(0.5, 0.5, "never fired", ha="center", va="center",
                    transform=ax.transAxes, fontsize=13, color="darkgreen")

        ax.set_title(
            f"{result.scenario_name}  —  fire rate: {result.fire_rate:.0%}  ({result.n_runs} runs)",
            fontsize=8,
        )

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    plt.tight_layout()
    plt.show()
