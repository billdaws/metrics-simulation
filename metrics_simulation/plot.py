import math

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
        t0 = float(result.scenario.timestamps[0])

        raw_mins = [(float(t) - t0) / 60.0 for t in result.scenario.timestamps]
        ax.plot(raw_mins, result.scenario.values, color="steelblue", alpha=0.35, linewidth=1, label="raw")

        if result.series.datapoints:
            valid = [(dp.timestamp, dp.value) for dp in result.series.datapoints if dp.value is not None]
            if valid:
                q_mins = [(t - t0) / 60.0 for t, _ in valid]
                q_vals = [v for _, v in valid]
                ax.plot(q_mins, q_vals, color="steelblue", linewidth=1.5, label="query output")

        ax.axhline(rule.threshold, color="orange", linestyle="--", linewidth=1, label=f"threshold ({rule.threshold})")

        for start_ts, end_ts in result.firing_windows:
            ax.axvspan((start_ts - t0) / 60.0, (end_ts - t0) / 60.0, color="red", alpha=0.2)

        status = "FIRED" if result.fired else "OK"
        title_color = "darkred" if result.fired else "darkgreen"
        ax.set_title(
            f"[{status}] {result.scenario.name}\n{result.scenario.description}",
            color=title_color,
            fontsize=8,
        )
        ax.set_xlabel("minutes elapsed", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
        ax.legend(fontsize=7)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_monte_carlo(results: list[MonteCarloResult], rule: AlertRule) -> None:
    n = len(results)
    ncols = min(2, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle(
        f"Monte Carlo: simulated paths  |  query: {rule.query}  |  threshold: {rule.comparator} {rule.threshold}",
        fontsize=10,
    )

    for i, result in enumerate(results):
        ax = axes[i // ncols][i % ncols]
        n_fired = sum(r.fired for r in result.runs)

        for run in result.runs:
            color = "tomato" if run.fired else "steelblue"
            ax.plot(run.raw_minutes, run.raw_values,
                    color=color, alpha=0.08, linewidth=0.5, linestyle="dotted")
            if run.query_minutes:
                ax.plot(run.query_minutes, run.query_values,
                        color=color, alpha=0.4 if run.fired else 0.25, linewidth=0.8)
            if run.fired and run.first_fire_min is not None and run.query_minutes:
                idx = min(range(len(run.query_minutes)),
                          key=lambda j: abs(run.query_minutes[j] - run.first_fire_min))
                ax.plot(run.query_minutes[idx], run.query_values[idx],
                        "o", color="darkred", markersize=3, alpha=0.6, zorder=5)

        ax.axhline(rule.threshold, color="orange", linestyle="--", linewidth=1)

        handles = [
            Line2D([0], [0], color="steelblue", alpha=0.7, label=f"not fired ({result.n_runs - n_fired})"),
            Line2D([0], [0], color="tomato", alpha=0.7, label=f"fired ({n_fired})"),
            Line2D([0], [0], color="darkred", marker="o", markersize=4, linestyle="none", label="first fire"),
            Line2D([0], [0], color="grey", alpha=0.5, linestyle="dotted", label="raw (dotted)"),
            Line2D([0], [0], color="orange", linestyle="--", label=f"threshold ({rule.threshold})"),
        ]
        ax.legend(handles=handles, fontsize=7)
        ax.set_title(
            f"{result.scenario_name}  —  fire rate: {result.fire_rate:.0%}  ({result.n_runs} runs)",
            fontsize=8,
        )
        ax.set_xlabel("minutes elapsed", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    plt.tight_layout()
    plt.show()
