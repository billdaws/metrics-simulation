import operator
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from metrics_simulation import carbon, graphite
from metrics_simulation.graphite import Series
from metrics_simulation.scenarios import Scenario
from metrics_simulation.server import GraphiteServer


@dataclass
class AlertRule:
    # Graphite expression. Use {metric} as a placeholder for the metric path,
    # e.g. "movingAverage({metric}, 5)". simulate() substitutes per scenario.
    query: str
    threshold: float
    comparator: Literal["gt", "lt", "gte", "lte"] = "gt"
    for_duration_points: int = 1


@dataclass
class ScenarioResult:
    scenario: Scenario
    series: Series
    firing_windows: list[tuple[int, int]]  # (start_ts, end_ts) unix seconds
    fired: bool
    first_fire_offset_min: float | None    # minutes from scenario start to first firing


_COMPARATORS: dict[str, Callable[[float, float], bool]] = {
    "gt": operator.gt,
    "lt": operator.lt,
    "gte": operator.ge,
    "lte": operator.le,
}


def firing_windows(series: Series, rule: AlertRule) -> list[tuple[int, int]]:
    compare = _COMPARATORS[rule.comparator]
    windows: list[tuple[int, int]] = []
    consecutive = 0
    firing = False
    fire_start = 0
    prev_ts = series.datapoints[0].timestamp if series.datapoints else 0

    for dp in series.datapoints:
        if dp.value is not None and compare(dp.value, rule.threshold):
            consecutive += 1
            if consecutive >= rule.for_duration_points and not firing:
                firing = True
                fire_start = dp.timestamp
        else:
            if firing:
                windows.append((fire_start, prev_ts))
                firing = False
            consecutive = 0
        prev_ts = dp.timestamp

    if firing:
        windows.append((fire_start, prev_ts))
    return windows


def simulate(
    rule: AlertRule,
    scenarios: list[Scenario],
    server: GraphiteServer,
    flush_wait_seconds: float = 5.0,
) -> list[ScenarioResult]:
    for scenario in scenarios:
        points = list(zip(scenario.values.tolist(), scenario.timestamps.tolist()))
        carbon.write_series(scenario.metric, points, host=server.carbon_host, port=server.carbon_port)

    time.sleep(flush_wait_seconds)

    results: list[ScenarioResult] = []
    for scenario in scenarios:
        target = rule.query.replace("{metric}", scenario.metric)
        series_list = graphite.query(
            target=target,
            from_time=int(scenario.timestamps[0]),
            until_time=int(scenario.timestamps[-1]),
            base_url=server.graphite_url,
        )

        series = series_list[0] if series_list else Series(target=target, datapoints=[])
        windows = firing_windows(series, rule) if series.datapoints else []
        fired = len(windows) > 0
        first_fire_offset_min = (
            (windows[0][0] - int(scenario.timestamps[0])) / 60.0 if fired else None
        )

        results.append(ScenarioResult(
            scenario=scenario,
            series=series,
            firing_windows=windows,
            fired=fired,
            first_fire_offset_min=first_fire_offset_min,
        ))

    return results


def summary(results: list[ScenarioResult]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "scenario": r.scenario.name,
            "description": r.scenario.description,
            "fired": r.fired,
            "first_fire_min": r.first_fire_offset_min,
        }
        for r in results
    ])


@dataclass
class MonteCarloResult:
    scenario_name: str
    description: str
    n_runs: int
    fire_rate: float            # fraction of runs where the alert fired
    first_fire_offsets: list[float]  # minutes from scenario start, one per firing run


def monte_carlo(
    rule: AlertRule,
    scenario_factories: list[Callable[[np.random.Generator], Scenario]],
    server: GraphiteServer,
    n: int = 100,
    seed: int | None = None,
    flush_wait_seconds: float = 5.0,
) -> list[MonteCarloResult]:
    master_rng = np.random.default_rng(seed)

    all_runs: list[list[Scenario]] = []
    for factory in scenario_factories:
        factory_runs: list[Scenario] = []
        for run_idx in range(n):
            rng = np.random.default_rng(master_rng.integers(0, 2**32))
            base = factory(rng)
            factory_runs.append(Scenario(
                name=base.name,
                description=base.description,
                metric=f"{base.metric}.mc_{run_idx}",
                timestamps=base.timestamps,
                values=base.values,
            ))
        all_runs.append(factory_runs)

    batch = [
        (s.metric, list(zip(s.values.tolist(), s.timestamps.tolist())))
        for factory_runs in all_runs
        for s in factory_runs
    ]
    carbon.write_batch(batch, host=server.carbon_host, port=server.carbon_port)
    time.sleep(flush_wait_seconds)

    results: list[MonteCarloResult] = []
    for factory_runs in all_runs:
        fire_offsets: list[float] = []
        for scenario in factory_runs:
            target = rule.query.replace("{metric}", scenario.metric)
            series_list = graphite.query(
                target=target,
                from_time=int(scenario.timestamps[0]),
                until_time=int(scenario.timestamps[-1]),
                base_url=server.graphite_url,
            )
            series = series_list[0] if series_list else Series(target=target, datapoints=[])
            windows = firing_windows(series, rule) if series.datapoints else []
            if windows:
                fire_offsets.append((windows[0][0] - int(scenario.timestamps[0])) / 60.0)

        results.append(MonteCarloResult(
            scenario_name=factory_runs[0].name,
            description=factory_runs[0].description,
            n_runs=n,
            fire_rate=len(fire_offsets) / n,
            first_fire_offsets=fire_offsets,
        ))

    return results


def mc_summary(results: list[MonteCarloResult]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "scenario": r.scenario_name,
            "n_runs": r.n_runs,
            "fire_rate": r.fire_rate,
            "median_first_fire_min": (
                float(np.median(r.first_fire_offsets)) if r.first_fire_offsets else None
            ),
        }
        for r in results
    ])
