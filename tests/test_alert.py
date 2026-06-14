from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from metrics_simulation.alert import (
    AlertRule,
    MonteCarloResult,
    ScenarioResult,
    firing_windows,
    mc_summary,
    monte_carlo,
    summary,
)
from metrics_simulation.graphite import Datapoint, Series
from metrics_simulation.scenarios import Scenario
from metrics_simulation.server import GraphiteServer


def _series(values: list[float | None], start_ts: int = 0, step: int = 60) -> Series:
    datapoints = [Datapoint(value=v, timestamp=start_ts + i * step) for i, v in enumerate(values)]
    return Series(target="sim.test", datapoints=datapoints)


def _rule(threshold: float = 0.05, comparator: str = "gt", for_duration_points: int = 1) -> AlertRule:
    return AlertRule(query="{metric}", threshold=threshold, comparator=comparator, for_duration_points=for_duration_points)


class TestFiringWindows:
    def test_no_breach_returns_empty(self) -> None:
        series = _series([0.01, 0.02, 0.03])
        assert firing_windows(series, _rule(threshold=0.05)) == []

    def test_all_breach_single_window(self) -> None:
        series = _series([0.1, 0.2, 0.3], start_ts=1000)
        windows = firing_windows(series, _rule(threshold=0.05))
        assert len(windows) == 1
        assert windows[0] == (1000, 1120)

    def test_for_duration_not_met_no_fire(self) -> None:
        series = _series([0.1, 0.1, 0.01], start_ts=0)
        rule = _rule(threshold=0.05, for_duration_points=3)
        assert firing_windows(series, rule) == []

    def test_for_duration_exactly_met(self) -> None:
        series = _series([0.1, 0.1, 0.1], start_ts=0)
        rule = _rule(threshold=0.05, for_duration_points=3)
        windows = firing_windows(series, rule)
        assert len(windows) == 1

    def test_fires_then_resolves(self) -> None:
        series = _series([0.1, 0.1, 0.01, 0.01], start_ts=0)
        windows = firing_windows(series, _rule(threshold=0.05))
        assert len(windows) == 1
        assert windows[0] == (0, 60)

    def test_fires_resolves_fires_again(self) -> None:
        series = _series([0.1, 0.01, 0.1], start_ts=0)
        windows = firing_windows(series, _rule(threshold=0.05))
        assert len(windows) == 2

    def test_null_datapoints_break_consecutive_count(self) -> None:
        series = _series([0.1, None, 0.1], start_ts=0)
        rule = _rule(threshold=0.05, for_duration_points=2)
        assert firing_windows(series, rule) == []

    def test_lt_comparator(self) -> None:
        series = _series([0.03, 0.03, 0.1], start_ts=0)
        rule = _rule(threshold=0.05, comparator="lt")
        windows = firing_windows(series, rule)
        assert len(windows) == 1
        assert windows[0] == (0, 60)

    def test_fire_start_respects_for_duration_offset(self) -> None:
        # Alert fires at the 3rd point (index 2), not at index 0
        series = _series([0.1, 0.1, 0.1, 0.1], start_ts=0)
        rule = _rule(threshold=0.05, for_duration_points=3)
        windows = firing_windows(series, rule)
        assert windows[0][0] == 120  # 3rd datapoint timestamp


class TestSummary:
    def _result(self, name: str, fired: bool, first_fire: float | None) -> ScenarioResult:
        ts = np.array([0, 60], dtype=np.int64)
        sc = Scenario(name=name, description="test", metric="sim.test", timestamps=ts, values=np.zeros(2))
        return ScenarioResult(
            scenario=sc,
            series=Series(target="sim.test", datapoints=[]),
            firing_windows=[],
            fired=fired,
            first_fire_offset_min=first_fire,
        )

    def test_summary_columns(self) -> None:
        results = [self._result("baseline", False, None), self._result("spike", True, 30.0)]
        df = summary(results)
        assert list(df.columns) == ["scenario", "description", "fired", "first_fire_min"]

    def test_summary_row_count(self) -> None:
        results = [self._result("a", False, None), self._result("b", True, 5.0)]
        assert len(summary(results)) == 2

    def test_summary_fired_values(self) -> None:
        results = [self._result("a", False, None), self._result("b", True, 5.0)]
        df = summary(results)
        assert df.loc[0, "fired"] == False  # noqa: E712 — np.bool_ doesn't support `is`
        assert df.loc[1, "fired"] == True  # noqa: E712
        assert df.loc[1, "first_fire_min"] == pytest.approx(5.0)


class TestMcSummary:
    def _mc_result(self, name: str, fire_rate: float, offsets: list[float]) -> MonteCarloResult:
        return MonteCarloResult(
            scenario_name=name,
            description="test scenario",
            n_runs=100,
            fire_rate=fire_rate,
            first_fire_offsets=offsets,
            runs=[],
        )

    def test_columns(self) -> None:
        df = mc_summary([self._mc_result("baseline", 0.0, [])])
        assert list(df.columns) == ["scenario", "n_runs", "fire_rate", "median_first_fire_min"]

    def test_fire_rate_stored_as_float(self) -> None:
        df = mc_summary([self._mc_result("baseline", 0.05, [10.0, 20.0])])
        assert df.loc[0, "fire_rate"] == pytest.approx(0.05)

    def test_median_first_fire_min_none_when_never_fired(self) -> None:
        df = mc_summary([self._mc_result("baseline", 0.0, [])])
        assert df.loc[0, "median_first_fire_min"] is None

    def test_median_first_fire_min_correct(self) -> None:
        df = mc_summary([self._mc_result("spike", 1.0, [10.0, 20.0, 30.0])])
        assert df.loc[0, "median_first_fire_min"] == pytest.approx(20.0)


class TestMonteCarlo:
    def _factory(self, rng: np.random.Generator) -> Scenario:
        ts = np.array([1000, 1060, 1120], dtype=np.int64)
        return Scenario(
            name="test",
            description="test scenario",
            metric="sim.test",
            timestamps=ts,
            values=np.array([0.1, 0.1, 0.1]),
        )

    def _server(self) -> GraphiteServer:
        # Unstarted server — no Docker contact. Network calls are mocked below.
        return GraphiteServer()

    def test_returns_one_result_per_factory(self) -> None:
        mock_series = Series(
            target="sim.test.mc_0",
            datapoints=[Datapoint(value=0.1, timestamp=1000)],
        )
        with (
            patch("metrics_simulation.alert.carbon.write_batch"),
            patch("metrics_simulation.alert.time.sleep"),
            patch("metrics_simulation.alert.graphite.query", return_value=[mock_series]),
        ):
            results = monte_carlo(
                AlertRule(query="{metric}", threshold=0.05),
                [self._factory, self._factory],
                server=self._server(),
                n=5,
                seed=0,
            )
        assert len(results) == 2

    def test_fire_rate_is_fraction(self) -> None:
        mock_series = Series(
            target="sim.test.mc_0",
            datapoints=[Datapoint(value=0.9, timestamp=1000)],
        )
        with (
            patch("metrics_simulation.alert.carbon.write_batch"),
            patch("metrics_simulation.alert.time.sleep"),
            patch("metrics_simulation.alert.graphite.query", return_value=[mock_series]),
        ):
            results = monte_carlo(
                AlertRule(query="{metric}", threshold=0.05),
                [self._factory],
                server=self._server(),
                n=10,
                seed=0,
            )
        assert results[0].fire_rate == pytest.approx(1.0)
        assert len(results[0].first_fire_offsets) == 10

    def test_unique_metric_paths_per_run(self) -> None:
        written: list[str] = []

        def capture_batch(series, host, port):  # type: ignore[no-untyped-def]
            written.extend(metric for metric, _ in series)

        mock_series = Series(target="x", datapoints=[])
        with (
            patch("metrics_simulation.alert.carbon.write_batch", side_effect=capture_batch),
            patch("metrics_simulation.alert.time.sleep"),
            patch("metrics_simulation.alert.graphite.query", return_value=[mock_series]),
        ):
            monte_carlo(
                AlertRule(query="{metric}", threshold=0.05),
                [self._factory],
                server=self._server(),
                n=3,
                seed=0,
            )

        assert written == ["sim.test_mc0", "sim.test_mc1", "sim.test_mc2"]
