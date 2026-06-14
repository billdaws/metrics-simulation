import numpy as np
import pytest

from metrics_simulation.scenarios import Scenario
from metrics_simulation.zscore import (
    ZScoreAlertRule,
    ZScoreResult,
    ZScoreMCResult,
    _detect_windows,
    simulate_zscore,
    zscore_summary,
    monte_carlo_zscore,
    zscore_mc_summary,
)


def _scenario(values: list[float], name: str = "test", metric: str = "sim.test") -> Scenario:
    ts = np.arange(len(values), dtype=np.int64) * 60
    return Scenario(
        name=name,
        description=f"{name} scenario",
        metric=metric,
        timestamps=ts,
        values=np.array(values, dtype=float),
    )


def _rule(sigma: float = 3.0) -> ZScoreAlertRule:
    # Small windows so tests are fast; burn_in == long_window so the rolling baseline
    # is fully populated before the alert is evaluated.
    return ZScoreAlertRule(short_window=1, long_window=20, sigma_threshold=sigma, burn_in_min=20)


class TestDetectWindows:
    def test_no_breach_returns_empty(self) -> None:
        z = np.array([0.0, 1.0, 2.9])
        assert _detect_windows(z, threshold=3.0) == []

    def test_all_breach_single_window(self) -> None:
        z = np.array([4.0, 5.0, 6.0])
        windows = _detect_windows(z, threshold=3.0)
        assert windows == [(0.0, 2.0)]

    def test_fires_then_resolves(self) -> None:
        z = np.array([4.0, 4.0, 1.0, 1.0])
        windows = _detect_windows(z, threshold=3.0)
        assert windows == [(0.0, 1.0)]

    def test_fires_resolves_fires_again(self) -> None:
        z = np.array([4.0, 1.0, 4.0])
        windows = _detect_windows(z, threshold=3.0)
        assert len(windows) == 2
        assert windows[0] == (0.0, 0.0)
        assert windows[1] == (2.0, 2.0)

    def test_negative_breach_fires(self) -> None:
        # |z| > threshold for negative z
        z = np.array([-4.0, -4.0])
        assert _detect_windows(z, threshold=3.0) == [(0.0, 1.0)]

    def test_window_open_at_end(self) -> None:
        z = np.array([1.0, 4.0])
        windows = _detect_windows(z, threshold=3.0)
        assert windows == [(1.0, 1.0)]


class TestSimulateZscore:
    def test_stable_series_never_fires(self) -> None:
        # Constant baseline → mean == value, std ≈ 0, z == 0 throughout
        values = [0.5] * 25
        results = simulate_zscore(_rule(), [_scenario(values)])
        assert not results[0].fired

    def test_large_spike_fires(self) -> None:
        # 20-point stable burn-in at 0.5, then spike to 100.
        # With long_window=20 the rolling mean ≈ 5.5 and std ≈ 22; z ≈ 4.3 > 3.
        values = [0.5] * 20 + [100.0]
        results = simulate_zscore(_rule(), [_scenario(values)])
        assert results[0].fired

    def test_spike_fires_at_correct_test_minute(self) -> None:
        # Spike at the 2nd point of the test window (test minute 1).
        values = [0.5] * 20 + [0.5, 100.0, 0.5]
        results = simulate_zscore(_rule(), [_scenario(values)])
        assert results[0].fired
        assert results[0].first_fire_test_min == pytest.approx(1.0)

    def test_large_drop_fires(self) -> None:
        # 20-point burn-in at 1.0, then drop to 0.
        # z = (0 - ~0.95) / ~0.22 ≈ -4.3 → |z| > 3.
        values = [1.0] * 20 + [0.0]
        results = simulate_zscore(_rule(), [_scenario(values)])
        assert results[0].fired

    def test_returns_one_result_per_scenario(self) -> None:
        s1 = _scenario([0.5] * 25, name="a", metric="sim.a")
        s2 = _scenario([0.5] * 25, name="b", metric="sim.b")
        results = simulate_zscore(_rule(), [s1, s2])
        assert len(results) == 2

    def test_test_window_length(self) -> None:
        # 20 burn-in + 5 test points → test window should have 5 entries
        values = [0.5] * 25
        results = simulate_zscore(_rule(), [_scenario(values)])
        assert len(results[0].test_minutes) == 5
        assert len(results[0].test_raw) == 5
        assert len(results[0].test_zscore) == 5

    def test_test_minutes_start_at_zero(self) -> None:
        values = [0.5] * 25
        results = simulate_zscore(_rule(), [_scenario(values)])
        assert results[0].test_minutes[0] == pytest.approx(0.0)

    def test_test_minutes_one_minute_apart(self) -> None:
        values = [0.5] * 25
        results = simulate_zscore(_rule(), [_scenario(values)])
        mins = results[0].test_minutes
        diffs = np.diff(mins)
        assert np.allclose(diffs, 1.0)

    def test_no_firing_windows_when_stable(self) -> None:
        values = [0.5] * 25
        results = simulate_zscore(_rule(), [_scenario(values)])
        assert results[0].firing_windows == []

    def test_firing_window_present_on_spike(self) -> None:
        values = [0.5] * 20 + [100.0]
        results = simulate_zscore(_rule(), [_scenario(values)])
        assert len(results[0].firing_windows) == 1

    def test_custom_sigma_threshold(self) -> None:
        # With threshold=10 the spike at 100 (z≈4.3) should NOT fire
        high_rule = ZScoreAlertRule(short_window=1, long_window=20, sigma_threshold=10.0, burn_in_min=20)
        values = [0.5] * 20 + [100.0]
        results = simulate_zscore(high_rule, [_scenario(values)])
        assert not results[0].fired


class TestZscoreSummary:
    def test_columns(self) -> None:
        values = [0.5] * 25
        results = simulate_zscore(_rule(), [_scenario(values)])
        df = zscore_summary(results)
        assert list(df.columns) == ["scenario", "description", "fired", "first_fire_test_min"]

    def test_row_count(self) -> None:
        s1 = _scenario([0.5] * 25, name="a", metric="sim.a")
        s2 = _scenario([0.5] * 25, name="b", metric="sim.b")
        df = zscore_summary(simulate_zscore(_rule(), [s1, s2]))
        assert len(df) == 2

    def test_fired_column_values(self) -> None:
        stable = _scenario([0.5] * 25, name="stable", metric="sim.stable")
        spiked = _scenario([0.5] * 20 + [100.0], name="spike", metric="sim.spike")
        df = zscore_summary(simulate_zscore(_rule(), [stable, spiked]))
        assert df.loc[0, "fired"] == False  # noqa: E712
        assert df.loc[1, "fired"] == True   # noqa: E712

    def test_first_fire_none_when_no_fire(self) -> None:
        df = zscore_summary(simulate_zscore(_rule(), [_scenario([0.5] * 25)]))
        assert df.loc[0, "first_fire_test_min"] is None

    def test_first_fire_populated_on_fire(self) -> None:
        df = zscore_summary(simulate_zscore(_rule(), [_scenario([0.5] * 20 + [100.0])]))
        assert df.loc[0, "first_fire_test_min"] == pytest.approx(0.0)


class TestMonteCarloZscore:
    def _stable_factory(self, rng: np.random.Generator) -> Scenario:
        return _scenario([0.5] * 25, metric="sim.stable")

    def _spike_factory(self, rng: np.random.Generator) -> Scenario:
        return _scenario([0.5] * 20 + [100.0], metric="sim.spike")

    def test_returns_one_result_per_factory(self) -> None:
        results = monte_carlo_zscore(_rule(), [self._stable_factory, self._spike_factory], n=3, seed=0)
        assert len(results) == 2

    def test_fire_rate_zero_for_stable(self) -> None:
        results = monte_carlo_zscore(_rule(), [self._stable_factory], n=10, seed=0)
        assert results[0].fire_rate == 0.0

    def test_fire_rate_one_for_guaranteed_spike(self) -> None:
        results = monte_carlo_zscore(_rule(), [self._spike_factory], n=10, seed=0)
        assert results[0].fire_rate == 1.0

    def test_fire_rate_is_fraction_of_n(self) -> None:
        results = monte_carlo_zscore(_rule(), [self._spike_factory], n=7, seed=0)
        assert results[0].n_runs == 7
        assert results[0].fire_rate == pytest.approx(1.0)

    def test_first_fire_offsets_populated_on_fire(self) -> None:
        results = monte_carlo_zscore(_rule(), [self._spike_factory], n=5, seed=0)
        assert len(results[0].first_fire_offsets) == 5
        for offset in results[0].first_fire_offsets:
            assert offset == pytest.approx(0.0)

    def test_first_fire_offsets_empty_when_never_fires(self) -> None:
        results = monte_carlo_zscore(_rule(), [self._stable_factory], n=5, seed=0)
        assert results[0].first_fire_offsets == []

    def test_scenario_name_captured(self) -> None:
        results = monte_carlo_zscore(_rule(), [self._stable_factory], n=2, seed=0)
        assert results[0].scenario_name == "test"

    def test_runs_list_length(self) -> None:
        results = monte_carlo_zscore(_rule(), [self._stable_factory], n=7, seed=0)
        assert len(results[0].runs) == 7


class TestZscoreMcSummary:
    def test_columns(self) -> None:
        results = monte_carlo_zscore(_rule(), [lambda rng: _scenario([0.5] * 25)], n=3, seed=0)
        df = zscore_mc_summary(results)
        assert list(df.columns) == ["scenario", "n_runs", "fire_rate", "median_first_fire_test_min"]

    def test_fire_rate_in_summary(self) -> None:
        results = monte_carlo_zscore(_rule(), [lambda rng: _scenario([0.5] * 20 + [100.0])], n=4, seed=0)
        df = zscore_mc_summary(results)
        assert df.loc[0, "fire_rate"] == pytest.approx(1.0)

    def test_median_none_when_never_fires(self) -> None:
        results = monte_carlo_zscore(_rule(), [lambda rng: _scenario([0.5] * 25)], n=4, seed=0)
        df = zscore_mc_summary(results)
        assert df.loc[0, "median_first_fire_test_min"] is None

    def test_median_populated_when_fires(self) -> None:
        results = monte_carlo_zscore(_rule(), [lambda rng: _scenario([0.5] * 20 + [100.0])], n=4, seed=0)
        df = zscore_mc_summary(results)
        assert df.loc[0, "median_first_fire_test_min"] == pytest.approx(0.0)
