import time

import numpy as np
import pytest

from metrics_simulation import scenarios
from metrics_simulation.scenarios import Scenario


def test_baseline_shape() -> None:
    s = scenarios.baseline(duration_min=60)
    assert isinstance(s, Scenario)
    assert len(s.timestamps) == len(s.values)
    assert len(s.timestamps) == 60


def test_baseline_timestamps_recent() -> None:
    s = scenarios.baseline(duration_min=10)
    now = int(time.time())
    assert s.timestamps[-1] <= now
    assert s.timestamps[0] >= now - 10 * 60 - 5  # allow a few seconds of skew


def test_baseline_values_nonnegative() -> None:
    rng = np.random.default_rng(0)
    s = scenarios.baseline(duration_min=60, rng=rng)
    assert (s.values >= 0).all()


def test_spike_has_spike_value() -> None:
    rng = np.random.default_rng(0)
    s = scenarios.spike(duration_min=60, spike_at_min=30, spike_value=1.0, rng=rng)
    assert s.values[30] == pytest.approx(1.0)


def test_gradual_creep_increases() -> None:
    rng = np.random.default_rng(0)
    s = scenarios.gradual_creep(
        duration_min=60, creep_start_min=30, start_rate=0.02, end_rate=0.15, rng=rng
    )
    pre_creep_mean = s.values[:30].mean()
    post_creep_mean = s.values[30:].mean()
    assert post_creep_mean > pre_creep_mean


def test_zero_drop_is_zero_after_drop() -> None:
    rng = np.random.default_rng(0)
    s = scenarios.zero_drop(duration_min=60, drop_at_min=30, rng=rng)
    assert (s.values[30:] == 0.0).all()


def test_timestamps_are_one_minute_apart() -> None:
    s = scenarios.baseline(duration_min=30)
    diffs = np.diff(s.timestamps)
    assert (diffs == 60).all()
