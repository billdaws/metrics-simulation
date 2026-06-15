import time
from dataclasses import dataclass

import numpy as np


@dataclass
class Scenario:
    name: str
    description: str
    metric: str
    timestamps: np.ndarray  # int64 unix timestamps, one per minute
    values: np.ndarray      # float64


def _timestamps(duration_min: int) -> np.ndarray:
    now = int(time.time())
    start = now - duration_min * 60
    return np.arange(start, now, 60, dtype=np.int64)


def _ar1_noise(size: int, std: float, phi: float, rng: np.random.Generator) -> np.ndarray:
    # AR(1): x[t] = phi * x[t-1] + epsilon[t]
    # epsilon_std is scaled so the marginal std of x equals `std`.
    epsilon_std = std * np.sqrt(1.0 - phi ** 2)
    noise = np.zeros(size)
    noise[0] = rng.normal(0.0, std)
    for i in range(1, size):
        noise[i] = phi * noise[i - 1] + rng.normal(0.0, epsilon_std)
    return noise


def baseline(
    duration_min: int = 120,
    rate: float = 0.02,
    noise_std: float = 0.003,
    phi: float = 0.7,
    metric: str = "sim.baseline",
    rng: np.random.Generator | None = None,
) -> Scenario:
    rng = rng or np.random.default_rng()
    ts = _timestamps(duration_min)
    values = (rate + _ar1_noise(len(ts), noise_std, phi, rng)).clip(0)
    return Scenario(
        name="baseline",
        description=f"Stable ~{rate:.0%} error rate. Alert should not fire.",
        metric=metric,
        timestamps=ts,
        values=values,
    )


def spike(
    duration_min: int = 120,
    spike_at_min: int = 60,
    spike_value: float = 1.0,
    base_rate: float = 0.02,
    noise_std: float = 0.003,
    phi: float = 0.7,
    metric: str = "sim.spike",
    rng: np.random.Generator | None = None,
) -> Scenario:
    rng = rng or np.random.default_rng()
    ts = _timestamps(duration_min)
    values = (base_rate + _ar1_noise(len(ts), noise_std, phi, rng)).clip(0)
    values[spike_at_min] = spike_value
    return Scenario(
        name="spike",
        description=f"Single-point 100% error rate at minute {spike_at_min}. Tests sensitivity to outliers.",
        metric=metric,
        timestamps=ts,
        values=values,
    )


def gradual_creep(
    duration_min: int = 120,
    creep_start_min: int = 60,
    start_rate: float = 0.02,
    end_rate: float = 0.15,
    noise_std: float = 0.003,
    phi: float = 0.7,
    metric: str = "sim.gradual_creep",
    rng: np.random.Generator | None = None,
) -> Scenario:
    rng = rng or np.random.default_rng()
    ts = _timestamps(duration_min)
    t = np.arange(len(ts))

    creep_len = max(duration_min - creep_start_min, 1)
    slope = (end_rate - start_rate) / creep_len
    trend = np.where(t < creep_start_min, start_rate, start_rate + slope * (t - creep_start_min))

    values = (trend + _ar1_noise(len(ts), noise_std, phi, rng)).clip(0)
    return Scenario(
        name="gradual_creep",
        description=(
            f"Error rate rises from {start_rate:.0%} to {end_rate:.0%} "
            f"starting at minute {creep_start_min}."
        ),
        metric=metric,
        timestamps=ts,
        values=values,
    )


def noisy_baseline(
    duration_min: int = 120,
    rate: float = 0.02,
    noise_std: float = 0.012,
    phi: float = 0.92,
    metric: str = "sim.noisy_baseline",
    rng: np.random.Generator | None = None,
) -> Scenario:
    rng = rng or np.random.default_rng()
    ts = _timestamps(duration_min)
    values = (rate + _ar1_noise(len(ts), noise_std, phi, rng)).clip(0)
    return Scenario(
        name="noisy_baseline",
        description=(
            f"Stable ~{rate:.0%} error rate with higher noise (std={noise_std}, φ={phi}). "
            "Alert should not fire."
        ),
        metric=metric,
        timestamps=ts,
        values=values,
    )


def zero_drop(
    duration_min: int = 120,
    drop_at_min: int = 60,
    base_rate: float = 0.02,
    noise_std: float = 0.003,
    phi: float = 0.7,
    metric: str = "sim.zero_drop",
    rng: np.random.Generator | None = None,
) -> Scenario:
    rng = rng or np.random.default_rng()
    ts = _timestamps(duration_min)
    t = np.arange(len(ts))
    trend = np.where(t < drop_at_min, base_rate, 0.0)
    values = (trend + _ar1_noise(len(ts), noise_std, phi, rng)).clip(0)
    values[t >= drop_at_min] = 0.0  # noise shouldn't re-introduce signal after the drop
    return Scenario(
        name="zero_drop",
        description=(
            f"Error rate (and likely traffic) drops to 0 at minute {drop_at_min}. "
            "A naive alert may fire or stay silent incorrectly."
        ),
        metric=metric,
        timestamps=ts,
        values=values,
    )
