import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .covariance import covariance_metrics, population_covariance


METHODS = ("baseline", "poisson")
DEFAULT_METHODS = METHODS


@dataclass(frozen=True)
class SamplingResult:
    method: str
    gradient_estimators: np.ndarray
    batch_sizes: np.ndarray
    covariance: np.ndarray
    metrics: dict[str, float]
    bootstrap_trace_ci95: tuple[float, float]


def poisson_window_count(
    batch_size: int, poisson_lambda: float, rounding: str = "ceil"
) -> int:

    if batch_size < 1:
        raise ValueError("batch_size K must be positive")
    if not 0.0 < poisson_lambda <= 1.0:
        raise ValueError("poisson_lambda must be in (0, 1]")
    if rounding not in ("ceil", "floor"):
        raise ValueError("poisson window rounding must be ceil or floor")
    return int(getattr(math, rounding)(batch_size / poisson_lambda))


def balanced_temporal_windows(sample_count: int, window_count: int) -> list[np.ndarray]:

    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if window_count < 1:
        raise ValueError("window_count must be positive")
    if window_count > sample_count:
        raise ValueError(
            f"Cannot form {window_count} non-empty windows from {sample_count} gradients. "
            "Increase lambda or reduce K."
        )
    return [part.astype(np.int64, copy=False) for part in np.array_split(np.arange(sample_count), window_count)]


def draw_baseline_indicator_mask(
    sample_count: int,
    batch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:

    if sample_count < batch_size:
        raise ValueError("sample_count must be at least K")
    q = batch_size / sample_count
    return rng.random(sample_count) < q


def draw_gradient_estimator(
    method: str,
    gradients: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    poisson_lambda: float = 0.5,
    poisson_window_rounding: str = "ceil",
) -> tuple[np.ndarray, int]:

    values = np.asarray(gradients, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("gradients must have shape [samples, dimensions]")
    if values.shape[0] < batch_size:
        raise ValueError("The gradient population must contain at least K samples")
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")


    if method == "poisson":
        window_count = poisson_window_count(
            batch_size, poisson_lambda, poisson_window_rounding
        )
        windows = balanced_temporal_windows(values.shape[0], window_count)
        xi = poisson_lambda
        included = rng.random(window_count) < xi
        selected: list[np.ndarray] = []
        for members in (windows[index] for index in np.flatnonzero(included)):
            selected.append(values[int(members[rng.integers(0, members.size)])])
        estimator = np.sum(selected, axis=0) / batch_size if selected else np.zeros(values.shape[1])
        return estimator, len(selected)


    included = draw_baseline_indicator_mask(values.shape[0], batch_size, rng)
    selected_count = int(included.sum())
    estimator = (
        values[included].sum(axis=0) / batch_size
        if selected_count
        else np.zeros(values.shape[1])
    )
    return estimator, selected_count


def _trace_bootstrap_ci(
    estimators: np.ndarray,
    rng: np.random.Generator,
    repetitions: int,
) -> tuple[float, float]:
    if repetitions <= 0 or estimators.shape[0] < 2:
        return (float("nan"), float("nan"))
    traces = _bootstrap_population_covariance_traces(
        estimators, rng, repetitions
    )
    low, high = np.quantile(traces, [0.025, 0.975])
    return float(low), float(high)


def _bootstrap_population_covariance_traces(
    values: np.ndarray,
    rng: np.random.Generator,
    repetitions: int,
) -> np.ndarray:

    array = np.asarray(values, dtype=np.float64)
    row_count = array.shape[0]
    probabilities = np.full(row_count, 1.0 / row_count)
    counts = rng.multinomial(row_count, probabilities, size=repetitions)
    squared_norms = np.einsum("ij,ij->i", array, array)
    mean_squared_norms = counts @ squared_norms / row_count
    means = counts @ array / row_count
    traces = mean_squared_norms - np.einsum("ij,ij->i", means, means)
    return np.maximum(traces, 0.0)


def simulate_method(
    gradients: np.ndarray,
    method: str,
    batch_size: int,
    trials: int,
    seed: int,
    bootstrap_repetitions: int,
    poisson_lambda: float = 0.5,
    poisson_window_rounding: str = "ceil",
) -> SamplingResult:
    values = np.asarray(gradients, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("gradients must have shape [samples, dimensions]")
    if trials < 2:
        raise ValueError("At least two sampling trials are required")
    rng = np.random.default_rng(seed)
    estimators = np.empty((trials, values.shape[1]), dtype=np.float64)
    batch_sizes = np.empty(trials, dtype=np.int64)
    for trial in range(trials):
        estimators[trial], batch_sizes[trial] = draw_gradient_estimator(
            method, values, batch_size, rng, poisson_lambda, poisson_window_rounding
        )
    covariance = population_covariance(estimators)
    bootstrap_rng = np.random.default_rng(seed + 1_000_003)
    ci = _trace_bootstrap_ci(estimators, bootstrap_rng, bootstrap_repetitions)
    return SamplingResult(
        method=method,
        gradient_estimators=estimators,
        batch_sizes=batch_sizes,
        covariance=covariance,
        metrics=covariance_metrics(covariance),
        bootstrap_trace_ci95=ci,
    )


def run_sampling_experiment(
    gradients: np.ndarray,
    methods: Iterable[str] = DEFAULT_METHODS,
    batch_size: int = 32,
    trials: int = 1_000,
    seed: int = 42,
    bootstrap_repetitions: int = 500,
    shared_method_seed: bool = False,
    poisson_lambda: float = 0.5,
    poisson_window_rounding: str = "ceil",
) -> dict[str, SamplingResult]:

    results: dict[str, SamplingResult] = {}
    for offset, method in enumerate(methods):
        results[method] = simulate_method(
            gradients=gradients,
            method=method,
            batch_size=batch_size,
            trials=trials,
            seed=seed if shared_method_seed else seed + 10_007 * offset,
            bootstrap_repetitions=bootstrap_repetitions,
            poisson_lambda=poisson_lambda,
            poisson_window_rounding=poisson_window_rounding,
        )
    return results


def analytical_sampling_traces(
    gradients: np.ndarray,
    methods: Iterable[str] = DEFAULT_METHODS,
    batch_size: int = 32,
    poisson_lambda: float = 0.5,
    poisson_window_rounding: str = "ceil",
) -> dict[str, float]:

    values = np.asarray(gradients)
    if values.ndim != 2:
        raise ValueError("gradients must have shape [samples, dimensions]")
    sample_count = values.shape[0]
    if sample_count < batch_size:
        raise ValueError("The gradient population must contain at least K samples")
    requested = tuple(methods)
    unknown = set(requested).difference(METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")

    squared_norms = np.einsum(
        "ij,ij->i", values, values, dtype=np.float64
    )
    denominator = float(batch_size * batch_size)
    traces: dict[str, float] = {}
    if "baseline" in requested:
        q = batch_size / sample_count
        traces["baseline"] = (
            q * (1.0 - q) * float(squared_norms.sum()) / denominator
        )
    for method in ("poisson",):
        if method not in requested:
            continue
        window_count = poisson_window_count(
            batch_size, poisson_lambda, poisson_window_rounding
        )
        xi = poisson_lambda
        total = 0.0
        for members in balanced_temporal_windows(sample_count, window_count):
            window = values[members]
            mean_squared_norm = float(squared_norms[members].mean())
            mean = window.mean(axis=0, dtype=np.float64)
            total += (
                xi * mean_squared_norm - xi * xi * float(mean @ mean)
            )
        traces[method] = max(total / denominator, 0.0)
    return {method: traces[method] for method in requested}


def bootstrap_trace_comparison(
    baseline_estimators: np.ndarray,
    method_estimators: np.ndarray,
    repetitions: int = 1_000,
    seed: int = 42,
) -> dict[str, float | list[float]]:
    if repetitions < 1:
        return {
            "difference_ci95": [float("nan"), float("nan")],
            "ratio_ci95": [float("nan"), float("nan")],
            "probability_trace_below_baseline": float("nan"),
        }
    baseline = np.asarray(baseline_estimators, dtype=np.float64)
    method = np.asarray(method_estimators, dtype=np.float64)
    rng = np.random.default_rng(seed)
    baseline_traces = _bootstrap_population_covariance_traces(
        baseline, rng, repetitions
    )
    method_traces = _bootstrap_population_covariance_traces(
        method, rng, repetitions
    )
    differences = method_traces - baseline_traces
    ratios = np.divide(
        method_traces,
        baseline_traces,
        out=np.full(repetitions, np.nan, dtype=np.float64),
        where=baseline_traces > 0,
    )
    finite_ratios = ratios[np.isfinite(ratios)]
    ratio_ci = (
        np.quantile(finite_ratios, [0.025, 0.975])
        if finite_ratios.size
        else np.asarray([np.nan, np.nan])
    )
    return {
        "difference_ci95": [float(value) for value in np.quantile(differences, [0.025, 0.975])],
        "ratio_ci95": [float(value) for value in ratio_ci],
        "probability_trace_below_baseline": float(np.mean(differences < 0)),
    }
