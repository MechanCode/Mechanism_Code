from __future__ import annotations

import numpy as np


def as_2d_float64(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D [samples, dimensions] array, got {array.shape}")
    if array.shape[0] < 1:
        raise ValueError("At least one gradient is required")
    if not np.isfinite(array).all():
        raise ValueError("Gradient vectors contain NaN or infinity")
    return array


def empirical_moments(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    array = as_2d_float64(values)
    mean = array.mean(axis=0)
    centered = array - mean
    covariance = centered.T @ centered / array.shape[0]
    return mean, (covariance + covariance.T) / 2.0


def population_covariance(values: np.ndarray) -> np.ndarray:

    return empirical_moments(values)[1]


def covariance_metrics(covariance: np.ndarray) -> dict[str, float]:

    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be square")
    symmetric = (matrix + matrix.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(symmetric)
    nonnegative = np.clip(eigenvalues, 0.0, None)
    trace = float(np.trace(symmetric))
    squared_sum = float(np.square(eigenvalues).sum())
    return {
        "trace": trace,
        "frobenius": float(np.sqrt(squared_sum)),
        "spectral": float(eigenvalues[-1]) if eigenvalues.size else 0.0,
        "minimum_eigenvalue": float(eigenvalues[0]) if eigenvalues.size else 0.0,
        "effective_rank": float(trace * trace / np.square(nonnegative).sum())
        if np.square(nonnegative).sum() > 0
        else 0.0,
    }
