from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Partition:
    sizes: tuple[int, ...]
    activation: float

    @property
    def bounds(self):
        return np.r_[0, np.cumsum(self.sizes, dtype=np.int64)]

    @property
    def expected_batch_size(self):
        return len(self.sizes) * self.activation


def pss_partition(candidates, batch_size, lam, mode="forecasting"):
    if not 1 <= batch_size <= candidates or not 0 < lam <= 1:
        raise ValueError("Require 1<=K<=M and 0<lambda<=1")
    if mode == "forecasting":
        count, activation = math.ceil(batch_size / lam), lam
    elif mode == "paper":
        count = math.floor(batch_size / lam)
        activation = batch_size / count
    else:
        raise ValueError(f"Unknown PSS mode: {mode}")
    if not batch_size <= count <= candidates:
        raise ValueError("PSS must have K<=N<=M nonempty windows")
    size, remainder = divmod(candidates, count)

    sizes = tuple(size + int(i < remainder if mode == "forecasting"
                             else i >= count - remainder) for i in range(count))
    return Partition(sizes, activation)


def lambda_grid(candidates, batch_size, step):
    if not 1 <= batch_size <= candidates or not 0 < step <= 1:
        raise ValueError("Invalid candidate count, batch size or lambda grid step")
    lower = 1.0 / (candidates // batch_size)
    values = [lower + i * step for i in range(math.floor((1 - lower) / step) + 1)]
    return sorted(set(min(1.0, v) for v in values) | {1.0})


class BatchSampler:
    def __init__(self, method, candidates, batch_size, rng, partition=None):
        if method not in ("poisson", "structured", "pss"):
            raise ValueError(f"Unknown method: {method}")
        if not 1 <= batch_size <= candidates:
            raise ValueError("Require 1<=K<=M")
        if method == "pss" and (partition is None or sum(partition.sizes) != candidates):
            raise ValueError("PSS needs the accounted partition of all candidates")
        self.method, self.candidates, self.batch_size = method, candidates, batch_size
        self.rng, self.partition = rng, partition
        self.bounds = None if partition is None else partition.bounds

    def draw(self):
        if self.method == "poisson":
            return np.flatnonzero(self.rng.random(self.candidates) < self.batch_size / self.candidates)
        if self.method == "structured":
            return self.rng.integers(0, self.candidates, size=self.batch_size, dtype=np.int64)
        active = np.flatnonzero(self.rng.random(len(self.partition.sizes)) < self.partition.activation)
        return np.asarray([self.rng.integers(self.bounds[i], self.bounds[i + 1])
                           for i in active], dtype=np.int64)
