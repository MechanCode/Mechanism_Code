"""data_sampler.py

Sampling utilities for differential privacy training with time series data.

This module provides custom batch samplers that implement privacy-preserving
sampling strategies:
- PoissonSamplerWithOutReplacement: Poisson subsampling for DP-SGD
- SpacedSamplingWithFixedSize: Stratified spaced sampling for improved privacy

DP notes:
- These samplers provide privacy amplification via subsampling.
- Noise injection occurs in the experiment files (exp/exp_main*.py), not here.
- For DP-SGD (Poisson), noise sigma = C after clipping.
- For spaced sampling (stratified), noise sigma = 2*C after clipping.

Reproducibility:
- Both samplers accept an optional `seed` parameter to ensure consistent
  sampling across different experiments with the same seed.
- The seed is used to initialize a local Random instance, isolating the
  sampler's randomness from the global random state.
"""

from torch.utils.data import Sampler, BatchSampler
import random
import math
from collections import defaultdict


# class PoissonSamplerWithReplacement(Sampler):
#     def __init__(self, dataset, w, r, sensitivity, size=None):
#         if size == None: 
#             self.sub_seq_len = 24*4*4
#             self.pred_len = 24*4
#         else:
#             self.sub_seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         self.dataset = dataset
#         self.r = r
#         self.L = self.sub_seq_len+self.pred_len
#         self.seq_len = len(dataset)
#         self.w = w
#         self.sensitivity = sensitivity

#     def __iter__(self):
#         indices = []
#         self.R = math.floor((self.seq_len-self.L-1)/self.r)
#         M = (self.L+self.w-2)/self.r
#         self.p = self.sensitivity/M

#         for idx in range(math.floor(self.R)):
#             if random.random() < self.p:
#                 indices.append(idx*self.r)
#         return iter(indices)

#     def __len__(self):
#         return int(self.R * self.p)
    

class PoissonSamplerWithOutReplacement(BatchSampler):
    """Poisson subsampler for DP-SGD with non-overlapping time series windows.

    Implements Poisson subsampling where each non-overlapping segment is
    independently included in a batch with probability p = batch_size/n_segments.

    DP notes:
    - This sampler provides privacy amplification via Poisson subsampling.
    - Actual noise injection occurs in exp/exp_main.py (sigma = C).

    Attributes:
        sampling_rate: Probability p of including each segment in a batch.
        n_segments: Total number of non-overlapping windows in the dataset.
        rng: Local Random instance for reproducible sampling.
    """

    def __init__(self, dataset, batch_size, size=None, seed=None):
        """Initialize Poisson sampler.

        Parameters:
            dataset: Time series dataset to sample from.
            batch_size: Expected number of samples per batch.
            size: Tuple of (seq_len, label_len, pred_len) defining window sizes.
            seed: Optional random seed for reproducible sampling. If None, uses
                  global random state.
        """
        # Window length configuration
        self.sub_seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.batch_size = batch_size
        self.dataset = dataset
        self.L = self.sub_seq_len + self.pred_len  # Total window length
        self.data_size = len(dataset)

        # DP: Number of non-overlapping segments (window space), each starting at i * L
        self.n_segments = self.data_size // self.L 

        # DP: Sampling rate p such that expected batch size = batch_size
        self.sampling_rate = self.batch_size / self.n_segments
        
        # Number of batches per epoch, approximately n_segments / batch_size
        self.planned_batches = math.ceil(self.n_segments / self.batch_size)
        
        # Local random generator for reproducibility
        self.rng = random.Random(seed) if seed is not None else random.Random()

    def __iter__(self):
        """Generate batches via Poisson subsampling.

        Each segment is independently included with probability sampling_rate.

        Yields:
            List of indices representing a batch of window start positions.
        """
        # DP: Poisson sampling - each non-overlapping segment included with prob p, without replacement
        if self.sampling_rate <= 0 or self.n_segments <= 0:
            return iter(())

        for _ in range(self.planned_batches):
            indices = []
            for seg in range(self.n_segments):
                if self.rng.random() < self.sampling_rate:
                    indices.append(seg * self.L)
            # print("batch size: ", len(indices))
            if indices:  # 只有当indices非空时才yield
                yield indices

    def __len__(self):
        return self.planned_batches
    
    def sampling_value(self):
        return self.sampling_rate


class SpacedSamplingWithFixedSize(BatchSampler):
    """Stratified spaced sampling for differential privacy with time series.

    Divides the window index space into k intervals and samples one window
    per interval, providing privacy amplification through stratified sampling.

    DP notes:
    - This sampler provides privacy amplification via stratified sampling.
    - Actual noise injection occurs in exp/exp_main_spaced_sampling.py (sigma = 2*C).

    Attributes:
        lam: Lambda parameter controlling sampling rate within intervals.
        L: Total window length (seq_len + pred_len).
        rng: Local Random instance for reproducible sampling.
    """

    def __init__(self, dataset, batch_size, lam=1, size=None, seed=None):
        """Initialize spaced sampler.

        Parameters:
            dataset: Time series dataset to sample from.
            batch_size: Expected number of samples per batch.
            lam: Lambda parameter for sampling rate control (default 1).
            size: Tuple of (seq_len, label_len, pred_len) defining window sizes.
            seed: Optional random seed for reproducible sampling. If None, uses
                  global random state.
        """
        self.sub_seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.batch_size = batch_size
        self.dataset = dataset
        self.L = self.sub_seq_len + self.pred_len  # sequence length
        self.data_size = len(dataset)
        self.lam = lam  # window sampling rate
        
        # Local random generator for reproducibility
        self.rng = random.Random(seed) if seed is not None else random.Random()

    def _build_spacing_interval(self):
        n = self.data_size // self.L  
        k = math.ceil(self.batch_size * 1 / self.lam)
        if k <= 0 or n <= 0:
            return []

        base, rem = divmod(n, k)
        intervals = []
        cur = 0
        for i in range(k):
            size = base + (1 if i < rem else 0)
            start = cur
            end = cur + size
            intervals.append((start, end))
            cur += size

        sampled_win_idx = []
        for start, end in intervals:
            if self.lam >= 1:
                idx = self.rng.randint(start, end - 1)
                sampled_win_idx.append(idx)
                continue
            else:
                if self.rng.random() > self.lam:
                    continue
                idx = self.rng.randint(start, end - 1)
                sampled_win_idx.append(idx)

        sampled_indices = [idx * self.L for idx in sampled_win_idx[:k]]
        return sampled_indices

    def __iter__(self):
        n_windows = self.data_size // self.L
        planned_batches = max(1, math.ceil(n_windows / self.batch_size)) if self.batch_size > 0 else 0
        for _ in range(planned_batches):
            batch_idx = self._build_spacing_interval()
            if batch_idx:
                yield batch_idx

    def __len__(self):
        n_windows = self.data_size // self.L
        return int(math.ceil(n_windows / self.batch_size))

