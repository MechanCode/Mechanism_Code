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
    """Poisson sampler over classification windows at a configurable stride."""

    def __init__(self, dataset, batch_size, size=None, stride=None, seed=None):
        self.seq_len = size
        self.batch_size = batch_size
        self.dataset = dataset
        self.data_size = len(dataset)
        self.stride = self.seq_len if stride is None else stride
        if self.stride <= 0:
            raise ValueError(f'sampling stride must be positive, got {self.stride}')
        if self.batch_size <= 0:
            raise ValueError(f'batch_size must be positive, got {self.batch_size}')

        self.n_segments = ((self.data_size - 1) // self.stride + 1) if self.data_size else 0

        self.sampling_rate = (self.batch_size / self.n_segments) if self.n_segments > 0 else 0.0
        self.sampling_rate = max(0.0, min(1.0, self.sampling_rate))
        self.rng = random.Random(seed)
        
        self.planned_batches = math.ceil(self.n_segments / self.batch_size) if self.n_segments else 0

    def __iter__(self):
        if self.sampling_rate <= 0 or self.n_segments <= 0:
            return iter(())

        for _ in range(self.planned_batches):
            indices = []
            for seg in range(self.n_segments):
                if self.rng.random() < self.sampling_rate:
                    indices.append(seg * self.stride)
            # print("batch size: ", len(indices))
            if indices:  
                yield indices

    def __len__(self):
        return self.planned_batches
    
    def sampling_value(self):
        return self.sampling_rate


class SpacedSamplingWithFixedSize(BatchSampler):
    """Spaced sampler over classification windows at a configurable stride."""

    def __init__(self, dataset, batch_size, lam=1, size=None, stride=None, seed=None):
        self.seq_len = size
        self.batch_size = batch_size
        self.dataset = dataset
        self.data_size = len(dataset)
        self.lam = lam
        self.stride = self.seq_len if stride is None else stride
        if self.stride <= 0:
            raise ValueError(f'sampling stride must be positive, got {self.stride}')
        if self.batch_size <= 0:
            raise ValueError(f'batch_size must be positive, got {self.batch_size}')
        self.rng = random.Random(seed)

    @property
    def n_segments(self):
        return ((self.data_size - 1) // self.stride + 1) if self.data_size else 0

    def _build_spacing_interval(self):
        """
        Divide the window index space evenly into floor(batch_size / lam)
        intervals. Activate each interval with probability lam, then randomly
        sample one window index from every activated interval.
        Return a list of window IDs for one batch (consistent with PoissonSampler).
        """
        n = self.n_segments
        # Keep the actual interval count identical to lambda selection and
        # privacy accounting, both of which use floor(batch_size / lambda).
        k = math.floor(self.batch_size / self.lam) if self.lam > 0 else self.batch_size
        if k <= 0 or n <= 0:
            return []

        base, rem = divmod(n, k)
        intervals = []
        cur = 0
        for i in range(k):
            size = base + (1 if i < rem else 0)
            start = cur
            end = cur + size
            if size > 0:
                intervals.append((start, end))
            cur += size

        sampled_win_ids = []
        for start, end in intervals:
            if self.lam >= 1:
                idx = self.rng.randint(start, end - 1)
                sampled_win_ids.append(idx)
                continue
            else:
                if self.rng.random() <= self.lam:
                    idx = self.rng.randint(start, end - 1)
                    sampled_win_ids.append(idx)

        return [idx * self.stride for idx in sampled_win_ids]

    def __iter__(self):
        planned_batches = math.ceil(self.n_segments / self.batch_size) if self.n_segments else 0
        for _ in range(planned_batches):
            batch_idx = self._build_spacing_interval()
            if batch_idx:
                yield batch_idx
    
    def __len__(self):
        planned_batches = math.ceil(self.n_segments / self.batch_size) if self.n_segments else 0
        return planned_batches
