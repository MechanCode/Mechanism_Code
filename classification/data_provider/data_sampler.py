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
    def __init__(self, dataset, batch_size, size=None):
        self.seq_len = size
        self.batch_size = batch_size
        self.dataset = dataset
        self.data_size = len(dataset)

        self.n_segments = self.data_size  

        self.sampling_rate = (self.batch_size / self.n_segments) if self.n_segments > 0 else 0.0
        self.sampling_rate = max(0.0, min(1.0, self.sampling_rate))
        print(self.batch_size, self.n_segments, self.data_size)
        
        self.planned_batches = math.ceil(self.n_segments / self.batch_size)

    def __iter__(self):
        if self.sampling_rate <= 0 or self.n_segments <= 0:
            return iter(())

        for _ in range(self.planned_batches):
            indices = []
            for seg in range(self.n_segments):
                if random.random() < self.sampling_rate:
                    # yield window ids; Dataset_Custom maps id -> offset internally
                    indices.append(seg)
            # print("batch size: ", len(indices))
            if indices:  
                yield indices

    def __len__(self):
        return self.planned_batches
    
    def sampling_value(self):
        return self.sampling_rate


class SpacedSamplingWithFixedSize(BatchSampler):
    def __init__(self, dataset, batch_size, lam=1, size=None):
        self.seq_len = size
        self.batch_size = batch_size
        self.dataset = dataset
        self.data_size = len(dataset) 
        self.lam = lam  

    def _build_spacing_interval(self):
        """
        Divide the window index space evenly into batch_size / lam intervals,
        and randomly sample one window index from each interval.
        Return a list of window IDs for one batch (consistent with PoissonSampler).
        """
        n = self.data_size  
        k = math.ceil(self.batch_size / self.lam) if self.lam > 0 else self.batch_size
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
                idx = random.randint(start, end - 1)
                sampled_win_ids.append(idx)
                continue
            else:
                if random.random() <= self.lam:
                    idx = random.randint(start, end - 1)
                    sampled_win_ids.append(idx)

        return sampled_win_ids

    def __iter__(self):
        planned_batches = max(1, math.ceil(self.data_size / self.batch_size)) if self.batch_size > 0 else 0
        for _ in range(planned_batches):
            batch_idx = self._build_spacing_interval()
            if batch_idx:
                yield batch_idx
    
    def __len__(self):
        planned_batches = max(1, math.ceil(self.data_size / self.batch_size)) if self.batch_size > 0 else 0
        return planned_batches



