"""Optional window-local features for the DLinear classification head."""
import torch
from torch import nn


class ClassificationFeatures(nn.Module):
    def __init__(self, mode, channels, projected_dim, moment_scale=1.0):
        super().__init__()
        if mode not in ('linear', 'quadratic', 'moments'):
            raise ValueError('dlinear_features must be linear, quadratic or moments')
        if moment_scale <= 0:
            raise ValueError('dlinear_moment_scale must be positive')
        self.mode = mode
        self.moment_scale = moment_scale
        self.output_dim = channels * projected_dim
        if mode == 'quadratic':
            self.output_dim *= 2
        elif mode == 'moments':
            self.output_dim += 3 * channels

    def forward(self, projected, window):
        features = projected.reshape(projected.size(0), -1)
        if self.mode == 'quadratic':
            return torch.cat((features, features.square()), dim=-1)
        if self.mode == 'moments':
            # Each statistic uses only the current input window. No fitted/global
            # statistics, dataset labels, or cross-window normalization are used.
            mean = window.mean(dim=1)
            std = window.std(dim=1, unbiased=False)
            variation = (window[:, 1:] - window[:, :-1]).abs().mean(dim=1)
            moments = torch.cat((mean, std, variation), dim=-1)
            return torch.cat((features, self.moment_scale * moments), dim=-1)
        return features


def classification_head(input_dim, classes, hidden):
    if hidden < 0:
        raise ValueError('dlinear_hidden must be nonnegative')
    if hidden:
        return nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Linear(hidden, classes))
    return nn.Linear(input_dim, classes)
