from __future__ import annotations

import torch
from torch import nn


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.average = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        padding = (self.kernel_size - 1) // 2
        front = inputs[:, :1, :].repeat(1, padding, 1)
        end = inputs[:, -1:, :].repeat(1, padding, 1)
        padded = torch.cat((front, inputs, end), dim=1)
        return self.average(padded.permute(0, 2, 1)).permute(0, 2, 1)


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.moving_average = MovingAverage(kernel_size)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trend = self.moving_average(inputs)
        return inputs - trend, trend


class Model(nn.Module):


    def __init__(self, configs) -> None:
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.enc_in = int(configs.enc_in)
        self.num_classes = int(configs.num_classes)
        self.decomposition = SeriesDecomposition(kernel_size=25)
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.seq_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.seq_len)
        self.classifier = nn.Linear(self.enc_in * self.seq_len, self.num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(
                f"inputs must have shape [batch, seq_len, channels], got {tuple(inputs.shape)}"
            )
        seasonal, trend = self.decomposition(inputs)
        seasonal = self.Linear_Seasonal(seasonal.permute(0, 2, 1))
        trend = self.Linear_Trend(trend.permute(0, 2, 1))
        features = (seasonal + trend).permute(0, 2, 1)
        return self.classifier(features.reshape(features.size(0), -1))
