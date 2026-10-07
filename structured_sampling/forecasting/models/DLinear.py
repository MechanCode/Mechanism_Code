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
        self.pred_len = configs.pred_len
        self.decomposition = SeriesDecomposition(kernel_size=25)


        self.Linear_Seasonal = nn.Linear(configs.seq_len, configs.pred_len)
        self.Linear_Trend = nn.Linear(configs.seq_len, configs.pred_len)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        seasonal, trend = self.decomposition(inputs)
        seasonal = self.Linear_Seasonal(seasonal.permute(0, 2, 1))
        trend = self.Linear_Trend(trend.permute(0, 2, 1))
        return (seasonal + trend).permute(0, 2, 1)
