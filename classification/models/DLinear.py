# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
#
# class moving_avg(nn.Module):
#     """
#     Moving average block to highlight the trend of time series
#     """
#     def __init__(self, kernel_size, stride):
#         super(moving_avg, self).__init__()
#         self.kernel_size = kernel_size
#         self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
#
#     def forward(self, x):
#         # padding on the both ends of time series
#         front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
#         end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
#         x = torch.cat([front, x, end], dim=1)
#         x = self.avg(x.permute(0, 2, 1))
#         x = x.permute(0, 2, 1)
#         return x
#
#
# class series_decomp(nn.Module):
#     """
#     Series decomposition block
#     """
#     def __init__(self, kernel_size):
#         super(series_decomp, self).__init__()
#         self.moving_avg = moving_avg(kernel_size, stride=1)
#
#     def forward(self, x):
#         moving_mean = self.moving_avg(x)
#         res = x - moving_mean
#         return res, moving_mean
#
# class Model(nn.Module):
#     """
#     Decomposition-Linear
#     """
#     def __init__(self, configs):
#         super(Model, self).__init__()
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#
#         # Decompsition Kernel Size
#         kernel_size = 25
#         self.decompsition = series_decomp(kernel_size)
#         self.individual = configs.individual
#         self.channels = configs.enc_in
#
#         if self.individual:
#             self.Linear_Seasonal = nn.ModuleList()
#             self.Linear_Trend = nn.ModuleList()
#
#             for i in range(self.channels):
#                 self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
#                 self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))
#
#                 # Use this two lines if you want to visualize the weights
#                 # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
#                 # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
#         else:
#             self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
#             self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
#
#             # Use this two lines if you want to visualize the weights
#             # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
#             # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
#
#     def forward(self, x):
#         # x: [Batch, Input length, Channel]
#         seasonal_init, trend_init = self.decompsition(x)
#         seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
#         if self.individual:
#             seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
#             trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
#             for i in range(self.channels):
#                 seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
#                 trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
#         else:
#             seasonal_output = self.Linear_Seasonal(seasonal_init)
#             trend_output = self.Linear_Trend(trend_init)
#
#         x = seasonal_output + trend_output
#         return x.permute(0,2,1) # to [Batch, Output length, Channel]


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):
    """
    Decomposition-Linear for Time Series Classification
    Optimized for DP-SGD stability:
    - Input LayerNorm to stabilize input distribution
    - Feature LayerNorm before classification
    - ReLU activation (simpler gradient flow)
    - Flatten-based classification for sequence mode
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.num_classes = configs.num_classes
        self.label_mode = getattr(configs, 'label_mode', 'sequence')
        
        # Input normalization layer
        self.input_norm = nn.LayerNorm(self.enc_in)

        # Decomposition Kernel Size
        kernel_size = min(25, self.seq_len // 4 * 2 + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size < 3:
            kernel_size = 3
        self.decompsition = series_decomp(kernel_size)

        # Linear layers for time series transformation
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.seq_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.seq_len)
        
        # Feature normalization
        self.feature_norm = nn.LayerNorm(self.seq_len)
        
        # Classification head
        hidden_dim = 256
        if self.label_mode == 'point':
            self.classifier = nn.Sequential(
                nn.LayerNorm(self.enc_in),
                nn.Linear(self.enc_in, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.num_classes)
            )
        else:
            # Flatten-based: use all temporal information
            flatten_dim = self.enc_in * self.seq_len
            self.classifier = nn.Sequential(
                nn.LayerNorm(flatten_dim),
                nn.Linear(flatten_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.num_classes)
            )
        
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization with smaller gain for stability"""
        gain = 0.5  # Smaller gain for reduced gradient variance
        nn.init.xavier_uniform_(self.Linear_Seasonal.weight, gain=gain)
        nn.init.zeros_(self.Linear_Seasonal.bias)
        nn.init.xavier_uniform_(self.Linear_Trend.weight, gain=gain)
        nn.init.zeros_(self.Linear_Trend.bias)
        
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        x = self.input_norm(x)
        
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)

        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output  # [Batch, Channel, seq_len]
        x = self.feature_norm(x)
        
        if self.label_mode == 'point':
            x = x.permute(0, 2, 1)
            logits = self.classifier(x)
        else:
            # Flatten: [Batch, Channel * seq_len]
            x = x.reshape(x.size(0), -1)
            logits = self.classifier(x)
        
        return logits

