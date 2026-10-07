import torch
from torch import nn


def positional_encoding(q_len, d_model):

    weights = torch.empty((q_len, d_model))
    nn.init.uniform_(weights, -0.02, 0.02)
    return nn.Parameter(weights, requires_grad=True)
