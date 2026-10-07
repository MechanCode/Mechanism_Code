from __future__ import annotations

import torch
from torch import nn

from layers.PatchTST_backbone import PatchTST_backbone


class Model(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        c_in = int(configs.enc_in)
        seq_len = int(configs.seq_len)
        self.num_classes = int(configs.num_classes)
        self.model = PatchTST_backbone(
            context_window=seq_len,
            target_window=seq_len,
            patch_len=16,
            stride=8,
            n_layers=2,
            d_model=128,
            n_heads=8,
            d_ff=2048,
            dropout=0.05,
            head_dropout=0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(c_in * seq_len, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(
                f"inputs must have shape [batch, seq_len, channels], got {tuple(inputs.shape)}"
            )
        features = self.model(inputs.permute(0, 2, 1)).permute(0, 2, 1)
        return self.classifier(features.reshape(features.size(0), -1))
