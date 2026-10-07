from __future__ import annotations

import torch
from torch import nn

from layers.PatchTST_backbone import TSTiEncoder

__all__ = ["Model"]


class Model(nn.Module):


    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.enc_in = int(configs.enc_in)
        self.num_classes = int(configs.num_classes)
        self.patch_len = int(configs.patch_len)
        self.stride = int(configs.stride)
        self.padding_patch = configs.padding_patch

        if self.seq_len < self.patch_len:
            raise ValueError(
                f"seq_len={self.seq_len} must be at least patch_len={self.patch_len}"
            )
        patch_num = int((self.seq_len - self.patch_len) / self.stride + 1)
        if self.padding_patch == "end":
            self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
            patch_num += 1
        elif self.padding_patch not in (None, "none"):
            raise ValueError("padding_patch must be 'end' or None")

        d_model = int(configs.d_model)
        self.input_norm = nn.LayerNorm(self.enc_in)
        self.encoder = TSTiEncoder(
            c_in=self.enc_in,
            patch_num=patch_num,
            patch_len=self.patch_len,
            n_layers=int(configs.e_layers),
            d_model=d_model,
            n_heads=int(configs.n_heads),
            d_ff=int(configs.d_ff),
            norm="LayerNorm",
            attn_dropout=0.0,
            dropout=0.0,
            res_attention=True,
        )
        self.classifier_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, self.num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        encoded = self.input_norm(inputs).permute(0, 2, 1)
        if self.padding_patch == "end":
            encoded = self.padding_patch_layer(encoded)
        encoded = encoded.unfold(
            dimension=-1, size=self.patch_len, step=self.stride
        )
        encoded = encoded.permute(0, 1, 3, 2)
        encoded = self.encoder(encoded)

        pooled = encoded.mean(dim=(1, 3))
        return self.classifier(self.classifier_norm(pooled))
