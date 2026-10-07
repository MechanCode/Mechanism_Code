from torch import nn

from layers.PatchTST_backbone import PatchTST_backbone


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.model = PatchTST_backbone(
            context_window=configs.seq_len,
            target_window=configs.pred_len,
            patch_len=16,
            stride=8,
            n_layers=2,
            d_model=128,
            n_heads=8,
            d_ff=2048,
            dropout=0.05,
            head_dropout=0.0,
        )

    def forward(self, x):
        return self.model(x.permute(0, 2, 1)).permute(0, 2, 1)
