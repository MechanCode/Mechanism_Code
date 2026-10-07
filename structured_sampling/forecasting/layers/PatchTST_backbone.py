import torch
from torch import nn
import torch.nn.functional as F

from layers.PatchTST_layers import positional_encoding


class PatchTST_backbone(nn.Module):
    def __init__(self, context_window, target_window, patch_len, stride,
                 n_layers, d_model, n_heads, d_ff, dropout, head_dropout):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        patch_num = int((context_window - patch_len) / stride + 1) + 1
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
        self.backbone = TSTiEncoder(
            patch_num, patch_len, n_layers, d_model, n_heads, d_ff, dropout
        )
        self.head = Flatten_Head(d_model * patch_num, target_window, head_dropout)

    def forward(self, z):
        z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z = z.permute(0, 1, 3, 2)
        return self.head(self.backbone(z))


class Flatten_Head(nn.Module):
    def __init__(self, nf, target_window, head_dropout):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        return self.dropout(self.linear(self.flatten(x)))


class TSTiEncoder(nn.Module):
    def __init__(self, patch_num, patch_len, n_layers, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.W_P = nn.Linear(patch_len, d_model)
        self.W_pos = positional_encoding(patch_num, d_model)
        self.dropout = nn.Dropout(dropout)
        self.encoder = TSTEncoder(d_model, n_heads, d_ff, dropout, n_layers)

    def forward(self, x):
        n_vars = x.shape[1]
        x = self.W_P(x.permute(0, 1, 3, 2))
        u = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        u = self.dropout(u + self.W_pos)
        z = self.encoder(u)
        z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))
        return z.permute(0, 1, 3, 2)


class TSTEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            TSTEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, src):
        output, scores = src, None
        for layer in self.layers:
            output, scores = layer(output, prev=scores)
        return output


class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.self_attn = _MultiheadAttention(d_model, n_heads, dropout)
        self.dropout_attn = nn.Dropout(dropout)
        self.norm_attn = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(self, src, prev=None):
        src2, scores = self.self_attn(src, prev=prev)
        src = self.norm_attn(src + self.dropout_attn(src2))
        src2 = self.ff(src)
        return self.norm_ffn(src + self.dropout_ffn(src2)), scores


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = self.d_v = d_model // n_heads
        self.W_Q = nn.Linear(d_model, self.d_k * n_heads)
        self.W_K = nn.Linear(d_model, self.d_k * n_heads)
        self.W_V = nn.Linear(d_model, self.d_v * n_heads)
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads)
        self.to_out = nn.Sequential(nn.Linear(n_heads * self.d_v, d_model), nn.Dropout(dropout))

    def forward(self, inputs, prev=None):
        batch_size = inputs.size(0)
        q = self.W_Q(inputs).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_K(inputs).view(batch_size, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        v = self.W_V(inputs).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        output, scores = self.sdp_attn(q, k, v, prev=prev)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        return self.to_out(output), scores


class _ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn_dropout = nn.Dropout(0.0)
        self.scale = nn.Parameter(torch.tensor((d_model // n_heads) ** -0.5), requires_grad=False)

    def forward(self, q, k, v, prev=None):
        scores = torch.matmul(q, k) * self.scale
        if prev is not None:
            scores = scores + prev
        weights = self.attn_dropout(F.softmax(scores, dim=-1))
        return torch.matmul(weights, v), scores
