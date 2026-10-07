__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp


class Model(nn.Module):
    """
    PatchTST for Time Series Classification
    Supports both sequence-level and point-level classification
    """
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='LayerNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto', padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = getattr(configs, 'classification_dim', None) or configs.seq_len
        head_hidden = getattr(configs, 'classifier_hidden', 256)
        classifier_dropout = getattr(configs, 'classifier_dropout', 0.2)
        if target_window < 1 or head_hidden < 0:
            raise ValueError('classification_dim must be positive; classifier_hidden must be nonnegative')
        if getattr(configs, 'label_mode', 'sequence') == 'point' and target_window != configs.seq_len:
            raise ValueError('classification_dim compression requires sequence labels')
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.individual
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size
        
        # Classification parameters
        self.num_classes = configs.num_classes
        self.label_mode = getattr(configs, 'label_mode', 'sequence')
        self.seq_len = configs.seq_len
        self.enc_in = c_in
        
        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
        else:
            self.model = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
        
        # Classification head
        if self.label_mode == 'point':
            # Point-wise classification: output [B, seq_len, num_classes]
            self.classifier = nn.Sequential(
                nn.Linear(c_in, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, self.num_classes)
            )
        else:
            # Sequence-level classification: output [B, num_classes]
            self.flatten_dim = c_in * target_window
            self.classifier = nn.Sequential(
                nn.Linear(self.flatten_dim, head_hidden),
                nn.ReLU(),
                nn.Dropout(classifier_dropout),
                nn.Linear(head_hidden, self.num_classes)
            ) if head_hidden else nn.Linear(self.flatten_dim, self.num_classes)
    
    
    def forward(self, x):           # x: [Batch, Input length, Channel]
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            x = self.model(x)
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        
        # Classification
        if self.label_mode == 'point':
            # x: [Batch, seq_len, Channel] -> [Batch, seq_len, num_classes]
            logits = self.classifier(x)
        else:
            # Flatten: [Batch, seq_len * Channel]
            x = x.reshape(x.size(0), -1)
            # Classification: [Batch, num_classes]
            logits = self.classifier(x)
        
        return logits
