"""
DLinear_Stable - Optimized for DP-SGD Training Stability

Key improvements for stable DP training:
1. Much smaller model (fewer parameters = less noise accumulation)
2. Deterministic initialization with fixed seeds
3. Global Average Pooling instead of flatten (reduces parameters drastically)
4. Batch normalization replaced with LayerNorm (works better with per-sample gradients)
5. No dropout (incompatible with DP-SGD)
6. Simpler architecture for cleaner gradient flow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class moving_avg(nn.Module):
    """Moving average block to highlight the trend of time series"""
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: [B, L, C]
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """Series decomposition block"""
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):
    """
    Stable DLinear for Time Series Classification with DP-SGD
    
    Design principles for DP stability:
    1. Minimize parameter count to reduce noise impact
    2. Use Global Average Pooling to aggregate temporal info
    3. Deterministic initialization
    4. Simple architecture with clean gradient flow
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.num_classes = configs.num_classes
        self.label_mode = getattr(configs, 'label_mode', 'sequence')
        
        self.init_seed = getattr(configs, 'seed', 42)
        
        # Decomposition kernel size
        kernel_size = min(25, max(3, self.seq_len // 10))
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.decomposition = series_decomp(kernel_size)
        
        # Feature dimension after pooling
        # Use smaller intermediate representation
        self.feature_dim = min(64, self.enc_in)
        
        # Channel mixing: reduce channel dimension
        self.channel_proj = nn.Linear(self.enc_in, self.feature_dim)
        
        # Temporal projection with MUCH smaller output
        # Instead of seq_len -> seq_len, use seq_len -> small_dim
        self.temporal_dim = 32  # Fixed small temporal dimension
        self.temporal_seasonal = nn.Linear(self.seq_len, self.temporal_dim)
        self.temporal_trend = nn.Linear(self.seq_len, self.temporal_dim)
        
        # Simple classifier
        # Input: feature_dim * temporal_dim (much smaller than before)
        classifier_input_dim = self.feature_dim * self.temporal_dim
        hidden_dim = 64  # Smaller hidden dimension
        
        if self.label_mode == 'point':
            # For point-wise classification
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.num_classes)
            )
        else:
            # For sequence classification
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.num_classes)
            )
        
        self._init_weights_deterministic(self.init_seed)
        
        # Print model info
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[DLinear_Stable] Total parameters: {total_params:,}")
        print(f"[DLinear_Stable] Classifier input dim: {classifier_input_dim}")
        print(f"[DLinear_Stable] Init seed: {self.init_seed}")

    def _init_weights_deterministic(self, seed=42):
        """
        Deterministic initialization using the provided seed.
        Each iteration uses a different seed (from args.seed),
        ensuring reproducibility within the same iteration while
        allowing variation across iterations.
        """
        # Save current RNG state
        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        
        # Use provided seed for initialization
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Small gain for stability with DP noise
        gain = 0.1
        
        # Initialize channel projection
        nn.init.xavier_uniform_(self.channel_proj.weight, gain=gain)
        nn.init.zeros_(self.channel_proj.bias)
        
        # Initialize temporal projections
        nn.init.xavier_uniform_(self.temporal_seasonal.weight, gain=gain)
        nn.init.zeros_(self.temporal_seasonal.bias)
        nn.init.xavier_uniform_(self.temporal_trend.weight, gain=gain)
        nn.init.zeros_(self.temporal_trend.bias)
        
        # Initialize classifier
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Restore RNG state
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state)

    def forward(self, x):
        # x: [Batch, seq_len, enc_in]
        batch_size = x.size(0)
        
        # Decomposition
        seasonal, trend = self.decomposition(x)
        # seasonal, trend: [B, seq_len, enc_in]
        
        # Channel projection (reduce channels)
        seasonal = self.channel_proj(seasonal)  # [B, seq_len, feature_dim]
        trend = self.channel_proj(trend)        # [B, seq_len, feature_dim]
        
        # Temporal projection
        # Transpose for temporal operation: [B, feature_dim, seq_len]
        seasonal = seasonal.permute(0, 2, 1)
        trend = trend.permute(0, 2, 1)
        
        # Project to smaller temporal dimension
        seasonal = self.temporal_seasonal(seasonal)  # [B, feature_dim, temporal_dim]
        trend = self.temporal_trend(trend)           # [B, feature_dim, temporal_dim]
        
        # Combine
        x = seasonal + trend  # [B, feature_dim, temporal_dim]
        
        if self.label_mode == 'point':
            # Point-wise: need to upsample back to seq_len
            # Use simple interpolation
            x = F.interpolate(x, size=self.seq_len, mode='linear', align_corners=False)
            x = x.permute(0, 2, 1)  # [B, seq_len, feature_dim]
            logits = self.classifier(x)  # [B, seq_len, num_classes]
        else:
            # Sequence classification: flatten and classify
            x = x.reshape(batch_size, -1)  # [B, feature_dim * temporal_dim]
            logits = self.classifier(x)    # [B, num_classes]
        
        return logits


class Model_GAP(nn.Module):
    """
    Even simpler model using Global Average Pooling
    Minimal parameters for maximum DP stability
    """
    
    def __init__(self, configs):
        super(Model_GAP, self).__init__()
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.num_classes = configs.num_classes
        self.label_mode = getattr(configs, 'label_mode', 'sequence')
        
        self.init_seed = getattr(configs, 'seed', 42)
        
        # Very simple: just project channels and pool
        hidden_dim = 32
        
        # Two-layer feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.enc_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Classifier
        if self.label_mode == 'point':
            self.classifier = nn.Linear(hidden_dim, self.num_classes)
        else:
            # After global average pooling
            self.classifier = nn.Linear(hidden_dim, self.num_classes)
        
        self._init_weights_deterministic(self.init_seed)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[Model_GAP] Total parameters: {total_params:,}")
        print(f"[Model_GAP] Init seed: {self.init_seed}")

    def _init_weights_deterministic(self, seed=42):
        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        gain = 0.1
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state)

    def forward(self, x):
        # x: [B, seq_len, enc_in]
        
        # Extract features at each time step
        features = self.feature_extractor(x)  # [B, seq_len, hidden_dim]
        
        if self.label_mode == 'point':
            logits = self.classifier(features)  # [B, seq_len, num_classes]
        else:
            # Global average pooling over time
            pooled = features.mean(dim=1)  # [B, hidden_dim]
            logits = self.classifier(pooled)  # [B, num_classes]
        
        return logits
