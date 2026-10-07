import torch
from torch import nn


class ConvAutoencoder(nn.Module):
    def __init__(self, channels, width=64):
        super().__init__()
        half, quarter = width // 2, width // 4
        self.network = nn.Sequential(
            nn.Conv1d(channels, width, 5, padding=2), nn.GELU(),
            nn.Conv1d(width, half, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv1d(half, quarter, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose1d(quarter, half, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose1d(half, width, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv1d(width, channels, 5, padding=2),
        )

    def forward(self, x):
        if x.shape[1] % 4:
            raise ValueError("Segment length must be divisible by four")
        return self.network(x.transpose(1, 2)).transpose(1, 2)


class BottleneckAutoencoder(nn.Module):


    def __init__(self, dimension, rank, temporal=False):
        super().__init__()
        self.temporal = temporal
        self.network = nn.Sequential(nn.Linear(dimension, rank), nn.GELU(),
                                     nn.Linear(rank, dimension))

    def forward(self, x):
        if self.temporal:
            return self.network(x.transpose(1, 2)).transpose(1, 2)
        return self.network(x)


class FeatureMean(nn.Module):


    def __init__(self, channels):
        super().__init__()
        self.center = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        return self.center.expand_as(x)


class LinearFeatureAutoencoder(nn.Module):


    def __init__(self, channels, rank):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(channels, rank), nn.Linear(rank, channels))

    def forward(self, x):
        return self.network(x)


class HybridAutoencoder(nn.Module):


    def __init__(self, channels, rank):
        super().__init__()
        self.encoder = nn.Linear(channels, rank)
        self.temporal = nn.Conv1d(rank, rank, kernel_size=5, padding=2)
        self.decoder = nn.Linear(rank, channels)
        self.activation = nn.GELU()

    def forward(self, x):
        latent = self.activation(self.encoder(x)).transpose(1, 2)
        filtered = self.activation(self.temporal(latent)).transpose(1, 2)
        return self.decoder(filtered)


class SharedAutoregression(nn.Module):


    def __init__(self):
        super().__init__()
        self.slope = nn.Parameter(torch.ones(1))
        self.intercept = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        previous = torch.cat((torch.zeros_like(x[:, :1]), x[:, :-1]), dim=1)
        return self.slope * previous + self.intercept


def initialize_model(channels, seed, config=None):

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if config is None or config.model_kind == "conv":
            return ConvAutoencoder(channels, 64 if config is None else config.model_width)
        if config.model_kind == "feature_ae":
            return BottleneckAutoencoder(channels, config.model_rank)
        if config.model_kind == "temporal_ae":
            return BottleneckAutoencoder(config.segment_length, config.model_rank, temporal=True)
        if config.model_kind == "feature_mean":
            return FeatureMean(channels)
        if config.model_kind == "feature_linear":
            return LinearFeatureAutoencoder(channels, config.model_rank)
        if config.model_kind == "hybrid_ae":
            return HybridAutoencoder(channels, config.model_rank)
        if config.model_kind == "shared_ar":
            return SharedAutoregression()
        raise ValueError(f"Unknown model kind: {config.model_kind}")
