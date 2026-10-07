from __future__ import annotations

import torch


def sample_classification_windows_with_replacement(
    features: torch.Tensor,
    labels: torch.Tensor,
    window_len: int,
    batch_size: int,
    *,
    sample_stride: int = 1,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    if features.ndim != 2:
        raise ValueError(f"features must have shape [L, D], got {tuple(features.shape)}")
    if labels.ndim != 1 or len(labels) != len(features):
        raise ValueError("labels must have shape [L] and match features")
    if window_len < 1 or batch_size < 1 or sample_stride < 1:
        raise ValueError("window_len, batch_size, and sample_stride must be positive")
    if len(features) < window_len:
        raise ValueError(
            f"series length {len(features)} is smaller than window_len={window_len}"
        )
    num_candidates = (len(features) - window_len) // sample_stride + 1
    candidate_ids = torch.randint(
        0, num_candidates, (batch_size,), generator=generator, device="cpu"
    )
    start_ids = candidate_ids * sample_stride
    feature_windows = torch.stack(
        [features[int(start) : int(start) + window_len] for start in start_ids]
    )
    label_windows = torch.stack(
        [labels[int(start) : int(start) + window_len] for start in start_ids]
    )
    targets = torch.mode(label_windows, dim=1).values
    return feature_windows, targets, start_ids


__all__ = ["sample_classification_windows_with_replacement"]
