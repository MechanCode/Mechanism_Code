from __future__ import annotations

import torch


def _validate_series(
    series: torch.Tensor,
    context_len: int,
    forecast_len: int,
    batch_size: int | None = None,
) -> tuple[int, int]:
    if series.ndim != 2:
        raise ValueError(f"series must have shape [L, D], got {tuple(series.shape)}")
    length, channels = map(int, series.shape)
    if context_len < 1 or forecast_len < 1:
        raise ValueError("context_len and forecast_len must be positive")
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be positive")
    if length - forecast_len + 1 < 1:
        raise ValueError(
            f"series length {length} is smaller than forecast_len={forecast_len}"
        )
    return length, channels


def build_padded_series(series: torch.Tensor, context_len: int) -> torch.Tensor:

    _validate_series(series, context_len, forecast_len=1)
    padding = series.new_zeros((context_len, series.shape[1]))
    return torch.cat((padding, series), dim=0)


def sample_windows_with_replacement(
    series: torch.Tensor,
    context_len: int,
    forecast_len: int,
    batch_size: int,
    *,
    sample_stride: int = 1,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    length, _ = _validate_series(series, context_len, forecast_len, batch_size)
    if sample_stride < 1:
        raise ValueError("sample_stride must be positive")
    num_candidates = (length - forecast_len) // sample_stride + 1


    candidate_ids = torch.randint(
        low=0,
        high=num_candidates,
        size=(batch_size,),
        generator=generator,
        device="cpu",
    )
    start_ids = candidate_ids * sample_stride
    padded = build_padded_series(series, context_len)
    total_len = context_len + forecast_len
    windows = torch.stack(
        [padded[int(start) : int(start) + total_len] for start in start_ids],
        dim=0,
    )
    return (
        windows[:, :context_len],
        windows[:, context_len:],
        start_ids,
    )
