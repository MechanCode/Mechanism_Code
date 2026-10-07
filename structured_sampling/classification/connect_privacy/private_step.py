from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .data import classification_loss
from .sampler import sample_classification_windows_with_replacement


@dataclass(frozen=True)
class PrivateStepStatistics:
    mean_loss: float
    clipped_fraction: float
    mean_gradient_norm: float
    start_ids: torch.Tensor


def _trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    return parameters


def private_classification_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    labels: torch.Tensor,
    seq_len: int,
    batch_size: int,
    clipping_norm: float,
    noise_multiplier: float,
    *,
    sample_stride: int = 1,
    generator: torch.Generator | None = None,
) -> PrivateStepStatistics:

    if clipping_norm <= 0 or noise_multiplier <= 0:
        raise ValueError("clipping_norm and noise_multiplier must be positive")
    windows, targets, start_ids = sample_classification_windows_with_replacement(
        features,
        labels,
        seq_len,
        batch_size,
        sample_stride=sample_stride,
        generator=generator,
    )
    device = next(model.parameters()).device
    windows = windows.to(device=device, dtype=torch.float32)
    targets = targets.to(device=device, dtype=torch.long)
    parameters = _trainable_parameters(model)
    clipped_sums = [torch.zeros_like(parameter) for parameter in parameters]
    losses: list[float] = []
    norms: list[float] = []
    clipped_count = 0

    model.train()
    for index in range(batch_size):
        logits = model(windows[index : index + 1])
        loss = classification_loss(logits, targets[index : index + 1])
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        squared_norm = torch.zeros((), device=device)
        for gradient in gradients:
            if gradient is not None:
                squared_norm.add_(gradient.detach().pow(2).sum())
        norm = squared_norm.sqrt()
        scale = torch.clamp(clipping_norm / norm.clamp_min(1e-12), max=1.0)
        norm_value = float(norm.detach())
        clipped_count += int(norm_value > clipping_norm)
        losses.append(float(loss.detach()))
        norms.append(norm_value)
        for accumulator, gradient in zip(clipped_sums, gradients):
            if gradient is not None:
                accumulator.add_(gradient.detach(), alpha=float(scale))

    optimizer.zero_grad(set_to_none=True)
    noise_std = noise_multiplier * clipping_norm
    for parameter, clipped_sum in zip(parameters, clipped_sums):
        noise = torch.randn(
            clipped_sum.shape,
            generator=generator,
            dtype=clipped_sum.dtype,
            device="cpu",
        ).to(device)
        parameter.grad = (clipped_sum + noise * noise_std) / batch_size
    optimizer.step()

    return PrivateStepStatistics(
        mean_loss=sum(losses) / batch_size,
        clipped_fraction=clipped_count / batch_size,
        mean_gradient_norm=sum(norms) / batch_size,
        start_ids=start_ids,
    )


__all__ = ["PrivateStepStatistics", "private_classification_step"]
