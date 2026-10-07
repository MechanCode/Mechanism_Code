from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .adapters import ExperimentAdapter
from .projection import CountSketchProjector


@dataclass(frozen=True)
class GradientCollection:
    embeddings: np.ndarray
    sample_indices: np.ndarray
    raw_norms: np.ndarray
    analyzed_norms: np.ndarray
    losses: np.ndarray


def candidate_indices(
    adapter: ExperimentAdapter,
    candidate_stride: int | None,
    seq_len: int,
    pred_len: int,
) -> np.ndarray:

    expected_sample_stride = seq_len + pred_len
    dataset_sample_stride = getattr(adapter.dataset, "sample_stride", None)
    if (
        dataset_sample_stride is not None
        and dataset_sample_stride != expected_sample_stride
    ):
        raise ValueError(
            "Dataset sample_stride must equal seq_len + pred_len: "
            f"got {dataset_sample_stride}, expected {expected_sample_stride}"
        )
    if candidate_stride is None:
        stride = 1
    else:
        stride = candidate_stride
    if stride < 1:
        raise ValueError("candidate_stride must be positive")
    candidates = np.arange(0, len(adapter.dataset), stride, dtype=np.int64)
    if candidates.size < 2:
        raise ValueError(
            f"Only {candidates.size} candidate gradient windows are available; need at least two"
        )
    return candidates


def collect_gradients(
    adapter: ExperimentAdapter,
    indices: np.ndarray,
    projection_dimension: int,
    projection_seed: int,
    clip: bool,
    clipping_norm: float | None,
    progress_every: int = 25,
    projection: str = "countsketch",
) -> GradientCollection:

    trainable = [(name, parameter) for name, parameter in adapter.model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("The model has no trainable parameters")
    if projection not in {"countsketch", "none"}:
        raise ValueError("projection must be 'countsketch' or 'none'")
    if clip and (clipping_norm is None or clipping_norm <= 0):
        raise ValueError("clipping_norm must be positive when clip=True")
    projector = (
        CountSketchProjector(projection_dimension, projection_seed)
        if projection == "countsketch"
        else None
    )
    embeddings: list[np.ndarray] = []
    raw_norms: list[float] = []
    analyzed_norms: list[float] = []
    losses: list[float] = []

    for position, sample_index in enumerate(indices):
        loss = adapter.sample_loss(int(sample_index))
        gradients = torch.autograd.grad(
            loss,
            [parameter for _, parameter in trainable],
            allow_unused=True,
            retain_graph=False,
            create_graph=False,
        )
        squared_norm = torch.zeros((), device=adapter.device, dtype=torch.float64)
        for gradient in gradients:
            if gradient is not None:
                squared_norm += gradient.detach().double().pow(2).sum()
        raw_norm = float(squared_norm.sqrt().cpu())
        scale = 1.0
        if clip:
            assert clipping_norm is not None
            scale = min(1.0, clipping_norm / (raw_norm + 1e-12))
        scaled_gradients = [None if gradient is None else gradient * scale for gradient in gradients]
        if projector is None:


            embedding = torch.cat(
                [
                    torch.zeros_like(parameter).reshape(-1)
                    if gradient is None
                    else gradient.detach().reshape(-1)
                    for (_, parameter), gradient in zip(trainable, scaled_gradients)
                ]
            )
        else:
            embedding = projector.project(
                (name, gradient) for (name, _), gradient in zip(trainable, scaled_gradients)
            )
        embeddings.append(embedding.detach().float().cpu().numpy())
        raw_norms.append(raw_norm)
        analyzed_norms.append(raw_norm * scale)
        losses.append(float(loss.detach().cpu()))

        if progress_every > 0 and ((position + 1) % progress_every == 0 or position + 1 == len(indices)):
            print(f"Collected gradients: {position + 1}/{len(indices)}")

    return GradientCollection(
        embeddings=np.stack(embeddings, axis=0),
        sample_indices=np.asarray(indices, dtype=np.int64),
        raw_norms=np.asarray(raw_norms, dtype=np.float64),
        analyzed_norms=np.asarray(analyzed_norms, dtype=np.float64),
        losses=np.asarray(losses, dtype=np.float64),
    )
