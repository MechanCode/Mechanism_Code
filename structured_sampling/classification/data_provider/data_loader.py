from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


def read_classification_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str], str]:

    csv_path = Path(path).expanduser()
    frame = pd.read_csv(csv_path)
    if frame.shape[1] < 2:
        raise ValueError(f"{csv_path} must contain features and a final label column")
    feature_columns = list(frame.columns[:-1])
    label_column = str(frame.columns[-1])
    features = frame.iloc[:, :-1].to_numpy(dtype=np.float32, copy=True)
    raw_labels = frame.iloc[:, -1].to_numpy(copy=True)
    if not np.isfinite(features).all():
        raise ValueError(f"{csv_path} contains NaN or infinite feature values")
    if not np.isfinite(raw_labels.astype(np.float64)).all():
        raise ValueError(f"{csv_path} contains invalid labels")
    labels = raw_labels.astype(np.int64)
    if not np.equal(raw_labels, labels).all():
        raise ValueError(f"{csv_path} labels must be integers")

    if labels.size and labels.min() >= 1:
        labels = labels - 1
    if labels.size and labels.min() < 0:
        raise ValueError(f"{csv_path} labels must be nonnegative (or 1-based)")
    return features, labels, feature_columns, label_column


def window_label(labels: np.ndarray) -> torch.Tensor:

    majority = int(np.bincount(labels).argmax())
    return torch.tensor(majority, dtype=torch.long)


__all__ = ["read_classification_csv", "window_label"]
