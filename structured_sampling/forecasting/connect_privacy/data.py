from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler


def split_single_series(
    series: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    total = len(series)
    train_length = int(total * 0.7)
    test_length = int(total * 0.2)
    validation_length = total - train_length - test_length
    return (
        series[:train_length],
        series[train_length : train_length + validation_length],
        series[train_length + validation_length :],
    )


def prepare_standardized_cache(
    root_path: str | Path,
    data_path: str,
    cache_dir: str | Path,
) -> dict[str, object]:

    source = (Path(root_path).expanduser() / data_path).resolve()
    cache = Path(cache_dir).expanduser().resolve()
    metadata_path = cache / "metadata.json"
    source_stat = source.stat()
    source_identity = {
        "source_path": str(source),
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
    }
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_files = [cache / f"{split}.npy" for split in ("train", "validation", "test")]
        if all(metadata.get(key) == value for key, value in source_identity.items()):
            if all(path.exists() for path in expected_files):
                return metadata

    with source.open(newline="", encoding="utf-8") as handle:
        columns = next(csv.reader(handle))
    if "date" not in columns:
        raise ValueError(f"{source} must contain a 'date' column")
    feature_columns = [column for column in columns if column != "date"]
    frame = pd.read_csv(
        source,
        usecols=feature_columns,
        dtype={column: np.float32 for column in feature_columns},
    )
    values = frame.to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"{source} contains NaN or infinite feature values")
    train_raw, validation_raw, test_raw = split_single_series(values)
    scaler = StandardScaler(copy=True)
    scaler.fit(train_raw)

    cache.mkdir(parents=True, exist_ok=True)
    for name, split in (
        ("train", train_raw),
        ("validation", validation_raw),
        ("test", test_raw),
    ):
        standardized = scaler.transform(split).astype(np.float32, copy=False)
        np.save(cache / f"{name}.npy", standardized, allow_pickle=False)

    metadata: dict[str, object] = {
        **source_identity,
        "total_length": len(values),
        "training_length": len(train_raw),
        "validation_length": len(validation_raw),
        "test_length": len(test_raw),
        "num_variables": len(feature_columns),
        "feature_columns": feature_columns,
        "preprocessing": "StandardScaler fitted on the 70% training split",
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def load_standardized_cache(
    cache_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    cache = Path(cache_dir).expanduser().resolve()
    metadata_path = cache / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"{metadata_path} does not exist; run prepare_n1_data.py first"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))


    arrays = tuple(
        np.load(cache / f"{split}.npy", mmap_mode="c", allow_pickle=False)
        for split in ("train", "validation", "test")
    )
    observed = tuple(len(array) for array in arrays)
    expected = (
        metadata["training_length"],
        metadata["validation_length"],
        metadata["test_length"],
    )
    if observed != expected:
        raise ValueError(f"cache split lengths {observed} do not match metadata {expected}")
    return arrays[0], arrays[1], arrays[2], metadata


class EvaluationWindows(Dataset):
    def __init__(
        self, series: np.ndarray, context_len: int, forecast_len: int
    ) -> None:
        self.series = torch.as_tensor(series, dtype=torch.float32)
        self.context_len = context_len
        self.forecast_len = forecast_len
        self.count = len(series) - context_len - forecast_len + 1
        if self.count < 1:
            raise ValueError("evaluation split contains no complete forecasting window")

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        boundary = index + self.context_len
        return (
            self.series[index:boundary],
            self.series[boundary : boundary + self.forecast_len],
        )


def evaluate_mse(
    model: nn.Module,
    series: np.ndarray,
    context_len: int,
    forecast_len: int,
    device: torch.device,
    *,
    batch_size: int,
    num_workers: int,
) -> float:
    loader = DataLoader(
        EvaluationWindows(series, context_len, forecast_len),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    model.eval()
    total_squared_error = 0.0
    element_count = 0
    with torch.no_grad():
        for contexts, targets in loader:
            contexts = contexts.to(device)
            targets = targets.to(device)
            predictions = model(contexts)[:, -forecast_len:, :]
            total_squared_error += float(
                nn.functional.mse_loss(predictions, targets, reduction="sum")
            )
            element_count += targets.numel()
    return total_squared_error / element_count
