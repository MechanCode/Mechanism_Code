from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from data_provider.data_loader import read_classification_csv, window_label


DATASET_PRESETS: dict[str, dict[str, object]] = {
    "HAR": {
        "seq_len": 200,
        "enc_in": 42,
        "num_classes": 8,
    },
    "USC": {
        "seq_len": 200,
        "enc_in": 6,
        "num_classes": 12,
    },
}


def dataset_preset(name: str) -> dict[str, object]:
    try:
        return DATASET_PRESETS[name.upper()].copy()
    except KeyError as exc:
        raise ValueError(f"unsupported dataset {name!r}; choose HAR or USC") from exc


def resolve_split_paths(
    dataset: str,
    train_path: str | Path | None,
    validation_path: str | Path | None,
    test_path: str | Path | None,
) -> tuple[Path, Path, Path]:
    dataset_preset(dataset)
    if not all((train_path, validation_path, test_path)):
        raise ValueError(
            "Provide all three split paths, or use scripts/launch_common.sh "
            "structured_classification with its central path settings."
        )
    return tuple(Path(path).expanduser().resolve()
                 for path in (train_path, validation_path, test_path))


def load_classification_splits(
    train_path: str | Path,
    validation_path: str | Path,
    test_path: str | Path,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    list[str],
    str,
]:
    loaded = [
        read_classification_csv(path)
        for path in (train_path, validation_path, test_path)
    ]
    feature_columns = loaded[0][2]
    label_column = loaded[0][3]
    for path, (_, labels, columns, observed_label_column) in zip(
        (train_path, validation_path, test_path), loaded
    ):
        if columns != feature_columns or observed_label_column != label_column:
            raise ValueError(f"CSV schema mismatch in {path}")
        if labels.size == 0:
            raise ValueError(f"classification split is empty: {path}")
    training_classes = set(np.unique(loaded[0][1]).tolist())
    for path, (_, labels, _, _) in zip(
        (validation_path, test_path), loaded[1:]
    ):
        unseen = set(np.unique(labels).tolist()) - training_classes
        if unseen:
            raise ValueError(f"{path} contains classes absent from training: {sorted(unseen)}")
    return (
        (loaded[0][0], loaded[0][1]),
        (loaded[1][0], loaded[1][1]),
        (loaded[2][0], loaded[2][1]),
        feature_columns,
        label_column,
    )


def _source_identity(prefix: str, path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        f"{prefix}_path": str(path),
        f"{prefix}_size": stat.st_size,
        f"{prefix}_mtime_ns": stat.st_mtime_ns,
    }


def prepare_classification_cache(
    train_path: str | Path,
    validation_path: str | Path,
    test_path: str | Path,
    cache_dir: str | Path,
) -> dict[str, object]:

    paths = tuple(
        Path(path).expanduser().resolve()
        for path in (train_path, validation_path, test_path)
    )
    cache = Path(cache_dir).expanduser().resolve()
    metadata_path = cache / "metadata.json"
    identity: dict[str, object] = {}
    for prefix, path in zip(("train_source", "validation_source", "test_source"), paths):
        identity.update(_source_identity(prefix, path))
    expected_files = [
        cache / f"{split}_{kind}.npy"
        for split in ("train", "validation", "test")
        for kind in ("features", "labels")
    ]
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            all(metadata.get(key) == value for key, value in identity.items())
            and not bool(metadata.get("standardized"))
            and all(path.exists() for path in expected_files)
        ):
            return metadata

    training, validation, test, feature_columns, label_column = (
        load_classification_splits(*paths)
    )
    splits = [training, validation, test]

    cache.mkdir(parents=True, exist_ok=True)
    for name, (features, labels) in zip(("train", "validation", "test"), splits):
        np.save(cache / f"{name}_features.npy", features.astype(np.float32, copy=False), allow_pickle=False)
        np.save(cache / f"{name}_labels.npy", labels.astype(np.int64, copy=False), allow_pickle=False)
    train_labels = splits[0][1]
    metadata: dict[str, object] = {
        **identity,
        "feature_columns": feature_columns,
        "label_column": label_column,
        "num_variables": len(feature_columns),
        "num_classes": int(train_labels.max()) + 1,
        "class_ids": sorted(int(value) for value in np.unique(train_labels)),
        "training_length": len(splits[0][0]),
        "validation_length": len(splits[1][0]),
        "test_length": len(splits[2][0]),
        "standardized": False,
        "preprocessing": "raw float32 features",
        "label_encoding": "zero-based (1-based source labels are shifted by one)",
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def load_classification_cache(
    cache_dir: str | Path,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    dict[str, object],
]:
    cache = Path(cache_dir).expanduser().resolve()
    metadata_path = cache / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"{metadata_path} does not exist; run prepare_n1_data.py first"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    splits = []
    for name in ("train", "validation", "test"):
        features = np.load(cache / f"{name}_features.npy", mmap_mode="c", allow_pickle=False)
        labels = np.load(cache / f"{name}_labels.npy", mmap_mode="c", allow_pickle=False)
        if len(features) != len(labels):
            raise ValueError(f"cached {name} features/labels have different lengths")
        splits.append((features, labels))
    observed = tuple(len(features) for features, _ in splits)
    expected = tuple(
        int(metadata[f"{name}_length"])
        for name in ("training", "validation", "test")
    )
    if observed != expected:
        raise ValueError(f"cache split lengths {observed} do not match metadata {expected}")
    return splits[0], splits[1], splits[2], metadata


class ClassificationWindows(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        seq_len: int,
        window_stride: int | None = None,
    ) -> None:
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.seq_len = int(seq_len)
        self.window_stride = self.seq_len if window_stride is None else int(window_stride)
        if self.seq_len < 1 or self.window_stride < 1:
            raise ValueError("seq_len and window_stride must be positive")
        self.count = (len(features) - self.seq_len) // self.window_stride + 1
        if self.count < 1:
            raise ValueError("split contains no complete classification window")

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(index) * self.window_stride
        stop = start + self.seq_len
        return self.features[start:stop], window_label(self.labels[start:stop])


def classification_loss(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, targets.reshape(-1))


def evaluate_classification(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    seq_len: int,
    device: torch.device,
    *,
    batch_size: int,
    num_workers: int,
    window_stride: int | None = None,
) -> dict[str, float | int]:
    dataset = ClassificationWindows(
        features, labels, seq_len, window_stride=window_stride
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    model.eval()
    total_loss = 0.0
    total_targets = 0
    predictions: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.long)
            logits = model(inputs)
            loss = classification_loss(logits, targets)
            flat_targets = targets.reshape(-1)
            flat_predictions = logits.argmax(dim=-1).reshape(-1)
            count = flat_targets.numel()
            total_loss += float(loss) * count
            total_targets += count
            predictions.append(flat_predictions.cpu().numpy())
            truths.append(flat_targets.cpu().numpy())
    predicted = np.concatenate(predictions)
    true = np.concatenate(truths)
    return {
        "loss": total_loss / total_targets,
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": float(f1_score(true, predicted, average="macro", zero_division=0)),
        "num_windows": len(dataset),
        "num_targets": total_targets,
    }


__all__ = [
    "ClassificationWindows",
    "DATASET_PRESETS",
    "classification_loss",
    "dataset_preset",
    "evaluate_classification",
    "load_classification_cache",
    "load_classification_splits",
    "prepare_classification_cache",
    "resolve_split_paths",
]
