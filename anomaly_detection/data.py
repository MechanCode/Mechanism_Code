from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Series:
    values: np.ndarray
    timestamps: np.ndarray
    features: list[str]
    source_sha256: str
    sequence_boundaries: tuple[tuple[str, int, int], ...] | None = None

    def identity(self):
        identity = {"rows": len(self.values), "channels": self.values.shape[1],
                "features": self.features, "sha256": self.source_sha256}
        if self.sequence_boundaries is not None:
            identity["sequence_boundaries"] = [list(b) for b in self.sequence_boundaries]
        return identity


def validate_boundaries(boundaries, rows):
    cursor, seen = 0, set()
    for name, start, stop in boundaries:
        if (not isinstance(name, str) or name in seen or type(start) is not int
                or type(stop) is not int or start != cursor or not start < stop <= rows):
            raise ValueError("Sequence boundaries must uniquely tile every row in order")
        cursor = stop
        seen.add(name)
    if cursor != rows:
        raise ValueError("Sequence boundaries do not cover the data")


def segment_starts(series, length, stride):
    boundaries = series.sequence_boundaries or (("sequence", 0, len(series.values)),)
    validate_boundaries(boundaries, len(series.values))
    if any(stop - start < length for _, start, stop in boundaries):
        raise ValueError("Every sequence must contain a complete segment")
    return np.concatenate([np.arange(start, stop-length+1, stride, dtype=np.int64)
                           for _, start, stop in boundaries])


def read_series(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Local CSV not found: {path}. No automatic download is performed.")
    frame = pd.read_csv(path)
    if frame.shape[1] < 2 or frame.empty:
        raise ValueError(f"{path}: expected timestamp column followed by numeric features")
    if not str(frame.columns[0]).lower().startswith(("timestamp", "time", "date")):
        raise ValueError(f"{path}: first column must be an explicit timestamp column")
    x = frame.iloc[:, 1:].to_numpy(dtype=np.float32, copy=True)
    if np.isinf(x).any():
        raise ValueError(f"{path}: infinite feature values are not supported")

    np.nan_to_num(x, copy=False, nan=0.0)
    boundaries = None
    boundary_path = path.parent / "boundaries.json"
    manifest_path = path.parent / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest.get("dataset") == "SMD_pooled_70_30" and not boundary_path.is_file():
        raise ValueError("Pooled data require boundaries.json")
    source_hash = sha256(path)
    if boundary_path.exists():
        payload = json.loads(boundary_path.read_text())
        if payload.get("version") != 1:
            raise ValueError("Unknown sequence boundary format")
        boundaries = tuple((r["sequence_id"], r["start"], r["stop"])
                           for r in payload["splits"][path.stem])
        validate_boundaries(boundaries, len(x))
        if manifest and (manifest["auxiliary_sha256"]["boundaries.json"] != sha256(boundary_path)
                         or manifest["csv_sha256"][path.stem] != source_hash):
            raise ValueError("Pooled data or boundary checksum mismatch")
    return Series(x, frame.iloc[:, 0].to_numpy(), list(frame.columns[1:]), source_hash, boundaries)


def read_test(data_dir, training):
    root = Path(data_dir)
    test = read_series(root / "test.csv")
    if test.features != training.features:
        raise ValueError("Training and test feature columns/order differ")
    label_path = root / "test_label.csv"
    if not label_path.is_file():
        raise FileNotFoundError(f"Local labels not found: {label_path}")
    frame = pd.read_csv(label_path)
    if frame.shape != (len(test.values), 2):
        raise ValueError("test_label.csv must contain matching timestamp and one label column")
    if not np.array_equal(frame.iloc[:, 0].to_numpy(), test.timestamps):
        raise ValueError("Test labels and test features have different timestamps")
    labels = frame.iloc[:, 1].to_numpy()
    if not np.isin(labels, [0, 1]).all() or np.unique(labels).size != 2:
        raise ValueError("Evaluation requires binary test labels with both classes present")
    return test, labels.astype(np.int64), sha256(label_path)


def evaluation_subset(test, labels, labels_hash, config):

    if config.evaluation_split == "full":
        return test, labels, labels_hash
    if test.sequence_boundaries is not None:
        intervals, boundaries, offset = [], [], 0
        gap = config.segment_length if config.evaluation_gap is None else config.evaluation_gap
        for name, begin, end in test.sequence_boundaries:
            cut = begin + int((end-begin) * config.development_fraction)
            start, stop = ((begin, cut) if config.evaluation_split == "development" else (cut+gap, end))
            if stop-start < config.segment_length:
                raise ValueError("Every machine's evaluation split needs complete segments")
            intervals.append((start, stop))
            boundaries.append((name, offset, offset+stop-start))
            offset += stop-start
        selected_labels = np.concatenate([labels[a:b] for a, b in intervals])
        if np.unique(selected_labels).size != 2:
            raise ValueError("Evaluation split needs both label classes")
        signature = json.dumps(intervals)
        selected = Series(np.concatenate([test.values[a:b] for a, b in intervals]),
                          np.concatenate([test.timestamps[a:b] for a, b in intervals]), test.features,
                          hashlib.sha256((test.source_sha256+signature).encode()).hexdigest(), tuple(boundaries))
        return selected, selected_labels, hashlib.sha256((labels_hash+signature).encode()).hexdigest()
    cut = int(len(test.values) * config.development_fraction)
    gap = config.segment_length if config.evaluation_gap is None else config.evaluation_gap
    start, stop = ((0, cut) if config.evaluation_split == "development"
                   else (cut + gap, len(test.values)))
    selected_labels = labels[start:stop]
    if stop - start < config.segment_length or np.unique(selected_labels).size != 2:
        raise ValueError("Evaluation split needs complete segments and both label classes")
    signature = f"{start}:{stop}"
    source_hash = hashlib.sha256((test.source_sha256 + signature).encode()).hexdigest()
    label_hash = hashlib.sha256((labels_hash + signature).encode()).hexdigest()
    selected = Series(test.values[start:stop], test.timestamps[start:stop], test.features, source_hash)
    return selected, selected_labels, label_hash


def gather_segments(values, candidate_ids, length, stride):
    starts = np.asarray(candidate_ids, dtype=np.int64) * stride
    return values[starts[:, None] + np.arange(length)[None, :]]
