from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class HARSequenceDataset(Dataset):


    def __init__(
        self,
        csv_path: str | Path,
        seq_len: int,
        label_column: str,
        expected_channels: int,
    ):
        self.csv_path = Path(csv_path).expanduser().resolve()
        self.seq_len = int(seq_len)
        self.sample_stride = self.seq_len
        self.label_column = label_column
        if self.seq_len < 1:
            raise ValueError("seq_len must be positive")
        if not self.csv_path.is_file():
            raise FileNotFoundError(self.csv_path)

        frame = pd.read_csv(self.csv_path)
        if label_column not in frame.columns:
            raise ValueError(
                f"Label column {label_column!r} not found in {self.csv_path}"
            )
        feature_columns = [
            column for column in frame.columns if column != label_column
        ]
        if len(feature_columns) != expected_channels:
            raise ValueError(
                f"{self.csv_path} has {len(feature_columns)} feature columns; "
                f"expected {expected_channels}"
            )
        self.feature_columns = feature_columns
        self.data = np.ascontiguousarray(
            frame[feature_columns].to_numpy(dtype=np.float32, copy=True)
        )
        raw_labels = frame[label_column].to_numpy(dtype=np.int64, copy=True)
        unique_labels = np.unique(raw_labels)
        self.label_values = unique_labels.tolist()
        label_map = {
            int(value): index for index, value in enumerate(unique_labels)
        }
        self.labels = np.asarray(
            [label_map[int(value)] for value in raw_labels], dtype=np.int64
        )

        boundaries = np.flatnonzero(
            np.r_[True, self.labels[1:] != self.labels[:-1], True]
        )
        starts: list[int] = []
        window_labels: list[int] = []
        for run_start, run_end in zip(boundaries[:-1], boundaries[1:]):
            run_label = int(self.labels[run_start])
            last_start = run_end - self.seq_len
            if last_start < run_start:
                continue
            for start in range(run_start, last_start + 1, self.seq_len):
                starts.append(start)
                window_labels.append(run_label)
        if not starts:
            raise ValueError(
                f"No complete seq_len={self.seq_len} windows in {self.csv_path}"
            )
        self.window_starts = np.asarray(starts, dtype=np.int64)
        self.window_labels = np.asarray(window_labels, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.window_starts.size)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self.window_starts[index])
        inputs = torch.from_numpy(self.data[start : start + self.seq_len])
        target = torch.tensor(
            int(self.window_labels[index]), dtype=torch.long
        )
        return inputs, target
