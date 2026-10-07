from __future__ import annotations

import csv
import importlib
import json
import sys
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional


_TRAINING_CONFIG_FIELDS = (
    "model",
    "features",
    "target",
    "freq",
    "embed",
    "seq_len",
    "label_len",
    "pred_len",
    "enc_in",
    "individual",
    "e_layers",
    "n_heads",
    "d_model",
    "d_ff",
    "dropout",
    "fc_dropout",
    "head_dropout",
    "patch_len",
    "patch_stride",
    "padding_patch",
    "revin",
    "affine",
    "subtract_last",
    "decomposition",
    "kernel_size",
)


@dataclass
class ExperimentAdapter:
    task: str
    dataset: Any
    model: torch.nn.Module
    device: torch.device
    features: str
    pred_len: int
    label_mode: str

    def sample_loss(self, index: int) -> torch.Tensor:
        sample = self.dataset[index]
        if self.task == "classification":
            inputs, target = sample
            inputs = torch.as_tensor(inputs, dtype=torch.float32, device=self.device).unsqueeze(0)
            target = torch.as_tensor(target, dtype=torch.long, device=self.device).unsqueeze(0)
            output = self.model(inputs)
            if self.label_mode == "point":
                return functional.cross_entropy(output.reshape(-1, output.shape[-1]), target.reshape(-1))
            return functional.cross_entropy(output, target.reshape(-1))

        inputs, target, _, _ = sample
        inputs = torch.as_tensor(inputs, dtype=torch.float32, device=self.device).unsqueeze(0)
        target = torch.as_tensor(target, dtype=torch.float32, device=self.device).unsqueeze(0)
        output = self.model(inputs)
        feature_start = -1 if self.features == "MS" else 0
        output = output[:, -self.pred_len :, feature_start:]
        target = target[:, -self.pred_len :, feature_start:]
        return functional.mse_loss(output, target, reduction="mean")


def _device_from_argument(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {value!r} requested, but CUDA is unavailable")
    return device


def _activate_source(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Source directory not found: {path}")
    source = str(path)
    if source in sys.path:
        sys.path.remove(source)
    sys.path.insert(0, source)


def _apply_training_config(arguments: Any) -> Path | None:

    if not arguments.checkpoint:
        return None
    checkpoint_path = Path(arguments.checkpoint).expanduser().resolve()
    config_path = checkpoint_path.parent / "config.json"
    if not config_path.is_file():
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"Training config must be a JSON object: {config_path}")
    normalization = config.get("normalization")
    if normalization not in (None, "LayerNorm"):
        raise ValueError(
            f"Checkpoint was trained with normalization={normalization!r}; "
            "the clean PatchTST now requires LayerNorm"
        )
    for field in _TRAINING_CONFIG_FIELDS:
        if field in config:
            setattr(arguments, field, config[field])
    expected_sample_stride = int(arguments.seq_len) + int(arguments.pred_len)
    if "sample_stride" in config:
        configured_sample_stride = int(config["sample_stride"])
        if configured_sample_stride != expected_sample_stride:
            raise ValueError(
                "Checkpoint sample_stride must equal seq_len + pred_len: "
                f"got {configured_sample_stride}, expected "
                f"{arguments.seq_len} + {arguments.pred_len} = "
                f"{expected_sample_stride}. Retrain this checkpoint without "
                "a fixed cross-configuration stride."
            )
    arguments.dataset_sample_stride = expected_sample_stride
    print(f"Loaded authoritative training parameters from {config_path}")
    return config_path


def _load_model_state(model: torch.nn.Module, checkpoint: str | None, device: torch.device) -> None:
    if not checkpoint:
        return
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError("Checkpoint must be a state_dict or contain a 'state_dict' entry")
    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint does not match the requested model. "
            f"Missing keys: {missing}; unexpected keys: {unexpected}"
        )


def _model_config(arguments: Any) -> Namespace:
    return Namespace(
        seq_len=arguments.seq_len,
        pred_len=arguments.pred_len,
        enc_in=arguments.enc_in,
        seed=arguments.seed,
        individual=arguments.individual,
        e_layers=arguments.e_layers,
        n_heads=arguments.n_heads,
        d_model=arguments.d_model,
        d_ff=arguments.d_ff,
        dropout=0.0,
        fc_dropout=0.0,
        head_dropout=0.0,
        patch_len=arguments.patch_len,
        stride=arguments.patch_stride,
        padding_patch=arguments.padding_patch,
        revin=arguments.revin,
        affine=arguments.affine,
        subtract_last=arguments.subtract_last,
        decomposition=arguments.decomposition,
        kernel_size=arguments.kernel_size,
    )


def _build_model(arguments: Any) -> torch.nn.Module:
    model_name = arguments.model
    if model_name != "PatchTST":
        raise ValueError("This experiment uses PatchTST")
    module = importlib.import_module(f"models.{model_name}")
    return module.Model(_model_config(arguments)).float()


def _build_dataset(arguments: Any) -> Any:
    loader_module = importlib.import_module("data_provider.data_loader")
    if not arguments.root_path or not arguments.data_path:
        raise ValueError("Forecasting requires --root-path and --data-path")
    target = arguments.target
    if target is None:
        csv_path = Path(arguments.root_path).expanduser() / arguments.data_path
        with csv_path.open(newline="", encoding="utf-8") as handle:
            columns = next(csv.reader(handle), [])
        feature_columns = [column for column in columns if column != "date"]
        if "date" not in columns or not feature_columns:
            raise ValueError("CSV must contain a 'date' column and at least one feature column")
        target = feature_columns[-1]
        arguments.target = target
    return loader_module.Dataset_Custom(
        root_path=arguments.root_path,
        data_path=arguments.data_path,
        flag=arguments.split,
        size=[arguments.seq_len, arguments.label_len, arguments.pred_len],
        features=arguments.features,
        target=target,
        timeenc=0 if arguments.embed != "timeF" else 1,
        freq=arguments.freq,
        sample_stride=getattr(arguments, "dataset_sample_stride", None),
    )


def build_adapter(arguments: Any) -> ExperimentAdapter:
    _apply_training_config(arguments)
    project_root = Path(__file__).resolve().parent
    if arguments.task != "forecasting":
        raise ValueError(
            "The clean models in model_code are forecasting models and do not provide "
            "a classification head; --task must be forecasting"
        )

    _activate_source(project_root / "model_code")
    for name in list(sys.modules):
        if name == "models" or name.startswith("models.") or name == "layers" or name.startswith("layers."):
            del sys.modules[name]
    device = _device_from_argument(arguments.device)
    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(arguments.seed)
    dataset = _build_dataset(arguments)
    if len(dataset) < 1:
        raise ValueError(f"The requested {arguments.split!r} split has no complete samples")
    first_input = dataset[0][0]
    observed_channels = int(np.asarray(first_input).shape[-1])
    if observed_channels != arguments.enc_in:
        raise ValueError(
            f"Dataset produces {observed_channels} channels but --enc-in={arguments.enc_in}. "
            "Use the actual number of non-date feature columns expected by the current loader."
        )
    model = _build_model(arguments).to(device)
    _load_model_state(model, arguments.checkpoint, device)
    model.eval()
    return ExperimentAdapter(
        task=arguments.task,
        dataset=dataset,
        model=model,
        device=device,
        features=arguments.features,
        pred_len=arguments.pred_len,
        label_mode="sequence",
    )
