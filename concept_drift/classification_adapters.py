from __future__ import annotations

import importlib
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .adapters import ExperimentAdapter


_CONFIG_FIELDS = (
    "model",
    "train_path",
    "val_path",
    "test_path",
    "label_column",
    "seq_len",
    "enc_in",
    "num_classes",
    "e_layers",
    "n_heads",
    "d_model",
    "d_ff",
    "patch_len",
    "patch_stride",
    "padding_patch",
    "individual",
    "kernel_size",
)


def _activate_source(path: Path) -> None:
    source = str(path)
    if source in sys.path:
        sys.path.remove(source)
    sys.path.insert(0, source)
    for name in list(sys.modules):
        if (
            name == "models"
            or name.startswith("models.")
            or name == "layers"
            or name.startswith("layers.")
            or name == "data_provider"
            or name.startswith("data_provider.")
        ):
            del sys.modules[name]


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {value!r} requested but unavailable")
    return device


def _apply_checkpoint_config(arguments: Any) -> Path:
    checkpoint = Path(arguments.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config_path = checkpoint.parent / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("task") != "classification":
        raise ValueError(f"Not a classification checkpoint: {config_path}")
    for field in _CONFIG_FIELDS:
        if field in config:
            setattr(arguments, field, config[field])
    print(f"Loaded authoritative training parameters from {config_path}")
    return checkpoint


def _model_config(arguments: Any) -> Namespace:
    return Namespace(
        seq_len=arguments.seq_len,
        enc_in=arguments.enc_in,
        num_classes=arguments.num_classes,
        e_layers=arguments.e_layers,
        n_heads=arguments.n_heads,
        d_model=arguments.d_model,
        d_ff=arguments.d_ff,
        patch_len=arguments.patch_len,
        stride=arguments.patch_stride,
        padding_patch=arguments.padding_patch,
        individual=getattr(arguments, "individual", 0),
        kernel_size=getattr(arguments, "kernel_size", 25),
    )


def build_classification_adapter(arguments: Any) -> ExperimentAdapter:
    checkpoint = _apply_checkpoint_config(arguments)
    project_root = Path(__file__).resolve().parent
    _activate_source(project_root / "classification_model_code")
    device = _resolve_device(arguments.device)
    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(arguments.seed)

    loader_module = importlib.import_module(
        "data_provider.classification_loader"
    )
    csv_path = getattr(arguments, f"{arguments.split}_path")
    dataset = loader_module.HARSequenceDataset(
        csv_path=csv_path,
        seq_len=arguments.seq_len,
        label_column=arguments.label_column,
        expected_channels=arguments.enc_in,
    )
    model_module = importlib.import_module(f"models.{arguments.model}")
    model = model_module.Model(_model_config(arguments)).float().to(device)
    try:
        state = torch.load(
            checkpoint, map_location=device, weights_only=True
        )
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    model.load_state_dict(state, strict=True)
    model.eval()
    return ExperimentAdapter(
        task="classification",
        dataset=dataset,
        model=model,
        device=device,
        features="",
        pred_len=0,
        label_mode="sequence",
    )
