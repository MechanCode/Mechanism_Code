from __future__ import annotations

import argparse
import importlib
import json
import random
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from data_provider.data_loader import Dataset_Custom


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("PatchTST",), required=True)
    parser.add_argument("--root-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--target", default=None, help="Defaults to the last non-date CSV column")
    parser.add_argument("--features", choices=("M", "S", "MS"), default="M")
    parser.add_argument("--freq", default="h")
    parser.add_argument("--embed", default="learned")
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--label-len", type=int, default=48)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--enc-in", type=int, required=True)
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=None,
        help="Window stride; defaults to seq_len + pred_len and must not be smaller",
    )

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or e.g. cuda:1")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)

    parser.add_argument("--individual", type=int, choices=(0, 1), default=0)
    parser.add_argument("--e-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.0, help="Must be 0")
    parser.add_argument("--fc-dropout", type=float, default=0.0, help="Must be 0")
    parser.add_argument("--head-dropout", type=float, default=0.0)
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--patch-stride", type=int, default=8)
    parser.add_argument("--padding-patch", choices=("end", "none"), default="end")
    parser.add_argument("--revin", type=int, choices=(0, 1), default=1)
    parser.add_argument("--affine", type=int, choices=(0, 1), default=0)
    parser.add_argument("--subtract-last", type=int, choices=(0, 1), default=0)
    parser.add_argument("--decomposition", type=int, choices=(0, 1), default=0)
    parser.add_argument("--kernel-size", type=int, default=25)
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {value!r} requested, but CUDA is unavailable")
    return device


def resolve_target(args: argparse.Namespace) -> str:
    csv_path = Path(args.root_path).expanduser() / args.data_path
    columns = list(pd.read_csv(csv_path, nrows=0).columns)
    feature_columns = [column for column in columns if column != "date"]
    if "date" not in columns or not feature_columns:
        raise ValueError("CSV must contain a 'date' column and at least one feature column")
    target = args.target or feature_columns[-1]
    if target not in feature_columns:
        raise ValueError(f"Target {target!r} is not a non-date CSV column")
    expected_channels = 1 if args.features == "S" else len(feature_columns)
    if args.enc_in != expected_channels:
        raise ValueError(
            f"--enc-in={args.enc_in} but features={args.features!r} produces "
            f"{expected_channels} channel(s)"
        )
    return target


def model_config(args: argparse.Namespace) -> Namespace:
    return Namespace(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        enc_in=args.enc_in,
        individual=args.individual,
        e_layers=args.e_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
        dropout=0.0,
        fc_dropout=0.0,
        head_dropout=0.0,
        patch_len=args.patch_len,
        stride=args.patch_stride,
        padding_patch=None if args.padding_patch == "none" else args.padding_patch,
        revin=args.revin,
        affine=args.affine,
        subtract_last=args.subtract_last,
        decomposition=args.decomposition,
        kernel_size=args.kernel_size,
    )


def make_dataset(args: argparse.Namespace, target: str, split: str) -> Dataset_Custom:
    return Dataset_Custom(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=split,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=target,
        timeenc=0 if args.embed != "timeF" else 1,
        freq=args.freq,
        sample_stride=args.sample_stride,
    )


def batch_loss(
    model: nn.Module,
    batch: tuple[torch.Tensor, ...],
    device: torch.device,
    pred_len: int,
    features: str,
) -> torch.Tensor:
    inputs, targets = batch[0].float().to(device), batch[1].float().to(device)
    feature_start = -1 if features == "MS" else 0
    predictions = model(inputs)[:, -pred_len:, feature_start:]
    targets = targets[:, -pred_len:, feature_start:]
    return nn.functional.mse_loss(predictions, targets)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pred_len: int,
    features: str,
) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            loss = batch_loss(model, batch, device, pred_len, features)
            size = int(batch[0].shape[0])
            total += float(loss) * size
            count += size
    if count == 0:
        raise ValueError("Evaluation split contains no complete samples")
    return total / count


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("--epochs and --patience must both be positive")
    if any(value != 0.0 for value in (args.dropout, args.fc_dropout, args.head_dropout)):
        raise ValueError("PatchTST dropout is disabled; all dropout arguments must be 0")
    args.sample_stride = args.sample_stride or args.seq_len + args.pred_len
    if args.sample_stride < args.seq_len + args.pred_len:
        raise ValueError("--sample-stride must be at least seq_len + pred_len")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.device)
    target = resolve_target(args)
    datasets = {split: make_dataset(args, target, split) for split in ("train", "val", "test")}
    for split, dataset in datasets.items():
        if len(dataset) < 1:
            raise ValueError(f"The {split!r} split contains no complete samples")
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.num_workers,
            generator=generator if split == "train" else None,
        )
        for split, dataset in datasets.items()
    }

    module = importlib.import_module(f"models.{args.model}")
    model = module.Model(model_config(args)).float().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_dir / "checkpoint.pth"

    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total, train_count = 0.0, 0
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(model, batch, device, args.pred_len, args.features)
            loss.backward()
            optimizer.step()
            size = int(batch[0].shape[0])
            train_total += float(loss.detach()) * size
            train_count += size
        train_loss = train_total / train_count
        val_loss = evaluate(model, loaders["val"], device, args.pred_len, args.features)
        history.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss})
        print(f"epoch={epoch} train_mse={train_loss:.8f} val_mse={val_loss:.8f}")
        if val_loss < best_val:
            best_val, best_epoch, stale_epochs = val_loss, epoch, 0
            torch.save(cpu_state_dict(model), checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state, strict=True)
    test_loss = evaluate(model, loaders["test"], device, args.pred_len, args.features)

    config = vars(args).copy()
    config["checkpoint_dir"] = str(args.checkpoint_dir)
    config["target"] = target
    config["sample_stride"] = args.sample_stride
    config["padding_patch"] = None if args.padding_patch == "none" else args.padding_patch
    config["normalization"] = "LayerNorm" if args.model == "PatchTST" else None
    config["dropout"] = 0.0
    config["fc_dropout"] = 0.0
    config["head_dropout"] = 0.0
    (args.checkpoint_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metrics = {
        "best_epoch": best_epoch,
        "best_val_mse": best_val,
        "test_mse": test_loss,
        "sample_stride": args.sample_stride,
        "sample_counts": {split: len(dataset) for split, dataset in datasets.items()},
        "history": history,
    }
    (args.checkpoint_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"saved checkpoint: {checkpoint_path}")
    print(f"best_epoch={best_epoch} test_mse={test_loss:.8f}")


if __name__ == "__main__":
    main()
