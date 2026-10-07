from __future__ import annotations

import argparse
import importlib
import json
import random
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from data_provider.classification_loader import HARSequenceDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=("PatchTST",), default="PatchTST"
    )
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--val-path", required=True)
    parser.add_argument("--test-path", required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--seq-len", type=int, default=200)
    parser.add_argument("--enc-in", type=int, required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--e-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--patch-stride", type=int, default=8)
    parser.add_argument("--padding-patch", choices=("end", "none"), default="end")
    parser.add_argument("--individual", type=int, choices=(0, 1), default=0)
    parser.add_argument("--kernel-size", type=int, default=25)
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {value!r} requested but unavailable")
    return device


def model_config(args: argparse.Namespace) -> Namespace:
    return Namespace(
        seq_len=args.seq_len,
        enc_in=args.enc_in,
        num_classes=args.num_classes,
        e_layers=args.e_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
        patch_len=args.patch_len,
        stride=args.patch_stride,
        padding_patch=None
        if args.padding_patch == "none"
        else args.padding_patch,
        individual=args.individual,
        kernel_size=args.kernel_size,
    )


def make_dataset(
    args: argparse.Namespace, split: str
) -> HARSequenceDataset:
    return HARSequenceDataset(
        csv_path=getattr(args, f"{split}_path"),
        seq_len=args.seq_len,
        label_column=args.label_column,
        expected_channels=args.enc_in,
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    count = 0
    predictions: list[int] = []
    targets: list[int] = []
    with torch.no_grad():
        for inputs, target in loader:
            inputs = inputs.float().to(device)
            target = target.long().to(device)
            logits = model(inputs)
            loss = nn.functional.cross_entropy(logits, target, reduction="sum")
            total_loss += float(loss)
            count += int(target.numel())
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            targets.extend(target.cpu().tolist())
    return {
        "cross_entropy": total_loss / count,
        "accuracy": float(np.mean(np.asarray(predictions) == np.asarray(targets))),
        "macro_f1": float(f1_score(targets, predictions, average="macro")),
    }


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("--epochs and --patience must both be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)

    datasets = {
        split: make_dataset(args, split)
        for split in ("train", "val", "test")
    }
    for split in ("val", "test"):
        if datasets[split].label_values != datasets["train"].label_values:
            raise ValueError(
                f"{split} labels {datasets[split].label_values} do not match "
                f"training labels {datasets['train'].label_values}"
            )
    observed_classes = sorted(
        set(datasets["train"].window_labels.tolist())
    )
    if len(observed_classes) != args.num_classes:
        raise ValueError(
            f"Training split has {len(observed_classes)} classes; "
            f"expected {args.num_classes}"
        )
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
        total_loss = 0.0
        count = 0
        for inputs, target in loaders["train"]:
            inputs = inputs.float().to(device)
            target = target.long().to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = nn.functional.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            batch_count = int(target.numel())
            total_loss += float(loss.detach()) * batch_count
            count += batch_count
        validation = evaluate(model, loaders["val"], device)
        record = {
            "epoch": epoch,
            "train_cross_entropy": total_loss / count,
            "val_cross_entropy": validation["cross_entropy"],
            "val_accuracy": validation["accuracy"],
            "val_macro_f1": validation["macro_f1"],
        }
        history.append(record)
        print(
            f"epoch={epoch} train_ce={record['train_cross_entropy']:.8f} "
            f"val_ce={record['val_cross_entropy']:.8f} "
            f"val_acc={record['val_accuracy']:.6f}"
        )
        if validation["cross_entropy"] < best_val:
            best_val = validation["cross_entropy"]
            best_epoch = epoch
            stale_epochs = 0
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
    validation = evaluate(model, loaders["val"], device)
    test = evaluate(model, loaders["test"], device)

    config = vars(args).copy()
    config["checkpoint_dir"] = str(args.checkpoint_dir)
    config["task"] = "classification"
    config["label_mode"] = "sequence"
    config["sample_stride"] = args.seq_len
    config["padding_patch"] = (
        None if args.padding_patch == "none" else args.padding_patch
    )
    config["dropout"] = 0.0
    config["normalization"] = "LayerNorm"
    (args.checkpoint_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics = {
        "best_epoch": best_epoch,
        "best_val_cross_entropy": best_val,
        "validation": validation,
        "test": test,
        "sample_stride": args.seq_len,
        "sample_counts": {
            split: len(dataset) for split, dataset in datasets.items()
        },
        "history": history,
    }
    (args.checkpoint_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"saved checkpoint: {checkpoint_path}")
    print(
        f"best_epoch={best_epoch} test_ce={test['cross_entropy']:.8f} "
        f"test_acc={test['accuracy']:.6f} test_f1={test['macro_f1']:.6f}"
    )


if __name__ == "__main__":
    main()
