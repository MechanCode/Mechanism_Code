from __future__ import annotations

import argparse
import importlib
import json
import math
import random
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from connect_privacy.accountant import PLDAccountant
from connect_privacy.data import (
    dataset_preset,
    evaluate_classification,
    load_classification_cache,
    resolve_split_paths,
)
from connect_privacy.mixtures import build_classification_wr_mixture_parameters
from connect_privacy.private_step import private_classification_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("HAR", "USC"), required=True)
    parser.add_argument("--model", choices=("DLinear", "PatchTST"), required=True)
    parser.add_argument("--train-path", default=None)
    parser.add_argument("--validation-path", default=None)
    parser.add_argument("--test-path", default=None)
    parser.add_argument(
        "--data-cache", type=Path, required=True,
        help="Prepared HAR/USC cache from prepare_n1_data.py",
    )
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--enc-in", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument(
        "--sample-stride", type=int, default=1,
        help="Stride between candidate training-window starts",
    )
    parser.add_argument("--target-epsilon", type=float, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)


    parser.set_defaults(
        clipping_norm=1.0, noise_multiplier=5.0, event_fraction=0.005,
        target_delta=1e-5, max_optimizer_steps=100_000,
        num_workers=0, eval_batch_size=128, pld_interval=0.001,
    )
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {value!r} requested but CUDA is unavailable")
    return device


def apply_dataset_defaults(args: argparse.Namespace) -> None:
    preset = dataset_preset(args.dataset)
    for name in ("seq_len", "enc_in", "num_classes"):
        if getattr(args, name) is None:
            setattr(args, name, int(preset[name]))
    args.eval_window_stride = args.seq_len


def validate_args(args: argparse.Namespace) -> None:
    apply_dataset_defaults(args)
    positive_ints = {
        "seq_len": args.seq_len,
        "enc_in": args.enc_in,
        "num_classes": args.num_classes,
        "batch_size": args.batch_size,
        "sample_stride": args.sample_stride,
        "eval_window_stride": args.eval_window_stride,
        "max_optimizer_steps": args.max_optimizer_steps,
        "eval_batch_size": args.eval_batch_size,
    }
    if any(value < 1 for value in positive_ints.values()):
        raise ValueError(f"positive integer parameters required: {positive_ints}")
    if args.clipping_norm <= 0 or args.noise_multiplier <= 0:
        raise ValueError("clipping norm and noise multiplier must be positive")
    if not 0 < args.event_fraction <= 1:
        raise ValueError("event_fraction must be in (0, 1]")
    if args.target_epsilon <= 0 or not 0 < args.target_delta < 1:
        raise ValueError("target epsilon must be positive and delta must be in (0, 1)")
    if args.learning_rate <= 0:
        raise ValueError("learning rate must be positive")


def model_config(args: argparse.Namespace) -> Namespace:
    return Namespace(
        seq_len=args.seq_len,
        enc_in=args.enc_in,
        num_classes=args.num_classes,
    )


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def _check_cache_sources(metadata: dict[str, object], paths: tuple[Path, Path, Path]) -> None:
    for prefix, expected in zip(
        ("train_source", "validation_source", "test_source"), paths
    ):
        observed = Path(str(metadata[f"{prefix}_path"]))
        if observed != expected:
            raise ValueError(f"cache source {observed} does not match requested split {expected}")


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    device = resolve_device(args.device)
    paths = resolve_split_paths(
        args.dataset, args.train_path, args.validation_path, args.test_path
    )

    training, validation, test, metadata = load_classification_cache(args.data_cache)
    _check_cache_sources(metadata, paths)
    feature_columns = list(metadata["feature_columns"])
    preprocessing = str(metadata["preprocessing"])

    train_features, train_labels = training
    channels = int(train_features.shape[1])
    inferred_classes = int(train_labels.max()) + 1
    if channels != args.enc_in:
        raise ValueError(
            f"--enc-in={args.enc_in}, but {args.dataset} data has {channels} feature columns"
        )
    if inferred_classes != args.num_classes:
        raise ValueError(
            f"--num-classes={args.num_classes}, but training labels imply {inferred_classes} classes"
        )
    if len(train_features) < args.seq_len:
        raise ValueError("training split contains no complete classification window")

    event_length = math.ceil(args.event_fraction * len(train_features))
    parameters = build_classification_wr_mixture_parameters(
        sequence_length=len(train_features),
        window_length=args.seq_len,
        event_length=event_length,
        batch_size=args.batch_size,
        noise_multiplier=args.noise_multiplier,
        sample_stride=args.sample_stride,
    )
    accountant = PLDAccountant(
        parameters, args.target_delta, privacy_loss_interval=args.pld_interval
    )
    optimizer_steps, budget_exhausted = accountant.maximum_steps(
        args.target_epsilon, maximum_steps=args.max_optimizer_steps
    )
    achieved_epsilon = accountant.epsilon(optimizer_steps)
    next_epsilon = accountant.epsilon(optimizer_steps + 1) if budget_exhausted else None
    if achieved_epsilon > args.target_epsilon + 1e-10:
        raise RuntimeError("accountant selected a step count above the privacy budget")
    if budget_exhausted and next_epsilon is not None and next_epsilon <= args.target_epsilon:
        raise RuntimeError("accountant stopped before the actual privacy boundary")
    print(
        f"dataset={args.dataset} target_epsilon={args.target_epsilon:g} "
        f"sigma={args.noise_multiplier:g} allowed_steps={optimizer_steps} "
        f"accounted_epsilon={achieved_epsilon:.6f}"
    )

    module = importlib.import_module(f"models.{args.model}")
    model = module.Model(model_config(args)).float().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    train_feature_tensor = torch.as_tensor(train_features, dtype=torch.float32)
    train_label_tensor = torch.as_tensor(train_labels, dtype=torch.long)
    history: list[dict[str, float | int]] = []
    for step in range(1, optimizer_steps + 1):
        statistics = private_classification_step(
            model,
            optimizer,
            train_feature_tensor,
            train_label_tensor,
            args.seq_len,
            args.batch_size,
            args.clipping_norm,
            args.noise_multiplier,
            sample_stride=args.sample_stride,
            generator=generator,
        )
        history.append(
            {
                "step": step,
                "mean_cross_entropy": statistics.mean_loss,
                "clipped_fraction": statistics.clipped_fraction,
                "mean_gradient_norm": statistics.mean_gradient_norm,
            }
        )
        if step == 1 or step % 10 == 0 or step == optimizer_steps:
            print(
                f"step={step}/{optimizer_steps} cross_entropy={statistics.mean_loss:.8f} "
                f"clipped={statistics.clipped_fraction:.3f}"
            )

    validation_metrics = evaluate_classification(
        model,
        validation[0],
        validation[1],
        args.seq_len,
        device,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        window_stride=args.eval_window_stride,
    )
    test_metrics = evaluate_classification(
        model,
        test[0],
        test[1],
        args.seq_len,
        device,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        window_stride=args.eval_window_stride,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(cpu_state_dict(model), args.output_dir / "checkpoint.pth")
    privacy_report = accountant.report(optimizer_steps, args.target_epsilon)
    privacy_report.update(
        {
            "event_fraction_of_training_sequence": args.event_fraction,
            "clipping_norm": args.clipping_norm,
            "next_step_epsilon": next_epsilon,
            "budget_exhausted": budget_exhausted,
            "noise_added_to": "sum_of_clipped_window_gradients",
            "noise_std": args.noise_multiplier * args.clipping_norm,
            "released_gradient": "(sum + noise) / batch_size",
        }
    )
    config = vars(args).copy()
    config.update(
        {
            "output_dir": str(args.output_dir),
            "data_cache": str(args.data_cache),
            "train_path": str(paths[0]),
            "validation_path": str(paths[1]),
            "test_path": str(paths[2]),
            "label_mode": "sequence",
            "optimizer": "sgd",
            "feature_columns": feature_columns,
            "preprocessing": preprocessing,
            "padding_patch": "end",
            "normalization": "LayerNorm" if args.model == "PatchTST" else None,
        }
    )
    metrics: dict[str, object] = {
        "validation": validation_metrics,
        "test": test_metrics,
        "validation_accuracy": validation_metrics["accuracy"],
        "validation_macro_f1": validation_metrics["macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "optimizer_steps": optimizer_steps,
        "accounted_epsilon": achieved_epsilon,
        "history": history,
    }
    for name, payload in (
        ("config.json", config),
        ("privacy_report.json", privacy_report),
        ("metrics.json", metrics),
    ):
        (args.output_dir / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(
        f"saved={args.output_dir} val_accuracy={validation_metrics['accuracy']:.6f} "
        f"val_macro_f1={validation_metrics['macro_f1']:.6f} "
        f"test_accuracy={test_metrics['accuracy']:.6f} "
        f"test_macro_f1={test_metrics['macro_f1']:.6f}"
    )
    return {**metrics, "privacy_report": privacy_report}


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
