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
    evaluate_mse,
    load_standardized_cache,
)
from connect_privacy.mixtures import build_wr_mixture_parameters
from connect_privacy.private_step import private_training_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("DLinear", "PatchTST"), required=True)
    parser.add_argument("--root-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument(
        "--data-cache",
        type=Path,
        required=True,
        help="Prepared StandardScaler cache from prepare_n1_data.py",
    )
    parser.add_argument("--context-len", type=int, default=80)
    parser.add_argument("--forecast-len", type=int, default=20)
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=1,
        help="Stride between candidate training-window starts (default: 1)",
    )
    parser.add_argument("--target-epsilon", type=float, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)


    parser.set_defaults(
        features='M',
        target=None,
        batch_size=50,
        clipping_norm=1.0,
        noise_multiplier=5.0,
        event_fraction=0.005,
        target_delta=1e-05,
        max_optimizer_steps=100000,
        eval_batch_size=128,
        pld_interval=0.001,
        individual=0,
        e_layers=2,
        n_heads=8,
        d_model=128,
        d_ff=2048,
        dropout=0.05,
        fc_dropout=0.05,
        head_dropout=0.0,
        patch_len=16,
        patch_stride=8,
        padding_patch='end',
        revin=0,
        affine=0,
        subtract_last=0,
        decomposition=0,
        kernel_size=25,
    )
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {value!r} requested but CUDA is unavailable")
    return device


def validate_args(args: argparse.Namespace) -> None:
    positive_ints = {
        "context_len": args.context_len,
        "forecast_len": args.forecast_len,
        "batch_size": args.batch_size,
        "sample_stride": args.sample_stride,
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


def model_config(args: argparse.Namespace, channels: int) -> Namespace:
    return Namespace(seq_len=args.context_len, pred_len=args.forecast_len, enc_in=channels)


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    device = resolve_device(args.device)

    train_array, validation_array, test_array, cache_metadata = (
        load_standardized_cache(args.data_cache)
    )
    expected_source = (
        Path(args.root_path).expanduser() / args.data_path
    ).resolve()
    if Path(str(cache_metadata["source_path"])) != expected_source:
        raise ValueError(
            f"cache source {cache_metadata['source_path']} does not match "
            f"requested dataset {expected_source}"
        )
    feature_columns = list(cache_metadata["feature_columns"])
    preprocessing = str(cache_metadata["preprocessing"])
    channels = int(train_array.shape[1])
    target = feature_columns[-1]

    event_length = math.ceil(args.event_fraction * len(train_array))
    parameters = build_wr_mixture_parameters(
        sequence_length=len(train_array),
        context_len=args.context_len,
        forecast_len=args.forecast_len,
        event_length=event_length,
        batch_size=args.batch_size,
        noise_multiplier=args.noise_multiplier,
        sample_stride=args.sample_stride,
    )
    accountant = PLDAccountant(
        parameters,
        args.target_delta,
        privacy_loss_interval=args.pld_interval,
    )
    optimizer_steps, budget_exhausted = accountant.maximum_steps(
        args.target_epsilon,
        maximum_steps=args.max_optimizer_steps,
    )
    achieved_epsilon = accountant.epsilon(optimizer_steps)
    next_epsilon = (
        accountant.epsilon(optimizer_steps + 1)
        if budget_exhausted
        else None
    )
    if achieved_epsilon > args.target_epsilon + 1e-10:
        raise RuntimeError("accountant selected a step count above the privacy budget")
    if budget_exhausted and next_epsilon is not None:
        if next_epsilon <= args.target_epsilon:
            raise RuntimeError("accountant stopped before the actual budget boundary")

    print(
        f"target_epsilon={args.target_epsilon:g} sigma={args.noise_multiplier:g} "
        f"allowed_steps={optimizer_steps} accounted_epsilon={achieved_epsilon:.6f}"
    )
    if optimizer_steps == 0:
        print("warning: the first optimizer step already exceeds the target budget")

    module = importlib.import_module(f"models.{args.model}")
    model = module.Model(model_config(args, channels)).float().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    train_series = torch.as_tensor(train_array, dtype=torch.float32)
    history: list[dict[str, float | int]] = []

    validation_interval_steps = 10
    best_validation_mse = float("inf")
    best_step = 0
    best_state = None

    def validate_and_select(step: int) -> float:
        nonlocal best_validation_mse, best_step, best_state
        value = evaluate_mse(
            model,
            validation_array,
            args.context_len,
            args.forecast_len,
            device,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
        )
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite validation MSE at step {step}: {value}")

        if value <= best_validation_mse:
            best_validation_mse = value
            best_step = step
            best_state = cpu_state_dict(model)
        print(
            f"step={step} validation_mse={value:.8f} "
            f"best_step={best_step} best_validation_mse={best_validation_mse:.8f}"
        )
        return value

    for step in range(1, optimizer_steps + 1):
        statistics = private_training_step(
            model,
            optimizer,
            train_series,
            args.context_len,
            args.forecast_len,
            args.batch_size,
            args.clipping_norm,
            args.noise_multiplier,
            sample_stride=args.sample_stride,
            generator=generator,
        )
        history.append(
            {
                "step": step,
                "mean_loss": statistics.mean_loss,
                "clipped_fraction": statistics.clipped_fraction,
                "mean_gradient_norm": statistics.mean_gradient_norm,
            }
        )
        if step == 1 or step % 10 == 0 or step == optimizer_steps:
            print(
                f"step={step}/{optimizer_steps} loss={statistics.mean_loss:.8f} "
                f"clipped={statistics.clipped_fraction:.3f}"
            )
        if step % validation_interval_steps == 0 or step == optimizer_steps:
            final_validation_mse = validate_and_select(step)
            history[-1]["validation_mse"] = final_validation_mse

    if optimizer_steps == 0:
        final_validation_mse = validate_and_select(0)
    assert best_state is not None
    model.load_state_dict(best_state, strict=True)
    validation_mse = best_validation_mse
    test_mse = evaluate_mse(
        model,
        test_array,
        args.context_len,
        args.forecast_len,
        device,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
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
            "noise_added_to": "sum_of_clipped_gradients",
            "noise_std": args.noise_multiplier * args.clipping_norm,
            "released_gradient": "(sum + noise) / batch_size",
        }
    )
    config = vars(args).copy()
    config["output_dir"] = str(args.output_dir)
    config["data_cache"] = str(args.data_cache)
    config["target"] = target
    config["channels"] = channels
    config["feature_columns"] = feature_columns
    config["preprocessing"] = preprocessing
    config["normalization"] = "LayerNorm" if args.model == "PatchTST" else None
    metrics = {
        "validation_mse": validation_mse,
        "test_mse": test_mse,
        "checkpoint_selection": "best_validation_mse",
        "validation_interval_steps": validation_interval_steps,
        "best_step": best_step,
        "final_validation_mse": final_validation_mse,
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
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        f"saved={args.output_dir} best_step={best_step} "
        f"validation_mse={validation_mse:.8f} "
        f"test_mse={test_mse:.8f}"
    )
    return {**metrics, "privacy_report": privacy_report}


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
