#!/usr/bin/env python3


from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from .sampling import (
    DEFAULT_METHODS,
    METHODS,
    analytical_sampling_traces,
    balanced_temporal_windows,
    bootstrap_trace_comparison,
    run_sampling_experiment,
)
from .poisson_config import add_poisson_arguments, resolve_poisson_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Empirical batch-gradient covariance with fixed model and dataset"
    )
    parser.add_argument("--task", choices=["forecasting", "classification"], required=True)
    parser.add_argument("--model", choices=["PatchTST"], default="PatchTST")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint fixed for every method")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, ...")
    parser.add_argument("--output-dir", required=True)

    data = parser.add_argument_group("data")
    data.add_argument("--root-path", default=None)
    data.add_argument("--data-path", default=None)
    data.add_argument("--features", choices=["M", "S", "MS"], default="M")
    data.add_argument("--target", default=None, help="Defaults to the last non-date CSV column")
    data.add_argument("--freq", default="h")
    data.add_argument("--embed", default="learned")
    data.add_argument("--seq-len", type=int, default=96)
    data.add_argument("--label-len", type=int, default=48)
    data.add_argument("--pred-len", type=int, default=96)
    data.add_argument("--enc-in", type=int, default=7)

    sampling = parser.add_argument_group("batch-gradient sampling and covariance")
    sampling.add_argument(
        "--methods", nargs="+", choices=list(METHODS), default=list(DEFAULT_METHODS)
    )
    add_poisson_arguments(sampling)
    sampling.add_argument("--batch-size", type=int, default=60, help="Expected batch size K")
    sampling.add_argument("--trials", type=int, default=10000)
    sampling.add_argument("--bootstrap", type=int, default=0)
    sampling.add_argument(
        "--candidate-stride",
        type=int,
        default=None,
        help="Stride in already non-overlapping Dataset indices; default 1",
    )
    sampling.add_argument(
        "--projection",
        choices=["countsketch", "none"],
        default="countsketch",
        help="Gradient representation; 'none' retains the full flattened gradient",
    )
    sampling.add_argument(
        "--projection-dim",
        type=int,
        default=128,
        help="CountSketch output dimension (ignored with --projection none)",
    )
    sampling.add_argument(
        "--projection-seed",
        type=int,
        default=2026,
        help="CountSketch seed (ignored with --projection none)",
    )
    sampling.add_argument(
        "--projection-seed-count",
        type=int,
        default=None,
        help=(
            "Number of consecutive CountSketch seeds to evaluate, starting at "
            "--projection-seed. Defaults to 30 for hybrid CountSketch analysis "
            "and 1 otherwise. The raw gradients are collected only once."
        ),
    )
    sampling.add_argument(
        "--clip",
        action="store_true",
        help="Enable per-sample gradient clipping using --clipping-norm",
    )
    sampling.add_argument(
        "--clipping-norm",
        type=float,
        default=None,
        help="Positive clipping threshold used when --clip is enabled",
    )
    sampling.add_argument("--seed", type=int, default=42)
    sampling.add_argument("--progress-every", type=int, default=25)
    sampling.add_argument(
        "--trace-only",
        action="store_true",
        help=(
            "Compute analytical covariance traces only. This avoids estimator "
            "and covariance matrices and makes --projection none practical."
        ),
    )

    model = parser.add_argument_group("PatchTST/model compatibility")
    model.add_argument("--individual", type=int, default=0)
    model.add_argument("--e-layers", type=int, default=2)
    model.add_argument("--n-heads", type=int, default=8)
    model.add_argument("--d-model", type=int, default=128)
    model.add_argument("--d-ff", type=int, default=512)
    model.add_argument("--dropout", type=float, default=0.0)
    model.add_argument("--fc-dropout", type=float, default=0.0)
    model.add_argument("--head-dropout", type=float, default=0.0)
    model.add_argument("--patch-len", type=int, default=16)
    model.add_argument("--patch-stride", type=int, default=8)
    model.add_argument("--padding-patch", default="end")
    model.add_argument("--revin", type=int, default=1)
    model.add_argument("--affine", type=int, default=0)
    model.add_argument("--subtract-last", type=int, default=0)
    model.add_argument("--decomposition", type=int, default=0)
    model.add_argument("--kernel-size", type=int, default=25)
    return parser


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _sampling_report(
    results: dict[str, Any],
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    baseline_trace = results["baseline"].metrics["trace"] if "baseline" in results else None
    for method, result in results.items():
        empirical_trace = result.metrics["trace"]
        comparison = None
        if method != "baseline" and "baseline" in results:
            comparison = bootstrap_trace_comparison(
                results["baseline"].gradient_estimators,
                result.gradient_estimators,
                repetitions=bootstrap,
                seed=seed + 50_021 * (len(report) + 1),
            )
            comparison = {
                key: (
                    [_finite_or_none(item) for item in value]
                    if isinstance(value, list)
                    else _finite_or_none(value)
                )
                for key, value in comparison.items()
            }
        report[method] = {
            "gradient_covariance": result.metrics,
            "trace_bootstrap_ci95": [
                _finite_or_none(value) for value in result.bootstrap_trace_ci95
            ],
            "batch_size_mean": float(result.batch_sizes.mean()),
            "batch_size_std": float(result.batch_sizes.std()),
            "batch_size_min": int(result.batch_sizes.min()),
            "batch_size_max": int(result.batch_sizes.max()),
            "trace_ratio_to_baseline": empirical_trace / baseline_trace
            if baseline_trace
            else None,
            "trace_reduction_vs_baseline": 1.0 - empirical_trace / baseline_trace
            if baseline_trace
            else None,
            "bootstrap_comparison_to_baseline": comparison,
        }
    return report


def _write_sampling_csv(path: Path, report: dict[str, Any]) -> None:
    fields = [
        "method",
        "covariance_trace",
        "covariance_trace_std_across_projection_seeds",
        "trace_ci95_low",
        "trace_ci95_high",
        "covariance_frobenius",
        "covariance_frobenius_std_across_projection_seeds",
        "covariance_spectral",
        "covariance_spectral_std_across_projection_seeds",
        "covariance_minimum_eigenvalue",
        "covariance_minimum_eigenvalue_std_across_projection_seeds",
        "covariance_effective_rank",
        "covariance_effective_rank_std_across_projection_seeds",
        "trace_ratio_to_baseline",
        "trace_ratio_to_baseline_std_across_projection_seeds",
        "trace_reduction_vs_baseline",
        "trace_reduction_vs_baseline_std_across_projection_seeds",
        "batch_size_mean",
        "batch_size_std",
        "batch_size_min",
        "batch_size_max",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, values in report.items():
            seed_statistics = values.get("projection_seed_statistics", {})

            def seed_std(name: str) -> float | None:
                statistics = seed_statistics.get(name)
                return statistics["std"] if statistics is not None else None

            writer.writerow(
                {
                    "method": method,
                    "covariance_trace": values["gradient_covariance"]["trace"],
                    "covariance_trace_std_across_projection_seeds": seed_std(
                        "trace"
                    ),
                    "trace_ci95_low": values["trace_bootstrap_ci95"][0],
                    "trace_ci95_high": values["trace_bootstrap_ci95"][1],
                    "covariance_frobenius": values["gradient_covariance"]["frobenius"],
                    "covariance_frobenius_std_across_projection_seeds": seed_std(
                        "frobenius"
                    ),
                    "covariance_spectral": values["gradient_covariance"]["spectral"],
                    "covariance_spectral_std_across_projection_seeds": seed_std(
                        "spectral"
                    ),
                    "covariance_minimum_eigenvalue": values["gradient_covariance"][
                        "minimum_eigenvalue"
                    ],
                    "covariance_minimum_eigenvalue_std_across_projection_seeds": seed_std(
                        "minimum_eigenvalue"
                    ),
                    "covariance_effective_rank": values["gradient_covariance"][
                        "effective_rank"
                    ],
                    "covariance_effective_rank_std_across_projection_seeds": seed_std(
                        "effective_rank"
                    ),
                    "trace_ratio_to_baseline": values["trace_ratio_to_baseline"],
                    "trace_ratio_to_baseline_std_across_projection_seeds": seed_std(
                        "trace_ratio_to_baseline"
                    ),
                    "trace_reduction_vs_baseline": values["trace_reduction_vs_baseline"],
                    "trace_reduction_vs_baseline_std_across_projection_seeds": seed_std(
                        "trace_reduction_vs_baseline"
                    ),
                    "batch_size_mean": values["batch_size_mean"],
                    "batch_size_std": values["batch_size_std"],
                    "batch_size_min": values["batch_size_min"],
                    "batch_size_max": values["batch_size_max"],
                }
            )


def _project_flat_gradients(
    flat_gradients: np.ndarray,
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    projection_dimension: int,
    projection_seed: int,
    device: torch.device,
) -> np.ndarray:

    from .projection import CountSketchProjector

    values = np.asarray(flat_gradients, dtype=np.float32)
    expected_dimension = sum(parameter.numel() for _, parameter in named_parameters)
    if values.ndim != 2 or values.shape[1] != expected_dimension:
        raise ValueError(
            "Raw gradient layout does not match the model's trainable parameters: "
            f"got {values.shape}, expected second dimension {expected_dimension}"
        )
    offsets: list[tuple[str, int, int]] = []
    start = 0
    for name, parameter in named_parameters:
        end = start + parameter.numel()
        offsets.append((name, start, end))
        start = end

    source = torch.as_tensor(values, device=device)
    projector = CountSketchProjector(projection_dimension, projection_seed)
    projected: list[np.ndarray] = []
    for row in source:
        embedding = projector.project(
            (name, row[start:end]) for name, start, end in offsets
        )
        projected.append(embedding.detach().float().cpu().numpy())
    return np.stack(projected, axis=0)


def _numeric_statistics(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _aggregate_projection_seed_reports(
    seed_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    if not seed_runs:
        raise ValueError("At least one projection-seed report is required")
    methods = tuple(seed_runs[0]["sampling_comparison"])
    metric_names = (
        "trace",
        "frobenius",
        "spectral",
        "minimum_eigenvalue",
        "effective_rank",
    )
    aggregate: dict[str, Any] = {}
    for method in methods:
        first = seed_runs[0]["sampling_comparison"][method]
        statistics = {
            metric: _numeric_statistics(
                [
                    run["sampling_comparison"][method]["gradient_covariance"][
                        metric
                    ]
                    for run in seed_runs
                ]
            )
            for metric in metric_names
        }
        for scalar in (
            "trace_ratio_to_baseline",
            "trace_reduction_vs_baseline",
        ):
            values = [
                run["sampling_comparison"][method][scalar] for run in seed_runs
            ]
            if all(value is not None for value in values):
                statistics[scalar] = _numeric_statistics(values)
        aggregate[method] = {
            "gradient_covariance": {
                metric: statistics[metric]["mean"] for metric in metric_names
            },
            "trace_bootstrap_ci95": [
                float(
                    np.mean(
                        [
                            run["sampling_comparison"][method][
                                "trace_bootstrap_ci95"
                            ][bound]
                            for run in seed_runs
                        ]
                    )
                )
                if all(
                    run["sampling_comparison"][method]["trace_bootstrap_ci95"][
                        bound
                    ]
                    is not None
                    for run in seed_runs
                )
                else None
                for bound in (0, 1)
            ],
            "batch_size_mean": first["batch_size_mean"],
            "batch_size_std": first["batch_size_std"],
            "batch_size_min": first["batch_size_min"],
            "batch_size_max": first["batch_size_max"],
            "trace_ratio_to_baseline": (
                statistics["trace_ratio_to_baseline"]["mean"]
                if "trace_ratio_to_baseline" in statistics
                else None
            ),
            "trace_reduction_vs_baseline": (
                statistics["trace_reduction_vs_baseline"]["mean"]
                if "trace_reduction_vs_baseline" in statistics
                else None
            ),
            "bootstrap_comparison_to_baseline": None,
            "projection_seed_statistics": statistics,
        }
    return aggregate


def _baseline_poisson_projection_summary(
    seed_runs: list[dict[str, Any]],
    unprojected_traces: dict[str, float],
) -> dict[str, Any] | None:
    if not seed_runs:
        return None
    methods = seed_runs[0]["sampling_comparison"]
    if "baseline" not in methods or "poisson" not in methods:
        return None
    comparisons = []
    projected_reductions = []
    for run in seed_runs:
        baseline_trace = run["sampling_comparison"]["baseline"][
            "gradient_covariance"
        ]["trace"]
        poisson_trace = run["sampling_comparison"]["poisson"][
            "gradient_covariance"
        ]["trace"]
        comparisons.append(baseline_trace > poisson_trace)
        projected_reductions.append(1.0 - poisson_trace / baseline_trace)
    count = int(sum(comparisons))
    raw_comparison = (
        unprojected_traces["baseline"] > unprojected_traces["poisson"]
        if "baseline" in unprojected_traces
        and "poisson" in unprojected_traces
        else None
    )
    raw_reduction = (
        1.0
        - unprojected_traces["poisson"]
        / unprojected_traces["baseline"]
        if "baseline" in unprojected_traces
        and unprojected_traces["baseline"] > 0
        and "poisson" in unprojected_traces
        else None
    )
    reduction_biases = (
        [reduction - raw_reduction for reduction in projected_reductions]
        if raw_reduction is not None
        else []
    )
    more_favorable = [bias > 0 for bias in reduction_biases]
    return {
        "interpretation": (
            "baseline_worse means baseline covariance trace is greater than "
            "poisson covariance trace. A positive reduction bias means the "
            "projection makes poisson look better than the unprojected trace."
        ),
        "projection_seed_count": len(seed_runs),
        "baseline_worse_count": count,
        "baseline_worse_fraction": count / len(seed_runs),
        "baseline_worse_for_every_projection_seed": all(comparisons),
        "baseline_worse_by_projection_seed": comparisons,
        "unprojected_baseline_worse": raw_comparison,
        "unprojected_trace_reduction": raw_reduction,
        "projected_trace_reduction_statistics": _numeric_statistics(
            projected_reductions
        ),
        "projected_minus_unprojected_trace_reduction_statistics": (
            _numeric_statistics(reduction_biases)
            if reduction_biases
            else None
        ),
        "projection_more_favorable_to_poisson_count": int(
            sum(more_favorable)
        ),
        "projection_more_favorable_to_poisson_fraction": (
            sum(more_favorable) / len(more_favorable)
            if more_favorable
            else None
        ),
        "projection_more_favorable_to_poisson_for_every_seed": (
            all(more_favorable) if more_favorable else None
        ),
        "projection_more_favorable_to_poisson_by_seed": more_favorable,
    }


def _write_projection_seed_txt(
    path: Path,
    seed_runs: list[dict[str, Any]],
    aggregate_report: dict[str, Any],
    unprojected_traces: dict[str, float],
    comparison_summary: dict[str, Any] | None,
) -> None:
    fields = [
        "projection_seed",
        "baseline_trace",
        "baseline_frobenius",
        "baseline_spectral",
        "baseline_minimum_eigenvalue",
        "baseline_effective_rank",
        "poisson_trace",
        "poisson_frobenius",
        "poisson_spectral",
        "poisson_minimum_eigenvalue",
        "poisson_effective_rank",
        "poisson_trace_ratio_to_baseline",
        "poisson_trace_reduction_vs_baseline",
        "projected_minus_unprojected_trace_reduction",
        "projection_more_favorable_to_poisson",
        "baseline_worse_by_trace",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        handle.write(
            "# Unprojected analytical covariance traces (no CountSketch)\n"
        )
        for method, trace in unprojected_traces.items():
            handle.write(f"# unprojected_{method}_trace={trace:.17g}\n")
        if comparison_summary is not None:
            handle.write(
                "# unprojected_baseline_worse="
                f"{str(comparison_summary['unprojected_baseline_worse']).lower()}\n"
            )
            handle.write(
                "# projected_baseline_worse_count="
                f"{comparison_summary['baseline_worse_count']}/"
                f"{comparison_summary['projection_seed_count']}\n"
            )
            handle.write(
                "# projected_baseline_worse_for_every_seed="
                f"{str(comparison_summary['baseline_worse_for_every_projection_seed']).lower()}\n"
            )
            bias_statistics = comparison_summary[
                "projected_minus_unprojected_trace_reduction_statistics"
            ]
            if bias_statistics is not None:
                favorable_count = comparison_summary[
                    "projection_more_favorable_to_poisson_count"
                ]
                favorable_for_every_seed = comparison_summary[
                    "projection_more_favorable_to_poisson_for_every_seed"
                ]
                handle.write(
                    "# projected_minus_unprojected_trace_reduction_mean="
                    f"{bias_statistics['mean']:.17g}\n"
                )
                handle.write(
                    "# projection_more_favorable_to_poisson_count="
                    f"{favorable_count}/"
                    f"{comparison_summary['projection_seed_count']}\n"
                )
                handle.write(
                    "# projection_more_favorable_to_poisson_for_every_seed="
                    f"{str(favorable_for_every_seed).lower()}\n"
                )
        handle.write("# Mean projected metrics across CountSketch seeds\n")
        for method, values in aggregate_report.items():
            for metric, value in values["gradient_covariance"].items():
                handle.write(
                    f"# projected_{method}_{metric}_mean={value:.17g}\n"
                )
            statistics = values["projection_seed_statistics"]
            for metric in (
                "trace",
                "frobenius",
                "spectral",
                "minimum_eigenvalue",
                "effective_rank",
            ):
                handle.write(
                    f"# projected_{method}_{metric}_std="
                    f"{statistics[metric]['std']:.17g}\n"
                )
            for scalar in (
                "trace_ratio_to_baseline",
                "trace_reduction_vs_baseline",
            ):
                if scalar in statistics:
                    handle.write(
                        f"# projected_{method}_{scalar}_mean="
                        f"{statistics[scalar]['mean']:.17g}\n"
                    )
                    handle.write(
                        f"# projected_{method}_{scalar}_std="
                        f"{statistics[scalar]['std']:.17g}\n"
                    )
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for run in seed_runs:
            comparison = run["sampling_comparison"]
            baseline = comparison.get("baseline")
            poisson = comparison.get("poisson")
            baseline_metrics = (
                baseline["gradient_covariance"] if baseline is not None else {}
            )
            poisson_metrics = (
                poisson["gradient_covariance"]
                if poisson is not None
                else {}
            )
            baseline_trace = baseline_metrics.get("trace")
            poisson_trace = poisson_metrics.get("trace")
            projected_reduction = (
                poisson["trace_reduction_vs_baseline"]
                if poisson is not None
                else None
            )
            raw_reduction = (
                comparison_summary["unprojected_trace_reduction"]
                if comparison_summary is not None
                else None
            )
            reduction_bias = (
                projected_reduction - raw_reduction
                if projected_reduction is not None
                and raw_reduction is not None
                else None
            )
            writer.writerow(
                {
                    "projection_seed": run["projection_seed"],
                    "baseline_trace": baseline_trace,
                    "baseline_frobenius": baseline_metrics.get("frobenius"),
                    "baseline_spectral": baseline_metrics.get("spectral"),
                    "baseline_minimum_eigenvalue": baseline_metrics.get(
                        "minimum_eigenvalue"
                    ),
                    "baseline_effective_rank": baseline_metrics.get(
                        "effective_rank"
                    ),
                    "poisson_trace": poisson_trace,
                    "poisson_frobenius": poisson_metrics.get(
                        "frobenius"
                    ),
                    "poisson_spectral": poisson_metrics.get("spectral"),
                    "poisson_minimum_eigenvalue": poisson_metrics.get(
                        "minimum_eigenvalue"
                    ),
                    "poisson_effective_rank": poisson_metrics.get(
                        "effective_rank"
                    ),
                    "poisson_trace_ratio_to_baseline": (
                        poisson["trace_ratio_to_baseline"]
                        if poisson is not None
                        else None
                    ),
                    "poisson_trace_reduction_vs_baseline": (
                        projected_reduction
                    ),
                    "projected_minus_unprojected_trace_reduction": (
                        reduction_bias
                    ),
                    "projection_more_favorable_to_poisson": (
                        reduction_bias > 0
                        if reduction_bias is not None
                        else None
                    ),
                    "baseline_worse_by_trace": (
                        baseline_trace > poisson_trace
                        if baseline_trace is not None
                        and poisson_trace is not None
                        else None
                    ),
                }
            )


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    if arguments.projection_seed_count is None:
        arguments.projection_seed_count = (
            30
            if arguments.projection == "countsketch"
            and not arguments.trace_only
            else 1
        )
    if arguments.clip and (
        arguments.clipping_norm is None or arguments.clipping_norm <= 0
    ):
        parser.error("--clipping-norm must be positive when --clip is enabled")
    if arguments.projection_seed_count < 1:
        parser.error("--projection-seed-count must be positive")
    if arguments.projection != "countsketch" and arguments.projection_seed_count != 1:
        parser.error(
            "--projection-seed-count can only exceed 1 with "
            "--projection countsketch"
        )
    if arguments.trace_only and arguments.projection_seed_count != 1:
        parser.error(
            "--trace-only supports one projection seed; omit --trace-only for "
            "the hybrid raw-trace plus multi-seed projected analysis"
        )
    from .adapters import build_adapter
    from .collector import candidate_indices, collect_gradients

    output_dir = Path(arguments.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if arguments.task == "classification":
        from .classification_adapters import build_classification_adapter

        adapter = build_classification_adapter(arguments)
        arguments.pred_len = 0
    else:
        adapter = build_adapter(arguments)
    effective_candidate_stride = (
        arguments.candidate_stride
        if arguments.candidate_stride is not None
        else 1
    )
    indices = candidate_indices(
        adapter,
        effective_candidate_stride,
        arguments.seq_len,
        arguments.pred_len,
    )

    balanced_temporal_windows(len(indices), arguments.batch_size)
    poisson_config = (
        resolve_poisson_config(arguments, adapter, len(indices), arguments.batch_size)
        if "poisson" in arguments.methods else None
    )
    poisson_lambda = poisson_config["lambda"] if poisson_config else 0.5
    poisson_window_rounding = poisson_config["window_rounding"] if poisson_config else "ceil"

    multi_seed_projection = (
        arguments.projection == "countsketch" and not arguments.trace_only
    )
    collection = collect_gradients(
        adapter=adapter,
        indices=indices,
        projection_dimension=arguments.projection_dim,
        projection_seed=arguments.projection_seed,
        clip=arguments.clip,
        clipping_norm=arguments.clipping_norm,
        progress_every=arguments.progress_every,


        projection="none" if multi_seed_projection else arguments.projection,
    )

    q = arguments.batch_size / collection.embeddings.shape[0]
    unprojected_traces: dict[str, float] | None = None
    projection_seed_runs: list[dict[str, Any]] = []
    projection_comparison_summary: dict[str, Any] | None = None
    saved_embeddings = collection.embeddings
    if multi_seed_projection:
        unprojected_traces = analytical_sampling_traces(
            gradients=collection.embeddings,
            methods=arguments.methods,
            batch_size=arguments.batch_size,
            poisson_lambda=poisson_lambda,
            poisson_window_rounding=poisson_window_rounding,
        )
        named_parameters = [
            (name, parameter)
            for name, parameter in adapter.model.named_parameters()
            if parameter.requires_grad
        ]
        sampling_results = {}
        for offset in range(arguments.projection_seed_count):
            projection_seed = arguments.projection_seed + offset
            print(
                "Evaluating CountSketch projection seed "
                f"{offset + 1}/{arguments.projection_seed_count}: "
                f"{projection_seed}"
            )
            projected_embeddings = _project_flat_gradients(
                collection.embeddings,
                named_parameters,
                arguments.projection_dim,
                projection_seed,
                adapter.device,
            )
            seed_results = run_sampling_experiment(
                gradients=projected_embeddings,
                methods=arguments.methods,
                batch_size=arguments.batch_size,
                trials=arguments.trials,
                seed=arguments.seed,
                poisson_lambda=poisson_lambda,
                poisson_window_rounding=poisson_window_rounding,
                bootstrap_repetitions=arguments.bootstrap,
                shared_method_seed=True,
            )
            seed_report = _sampling_report(
                seed_results, arguments.bootstrap, arguments.seed
            )
            projection_seed_runs.append(
                {
                    "projection_seed": projection_seed,
                    "sampling_comparison": seed_report,
                }
            )

            if offset == 0:
                sampling_results = seed_results
                saved_embeddings = projected_embeddings
        sampling_report = _aggregate_projection_seed_reports(
            projection_seed_runs
        )
        projection_comparison_summary = (
            _baseline_poisson_projection_summary(
                projection_seed_runs, unprojected_traces
            )
        )
    elif arguments.trace_only:
        trace_values = analytical_sampling_traces(
            gradients=collection.embeddings,
            methods=arguments.methods,
            batch_size=arguments.batch_size,
            poisson_lambda=poisson_lambda,
            poisson_window_rounding=poisson_window_rounding,
        )
        baseline_trace = trace_values.get("baseline")
        sampling_results = {}
        sampling_report = {}
        for method, trace in trace_values.items():
            ratio = (
                trace / baseline_trace
                if baseline_trace is not None and baseline_trace > 0
                else None
            )
            sampling_report[method] = {
                "gradient_covariance": {
                    "trace": trace,
                    "frobenius": None,
                    "spectral": None,
                    "minimum_eigenvalue": None,
                    "effective_rank": None,
                },
                "trace_bootstrap_ci95": [None, None],
                "batch_size_mean": (
                    poisson_config["expected_batch_size"]
                    if method == "poisson" else float(arguments.batch_size)
                ),
                "batch_size_std": None,
                "batch_size_min": None,
                "batch_size_max": None,
                "trace_ratio_to_baseline": ratio,
                "trace_reduction_vs_baseline": (
                    1.0 - ratio if ratio is not None else None
                ),
                "bootstrap_comparison_to_baseline": None,
            }
    else:
        sampling_results = run_sampling_experiment(
            gradients=collection.embeddings,
            methods=arguments.methods,
            batch_size=arguments.batch_size,
            trials=arguments.trials,
            seed=arguments.seed,
            poisson_lambda=poisson_lambda,
            poisson_window_rounding=poisson_window_rounding,
            bootstrap_repetitions=arguments.bootstrap,
        )
        sampling_report = _sampling_report(
            sampling_results, arguments.bootstrap, arguments.seed
        )

    experiment = {
            "task": arguments.task,
            "model": arguments.model,
            "checkpoint": arguments.checkpoint,
            "split": arguments.split,
            "seed": arguments.seed,
            "shared_sampling_seed_across_methods": bool(multi_seed_projection),
            "projection": arguments.projection,
            "projection_dimension": (
                arguments.projection_dim if arguments.projection == "countsketch" else None
            ),
            "projection_seed": (
                arguments.projection_seed if arguments.projection == "countsketch" else None
            ),
            "projection_seed_count": (
                arguments.projection_seed_count
                if arguments.projection == "countsketch"
                else None
            ),
            "projection_seeds": (
                [
                    arguments.projection_seed + offset
                    for offset in range(arguments.projection_seed_count)
                ]
                if arguments.projection == "countsketch"
                else None
            ),
            "gradient_dimension": int(collection.embeddings.shape[1]),
            "projected_gradient_dimension": (
                arguments.projection_dim
                if arguments.projection == "countsketch"
                else None
            ),
            "candidate_stride": effective_candidate_stride,
            "dataset_sample_stride": getattr(adapter.dataset, "sample_stride", None),
            "candidate_count": int(collection.embeddings.shape[0]),
            "batch_size_K": arguments.batch_size,
            "baseline_segment_inclusion_probability_q": q,
            "clip": arguments.clip,
            "clipping_norm": arguments.clipping_norm,
            "trials": None if arguments.trace_only else arguments.trials,
            "trace_only": arguments.trace_only,
            "normalization": "Every stochastic gradient estimator is sum/K",
            "independence_assumption": (
                "Baseline uses one independent Bernoulli(q) indicator per fixed candidate. "
                "No sample covariance cross terms are estimated."
            ),
    }
    if poisson_config is not None:
        experiment["poisson"] = poisson_config

    report = {
        "experiment": experiment,
        "gradient_summary": {
            "loss_mean": float(collection.losses.mean()),
            "loss_std": float(collection.losses.std()),
            "raw_gradient_norm_mean": float(collection.raw_norms.mean()),
            "raw_gradient_norm_std": float(collection.raw_norms.std()),
            "analyzed_gradient_norm_mean": float(collection.analyzed_norms.mean()),
            "analyzed_gradient_norm_std": float(collection.analyzed_norms.std()),
        },
        "sampling_comparison": sampling_report,
    }
    if unprojected_traces is not None:
        baseline_raw_trace = unprojected_traces.get("baseline")
        report["unprojected_trace"] = {
            method: {
                "trace": trace,
                "trace_ratio_to_baseline": (
                    trace / baseline_raw_trace
                    if baseline_raw_trace is not None and baseline_raw_trace > 0
                    else None
                ),
                "trace_reduction_vs_baseline": (
                    1.0 - trace / baseline_raw_trace
                    if baseline_raw_trace is not None and baseline_raw_trace > 0
                    else None
                ),
            }
            for method, trace in unprojected_traces.items()
        }
        report["projection_seed_comparison"] = projection_comparison_summary
        report["projection_seed_runs"] = projection_seed_runs

    with (output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    _write_sampling_csv(output_dir / "sampling_comparison.csv", sampling_report)
    if multi_seed_projection:
        assert unprojected_traces is not None
        _write_projection_seed_txt(
            output_dir / "projection_seed_results.txt",
            projection_seed_runs,
            sampling_report,
            unprojected_traces,
            projection_comparison_summary,
        )
    else:
        (output_dir / "projection_seed_results.txt").unlink(missing_ok=True)
    (output_dir / "batch_seed_results.txt").unlink(missing_ok=True)
    (output_dir / "batch_seed_sampling_comparison.csv").unlink(missing_ok=True)
    if arguments.trace_only:
        (output_dir / "gradient_embeddings.npz").unlink(missing_ok=True)
        np.savez_compressed(
            output_dir / "gradient_statistics.npz",
            sample_indices=collection.sample_indices,
            raw_norms=collection.raw_norms,
            analyzed_norms=collection.analyzed_norms,
            losses=collection.losses,
        )
    else:
        (output_dir / "gradient_statistics.npz").unlink(missing_ok=True)
        np.savez_compressed(
            output_dir / "gradient_embeddings.npz",
            embeddings=saved_embeddings,
            sample_indices=collection.sample_indices,
            raw_norms=collection.raw_norms,
            analyzed_norms=collection.analyzed_norms,
            losses=collection.losses,
        )
    for method in METHODS:
        if method not in sampling_results:
            (output_dir / f"{method}_gradient_estimators.npy").unlink(missing_ok=True)
            (output_dir / f"{method}_gradient_covariance.npy").unlink(missing_ok=True)
    for method, result in sampling_results.items():
        np.save(
            output_dir / f"{method}_gradient_estimators.npy",
            result.gradient_estimators,
        )
        np.save(
            output_dir / f"{method}_gradient_covariance.npy",
            result.covariance,
        )

    print(json.dumps(report["sampling_comparison"], ensure_ascii=False, indent=2))
    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()
