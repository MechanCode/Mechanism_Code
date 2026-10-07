from dataclasses import asdict
import warnings

from .sampling import balanced_temporal_windows, poisson_window_count


def add_poisson_arguments(parser):
    parser.add_argument(
        "--poisson-lambda", default="auto",
        help="auto: run the task's original privacy-based selector; or a number in (0, 1]",
    )

    parser.add_argument("--dp-sigma", type=float, default=5.0)
    parser.add_argument("--dp-delta", type=float, default=1e-5)
    parser.add_argument("--privacy-budget-limit", type=float, default=6.0)
    parser.add_argument("--w", type=float, default=0.005)


def resolve_poisson_config(arguments, adapter, candidate_count, batch_size):

    if str(arguments.poisson_lambda) == "auto" and arguments.split != "train":
        raise ValueError("Automatic poisson lambda requires the train split")
    if str(arguments.poisson_lambda) == "auto" and adapter.task == "classification":
        from .classification_lambda import select_classification_lambda

        metadata = select_classification_lambda(
            arguments, adapter.dataset, candidate_count, batch_size
        )
        lam = metadata["lambda"]
    elif str(arguments.poisson_lambda) == "auto":
        from .lambda_cal import LambdaConfig, select_lambda

        stride = int(adapter.dataset.sample_stride) * (arguments.candidate_stride or 1)
        config = LambdaConfig(
            time_series_length=len(adapter.dataset.data_x),
            segment_length=arguments.seq_len + arguments.pred_len,
            stride_length=stride,
            batch_size=batch_size,
            dp_sigma=arguments.dp_sigma,
            dp_delta=arguments.dp_delta,
            epsilon=arguments.privacy_budget_limit,
            w=arguments.w,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            selected = select_lambda(config)
        if selected.candidate_windows != candidate_count:
            raise ValueError("Lambda selector and gradient population have different candidate counts")
        lam = float(selected.lam)
        metadata = {
            "selection": "auto",
            "selector": "lambda_cal.select_lambda (forecasting select_space_distance)",
            "selection_config": asdict(config),
            "selection_result": asdict(selected),
        }
        if not selected.feasible:
            warnings.warn(
                "Lambda search found no feasible positive step; using the original "
                "selector's fallback, recorded as feasible=false in the report.",
                RuntimeWarning,
            )
    else:
        lam = float(arguments.poisson_lambda)
        metadata = {"selection": "manual"}
    rounding = "floor" if adapter.task == "classification" else "ceil"
    window_count = poisson_window_count(batch_size, lam, rounding)
    balanced_temporal_windows(candidate_count, window_count)
    metadata.update({
        "lambda": lam,
        "window_count": window_count,
        "window_rounding": rounding,
        "window_inclusion_probability": lam,
        "expected_batch_size": window_count * lam,
        "empty_batch_probability": (1.0 - lam) ** window_count,
        "empty_batch_policy": "zero estimator, included in covariance (unconditional draws)",
        "normalization": "sum/K",
    })
    return metadata
