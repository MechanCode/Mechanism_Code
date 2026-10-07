import ast
from contextlib import redirect_stdout
import hashlib
import io
import math
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
from scipy.special import comb, logsumexp


def select_classification_lambda(arguments, dataset, candidate_count, batch_size):
    source = Path(__file__).resolve().parent / "lambda_cal/classification_selector.py"
    contents = source.read_text(encoding="utf-8")
    tree = ast.parse(contents)
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                    and node.name == "Exp_Poisson_TSC")
    methods = [node for node in original.body if isinstance(node, ast.FunctionDef)
               and node.name in ("rho_stable_b0_from_c", "select_space_distance")]
    if len(methods) != 2:
        raise ValueError(f"Cannot find original lambda selection methods in {source}")

    cls = ast.ClassDef(name="Selector", bases=[], keywords=[], body=methods,
                       decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    namespace = {"np": np, "math": math, "comb": comb, "logsumexp": logsumexp}
    exec(compile(module, str(source), "exec"), namespace)
    runner = namespace["Selector"]()
    stride = int(dataset.sample_stride) * (arguments.candidate_stride or 1)
    length = len(dataset.data)
    runner.args = SimpleNamespace(seq_len=arguments.seq_len, sampling_stride=stride,
                                  batch_size=batch_size)
    runner.data_size = length
    runner.w = int(length * arguments.w)
    runner.dp_sigma = arguments.dp_sigma
    runner.dp_delta = arguments.dp_delta
    runner.privacy_budget_limit = arguments.privacy_budget_limit
    selector_count = (length - arguments.seq_len) // stride + 1
    if selector_count < batch_size:
        raise ValueError("Too few windows for classification lambda selection")
    output = io.StringIO()
    with warnings.catch_warnings(), redirect_stdout(output):
        warnings.simplefilter("ignore", RuntimeWarning)
        lam = float(runner.select_space_distance())
    max_steps = int(output.getvalue().split("with max steps:")[-1].strip())
    return {
        "selection": "auto",
        "selector": str(source),
        "selector_source_sha256": hashlib.sha256(contents.encode()).hexdigest(),
        "selection_config": {
            "time_series_length": length, "segment_length": arguments.seq_len,
            "stride_length": stride, "batch_size": batch_size,
            "dp_sigma": runner.dp_sigma, "dp_delta": runner.dp_delta,
            "epsilon": runner.privacy_budget_limit, "w": arguments.w,
        },
        "selection_result": {
            "lam": lam, "max_steps": max_steps, "feasible": max_steps > 0,
            "candidate_windows": selector_count,
            "gradient_candidate_count": candidate_count,
        },
        "geometry_note": (
            "Original classification selector uses raw timestamp length and regular stride. "
            "The fixed clean checkpoint dataset uses windows within same-label runs; "
            "its actual gradient candidate count may differ. Lambda selection preserves "
            "the original rule; both estimators use the same actual gradient population."
        ),
        "lambda": lam,
    }
