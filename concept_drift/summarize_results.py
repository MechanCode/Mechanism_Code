import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .covariance import covariance_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, help="Optional relocated root containing traffic/checkpoint and usc/checkpoint")
    args = parser.parse_args()
    root = args.result_root.resolve()
    methods = ("baseline", "poisson")
    rows, details = [], {}
    for dataset in ("traffic", "usc"):
        directory = root / dataset / "covariance"
        report = json.loads((directory / "report.json").read_text())
        experiment = report["experiment"]
        assert experiment["batch_size_K"] == 60
        assert experiment["trials"] == 10000
        assert experiment["projection_dimension"] == 128
        assert experiment["projection_seeds"] == list(range(2026, 2056))
        assert experiment["clip"] is False
        assert set(methods).issubset(report["sampling_comparison"])
        checkpoint = (args.checkpoint_root / dataset / "checkpoint/checkpoint.pth"
                      if args.checkpoint_root else Path(experiment["checkpoint"]))
        config = json.loads((checkpoint.parent / "config.json").read_text())
        training = json.loads((checkpoint.parent / "metrics.json").read_text())
        assert config["model"] == "PatchTST"
        assert config["seq_len"] == 160 and config["batch_size"] == 50
        assert config["enc_in"] == (371 if dataset == "traffic" else 6)
        if dataset == "traffic":
            assert config["pred_len"] == 20
        else:
            assert config["num_classes"] == 12
        assert len(report["projection_seed_runs"]) == 30
        poisson = experiment["poisson"]
        assert poisson["selection_result"]["feasible"]
        details[dataset] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "training_config": config,
            "best_epoch": training["best_epoch"],
            "poisson": poisson,
            "candidate_count": experiment["candidate_count"],
            "gradient_dimension": experiment["gradient_dimension"],
            "normalization": experiment["normalization"],
        }
        for method in methods:
            first = report["projection_seed_runs"][0]["sampling_comparison"][method]
            matrix = np.load(directory / f"{method}_gradient_covariance.npy")
            estimators = np.load(directory / f"{method}_gradient_estimators.npy")
            centered = estimators - estimators.mean(axis=0)
            np.testing.assert_allclose(matrix, centered.T @ centered / 10000,
                                       rtol=1e-10, atol=1e-12)
            for metric, value in covariance_metrics(matrix).items():
                np.testing.assert_allclose(value, first["gradient_covariance"][metric],
                                           rtol=1e-9, atol=1e-12)
            trace = report["unprojected_trace"][method]["trace"]
            baseline_trace = report["unprojected_trace"]["baseline"]["trace"]
            row = {
                "dataset": dataset, "method": method,
                "lambda": poisson["lambda"] if method == "poisson" else None,
                "expected_batch_size": poisson["expected_batch_size"] if method == "poisson" else 60,
                "trace_original": trace,
                "trace_ratio_to_baseline": trace / baseline_trace,
                "trace_reduction_pct": 100 * (1 - trace / baseline_trace),
            }
            for metric in ("frobenius", "spectral"):
                values = np.array([seed["sampling_comparison"][method]["gradient_covariance"][metric]
                                   for seed in report["projection_seed_runs"]])
                baseline = np.array([seed["sampling_comparison"]["baseline"]["gradient_covariance"][metric]
                                     for seed in report["projection_seed_runs"]])
                assert np.isfinite(values).all()
                aggregate = report["sampling_comparison"][method]["gradient_covariance"][metric]
                np.testing.assert_allclose(values.mean(), aggregate, rtol=1e-12)
                row.update({
                    f"{metric}_projected_mean": float(values.mean()),
                    f"{metric}_projected_std": float(values.std()),
                    f"{metric}_ratio_of_means": float(values.mean() / baseline.mean()),
                    f"{metric}_paired_ratio_mean": float((values / baseline).mean()),
                    f"{metric}_paired_ratio_std": float((values / baseline).std()),
                    f"{metric}_below_baseline_seed_count": int((values < baseline).sum()),
                })
            rows.append(row)
    with (root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "summary.json").write_text(json.dumps({
        "checkpoint_policy": "reuse existing best-validation non-DP checkpoints; no retraining",
        "noise_and_clipping": "none; privacy parameters only select lambda",
        "projected_standard_deviation": "population SD across 30 projection seeds, not training uncertainty",
        "datasets": details, "results": rows,
    }, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["Poisson", "Poisson spaced"]
    colors = ["#48A999", "#D96C75"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), constrained_layout=True)
    fig.suptitle(
        "Fixed non-DP PatchTST checkpoints | K=60 | CountSketch d=128\n"
        "Projected norms: ratio of means over 30 seeds, 10K draws per seed",
        fontsize=11,
    )
    for i, dataset in enumerate(("traffic", "usc")):
        subset = [row for row in rows if row["dataset"] == dataset]
        for j, metric in enumerate(("trace", "frobenius", "spectral")):
            ax = axes[i, j]
            field = "trace_ratio_to_baseline" if metric == "trace" else f"{metric}_ratio_of_means"
            heights = [row[field] for row in subset]
            bars = ax.bar(labels, heights, color=colors)
            ax.axhline(1, color="gray", linestyle="--", linewidth=0.8)
            ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
            ax.set_ylim(0, max(heights) * 1.17)
            ax.set_title(f"{dataset.upper()} · {metric}" + (" (original)" if j == 0 else " (projected)"))
            ax.set_ylabel("Ratio to Poisson baseline")
            ax.tick_params(axis="x", labelsize=8)
    fig.savefig(root / "comparison.pdf")
    fig.savefig(root / "comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps(rows, indent=2))
    print(f"Validated both tasks, 30 seeds, 10K trials. Results: {root}")


if __name__ == "__main__":
    main()
