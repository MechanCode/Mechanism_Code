from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    groups: dict[tuple[str, float], list[dict[str, object]]] = {}
    for metrics_path in args.output_root.glob("*/epsilon_*/seed_*/metrics.json"):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        config = json.loads(
            (metrics_path.parent / "config.json").read_text(encoding="utf-8")
        )
        key = (str(config["model"]), float(config["target_epsilon"]))
        groups.setdefault(key, []).append(
            {
                "seed": int(config["seed"]),
                "test_mse": float(metrics["test_mse"]),
                "validation_mse": float(metrics["validation_mse"]),
                "optimizer_steps": int(metrics["optimizer_steps"]),
                "accounted_epsilon": float(metrics["accounted_epsilon"]),
            }
        )

    summaries: list[dict[str, object]] = []
    for (model, epsilon), runs in sorted(groups.items()):
        test_values = [float(run["test_mse"]) for run in runs]
        validation_values = [float(run["validation_mse"]) for run in runs]
        steps = {int(run["optimizer_steps"]) for run in runs}
        accounted = {float(run["accounted_epsilon"]) for run in runs}
        if len(steps) != 1 or len(accounted) != 1:
            raise ValueError(f"privacy metadata differs across seeds for {(model, epsilon)}")
        summaries.append(
            {
                "model": model,
                "target_epsilon": epsilon,
                "num_runs": len(runs),
                "optimizer_steps": steps.pop(),
                "accounted_epsilon": accounted.pop(),
                "mean_test_mse": statistics.fmean(test_values),
                "std_test_mse": statistics.stdev(test_values)
                if len(test_values) > 1
                else 0.0,
                "mean_validation_mse": statistics.fmean(validation_values),
                "std_validation_mse": statistics.stdev(validation_values)
                if len(validation_values) > 1
                else 0.0,
            }
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "aggregate_results.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fieldnames = list(summaries[0]) if summaries else []
    with (args.output_root / "aggregate_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(summaries)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
