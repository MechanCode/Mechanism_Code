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
    groups: dict[tuple[str, str, float], list[dict[str, float | int]]] = {}
    for metrics_path in args.output_root.glob("*/epsilon_*/seed_*/metrics.json"):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        config = json.loads((metrics_path.parent / "config.json").read_text(encoding="utf-8"))
        key = (
            str(config["dataset"]),
            str(config["model"]),
            float(config["target_epsilon"]),
        )
        groups.setdefault(key, []).append(
            {
                "seed": int(config["seed"]),
                "test_accuracy": float(metrics["test_accuracy"]),
                "test_macro_f1": float(metrics["test_macro_f1"]),
                "validation_accuracy": float(metrics["validation_accuracy"]),
                "validation_macro_f1": float(metrics["validation_macro_f1"]),
                "optimizer_steps": int(metrics["optimizer_steps"]),
                "accounted_epsilon": float(metrics["accounted_epsilon"]),
            }
        )

    summaries: list[dict[str, object]] = []
    for (dataset, model, epsilon), runs in sorted(groups.items()):
        steps = {int(run["optimizer_steps"]) for run in runs}
        accounted = {float(run["accounted_epsilon"]) for run in runs}
        if len(steps) != 1 or len(accounted) != 1:
            raise ValueError(f"privacy metadata differs across seeds for {(dataset, model, epsilon)}")
        summary: dict[str, object] = {
            "dataset": dataset,
            "model": model,
            "target_epsilon": epsilon,
            "num_runs": len(runs),
            "optimizer_steps": steps.pop(),
            "accounted_epsilon": accounted.pop(),
        }
        for split in ("test", "validation"):
            for metric in ("accuracy", "macro_f1"):
                values = [float(run[f"{split}_{metric}"]) for run in runs]
                summary[f"mean_{split}_{metric}"] = statistics.fmean(values)
                summary[f"std_{split}_{metric}"] = (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                )
        summaries.append(summary)

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "aggregate_results.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
