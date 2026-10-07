from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--data-cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("DLinear", "PatchTST"),
        default=["DLinear", "PatchTST"],
    )
    parser.add_argument(
        "--epsilons", nargs="+", type=float, default=[1, 2, 3, 4, 5, 6]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--context-len", type=int, default=80)
    parser.add_argument("--forecast-len", type=int, default=20)
    parser.add_argument("--dlinear-learning-rate", type=float, default=0.02)
    parser.add_argument("--patchtst-learning-rate", type=float, default=0.005)
    parser.add_argument("--summary-file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    learning_rates = {
        "DLinear": args.dlinear_learning_rate,
        "PatchTST": args.patchtst_learning_rate,
    }
    script_dir = Path(__file__).resolve().parent
    completed: list[dict[str, object]] = []
    for model in args.models:
        for epsilon in args.epsilons:
            for seed in args.seeds:
                epsilon_label = f"{epsilon:g}"
                output_dir = (
                    args.output_root
                    / model
                    / f"epsilon_{epsilon_label}"
                    / f"seed_{seed}"
                )
                metrics_path = output_dir / "metrics.json"
                if metrics_path.exists():
                    print(
                        f"ignoring existing result and rerunning: {output_dir}",
                        flush=True,
                    )
                command = [
                    sys.executable,
                    str(script_dir / "train_n1_structured_wr.py"),
                    "--model",
                    model,
                    "--root-path",
                    args.root_path,
                    "--data-path",
                    args.data_path,
                    "--data-cache",
                    str(args.data_cache),
                    "--context-len",
                    str(args.context_len),
                    "--forecast-len",
                    str(args.forecast_len),
                    "--sample-stride",
                    str(args.sample_stride),
                    "--target-epsilon",
                    str(epsilon),
                    "--learning-rate",
                    str(learning_rates[model]),
                    "--seed",
                    str(seed),
                    "--device",
                    args.device,
                    "--num-workers",
                    str(args.num_workers),
                    "--output-dir",
                    str(output_dir),
                ]
                print("running:", " ".join(command), flush=True)
                subprocess.run(command, check=True, cwd=script_dir)
                metrics = json.loads(
                    metrics_path.read_text(encoding="utf-8")
                )
                completed.append(
                    {
                        "model": model,
                        "target_epsilon": epsilon,
                        "seed": seed,
                        "optimizer_steps": metrics["optimizer_steps"],
                        "accounted_epsilon": metrics["accounted_epsilon"],
                        "test_mse": metrics["test_mse"],
                    }
                )
                summary_file = args.summary_file or (
                    args.output_root / "sweep_results.json"
                )
                summary_file.parent.mkdir(parents=True, exist_ok=True)
                summary_file.write_text(
                    json.dumps(completed, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )


if __name__ == "__main__":
    main()
