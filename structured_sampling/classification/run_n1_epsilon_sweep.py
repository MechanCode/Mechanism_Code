from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("HAR", "USC"), required=True)
    parser.add_argument("--train-path", default=None)
    parser.add_argument("--validation-path", default=None)
    parser.add_argument("--test-path", default=None)
    parser.add_argument("--data-cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--models", nargs="+", choices=("DLinear", "PatchTST"),
        default=["DLinear", "PatchTST"],
    )
    parser.add_argument("--epsilons", nargs="+", type=float, default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--enc-in", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--dlinear-learning-rate", type=float, default=0.02)
    parser.add_argument("--patchtst-learning-rate", type=float, default=0.01)
    parser.add_argument("--summary-file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_stride < 1 or args.batch_size < 1:
        raise ValueError("sample_stride and batch_size must be positive")
    learning_rates = {
        "DLinear": args.dlinear_learning_rate,
        "PatchTST": args.patchtst_learning_rate,
    }
    script_dir = Path(__file__).resolve().parent
    completed: list[dict[str, object]] = []
    for model in args.models:
        for epsilon in args.epsilons:
            for seed in args.seeds:
                output_dir = (
                    args.output_root / model / f"epsilon_{epsilon:g}" / f"seed_{seed}"
                )
                metrics_path = output_dir / "metrics.json"
                command = [
                    sys.executable,
                    str(script_dir / "train_n1_structured_wr.py"),
                    "--dataset", args.dataset,
                    "--model", model,
                    "--data-cache", str(args.data_cache),
                    "--batch-size", str(args.batch_size),
                    "--sample-stride", str(args.sample_stride),
                    "--target-epsilon", str(epsilon),
                    "--learning-rate", str(learning_rates[model]),
                    "--seed", str(seed),
                    "--device", args.device,
                    "--output-dir", str(output_dir),
                ]
                for option, value in (
                    ("--train-path", args.train_path),
                    ("--validation-path", args.validation_path),
                    ("--test-path", args.test_path),
                    ("--seq-len", args.seq_len),
                    ("--enc-in", args.enc_in),
                    ("--num-classes", args.num_classes),
                ):
                    if value is not None:
                        command.extend((option, str(value)))
                print("running:", " ".join(command), flush=True)
                subprocess.run(command, check=True, cwd=script_dir)
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                completed.append(
                    {
                        "dataset": args.dataset,
                        "model": model,
                        "target_epsilon": epsilon,
                        "seed": seed,
                        "optimizer_steps": metrics["optimizer_steps"],
                        "accounted_epsilon": metrics["accounted_epsilon"],
                        "test_accuracy": metrics["test_accuracy"],
                        "test_macro_f1": metrics["test_macro_f1"],
                    }
                )
                summary_file = args.summary_file or args.output_root / "sweep_results.json"
                summary_file.parent.mkdir(parents=True, exist_ok=True)
                summary_file.write_text(
                    json.dumps(completed, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )


if __name__ == "__main__":
    main()
