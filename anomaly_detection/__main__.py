import argparse
import json
from pathlib import Path

from .config import Config, METHODS
from .data import read_series, read_test, evaluation_subset
from .io_utils import atomic_json, code_identity, fingerprint
from .privacy import build_plan


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    for name in ("account", "train", "sweep", "summarize"):
        command = sub.add_parser(name)
        command.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
        if name == "summarize":
            command.add_argument("--allow-incomplete", action="store_true")
            continue
        command.add_argument("--data-dir", type=Path, required=True, help="Existing local PSM CSV directory; never downloaded")
        command.add_argument("--config", type=Path, help="JSON overrides of the default protocol")
        command.add_argument("--split", choices=("confirmation", "development"),
                             help="Built-in experiment configuration (default: confirmation without --config)")
        if name in ("account", "sweep"):
            command.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
        if name in ("train", "sweep"):
            command.add_argument("--device", default="auto", help="auto, cpu, or cuda:0")
            command.add_argument("--num-threads", type=int, default=4)
            command.add_argument("--resume", action="store_true", help="Resume matching local checkpoints")
            command.add_argument("--save-scores", action="store_true")
        if name == "train":
            command.add_argument("--method", choices=METHODS, required=True)
            command.add_argument("--epsilon", type=float, required=True)
            command.add_argument("--seed", type=int, required=True)
        if name == "sweep":
            command.add_argument("--seeds", type=int, nargs="+", help="Optional subset of the configured 30 seeds")
    return cli


def obtain_plan(config, training, output_dir, methods):
    path = output_dir / "privacy_plan.json"
    source_id = code_identity()
    if path.exists():
        plan = json.loads(path.read_text())
        if plan["protocol"] != config.to_dict() or plan["training"] != training.identity():
            raise ValueError("Existing privacy plan has different data/config. Use a new output directory.")
        if plan.get("code_id") != source_id:
            if not str(plan.get("code_id", "")).startswith("ast-v1:"):
                raise ValueError("Existing privacy plan uses a legacy source fingerprint. "
                                 "Its code equivalence cannot be verified from that hash; "
                                 "use a new output directory for the new fingerprint format.")
            raise ValueError("Existing privacy plan has different executable code. Use a new output directory.")
        if not set(methods) <= set(plan["methods"]):
            raise ValueError("Existing plan lacks requested methods. Account all desired methods in a new directory.")
        print(f"Reusing privacy plan: {path}", flush=True)
        return plan


    plan = build_plan(config, training.identity(), methods, progress=lambda message: print(message, flush=True))
    plan["code_id"] = source_id
    atomic_json(path, plan)
    return plan


def print_budgets(plan):
    print("method       epsilon    steps    accounted_epsilon    next_epsilon    lambda    exhausted")
    for method, details in plan["methods"].items():
        for b in details["budgets"]:
            print(f"{method:12} {b['target_epsilon']:7g} {b['steps']:8d} "
                  f"{b['epsilon']:20.8f} {b['next_epsilon']:15.8f} "
                  f"{b.get('lambda', '-'):>9} {b['budget_exhausted']}")
    if any(b["steps"] == 0 for details in plan["methods"].values() for b in details["budgets"]):
        print("NOTE: zero-step entries evaluate the initial untrained model; training_performed=false.")


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "summarize":
        from .reporting import summarize
        print(summarize(args.output_dir, allow_incomplete=args.allow_incomplete).to_string(index=False))
        return
    config = Config.load(args.config, split=args.split)
    methods = args.methods if args.command != "train" else [args.method]
    if len(set(methods)) != len(methods):
        raise ValueError("Methods must be unique")
    training = read_series(args.data_dir / "train.csv")
    plan = obtain_plan(config, training, args.output_dir, methods)
    print_budgets(plan)
    if args.command == "account":
        print(f"Saved plan {fingerprint(plan)}; no training or test access")
        return
    import torch
    from .training import run_experiment
    if args.num_threads < 1:
        raise ValueError("--num-threads must be positive")
    torch.set_num_threads(args.num_threads)
    test, labels, labels_hash = read_test(args.data_dir, training)
    test, labels, labels_hash = evaluation_subset(test, labels, labels_hash, config)
    if len(test.values) < config.segment_length:
        raise ValueError("Test sequence is shorter than segment_length")
    if args.command == "train":
        jobs = [(args.method, args.epsilon, args.seed)]
    else:
        seeds = (list(range(config.seed_start, config.seed_start + config.num_seeds))
                 if args.seeds is None else args.seeds)
        if len(set(seeds)) != len(seeds) or any(
                not config.seed_start <= seed < config.seed_start + config.num_seeds for seed in seeds):
            raise ValueError("Seeds must be unique and within the configured range")
        jobs = [(method, epsilon, seed) for method in methods for epsilon in config.epsilons for seed in seeds]
    for method, epsilon, seed in jobs:
        run_experiment(config, plan, training, test, labels, labels_hash, args.output_dir,
                       method, epsilon, seed, device=args.device, resume=args.resume,
                       save_scores=args.save_scores, progress=lambda message: print(message, flush=True))
    if args.command == "sweep":
        from .reporting import summarize

        print(summarize(args.output_dir, allow_incomplete=True).to_string(index=False))


if __name__ == "__main__":
    main()
