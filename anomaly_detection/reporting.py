import json
from pathlib import Path

import numpy as np
import pandas as pd

from .io_utils import atomic_json, fingerprint


def summarize(output_dir, *, allow_incomplete=False):
    root = Path(output_dir)
    plan = json.loads((root / "privacy_plan.json").read_text())
    config, plan_id = plan["protocol"], fingerprint(plan)
    expected = {(method, epsilon, seed) for method in plan["methods"]
                for epsilon in config["epsilons"]
                for seed in range(config["seed_start"], config["seed_start"] + config["num_seeds"])}
    seen, records, test_ids, environments = set(), [], set(), set()
    for path in sorted(root.glob("*/epsilon_*/seed_*/result.json")):
        result = json.loads(path.read_text())
        key = (result["method"], result["epsilon"], result["seed"])
        if result["plan_id"] != plan_id or result["status"] != "complete":
            raise ValueError(f"Incompatible or unfinished result: {path}")
        if key not in expected or key in seen:
            raise ValueError(f"Unexpected or duplicate result: {path}")
        budget = next(b for b in plan["methods"][key[0]]["budgets"] if b["target_epsilon"] == key[1])
        if result["steps"] != budget["steps"] or result["budget"] != budget:
            raise ValueError(f"Result did not follow the accounted budget: {path}")
        seen.add(key)
        test_ids.add(fingerprint([result["test"], result["labels_sha256"]]))
        environments.add(fingerprint(result["environment"]))
        records.append({"method": key[0], "epsilon": key[1], "seed": key[2],
                        "steps": result["steps"], "accounted_epsilon": budget["epsilon"],
                        "training_performed": result["steps"] > 0,
                        "budget_exhausted": budget["budget_exhausted"],
                        "lambda": budget.get("lambda", np.nan), **result["metrics"]})
    missing = sorted(expected - seen)
    if not records:
        raise ValueError("No completed runs to summarize")
    if len(test_ids) != 1:
        raise ValueError("Cannot aggregate results evaluated on different test data")
    if len(environments) != 1:
        raise ValueError("Cannot aggregate different software/device environments in one experiment")
    if missing and not allow_incomplete:
        raise ValueError(f"Missing {len(missing)} of {len(expected)} runs; use --allow-incomplete only for development")
    runs = pd.DataFrame(records).sort_values(["method", "epsilon", "seed"])
    metrics = ["auprc_ap", "auroc"] + (["f1"] if config["f1_threshold"] is not None else [])
    rows = []
    for (method, epsilon), group in runs.groupby(["method", "epsilon"]):
        row = {"method": method, "epsilon": epsilon, "n": len(group),
               "expected_n": config["num_seeds"], "complete": len(group) == config["num_seeds"],
               "steps": int(group.steps.iloc[0]), "training_performed": bool(group.steps.iloc[0] > 0),
               "budget_exhausted": bool(group.budget_exhausted.iloc[0])}
        for metric in metrics:
            if not np.isfinite(group[metric]).all():
                raise ValueError(f"Non-finite metric: {metric}")
            row[metric + "_mean"] = group[metric].mean()
            row[metric + "_std"] = group[metric].std(ddof=1)
        rows.append(row)
    summary = pd.DataFrame(rows)
    runs.to_csv(root / "runs.csv", index=False)
    summary.to_csv(root / "summary.csv", index=False)
    atomic_json(root / "completion.json", {"expected": len(expected), "completed": len(seen),
                                          "complete": not missing, "missing": missing})
    plot_summary(summary, metrics, root, incomplete=bool(missing))
    return summary


def plot_summary(frame, metrics, output_dir, *, incomplete):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = {"poisson": "Poisson", "structured": "Structured", "pss": "Poisson spaced"}
    labels = {"auprc_ap": "AUPRC (average precision)", "auroc": "AUROC", "f1": "F1 (fixed threshold)"}
    for metric in metrics:
        fig, axis = plt.subplots(figsize=(5.6, 3.8), constrained_layout=True)
        for method, group in frame.groupby("method"):
            group = group.sort_values("epsilon")
            axis.errorbar(group.epsilon, group[metric + "_mean"],
                          yerr=group[metric + "_std"].fillna(0), marker="o", capsize=3,
                          label=names[method])
        axis.set(xlabel="Privacy budget ε", ylabel=labels[metric], ylim=(0, 1))
        if incomplete:
            axis.set_title("Incomplete development results")
        axis.grid(alpha=0.25)
        axis.legend()
        for suffix in ("png", "pdf"):
            fig.savefig(Path(output_dir) / f"{metric}.{suffix}", dpi=180)
        plt.close(fig)
